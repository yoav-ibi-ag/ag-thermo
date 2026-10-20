"""Robustness check on the within-target nanobody stability ranker.

The headline rho~0.45 comes from 77 labels across 8 targets, so it needs error
bars before anyone designs against it. Three checks:

1. Bootstrap CI  - resample binders within each target, recompute the LOO
   within-target Spearman from the already-computed LOO scores.
2. Permutation null - reshuffle Tm within each target and refit the whole LOO
   pipeline, giving an empirical p-value that is immune to the pairwise
   ranker's optimism.
3. Leave-one-target-out - refit excluding each target in turn and score that
   held-out target, i.e. does the ranker transfer to an unseen target.

Output: data/tm_predictor/rank_bakeoff/robustness_{metrics.json,.png}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tm_rank_bakeoff import (FEATURE_COLS, STRUCT_COLS, esmc6b_matrix, load_master,
                             npz_matrix, pairwise_rank_loo, within_target)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/tm_predictor/rank_bakeoff"
N_BOOT = 2000
N_PERM = 300
MIN_N = 5
SEED = 0


def build_reps(df, names):
    esmc600 = npz_matrix(ROOT / "data/tm_predictor/inhouse_esmc_600m_tagstripped.npz", names)
    nm = df[["nanomelt_tm"]].to_numpy(float)
    bio = df[FEATURE_COLS].to_numpy(float)
    st = df[STRUCT_COLS].to_numpy(float)
    return {
        "esmc600": (esmc600, 16),
        "esmc600+struct": (np.c_[esmc600, st], 20),
        "esmc600+nanomelt+biophys+struct": (np.c_[esmc600, nm, bio, st], 20),
        "esmc6b+nanomelt+biophys": (np.c_[esmc6b_matrix(names), nm, bio], 20),
        "struct": (st, None),
    }


def boot_ci(y, s, tg, rng):
    """Resample binders within target; CI on the n-weighted within-target rho."""
    vals = []
    groups = {t: np.where(tg == t)[0] for t in np.unique(tg)}
    for _ in range(N_BOOT):
        num = den = 0.0
        for t, idx in groups.items():
            if len(idx) < MIN_N:
                continue
            pick = rng.choice(idx, size=len(idx), replace=True)
            if np.std(s[pick]) == 0 or np.std(y[pick]) == 0:
                continue
            rho = spearmanr(y[pick], s[pick]).statistic
            if np.isnan(rho):
                continue
            num += rho * len(idx)
            den += len(idx)
        if den:
            vals.append(num / den)
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(v.std())


def perm_null(X, y, tg, pca_k, observed, rng):
    """Refit the full LOO pipeline on Tm shuffled within target."""
    null = []
    for _ in range(N_PERM):
        yp = y.copy()
        for t in np.unique(tg):
            idx = np.where(tg == t)[0]
            yp[idx] = rng.permutation(y[idx])
        s = pairwise_rank_loo(X, yp, tg, pca_k=pca_k)
        _, nw = within_target(yp, s, tg)
        if nw is not None:
            null.append(nw)
    null = np.array(null)
    p = float((np.sum(null >= observed) + 1) / (len(null) + 1))
    return {"null_mean": float(null.mean()), "null_sd": float(null.std()),
            "null_p95": float(np.percentile(null, 95)), "p_value": p,
            "n_perm": int(len(null))}


def loto(X, y, tg, pca_k):
    """Leave-one-target-out: fit on other targets, rank the held-out target."""
    import itertools

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    per = {}
    for t in np.unique(tg):
        te = np.where(tg == t)[0]
        if len(te) < MIN_N:
            continue
        tr = np.where(tg != t)[0]
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = np.nan_to_num(sc.transform(X[tr])), np.nan_to_num(sc.transform(X[te]))
        if pca_k is not None and X.shape[1] > pca_k:
            pca = PCA(n_components=min(pca_k, len(tr) - 1)).fit(Xtr)
            Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        D, L = [], []
        for tt in np.unique(tg[tr]):
            idx = np.where(tg[tr] == tt)[0]
            for a, b in itertools.combinations(idx, 2):
                if y[tr][a] == y[tr][b]:
                    continue
                d = Xtr[a] - Xtr[b]
                lab = 1 if y[tr][a] > y[tr][b] else 0
                D += [d, -d]
                L += [lab, 1 - lab]
        clf = LogisticRegression(C=1.0, fit_intercept=False, max_iter=2000).fit(
            np.array(D), np.array(L))
        s = Xte @ clf.coef_[0]
        per[str(t)] = {"n": int(len(te)), "spearman": float(spearmanr(y[te], s).statistic)}
    num = sum(v["spearman"] * v["n"] for v in per.values())
    den = sum(v["n"] for v in per.values())
    return per, (num / den if den else None)


def main():
    rng = np.random.default_rng(SEED)
    df = load_master()
    y = df["tm_C"].to_numpy(float)
    tg = df["target_id"].to_numpy()
    names = df["nb_name"].tolist()
    reps = build_reps(df, names)

    out = {"n_binders": int(len(df)), "n_targets": int(len(np.unique(tg))),
           "n_boot": N_BOOT, "n_perm": N_PERM, "reps": {}}

    for name, (X, k) in reps.items():
        s = pairwise_rank_loo(X, y, tg, pca_k=k)
        _, obs = within_target(y, s, tg)
        lo, hi, sd = boot_ci(y, s, tg, rng)
        lt_per, lt = loto(X, y, tg, k)
        out["reps"][name] = {
            "observed_rho": obs, "boot_ci95": [lo, hi], "boot_sd": sd,
            "loto_rho": lt, "loto_per_target": lt_per,
        }
        print(f"{name:34s} rho={obs:.3f}  CI95=[{lo:.3f},{hi:.3f}]  LOTO={lt:.3f}", flush=True)

    # Permutation null only for the headline representation (it is the expensive one).
    head = "esmc600"
    X, k = reps[head]
    out["permutation_null"] = {head: perm_null(X, y, tg, k, out["reps"][head]["observed_rho"], rng)}
    pn = out["permutation_null"][head]
    print(f"\npermutation null ({head}): mean={pn['null_mean']:.3f} sd={pn['null_sd']:.3f} "
          f"p95={pn['null_p95']:.3f}  p={pn['p_value']:.4f}")

    (OUT / "robustness_metrics.json").write_text(json.dumps(out, indent=2))
    _plot(out)


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reps = out["reps"]
    names = list(reps.keys())
    obs = [reps[n]["observed_rho"] for n in names]
    lo = [reps[n]["observed_rho"] - reps[n]["boot_ci95"][0] for n in names]
    hi = [reps[n]["boot_ci95"][1] - reps[n]["observed_rho"] for n in names]
    lt = [reps[n]["loto_rho"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    ypos = np.arange(len(names))
    ax.errorbar(obs, ypos, xerr=[lo, hi], fmt="o", color="#2c7fb8", capsize=4,
                label="within-target LOO (95% CI)")
    ax.scatter(lt, ypos, marker="s", color="#d95f0e", label="leave-one-target-out")
    pn = list(out["permutation_null"].values())[0]
    ax.axvline(pn["null_p95"], ls="--", color="gray", label=f"perm null p95 = {pn['null_p95']:.2f}")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlabel("within-target Spearman")
    ax.set_title(f"Ranker robustness (n={out['n_binders']} binders, {out['n_targets']} targets)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "robustness.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
