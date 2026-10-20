"""Within-target learning-to-rank bake-off for nanobody thermostability.

Goal: rank binders by stability WITHIN a target, to help design toward more
stable binders. We compare representations through one ranking head and evaluate
with leave-one-binder-out (LOO) within-target Spearman (headline), against the
zero-shot baseline (rho0=0.21) and the noise/range ceiling (~0.94).

Ranking head: pooled within-target PAIRWISE logistic ranking. For same-target
pairs (i,j) we learn w so that w.x scores more-stable binders higher; the target
offset cancels in x_i - x_j, so all targets share one ranker (uses all data).

Representations: NanoMelt score, biophysical/SAP features, ESM-C 600M, ESM-C 6B,
nanoBERT, and combinations.

Also emits design levers: standardized coefficients of the interpretable
biophysical ranker (which features push toward higher stability).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/tm_predictor/rank_bakeoff"
OUT.mkdir(parents=True, exist_ok=True)

RHO0 = 0.214
CEILING = 0.94
MIN_N = 5
FEATURE_COLS = [
    "sap_mean", "sap_sum", "sap_pos_sum", "sap_max",
    "surfscore_sum", "surfscore_pos_sum",
    "n_hyd_patches", "hyd_patch_sasa_max", "hyd_patch_sasa_sum", "hyd_patch_sasa_mean",
    "iptm", "sheet_pI", "sheet_instability", "sheet_aliphatic", "sheet_hydropathicity",
    "nb_len", "SAP_B_sap_score",
]
# Rosetta FastRelax monomer energy; ThermoMPNN site-saturation ddG landscape
# (positive ddG = destabilizing) collapsed to per-nanobody summaries.
ROSETTA_COLS = ["rosetta_total_reu", "rosetta_reu_per_res"]
TMPNN_COLS = [
    "tmpnn_ddg_mean", "tmpnn_ddg_std", "tmpnn_ddg_max", "tmpnn_ddg_min",
    "tmpnn_frac_destab", "tmpnn_site_worst_mean", "tmpnn_site_worst_sum",
]
STRUCT_COLS = ROSETTA_COLS + TMPNN_COLS


# --------------------------------------------------------------------- loaders
def load_master():
    tm = pd.read_csv(ROOT / "data/pep_patch/tm_joined.csv")
    preds = pd.read_csv(ROOT / "data/nanomelt_eval/predictions.csv")[
        ["pair_id", "nanomelt_tm"]]
    df = tm.merge(preds, on="pair_id", how="inner")
    assert len(df) == len(tm)
    for rel, cols in ((ROOT / "data/tm_predictor/rosetta_features.csv", ROSETTA_COLS),
                      (ROOT / "data/tm_predictor/thermompnn_features.csv", TMPNN_COLS)):
        if rel.exists():
            df = df.merge(pd.read_csv(rel)[["pair_id"] + cols], on="pair_id", how="left")
    return df.reset_index(drop=True)


def npz_matrix(path, nb_names):
    z = np.load(path, allow_pickle=True)
    m = {n: e for n, e in zip(z["nb_name"], z["emb"])}
    return np.stack([m[n] for n in nb_names]).astype(float)


def esmc6b_matrix(nb_names):
    ids = pd.read_csv(ROOT / "embeddings_esmc_6b/nb_embedding_ids.csv")
    E = np.load(ROOT / "embeddings_esmc_6b/nb_embeddings.npy")
    m = {n: E[i] for i, n in enumerate(ids["nb_name"])}
    return np.stack([m[n] for n in nb_names]).astype(float)


# ------------------------------------------------------------------ rank model
def pairwise_rank_loo(X, y, tg, pca_k=None, C=1.0):
    """LOO within-target scores from a pooled pairwise logistic ranker."""
    n = len(y)
    scores = np.full(n, np.nan)
    for h in range(n):
        tr = np.setdiff1d(np.arange(n), [h])
        sc = StandardScaler().fit(X[tr])
        Xtr, Xh = sc.transform(X[tr]), sc.transform(X[h][None])
        Xtr, Xh = np.nan_to_num(Xtr), np.nan_to_num(Xh)  # scaled mean-impute
        if pca_k is not None and X.shape[1] > pca_k:
            k = min(pca_k, len(tr) - 1)
            pca = PCA(n_components=k).fit(Xtr)
            Xtr, Xh = pca.transform(Xtr), pca.transform(Xh)
        # build within-target pairs among training rows
        D, L = [], []
        for t in np.unique(tg[tr]):
            idx = np.where(tg[tr] == t)[0]
            for a, b in itertools.combinations(idx, 2):
                if y[tr][a] == y[tr][b]:
                    continue
                d = Xtr[a] - Xtr[b]
                lab = 1 if y[tr][a] > y[tr][b] else 0
                D.append(d); L.append(lab)
                D.append(-d); L.append(1 - lab)
        D, L = np.array(D), np.array(L)
        if len(np.unique(L)) < 2:
            scores[h] = 0.0
            continue
        clf = LogisticRegression(C=C, fit_intercept=False, max_iter=2000).fit(D, L)
        w = clf.coef_[0]
        scores[h] = float(Xh[0] @ w)
    return scores


def within_target(y, s, tg):
    per, num, den = {}, 0.0, 0.0
    for t in np.unique(tg):
        m = tg == t
        if m.sum() >= MIN_N and np.std(s[m]) > 0:
            rho = float(spearmanr(y[m], s[m]).statistic)
            per[t] = {"n": int(m.sum()), "spearman": rho}
            num += rho * m.sum(); den += m.sum()
    return per, (num / den if den else None)


# ---------------------------------------------------------------------- driver
def main():
    df = load_master()
    y = df["tm_C"].to_numpy(float)
    tg = df["target_id"].to_numpy()
    names = df["nb_name"].tolist()

    reps = {
        "nanomelt": (df[["nanomelt_tm"]].to_numpy(float), None),
        "biophys": (df[FEATURE_COLS].to_numpy(float), None),
        "esmc600": (npz_matrix(ROOT / "data/tm_predictor/inhouse_esmc_600m_tagstripped.npz", names), 16),
        "esmc6b": (esmc6b_matrix(names), 16),
        "nanobert": (npz_matrix(ROOT / "data/tm_predictor/inhouse_nanobert.npz", names), 12),
        "rosetta": (df[ROSETTA_COLS].to_numpy(float), None),
        "thermompnn": (df[TMPNN_COLS].to_numpy(float), None),
        "struct": (df[STRUCT_COLS].to_numpy(float), None),
    }
    # combinations (concatenate standardized blocks is handled inside via StandardScaler)
    combos = {
        "esmc600+nanomelt+biophys": np.c_[reps["esmc600"][0], df[["nanomelt_tm"]].to_numpy(float), df[FEATURE_COLS].to_numpy(float)],
        "nanobert+nanomelt+biophys": np.c_[reps["nanobert"][0], df[["nanomelt_tm"]].to_numpy(float), df[FEATURE_COLS].to_numpy(float)],
        "esmc6b+nanomelt+biophys": np.c_[reps["esmc6b"][0], df[["nanomelt_tm"]].to_numpy(float), df[FEATURE_COLS].to_numpy(float)],
        "esmc600+struct": np.c_[reps["esmc600"][0], df[STRUCT_COLS].to_numpy(float)],
        "biophys+struct": np.c_[df[FEATURE_COLS].to_numpy(float), df[STRUCT_COLS].to_numpy(float)],
        "esmc600+nanomelt+biophys+struct": np.c_[
            reps["esmc600"][0], df[["nanomelt_tm"]].to_numpy(float),
            df[FEATURE_COLS].to_numpy(float), df[STRUCT_COLS].to_numpy(float)],
    }

    results = {}
    loo_scores = {}
    for name, (X, k) in reps.items():
        s = pairwise_rank_loo(X, y, tg, pca_k=k)
        per, nw = within_target(y, s, tg)
        results[name] = {"n_weighted_mean_spearman": nw, "per_target": per}
        loo_scores[name] = s
    for name, X in combos.items():
        k = 20 if "esmc" in name else 16
        s = pairwise_rank_loo(X, y, tg, pca_k=k)
        per, nw = within_target(y, s, tg)
        results[name] = {"n_weighted_mean_spearman": nw, "per_target": per}
        loo_scores[name] = s

    # ----- design levers: interpretable biophysical ranker on ALL data -------
    lever_cols = FEATURE_COLS + STRUCT_COLS
    Xl = df[lever_cols].to_numpy(float)
    Xs = np.nan_to_num(StandardScaler().fit_transform(Xl))
    D, L = [], []
    for t in np.unique(tg):
        idx = np.where(tg == t)[0]
        for a, b in itertools.combinations(idx, 2):
            if y[a] == y[b]:
                continue
            d = Xs[a] - Xs[b]; lab = 1 if y[a] > y[b] else 0
            D.append(d); L.append(lab); D.append(-d); L.append(1 - lab)
    clf = LogisticRegression(C=1.0, fit_intercept=False, max_iter=5000).fit(np.array(D), np.array(L))
    levers = sorted(zip(lever_cols, clf.coef_[0].tolist()),
                    key=lambda kv: abs(kv[1]), reverse=True)

    # The multivariate lever coefficients are unstable because the ThermoMPNN
    # summaries are strongly collinear, so also report each feature on its own.
    uni = []
    for j, f in enumerate(lever_cols):
        _, nw = within_target(y, Xl[:, j], tg)
        if nw is not None:
            uni.append({"feature": f, "within_target_spearman": nw})
    uni.sort(key=lambda d: abs(d["within_target_spearman"]), reverse=True)

    report = {
        "baseline_zeroshot_rho": RHO0,
        "ceiling_rho": CEILING,
        "headline": "LOO within-target Spearman (n-weighted), pairwise ranker",
        "results": dict(sorted(results.items(),
                               key=lambda kv: (kv[1]["n_weighted_mean_spearman"] or -9),
                               reverse=True)),
        "design_levers_biophys_stdcoef": [
            {"feature": f, "coef": c, "direction": "higher->more stable" if c > 0 else "lower->more stable"}
            for f, c in levers],
        "univariate_within_target_spearman": uni,
    }
    (OUT / "rank_metrics.json").write_text(json.dumps(report, indent=2))

    sdf = df[["pair_id", "nb_name", "target_id", "tm_C"]].copy()
    for k2, v in loo_scores.items():
        sdf[f"score_{k2}"] = v
    sdf.to_csv(OUT / "loo_rank_scores.csv", index=False)

    _plots(report, levers)
    print(json.dumps({k: round(v["n_weighted_mean_spearman"] or float("nan"), 3)
                      for k, v in report["results"].items()}, indent=2))
    print("\nTop design levers (biophys, standardized):")
    for f, c in levers[:8]:
        print(f"  {f:22s} {c:+.3f}  ({'higher' if c>0 else 'lower'} -> more stable)")


def _plots(report, levers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = report["results"]
    names = list(res.keys())
    vals = [res[n]["n_weighted_mean_spearman"] or 0 for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#2c7fb8" if v >= 0 else "#d95f0e" for v in vals]
    ax[0].barh(names[::-1], vals[::-1], color=colors[::-1])
    ax[0].axvline(report["baseline_zeroshot_rho"], ls="--", color="gray", label="zero-shot 0.21")
    ax[0].axvline(report["ceiling_rho"], ls=":", color="green", label="ceiling ~0.94")
    ax[0].set_xlabel("within-target Spearman (LOO, pairwise ranker)")
    ax[0].set_title("Representation bake-off")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis="x")

    top = levers[:10][::-1]
    fnames = [f for f, _ in top]; coefs = [c for _, c in top]
    lc = ["#2c7fb8" if c > 0 else "#d95f0e" for c in coefs]
    ax[1].barh(fnames, coefs, color=lc)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_xlabel("std. ranker coef  (+ = higher value -> more stable)")
    ax[1].set_title("Design levers (biophysical)")
    ax[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "rank_bakeoff.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
