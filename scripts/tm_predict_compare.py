"""Head-to-head comparison of nanobody thermostability (Tm) predictors.

Goal (per Yoav): rank *new binders against already-characterized targets*. The
honest test for that is leave-one-binder-out WITHIN each target (the per-target
offset is known at prediction time). We therefore report, as the headline
metric, the n-weighted mean of per-target Spearman under leave-one-binder-out
(LOO). We also run leave-one-target-out (LOTO) as a harder "new target" stress
test.

Models compared
  baselines:     per-target-mean (ranking floor), raw NanoMelt
  recal:         global affine NanoMelt, within-target affine NanoMelt
  tabular:       Ridge / HistGBR on biophysical+SAP features (+NanoMelt)
  esm-c:         ESM-C 600M in-house linear probe
                 ESM-C 600M public-pretrained -> within-target transfer
  stack:         NanoMelt + ESM-C(public pred) + features (recommended)

Data is entirely local: in-house 77 campaign nanobodies (measured Tm + features
+ NanoMelt score), public 764 experimental Tm set, ESM-C 600M embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/tm_predictor"
OUT.mkdir(parents=True, exist_ok=True)

ALPHAS = np.logspace(-2, 4, 25)
MIN_N = 5  # min binders per target to report its within-target Spearman
FEATURE_COLS = [
    "sap_mean", "sap_sum", "sap_pos_sum", "sap_max",
    "surfscore_sum", "surfscore_pos_sum",
    "n_hyd_patches", "hyd_patch_sasa_max", "hyd_patch_sasa_sum", "hyd_patch_sasa_mean",
    "iptm", "sheet_pI", "sheet_instability", "sheet_aliphatic", "sheet_hydropathicity",
    "nb_len", "SAP_B_sap_score",
]


# ---------------------------------------------------------------- data loading
def load_inhouse() -> pd.DataFrame:
    tm = pd.read_csv(ROOT / "data/pep_patch/tm_joined.csv")
    preds = pd.read_csv(ROOT / "data/nanomelt_eval/predictions.csv")[
        ["pair_id", "nb_name", "vhh_seq", "nanomelt_tm"]
    ]
    df = tm.merge(preds, on="pair_id", how="inner", suffixes=("", "_p"))
    assert len(df) == len(tm), f"merge changed row count: {len(df)} vs {len(tm)}"
    df["tm_C"] = df["tm_C"].astype(float)
    return df


def load_inhouse_esmc() -> dict[str, np.ndarray]:
    z = np.load(OUT / "inhouse_esmc_600m_tagstripped.npz", allow_pickle=True)
    return {n: e for n, e in zip(z["nb_name"], z["emb"])}


def load_pptstab(nb_names) -> np.ndarray:
    """PPTStab (ProtBERT, Meltome-trained) Tm, aligned to nb_names. NaN if absent."""
    p = OUT / "pptstab_pred.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["seq_ID"] = df["seq_ID"].str.lstrip(">")
    m = dict(zip(df["seq_ID"], df["Tm(\u00b0C)"].astype(float)))
    san = lambda s: "".join(c if c.isalnum() else "_" for c in s)
    return np.array([m.get(san(n), np.nan) for n in nb_names], dtype=float)


def load_public() -> tuple[np.ndarray, np.ndarray]:
    pub = pd.read_csv(ROOT / "data/public_tm/public_nb_tm.csv")
    ids = pd.read_csv(ROOT / "embeddings_esmc_600m/public_tm/nb_embedding_ids.csv")
    emb = np.load(ROOT / "embeddings_esmc_600m/public_tm/nb_embeddings.npy")
    seq2emb = {s: emb[i] for i, s in enumerate(ids["nb_seq"])}
    keep = pub["nb_seq"].isin(seq2emb)
    pub = pub[keep].reset_index(drop=True)
    X = np.stack([seq2emb[s] for s in pub["nb_seq"]], axis=0).astype(np.float32)
    y = pub["tm_C"].to_numpy(float)
    return X, y


# ------------------------------------------------------------------- utilities
def target_mean_loo(y: np.ndarray, targets: np.ndarray, i: int) -> float:
    """Mean Tm of same-target binders excluding i."""
    m = (targets == targets[i])
    m[i] = False
    return float(y[m].mean()) if m.any() else float(y.mean())


def within_target_spearman(y_true, y_pred, targets) -> dict:
    """Per-target Spearman (n>=MIN_N) + n-weighted mean + pooled centered rho."""
    per = {}
    num = den = 0.0
    yc_t, yc_p = [], []
    for t in np.unique(targets):
        m = targets == t
        n = int(m.sum())
        yt, yp = y_true[m], y_pred[m]
        yc_t.append(yt - yt.mean())
        yc_p.append(yp - yp.mean())
        if n >= MIN_N and np.std(yp) > 0 and np.std(yt) > 0:
            rho = float(spearmanr(yt, yp).statistic)
            per[t] = {"n": n, "spearman": rho}
            num += rho * n
            den += n
    pooled = float(spearmanr(np.concatenate(yc_t), np.concatenate(yc_p)).statistic)
    return {
        "per_target": per,
        "n_weighted_mean_spearman": (num / den) if den else None,
        "pooled_centered_spearman": pooled,
    }


def global_metrics(y_true, y_pred) -> dict:
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "pearson": float(pearsonr(y_true, y_pred).statistic),
    }


def summarize(name, y_true, y_pred, targets) -> dict:
    return {"model": name, **within_target_spearman(y_true, y_pred, targets),
            "global": global_metrics(y_true, y_pred)}


# ------------------------------------------------------------- LOO predictors
def loo_predict(df, kind, esmc_inhouse, esmc_pub_pred):
    """Return leave-one-binder-out predictions (absolute Tm) for a model `kind`."""
    y = df["tm_C"].to_numpy(float)
    tg = df["target_id"].to_numpy()
    nm = df["nanomelt_tm"].to_numpy(float)
    feats = df[FEATURE_COLS].to_numpy(float)
    emb = np.stack([esmc_inhouse[n] for n in df["nb_name"]], axis=0)
    n = len(df)
    yhat = np.full(n, np.nan)

    for i in range(n):
        tr = np.arange(n) != i
        tmean = target_mean_loo(y, tg, i)

        if kind == "raw_nanomelt":
            yhat[i] = nm[i]
        elif kind == "target_mean":
            yhat[i] = tmean
        elif kind == "recal_global":
            lr = LinearRegression().fit(nm[tr, None], y[tr])
            yhat[i] = lr.predict(nm[i, None, None])[0]
        elif kind == "recal_within":
            # pooled within-target slope on train, applied around target mean
            tmeans = {t: y[tr][tg[tr] == t].mean() for t in np.unique(tg[tr])}
            nmeans = {t: nm[tr][tg[tr] == t].mean() for t in np.unique(tg[tr])}
            dy = np.array([y[j] - tmeans[tg[j]] for j in np.where(tr)[0]])
            dx = np.array([nm[j] - nmeans[tg[j]] for j in np.where(tr)[0]])
            slope = float(np.polyfit(dx, dy, 1)[0]) if np.std(dx) > 0 else 0.0
            yhat[i] = tmean + slope * (nm[i] - nmeans.get(tg[i], nm[tr].mean()))
        elif kind in ("feat_ridge", "feat_hgbt", "esmc_probe", "stack"):
            if kind == "feat_ridge":
                Xtr = np.c_[feats[tr], nm[tr]]
                Xte = np.concatenate([feats[i], [nm[i]]])[None, :]
                sc = StandardScaler().fit(Xtr)
                m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), _center(y, tg, tr))
                yhat[i] = tmean + m.predict(sc.transform(Xte))[0]
            elif kind == "feat_hgbt":
                Xtr = np.c_[feats[tr], nm[tr]]
                Xte = np.concatenate([feats[i], [nm[i]]])[None, :]
                m = HistGradientBoostingRegressor(
                    max_depth=3, max_iter=200, learning_rate=0.05,
                    min_samples_leaf=5, l2_regularization=1.0, random_state=0
                ).fit(Xtr, _center(y, tg, tr))
                yhat[i] = tmean + m.predict(Xte)[0]
            elif kind == "esmc_probe":
                k = min(16, tr.sum() - 1, emb.shape[1])
                pca = PCA(n_components=k).fit(emb[tr])
                Ztr, Zte = pca.transform(emb[tr]), pca.transform(emb[i][None])
                sc = StandardScaler().fit(Ztr)
                m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Ztr), _center(y, tg, tr))
                yhat[i] = tmean + m.predict(sc.transform(Zte))[0]
            elif kind == "stack":
                k = min(8, tr.sum() - 1, emb.shape[1])
                pca = PCA(n_components=k).fit(emb[tr])
                def feat_block(idx, mask=None):
                    z = pca.transform(emb[idx])
                    base = np.c_[nm[idx], esmc_pub_pred[idx],
                                 feats[idx][:, :6], z]
                    return base
                Xtr = feat_block(tr)
                Xte = feat_block(np.array([i]))
                sc = StandardScaler().fit(Xtr)
                m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), _center(y, tg, tr))
                yhat[i] = tmean + m.predict(sc.transform(Xte))[0]
        elif kind == "esmc_transfer":
            # public-pretrained prediction recalibrated within target (LOO)
            tmeans = {t: y[tr][tg[tr] == t].mean() for t in np.unique(tg[tr])}
            pmeans = {t: esmc_pub_pred[tr][tg[tr] == t].mean() for t in np.unique(tg[tr])}
            dy = np.array([y[j] - tmeans[tg[j]] for j in np.where(tr)[0]])
            dx = np.array([esmc_pub_pred[j] - pmeans[tg[j]] for j in np.where(tr)[0]])
            slope = float(np.polyfit(dx, dy, 1)[0]) if np.std(dx) > 0 else 0.0
            yhat[i] = tmean + slope * (esmc_pub_pred[i] - pmeans.get(tg[i], esmc_pub_pred[tr].mean()))
        else:
            raise ValueError(kind)
    return yhat


def _center(y, tg, tr):
    """Target-mean-centered training targets (train rows only)."""
    tmeans = {t: y[tr][tg[tr] == t].mean() for t in np.unique(tg[tr])}
    return np.array([y[j] - tmeans[tg[j]] for j in np.where(tr)[0]])


# ------------------------------------------------- leave-one-target-out stress
def loto(df, esmc_pub_pred):
    """New-target stress test: predict absolute Tm for a held-out target's
    binders (its mean is unknown) and measure within-target Spearman."""
    y = df["tm_C"].to_numpy(float)
    tg = df["target_id"].to_numpy()
    nm = df["nanomelt_tm"].to_numpy(float)
    feats = df[FEATURE_COLS].to_numpy(float)
    results = {m: {} for m in ["raw_nanomelt", "recal_global", "feat_abs",
                               "esmc_transfer_raw", "stack_abs"]}
    for t in np.unique(tg):
        te = tg == t
        tr = ~te
        if te.sum() < MIN_N:
            continue
        yt = y[te]
        preds = {
            "raw_nanomelt": nm[te],
            "recal_global": LinearRegression().fit(nm[tr, None], y[tr]).predict(nm[te, None]),
            "esmc_transfer_raw": esmc_pub_pred[te],
        }
        sc = StandardScaler().fit(feats[tr])
        preds["feat_abs"] = RidgeCV(alphas=ALPHAS).fit(
            sc.transform(feats[tr]), y[tr]).predict(sc.transform(feats[te]))
        Xtr = np.c_[nm[tr], esmc_pub_pred[tr], feats[tr]]
        Xte = np.c_[nm[te], esmc_pub_pred[te], feats[te]]
        scs = StandardScaler().fit(Xtr)
        preds["stack_abs"] = RidgeCV(alphas=ALPHAS).fit(
            scs.transform(Xtr), y[tr]).predict(scs.transform(Xte))
        for m, p in preds.items():
            if np.std(p) > 0 and np.std(yt) > 0:
                results[m][t] = {"n": int(te.sum()),
                                 "spearman": float(spearmanr(yt, p).statistic)}
    summ = {}
    for m, d in results.items():
        if d:
            num = sum(v["spearman"] * v["n"] for v in d.values())
            den = sum(v["n"] for v in d.values())
            summ[m] = {"per_target": d, "n_weighted_mean_spearman": num / den}
    return summ


# ---------------------------------------------------------------------- driver
def main() -> None:
    df = load_inhouse()
    esmc_inhouse = load_inhouse_esmc()
    missing = [n for n in df["nb_name"] if n not in esmc_inhouse]
    assert not missing, f"missing ESM-C embeddings for: {missing[:5]}"

    # Public-pretrained ESM-C regressor (fit ONCE; public is independent of in-house)
    Xpub, ypub = load_public()
    sc = StandardScaler().fit(Xpub)
    pca = PCA(n_components=32).fit(sc.transform(Xpub))
    pub_model = RidgeCV(alphas=ALPHAS).fit(pca.transform(sc.transform(Xpub)), ypub)
    emb_ih = np.stack([esmc_inhouse[n] for n in df["nb_name"]], axis=0)
    esmc_pub_pred = pub_model.predict(pca.transform(sc.transform(emb_ih)))

    y = df["tm_C"].to_numpy(float)
    tg = df["target_id"].to_numpy()

    models = ["target_mean", "raw_nanomelt", "recal_global", "recal_within",
              "feat_ridge", "feat_hgbt", "esmc_probe", "esmc_transfer", "stack"]
    loo_preds = {}
    summaries = []
    for kind in models:
        yhat = loo_predict(df, kind, esmc_inhouse, esmc_pub_pred)
        loo_preds[kind] = yhat
        summaries.append(summarize(kind, y, yhat, tg))

    # add raw public-pretrained ESM-C (no in-house adaptation) for reference
    loo_preds["esmc_public_raw"] = esmc_pub_pred
    summaries.append(summarize("esmc_public_raw", y, esmc_pub_pred, tg))

    # add PPTStab (external, Meltome-trained, zero-shot) if available
    pptstab = load_pptstab(df["nb_name"].tolist())
    if pptstab is not None and not np.isnan(pptstab).any():
        loo_preds["pptstab_raw"] = pptstab
        summaries.append(summarize("pptstab_raw", y, pptstab, tg))
        report_pptstab_spread = {
            "min": float(pptstab.min()), "max": float(pptstab.max()),
            "std": float(pptstab.std()),
        }
    else:
        report_pptstab_spread = None

    loto_summ = loto(df, esmc_pub_pred)

    # ---- persist
    pred_df = df[["pair_id", "nb_name", "target_id", "tm_C", "nanomelt_tm"]].copy()
    for k, v in loo_preds.items():
        pred_df[f"pred_{k}"] = v
    pred_df.to_csv(OUT / "loo_predictions.csv", index=False)

    report = {
        "n_inhouse": int(len(df)),
        "n_public": int(len(ypub)),
        "tm_range": [float(y.min()), float(y.max())],
        "targets": {t: int((tg == t).sum()) for t in np.unique(tg)},
        "headline_metric": "leave-one-binder-out within-target Spearman (n-weighted mean)",
        "pptstab_tm_spread": report_pptstab_spread,
        "loo": sorted(summaries,
                      key=lambda s: (s["n_weighted_mean_spearman"] or -9), reverse=True),
        "loto_new_target": loto_summ,
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({s["model"]: round(s["n_weighted_mean_spearman"] or float("nan"), 3)
                      for s in report["loo"]}, indent=2))
    print("\nLOTO (new-target) n-weighted Spearman:")
    print(json.dumps({m: round(d["n_weighted_mean_spearman"], 3)
                      for m, d in loto_summ.items()}, indent=2))

    _plots(report, pred_df, loo_preds, y, tg)


def _plots(report, pred_df, loo_preds, y, tg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loo = report["loo"]
    names = [s["model"] for s in loo]
    vals = [s["n_weighted_mean_spearman"] or 0 for s in loo]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2c7fb8" if v >= 0 else "#d95f0e" for v in vals]
    ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("within-target Spearman (LOO, n-weighted mean)")
    ax.set_title("Tm ranking of new binders vs known targets")
    fig.tight_layout()
    fig.savefig(OUT / "within_target_spearman_bar.png", dpi=150)
    plt.close(fig)

    best = loo[0]["model"]
    fig, ax = plt.subplots(figsize=(6, 6))
    for t in np.unique(tg):
        m = tg == t
        ax.scatter(y[m], loo_preds[best][m], s=28, label=t.split("__")[-1][:14], alpha=0.8)
    lims = [min(y.min(), np.nanmin(loo_preds[best])), max(y.max(), np.nanmax(loo_preds[best]))]
    ax.plot(lims, lims, "k--", lw=0.7)
    ax.set_xlabel("measured Tm (C)")
    ax.set_ylabel(f"predicted Tm (C) — {best}")
    ax.set_title(f"Best model: {best}")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "best_model_scatter.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved figures + metrics to {OUT}")


if __name__ == "__main__":
    main()
