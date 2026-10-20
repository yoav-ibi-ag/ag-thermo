"""Learning-curve / power analysis for a target-specific nanobody Tm predictor.

Answers, from the current 77-label data, three Phase-0 questions:

1. CEILING: given the per-target Tm spread and the assay measurement error,
   what within-target Spearman can ANY predictor reach? (noise+range cap)
2. POWER: how many binders per target are needed to (a) reliably estimate a
   within-target ranking, (b) show a model beats the zero-shot baseline rho0?
3. LEARNING CURVE: how does the zero-shot within-target rho estimate (and its
   CI) behave as n per target grows? (bootstrap + Fisher-z projection)

Outputs figures + a markdown summary under data/tm_predictor/power/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/tm_predictor/power"
OUT.mkdir(parents=True, exist_ok=True)

RHO0 = 0.214          # zero-shot NanoMelt within-target baseline (from comparison)
ALPHA = 0.05
MIN_N = 5
Z_A2 = norm.ppf(1 - ALPHA / 2)
Z_A1 = norm.ppf(1 - ALPHA)


def fisher_ci(rho, n):
    if n <= 3:
        return (np.nan, np.nan)
    z, se = np.arctanh(rho), 1 / np.sqrt(n - 3)
    return np.tanh(z - Z_A2 * se), np.tanh(z + Z_A2 * se)


def n_to_detect(rho1, rho0=0.0, power=0.8, one_sided=False):
    """Binders needed for a within-target rho1 to be distinguished from rho0."""
    z1, z0 = np.arctanh(rho1), np.arctanh(rho0)
    if z1 == z0:
        return np.inf
    za = Z_A1 if one_sided else Z_A2
    zb = norm.ppf(power)
    return 3 + ((za + zb) / (z1 - z0)) ** 2


def main():
    df = pd.read_csv(ROOT / "data/tm_predictor/loo_predictions.csv")
    tg = df["target_id"].to_numpy()
    y = df["tm_C"].to_numpy(float)
    nm = df["pred_raw_nanomelt"].to_numpy(float)

    targets = [t for t in np.unique(tg) if (tg == t).sum() >= MIN_N]
    rep = {"baseline_rho0": RHO0, "assumptions": {}, "per_target": {}, "ceiling": {},
           "power": {}, "learning_curve": {}}

    # ---- per-target observed Tm spread ------------------------------------
    for t in targets:
        m = tg == t
        rep["per_target"][t] = {
            "n": int(m.sum()),
            "tm_sd": float(np.std(y[m], ddof=1)),
            "tm_range": [float(y[m].min()), float(y[m].max())],
            "nanomelt_within_rho": float(spearmanr(y[m], nm[m]).statistic),
        }
    pooled_sd = float(np.sqrt(np.mean([rep["per_target"][t]["tm_sd"] ** 2 for t in targets])))
    rep["assumptions"]["pooled_within_target_tm_sd"] = pooled_sd

    # ---- 1. CEILING: rho_max = sqrt(1 - sigma_err^2 / sigma_obs^2) ---------
    # applied at the pooled within-target Tm SD, for a grid of assay errors
    for serr in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        r = max(0.0, 1 - (serr ** 2) / (pooled_sd ** 2))
        rep["ceiling"][f"assay_err_{serr}C"] = {
            "reliability": r, "rho_ceiling": float(np.sqrt(r))}
    # how much WIDER the Tm spread must be to reach a target ceiling at a fixed err
    serr_ref = 1.5
    widen = {}
    for rho_goal in [0.5, 0.6, 0.7, 0.8]:
        # rho_goal = sqrt(1 - serr^2/sd^2)  ->  sd = serr / sqrt(1 - rho_goal^2)
        sd_needed = serr_ref / np.sqrt(1 - rho_goal ** 2)
        widen[f"rho_{rho_goal}"] = {
            "required_within_target_tm_sd": float(sd_needed),
            "fold_vs_current": float(sd_needed / pooled_sd)}
    rep["ceiling"]["widen_range_at_1.5C_err"] = widen

    # ---- 2. POWER: n per target to detect / beat baseline -----------------
    rep["power"]["n_to_detect_vs_zero_80pct"] = {
        f"rho_{r}": round(n_to_detect(r, 0.0, 0.8), 1)
        for r in [0.3, 0.4, 0.5, 0.6, 0.7]}
    rep["power"]["n_to_beat_baseline_rho0_80pct"] = {
        f"rho_{r}": round(n_to_detect(r, RHO0, 0.8, one_sided=True), 1)
        for r in [0.4, 0.5, 0.6, 0.7, 0.8]}
    # combined across T targets: total binders N with T targets -> SE=1/sqrt(N-3T)
    def n_total_to_beat(rho1, T, power=0.8):
        z1, z0 = np.arctanh(rho1), np.arctanh(RHO0)
        per = ((Z_A1 + norm.ppf(power)) / (z1 - z0)) ** 2
        return 3 * T + per  # total binders across T targets
    rep["power"]["total_binders_to_beat_baseline_80pct"] = {
        f"T{T}_targets": {f"rho_{r}": round(n_total_to_beat(r, T), 0)
                          for r in [0.4, 0.5, 0.6]}
        for T in [4, 6, 8]}

    # ---- 3. LEARNING CURVE: zero-shot rho estimate vs n (bootstrap) --------
    rng = np.random.default_rng(0)
    ns = [5, 8, 10, 12, 15, 20, 25, 30]
    lc = {}
    pooled = np.concatenate([
        np.c_[y[tg == t] - y[tg == t].mean(), nm[tg == t] - nm[tg == t].mean()]
        for t in targets])  # target-centered points
    yc, xc = pooled[:, 0], pooled[:, 1]
    for n in ns:
        rhos = []
        for _ in range(2000):
            idx = rng.integers(0, len(yc), size=n)
            if np.std(xc[idx]) > 0 and np.std(yc[idx]) > 0:
                rhos.append(spearmanr(yc[idx], xc[idx]).statistic)
        rhos = np.array(rhos)
        lc[n] = {"median_rho": float(np.median(rhos)),
                 "ci95": [float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))],
                 "p_rho_gt_0": float(np.mean(rhos > 0))}
    rep["learning_curve"]["zeroshot_bootstrap_centered"] = lc

    (OUT / "power_metrics.json").write_text(json.dumps(rep, indent=2))
    _plots(rep, lc, ns)
    _print_summary(rep, pooled_sd)


def _plots(rep, lc, ns):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (a) ceiling vs assay error
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    errs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ceil = [rep["ceiling"][f"assay_err_{e}C"]["rho_ceiling"] for e in errs]
    ax[0].plot(errs, ceil, "o-", color="#2c7fb8")
    ax[0].axhline(rep["baseline_rho0"], ls="--", color="gray",
                  label=f"zero-shot now ({rep['baseline_rho0']:.2f})")
    ax[0].set_xlabel("assay error sigma_err (C)")
    ax[0].set_ylabel("max achievable within-target rho")
    ax[0].set_title(f"Ceiling at current Tm spread (SD={rep['assumptions']['pooled_within_target_tm_sd']:.1f} C)")
    ax[0].set_ylim(0, 1); ax[0].legend(); ax[0].grid(alpha=0.3)

    # (b) learning curve
    med = [lc[n]["median_rho"] for n in ns]
    lo = [lc[n]["ci95"][0] for n in ns]
    hi = [lc[n]["ci95"][1] for n in ns]
    ax[1].plot(ns, med, "o-", color="#d95f0e")
    ax[1].fill_between(ns, lo, hi, alpha=0.2, color="#d95f0e")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axhline(rep["baseline_rho0"], ls="--", color="gray",
                  label=f"baseline {rep['baseline_rho0']:.2f}")
    ax[1].set_xlabel("binders measured per target (n)")
    ax[1].set_ylabel("within-target rho (95% CI)")
    ax[1].set_title("Estimate stability of zero-shot ranking vs n")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "power_ceiling_learningcurve.png", dpi=150)
    plt.close(fig)


def _print_summary(rep, pooled_sd):
    print(f"Pooled within-target Tm SD (current): {pooled_sd:.2f} C")
    print("\nCeiling rho by assay error (at current spread):")
    for e in [0.5, 1.0, 1.5, 2.0, 3.0]:
        c = rep["ceiling"][f"assay_err_{e}C"]["rho_ceiling"]
        print(f"  sigma_err={e} C -> rho_max={c:.2f}")
    print("\nn per target to DETECT a within-target rho (80% power, vs 0):")
    for k, v in rep["power"]["n_to_detect_vs_zero_80pct"].items():
        print(f"  {k}: n>={v}")
    print("\nn per target to BEAT zero-shot rho0=0.21 (80% power, one-sided):")
    for k, v in rep["power"]["n_to_beat_baseline_rho0_80pct"].items():
        print(f"  {k}: n>={v}")
    print(f"\nFigures + JSON -> {OUT}")


if __name__ == "__main__":
    main()
