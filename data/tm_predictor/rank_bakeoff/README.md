# Within-target learning-to-rank bake-off

Objective switched from absolute Tm regression to **within-target ranking**, via a
pooled pairwise logistic ranker (same-target pairs; target offset cancels).
Evaluation: leave-one-binder-out (LOO) within-target Spearman, n-weighted over the
6 targets with n>=5 (73 binders). Baseline zero-shot = 0.21; noise/range ceiling ~0.94.

## Result: the ranking head ~doubles zero-shot, and fixes the failing targets

| representation | within-target rho (LOO) |
|---|---|
| ESM-C 6B + NanoMelt + biophys | **0.46** |
| ESM-C 600M + NanoMelt + biophys | 0.45 |
| ESM-C 600M (alone) | 0.44 |
| ESM-C 6B (alone) | 0.39 |
| nanoBERT + NanoMelt + biophys | 0.21 |
| NanoMelt (as ranker) | 0.13 |
| nanoBERT (alone) | 0.08 |
| biophys (alone) | 0.03 |

ESM-C representations roughly double the zero-shot baseline (0.21 -> ~0.46),
about **halfway to the ceiling**. Crucially, the learned ranker turns **every**
target positive, including the three that were negative zero-shot:

| target | n | zero-shot NanoMelt | ESM-C 6B ranker | ESM-C 600M ranker |
|---|---|---|---|---|
| Bemisia HSP70 | 13 | +0.63 | +0.21 | +0.20 |
| Bemisia Trehalase | 13 | -0.12 | +0.36 | +0.56 |
| Helicoverpa CDA | 19 | +0.27 | +0.65 | +0.56 |
| Helicoverpa Chitin synthase | 8 | -0.25 | +0.46 | +0.74 |
| Helicoverpa Trehalase | 14 | +0.51 | +0.64 | +0.04 |
| Spodoptera CDA | 6 | -0.21 | +0.21 | +0.83 |

Takeaways:
- The signal that beats zero-shot lives in the **ESM-C embedding**; NanoMelt,
  nanoBERT, and biophysical features add little on their own.
- nanoBERT underperforms despite being nanobody-specific (small 320-d, 6-layer
  model; less informative than ESM-C here).
- ESM-C 6B and 600M are ~tied; 600M is cheaper and tag-stripped (cleaner), so it
  is the practical default.

## Caveats (honest)

- LOO with small per-target n => wide per-target CIs; the reliable number is the
  aggregate over 73 binders (~0.46). Needs **prospective** validation on a new
  measured batch (Phase 2).
- The biophysical-only ranker is weak in LOO (0.03), so the "design levers" below
  are **in-sample associations / hypotheses**, not validated predictors. The real
  predictive power is in ESM-C.

## Design levers (biophysical, standardized ranker coefficients; hypotheses)

Direction that associates with higher Tm:
- lower surface aggregation propensity (sap_mean, SAP_B_sap_score)
- smaller / less-exposed hydrophobic patches (hyd_patch_sasa_sum)
- lower instability index (sheet_instability)
- lower sheet aliphatic index; lower sheet pI
(surfscore_sum and iptm point positive)

These are directionally sensible (less surface hydrophobicity/aggregation = more
stable) but should be treated as leads, since biophys alone barely ranks.

## Practical use (design toward more stable binders)

Train the ESM-C 600M pairwise ranker on all 77 labels and use `w . x` as a
**stability score** to rank candidate VHH for a known target in silico, before
synthesis. Expected within-target ranking quality ~0.46 (vs 0.21 zero-shot).
Pair with the active-learning loop (Phase 2) to keep improving.

Artifacts: `rank_metrics.json`, `loo_rank_scores.csv`, `rank_bakeoff.png`.
