# ag-thermo

Nanobody thermostability (Tm) prediction and ranking for in-house binder design.

The practical deliverable is a **pure-sequence stability ranker** (ESM-C 600M +
pairwise learning-to-rank) that scores candidate VHH *within a known target* to
prioritize them before synthesis. This repo also holds the full comparison of
predictors, a Phase-0 power analysis, and the structure-feature experiments.

## TL;DR findings

- **Zero-shot off-the-shelf predictors ceiling out at within-target Spearman ~0.21**
  (NanoMelt ≈ ESM-C). PPTStab is unusable (predicts a flat ~50 °C for all VHH).
- **Naively fine-tuning / regressing absolute Tm on the 77 labels makes ranking
  *worse*** (overfitting on a narrow 46–68 °C range, 6–19 binders/target).
- **Switching the objective to within-target ranking + using ESM-C embeddings
  roughly doubles ranking to ρ≈0.46** (LOO, n-weighted over 73 binders) — about
  halfway to the noise/range ceiling (~0.94) — and turns every target positive,
  including the three that were negative zero-shot. No new data required.
- Power analysis: the bottleneck is **model/representation + label count, not Tm
  range**. ~30 binders/target over 4–6 targets (one assay, error ≤1.5 °C) is the
  target for a reliable target-specific model.
- **Structure energetics (Rosetta FastRelax + ThermoMPNN site-saturation ddG) do
  not improve the ranker.** On their own they beat NanoMelt (ρ=0.24 vs 0.13), but
  added to ESM-C they change nothing within-target (0.438 → 0.444) and they
  *halve* transfer to an unseen target (LOTO 0.446 → 0.238). Production model
  stays **ESM-C 600M only**.
- Robustness: ρ=0.438 with bootstrap 95% CI [0.18, 0.63], permutation null mean
  −0.06 (p95 = 0.19), **p = 0.003**. Real signal, wide error bars — exactly the
  precision the power analysis predicted for n=77.

See `data/tm_predictor/report.md`, `.../rank_bakeoff/README.md`,
`.../rank_bakeoff/robustness_metrics.json`, and `.../power/README.md` for the
full write-ups and figures.

## Design levers

Univariate within-target Spearman against Tm (robust to the collinearity that
makes the multivariate coefficients unstable):

| feature | ρ | direction |
| --- | --- | --- |
| `SAP_B_sap_score` | −0.29 | lower surface aggregation propensity → more stable |
| `sheet_aliphatic` | −0.27 | lower aliphatic index → more stable |
| `sap_pos_sum` | −0.25 | lower positive-SAP surface → more stable |
| `rosetta_reu_per_res` | −0.24 | more negative relaxed energy/residue → more stable |
| `tmpnn_ddg_max` | +0.18 | (weak) |

So even though the energetic features do not help the ranker, Rosetta energy per
residue and SAP are the most actionable single knobs for designing toward
stability.

## The scorer

```bash
# train the artifact from the 77 in-house labels (sklearn only)
python scripts/tm_rank_score.py train

# score candidate VHH (FASTA or one-seq-per-line); embeds with ESM-C 600M
python scripts/tm_rank_score.py score -i candidates.fasta -o scored.csv \
    --target Helicoverpa_armigera__Chitin_deacetylases
```

Output: `stability_score`, `percentile_global`, and (if `--target` is a known
target) `percentile_in_target`. Expected within-target ranking quality ~0.44.
Model artifact: `data/tm_predictor/rank_model/`.

## Layout

```
scripts/
  tm_rank_score.py       # the deliverable: train/score CLI (ESM-C 600M ranker)
  tm_rank_bakeoff.py     # within-target learning-to-rank representation bake-off
  tm_predict_compare.py  # full predictor comparison (baselines/recal/ESM-C/stack/PPTStab)
  tm_power_analysis.py   # Phase-0 ceiling / power / learning-curve analysis
  tm_rank_robustness.py  # bootstrap CI + permutation null + leave-one-target-out
  tm_rosetta_features.py / relax_score_monomer.xml  # FastRelax monomer energy
  tm_thermompnn_features.py  # ThermoMPNN site-saturation ddG -> per-VHH summaries
  tm_embed_inhouse_esmc.py / tm_embed_nanobert.py  # embedding helpers
  02_embed_esmc.py       # ESM-C loader (local weights) reused for embedding
data/
  pep_patch/   tm_joined.csv (labels+features), features.csv, pdb/ (monomer structures)
  nanomelt_eval/  predictions.csv (NanoMelt zero-shot), metrics.json
  public_tm/   public experimental Tm (764) + NanoMelt datasets
  tm_predictor/  all derived outputs: metrics, reports, figures, model artifact
embeddings_esmc_600m/public_tm/  ESM-C 600M embeddings of the public 764
embeddings_esmc_6b/              ESM-C 6B embeddings of the in-house 77
```

## Environments

- Modeling (sklearn/scipy/pandas/matplotlib/joblib): any env, e.g. conda `esmfold2`.
- ESM-C embedding (`esm` + `accelerate` + `torch`): conda `esmfold2`; ESM-C weights
  loaded from local HF cache. GPU is much faster — run outside restrictive sandboxes.
- PPTStab comparison uses its own pinned env (`pptstab`); see its upstream repo.
- Rosetta features need `rosetta_scripts` on disk; Boltz PDBs use `HIE`/`HID`
  residue names that Rosetta rejects, so `tm_rosetta_features.clean_pdb` maps
  them back to `HIS` before scoring.
- ThermoMPNN is run from a local clone of the upstream repo (`analysis/` on
  `sys.path`); `esmfold2` already satisfies its deps (torch/omegaconf/biopython).
  Full 77-protein site-saturation takes ~2 min on one GPU.

Run compute-heavy embedding on GPU; the sklearn ranking/analysis is single-thread
friendly (`OMP_NUM_THREADS=1`).
