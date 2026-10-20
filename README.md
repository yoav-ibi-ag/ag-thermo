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

See `data/tm_predictor/report.md`, `.../rank_bakeoff/README.md`, and
`.../power/README.md` for the full write-ups and figures.

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

Run compute-heavy embedding on GPU; the sklearn ranking/analysis is single-thread
friendly (`OMP_NUM_THREADS=1`).
