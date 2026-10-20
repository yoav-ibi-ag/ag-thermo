# Nanobody Tm predictor — head-to-head comparison

**Goal:** rank *new binders against already-characterized targets*.
**Headline metric:** leave-one-binder-out (LOO) within-target Spearman,
n-weighted mean over the 6 targets with n>=5 (73 of 77 binders).
**Data:** 77 in-house measured Tm (range 46-68 C, 8 targets), 764 public
experimental Tm, ESM-C 600M embeddings (1152-d), NanoMelt zero-shot score.

## Result: zero-shot wins, in-house training hurts

| model | within-target rho (LOO) | global MAE (C) | note |
|---|---|---|---|
| raw NanoMelt (zero-shot) | **0.214** | 9.4 | best ranking; +9 C bias |
| ESM-C 600M, public-pretrained (zero-shot) | **0.193** | 8.5 | ~ties NanoMelt |
| PPTStab (ProtBERT, Meltome; zero-shot) | 0.017 | 7.9 | predictions nearly flat (~50 C) |
| ESM-C in-house linear probe | 0.085 | 3.1 | overfit |
| NanoMelt global affine recal | 0.030 | 3.9 | fixes bias, not ranking |
| stack (NanoMelt+ESM-C+features) | -0.152 | 3.3 | overfit |
| feature HistGBR | -0.202 | 3.9 | overfit |
| NanoMelt within-target recal | -0.320 | 3.2 | overfit |
| feature Ridge | -0.687 | 3.3 | overfit |
| ESM-C within-target transfer | -0.941 | 3.4 | overfit |
| per-target-mean (floor) | -1.0 | 3.3 | no ranking by construction |

Two things stand out:

1. **Nothing trained on the 77 in-house labels beats an off-the-shelf
   zero-shot predictor.** Every model that fits a slope/weights on the
   in-house data ranks *worse* under LOO, often strongly negative. This is
   overfitting: ~6-19 points/target over a 22 C range is too little to learn a
   generalizing map.
2. **Low MAE is a trap here.** The learned models cut MAE from 9.4 to ~3 C, but
   only by regressing to the per-target mean (the dummy per-target-mean already
   gives MAE 3.3 C). That destroys within-target ranking, which is the thing you
   actually need.

## The usable signal is target-specific

Per-target within-target Spearman (raw NanoMelt / ESM-C-public):

| target | n | NanoMelt | ESM-C |
|---|---|---|---|
| Bemisia HSP70 | 13 | **0.63** | 0.52 |
| Helicoverpa Trehalase | 14 | **0.51** | 0.41 |
| Spodoptera CDA | 6 | -0.21 | 0.41 |
| Helicoverpa CDA | 19 | 0.27 | 0.14 |
| Bemisia Trehalase | 13 | -0.12 | 0.17 |
| Helicoverpa Chitin synthase | 8 | -0.25 | -0.71 |

So the zero-shot handle works on some targets (HSP70, Helicoverpa Trehalase) and
not others. Per-target n is small, so every rho has a wide CI (+/- ~0.4-0.6);
treat the ordering qualitatively, not as precise numbers.

## New-target generalization (LOTO)

Only zero-shot carries over to an unseen target: NanoMelt 0.21, ESM-C 0.19;
models trained on the other targets collapse to ~0.

## PPTStab (external comparator)

PPTStab (ProtBERT embeddings + ANN/MLP ensemble, trained on the Meltome Atlas of
whole-proteome Tm) was run zero-shot on the 77 VHH (standalone, EMB method, -d 2).
Its predictions span only **48.3-51.5 C (stdev 0.82)** — essentially a constant
~50 C for every nanobody. Consequence:

- within-target ranking rho = **0.017** (noise); per-target rho scattered
  -0.19..0.41 with no consistent signal.
- global rho 0.22 is a between-target composition artifact, not usable ranking;
  MAE 7.9 C (it centers ~50 C, below your median 58 C).

Why: Meltome Tm is dominated by gross composition/length that separates thermophile
from mesophile proteins. Your 77 nanobodies are near-identical ~125-residue VHH
scaffolds differing only in CDR loops — exactly the variation PPTStab cannot see,
so it collapses them to one value. NanoMelt / ESM-C (nanobody / protein-LM
representations) at least capture some CDR-level signal. **PPTStab adds nothing
here and should be dropped from the candidate set.**

## ESM-C vs ESM-2

ESM-C 600M (public-pretrained) essentially ties NanoMelt (which is ESM-1b/ESM-2
based): 0.19 vs 0.21. The backbone was never the bottleneck — the 77-label,
narrow-range dataset is. ESM-C is a fine license-flexible in-house alternative,
but it does not unlock better ranking here.

## Recommendation

- **Do not fine-tune / train a target-specific Tm model on the current 77
  labels.** It measurably hurts ranking.
- **Use zero-shot NanoMelt (or ESM-C) as the handle now**, applied only where it
  has shown signal (HSP70, Helicoverpa Trehalase; partial on Helicoverpa CDA).
  Flag targets where it is unreliable (Bemisia Trehalase, chitin synthase).
- **Fine-tuning becomes worth revisiting once labels grow** — wider Tm range and
  ~30-50+ binders per target, measured by a single assay. At that point the
  within-target recal / stack design here should start to beat zero-shot.

Artifacts: `metrics.json`, `loo_predictions.csv`,
`within_target_spearman_bar.png`, `best_model_scatter.png`.
