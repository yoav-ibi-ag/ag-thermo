# Phase-0 power / learning-curve analysis

Run on the 77 in-house labels (6 targets with n>=5). Baseline = zero-shot
NanoMelt within-target rho0 = 0.21.

## Key finding: measurement range is NOT the bottleneck; model skill + label count are

- Pooled within-target Tm **SD = 4.38 C** (per-target 2.1-6.9 C). With a realistic
  nanoDSF assay error of 1-1.5 C, the achievable within-target rho **ceiling is
  ~0.94-0.97**. Current predictors reach only 0.21 of that — so there is large
  headroom, and widening the Tm range is a minor lever.
- Exceptions: two low-spread targets (Spodoptera CDA SD 2.1 over 58-62 C;
  Helicoverpa Trehalase SD 2.9) sit closer to the noise floor, so assay error
  matters most there.

## Per-target spread and current zero-shot signal

| target | n | Tm SD | range | NanoMelt within-rho |
|---|---|---|---|---|
| Bemisia HSP70 | 13 | 3.5 | 51-64 | +0.63 |
| Bemisia Trehalase | 13 | 5.0 | 47-68 | -0.12 |
| Helicoverpa CDA | 19 | 4.2 | 52-67 | +0.27 |
| Helicoverpa Chitin synthase | 8 | 6.9 | 46-67 | -0.25 |
| Helicoverpa Trehalase | 14 | 2.9 | 50-58 | +0.51 |
| Spodoptera CDA | 6 | 2.1 | 58-62 | -0.21 |

## How many binders per target?

Binders/target for 80% power (Fisher-z):

| goal | n per target |
|---|---|
| detect within-target rho=0.5 (vs 0) | ~29 |
| detect rho=0.6 | ~19 |
| prove a model beats zero-shot 0.21, if model rho=0.6 | ~30 |
| prove it beats 0.21, if model rho=0.5 | ~59 |
| prove it beats 0.21, if model rho=0.7 | ~18 |
| prove it beats 0.21, if model rho=0.4 | ~148 (impractical) |

Pooling across targets (combined test) is cheaper than proving each target:

| model rho | total binders to beat baseline (4-8 targets) |
|---|---|
| 0.5 | ~68-80 |
| 0.6 | ~39-51 |

## Learning curve (zero-shot, bootstrap)

The zero-shot rho estimate is stable (~0.22) but its 95% CI only clears 0 around
**n~20-30 per target** (P(rho>0) = 0.83 at n=20, 0.88 at n=30). Below n~15 the
within-target ranking is statistically indistinguishable from noise — which is
why the current per-target numbers (n=6-19) are so scattered.

## Phase-0 targets (recommendation)

- **~30 binders/target on 4-6 priority targets** (~120-180 total). Enough to detect
  rho>=0.5 per target AND to pool evidence that a rho~0.5-0.6 model beats zero-shot.
- **One assay, error <=1.5 C, with replicates.** Protects the ceiling, critical for
  the low-SD targets.
- **Do not prioritize widening Tm range** generally (within-target spread already
  adequate); instead prioritize count + assay consistency. Only the two low-SD
  targets benefit from deliberately including stability-diverse designs.

Artifacts: `power_metrics.json`, `power_ceiling_learningcurve.png`.
