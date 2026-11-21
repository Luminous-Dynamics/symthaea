# RCAF synthetic null, dose-response and estimator-surface results

**Run date:** 2026-11-21  
**Repository:** `Luminous-Dynamics/symthaea`  
**PR:** [#13041](https://github.com/Luminous-Dynamics/symthaea/pull/13041)  
**Exact checked-out source commit:** `335d7430246ff01f013b96a934601de256f3ea23`  
**Focused workflow run:** `44899300679` — passed  
**Artifact:** `rcaf-simulation-evidence-cf881d34d06c100aab9e4cb3134fd6688c8b57e8`  
**Artifact ID:** `14705510877`  
**Artifact SHA-256:** `47c8f220bb17bf36814d073b8540765df8e6a64a9945da8b0c8baa4e501b80c9`

## Exact-head validation

The focused research workflow completed all 20 steps successfully.

- RCAF report/power Python tests: 18 passed.
- EIG/design tests: 13 passed.
- Defensible multiverse manifest tests: 7 passed.
- Measurement-qualification receipt tests: 8 passed.
- Rust kernel and simulation tests: 26 passed, 0 failed.
- Draft 2020-12 schema and semantic validation: passed.
- Theory-design CLI and synthetic ranking: passed.
- Rust formatting and strict Clippy: passed.
- Single-point injection recovery, dose-response, three-estimator tournament, estimator-by-dose surface: passed.
- Content-addressed execution receipt and artifact upload: passed.

Broad Symthaea CI, governance, showroom and benchmark jobs were skipped. This is a focused research-package pass, not a whole-repository pass.

## Generative model

The simulation uses a Gaussian two-factor network model with stationary AR(1) shared factors and residuals:
- 24 independent synthetic participants per replicate;
- 12 regions;
- 160 time points per segment;
- `rho=0.30`;
- five-region injected module;
- baseline global loading 0.20, module loading 0.18;
- participant loading jitter 0.06;
- two-sided exact participant sign test at `alpha=0.05`.

These are simulated observations. They are not EEG, MEG or fMRI recordings and do not model the full neuroimaging acquisition/preprocessing pipeline.

## Single-point null and injection

At `delta=0.40`, 100 Monte-Carlo replicates produced:

| Metric | Result | 95% Wilson interval |
|---|---:|---:|
| Null false-positive rate | 1/100 = 1.0% | 0.18%–5.45% |
| Injection detection power | 100/100 = 100% | 96.30%–100% |
| Median null paired gap | -0.002685 | — |
| Median injected paired gap | 0.068384 | — |

This verifies detection of a strong perturbation in this generator. It does not establish biological sensitivity or general null calibration.

## Absolute-WJ dose-response

Each effect size used 40 Monte-Carlo replicates. The same seed/draw order was reused across dose values.

| Module-loading increment | Rejections / 40 | Detection rate | Median injected gap |
|---:|---:|---:|---:|
| 0.00 | 2/40 | 5% | 0.001597 |
| 0.10 | 0/40 | 0% | 0.001466 |
| 0.20 | 2/40 | 5% | 0.003109 |
| 0.30 | 18/40 | 45% | 0.025868 |
| 0.40 | 40/40 | 100% | 0.065091 |

The curve is broadly increasing but not strictly monotonic at the smallest perturbations. At 40 replicates, intervals are wide; this is an engineering smoke curve, not a precision power study.

## Matched-input three-estimator × five-dose surface

All three metrics use the same synthetic observations, seed, participants and effect-size points.

<small>Null rejection rate is 1/40 = 2.5% for each estimator (95% Wilson interval 0.44%–12.88%). The dose-specific detection interval is computed from 40 replicates per point.</small>

| Estimator | Delta 0.00 | Delta 0.10 | Delta 0.20 | Delta 0.30 | Delta 0.40 |
|---|---:|---:|---:|---:|---:|
| Absolute Weighted Jaccard | 5% | 0% | 5% | 45% | 100% |
| Signed-shift Weighted Jaccard | 0% | 0% | 22.5% | 85% | 100% |
| Normalized mean absolute edge error | 0% | 0% | 22.5% | 92.5% | 100% |

95% Wilson intervals for the middle-to-strong dose points:

| Estimator | Delta 0.20 | Delta 0.30 | Delta 0.40 |
|---|---:|---:|---:|
| Absolute Weighted Jaccard | 1.4%–16.5% | 30.7%–60.2% | 91.2%–100% |
| Signed-shift Weighted Jaccard | 12.3%–37.5% | 70.9%–92.9% | 91.2%–100% |
| Normalized mean absolute edge error | 12.3%–37.5% | 80.1%–97.4% | 91.2%–100% |

The new surface suggests that, in this factor model, signed-shift WJ and normalized edge error recover weaker/moderate perturbations more often than absolute WJ. The strong-injection point saturates all three. This is a useful method-development hypothesis, not a universal ranking: the estimators measure different aspects of network similarity, the intervals overlap at some doses, and each has only 40 Monte-Carlo replicates per point.

For a polarity-only flip fixture, the three estimators also differ deterministically:
- absolute WJ = 1.0;
- signed-shift WJ = 1/3;
- normalized mean absolute edge similarity = 0.5.

They must not be averaged into one score or treated as equivalent constructs.

## What this changes

1. We now test a full estimator operating curve, not only one strong known-effect point.
2. The null false-positive rate is measured alongside injection sensitivity.
3. The same input draws are shared across estimators and dose settings.
4. The results show why metric choice should depend on the declared construct and the operating region, rather than a retrospective maximum score.
5. The next required step is to increase Monte-Carlo replicates (at least 1,000 for final simulation calibration), expand the nuisance grid (especially AR(1), missing edges, nonstationarity and common-mode nuisance), and compare with additional conventional estimators under matched semantics.

## Claim ceiling

This report establishes software behaviour under the declared Gaussian-factor/AR(1) synthetic model only. It does not establish biological validity, causal interpretation, consciousness detection, clinical utility, or SOTA superiority. No EEG/fMRI/MEG/iEEG data were downloaded or analysed in this run, and no published COGITATE or ds006623 result was reproduced.
