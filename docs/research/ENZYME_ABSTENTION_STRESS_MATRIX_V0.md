# Enzyme Abstention Stress Matrix V0

**Status:** exploratory software benchmark specification; not a validated statistical decision rule.  
**Implementation:** `xtask enzyme-abstention-matrix`, in `xtask/src/enzyme_abstention_matrix.rs`.  
**Output:** machine-readable JSON with every scenario, fitted candidate, chosen action, reason, and denominator.  
**Relationship:** stacked on [Symthaea PR #7456](https://github.com/Luminous-Dynamics/symthaea/pull/7456); do not read its existence as a passing benchmark.

## Research question

Can the pipeline distinguish three scientifically different outcomes?

1. **Select a candidate model** when available observations distinguish it from its competitor and at least one candidate has adequate absolute fit.
2. **Abstain because the evidence is ambiguous** when fitted predictions are too similar relative to the declared measurement-noise scale, or the information-criterion difference is not decisive.
3. **Abstain because the candidate set is inadequate** when even the best candidate fails an absolute-fit gate.

A low residual or a large information-criterion gap alone is not enough. Model-selection criteria compare the candidates supplied to them; they cannot establish that one of those candidates is a reasonable scientific explanation.

## Current synthetic matrix

The first executable matrix contains 96 cases: four frozen scenario strata, three declared noise scales (0.25, 1.0, 5.0 rate units), and eight deterministic seeds per stratum/noise combination.

| Stratum | Data-generating curve | Intended decision |
|---|---|---|
| Well-separated Michaelis–Menten | Michaelis–Menten, (V_{max}=100, K_m=2) | Select Michaelis–Menten when evidence is decisive |
| Well-separated substrate inhibition | Substrate inhibition, (V_{max}=100, K_m=2, K_i=20) | Select substrate inhibition when evidence is decisive |
| Observationally ambiguous | Substrate inhibition with (K_i=10{,}000), using only low substrate concentrations | Preserve ambiguity and abstain |
| Candidate-set misspecification | Hill curve with exponent 1.6 | Abstain when both candidate families fail the absolute-fit gate |

Noise is deterministic bounded uniform noise, scaled to the declared standard deviation. This makes the software replayable; it is not a claim that real enzyme assay errors are uniform, independent, homoscedastic, or fully described by a single standard deviation.

Candidate parameters are fit using a coarse exhaustive grid. The fitter is intentionally an understandable baseline, not a production estimator. Grid resolution can affect both residuals and model selection, so these values must be varied in a later sensitivity analysis.

## Decision policy under test

The current policy is deliberately explicit and ordered:

1. **Absolute fit:** abstain if the lower of the two candidates' RMSE values exceeds 3 times the declared noise SD.
2. **Observational resolution:** abstain if the RMS difference between the two fitted predictions on tested substrate conditions is below 1 declared noise SD.
3. **Model-selection evidence:** abstain if (|\Delta BIC| < 10).
4. Otherwise choose the candidate favored by BIC, where `ΔBIC = BIC(Michaelis–Menten) - BIC(substrate inhibition)`; negative values favor Michaelis–Menten.

These cutoffs are **heuristics, not calibrated confidence levels**. The fitted-prediction gap is a screening proxy—not a power calculation or a formal identifiability result. The absolute-fit gate is not a substitute for inspecting residual structure. The use of BIC also needs review against each model's likelihood assumptions and the actual experimental error model.

A hard boundary is intentional: **noise can conceal model misspecification**. In a high-noise Hill-generated case, both candidate fits may look plausible at the declared noise scale even though the known generator is outside the candidate set. The policy can then choose the better of two inadequate explanations. The report exposes this as a false-decisive outcome against the known synthetic target; that outcome must not be hidden by reporting only the winners. It also demonstrates why these controls need a power/detectability annotation before they become a pass/fail gate for a calibrated decision rule.

The report stores both the *expected decision* for each prespecified stratum and the *observed decision*, plus the reason for each outcome. A target hit-rate is descriptive. The matrix generator does not turn its own target rate into a scientific pass.

## Why this is a paradigm shift

Conventional model ranking asks which supplied model wins. An evidence-bounded workflow must also ask whether the data could have distinguished the models and whether the supplied model set explains the observations at all.

That creates separate, auditable error categories:

- **False decisive:** the system selects a named mechanism when its preregistered target requires abstention.
- **False abstention:** the system refuses to decide on a genuinely separable case.
- **Wrong selection:** the system confidently selects the wrong candidate on a separable case.
- **Candidate-set failure:** both models fit poorly, regardless of their relative ranking.
- **Unresolved parameter:** model-family selection is possible but one or more parameter values are not practically identifiable.

Do not collapse these into a single accuracy number. Preserve the full confusion table, all abstentions, failed fits, exclusions, and denominators. The report's current 96 cases are a smoke matrix for the logic, not evidence that the proposed ≤5% future target has been achieved.

## Proposed benchmark gates

These are future acceptance targets, not current results.

| Track | Proposed target | Required evidence |
|---|---|---|
| Deterministic replay | 100% byte-identical reports for repeated runs on the same pinned toolchain; every specified input mutation detected | Frozen fixture, toolchain, artifact hashes, mutation suite |
| Scenario accounting | 100% of generated cases appear exactly once in the report and in a declared stratum | Independent report validator and count-conservation tests |
| Separable model discrimination | Balanced accuracy ≥ 0.90 on preregistered separable cases at the declared operating conditions | At least 1,000 scenarios across independent seeds, model parameters, substrate grids, and noise regimes |
| False-decisive control | ≤ 5% decisive selections on *detectably* ambiguous or candidate-misspecified held-out scenarios | At least 1,000 held-out control cases, disjoint from threshold tuning; preregister an independent detectability/power criterion. Also report the rate for noise-obscured misspecification, without silently dropping those cases. |
| Noise robustness | Report selection, abstention, and wrong-selection rates separately at each noise level | No aggregation that hides high-noise failure modes |
| Uncertainty calibration | 90–98% empirical coverage for nominal 95% intervals, with interval width reported | Proper interval procedure and independent synthetic coverage suite |
| Experiment-design utility | ≥ 25% fewer measurements than a preregistered uniform-random baseline to meet the same recovery/coverage gates | Matched measurement budgets, repeated independent simulated studies, no degraded coverage |
| External reproduction | Independently reproduced parameter/rate extraction within preregistered tolerances on at least three rights-cleared assay series across two sources | Raw signal, calibration, blank/control policy, rate extraction and model fit all replayed |
| Transport | Improvement over the strongest fair baseline on whole-study holdouts, with uncertainty and complete denominators | Frozen dataset versions, deduplication, study grouping, leakage audit |
| Prospective utility | Positive preregistered cost-normalized decision improvement across at least two independent campaigns | Predictions/experiment choices frozen before outcomes are observed |

A threshold is not a result. A passing synthetic gate is only evidence that the implementation behaved as specified on that synthetic matrix.

## Next scientific and engineering gates

1. Expand from 96 cases to a larger preregistered scenario space, including randomized parameter ranges, heteroscedastic and correlated residuals, outliers, censoring/detection limits, sparse substrate coverage, enzyme inactivation, and multi-substrate mechanisms where data warrant them.
2. Split scenario generation into **threshold-tuning** and **locked evaluation** sets. The held-out scenarios must not be used to choose cutoffs.
3. Compare the current baseline with AICc, likelihood-ratio tests where assumptions permit, bootstrap model weights, profile likelihood, and posterior predictive checks. Report where each technique's assumptions apply; do not crown a universal criterion.
4. Measure false-decisive rate, false-abstention rate, wrong-selection rate, and absolute-fit detection separately by stratum and noise scale. Include confidence intervals for those rates.
5. Add identifiability diagnostics for parameters conditional on the selected model. Model-family selection must not imply that every parameter is estimable.
6. Keep the real-data path distinct: use the rights-cleared SLAC/EnzymeML source already documented by PR #7456, then independently replay raw signal → calibration → control correction → concentration/rate extraction → kinetic fit. A calibration-only replay is not a full kinetic reproduction.
7. Freeze external-study splits and fair baselines before adding any enzyme-function predictor. Report unseen-study, sequence-family, protocol/laboratory, and context-transfer results as separate tasks.
8. Only after these gates, test prospective experiment choice by value of information or expected decision-loss reduction, including assay cost and feasibility.

## Standards and literature

- [STRENDA guidelines](https://www.beilstein-strenda-db.org/strenda/public/guidelines.xhtml): reporting requirements help make kinetic results interpretable; STRENDA explicitly does not define a universal quality score.
- [EnzymeML V2 specification](https://enzymeml.org/versions/v2/): reuse standardized representations of enzyme experiments, measurements, equations, kinetic models and parameters instead of inventing a competing biochemical data format.
- [EnzymeML-based modeling workflow dataset, DaRUS DOI 10.18419/DARUS-3867](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2Fdarus-3867): a plausible public, licensed starting point for replay; preserve the official record as the canonical rights/attribution anchor.
- [Liu, Maini & Baker, *Optimal experiment design for practical parameter identifiability and model discrimination*](https://www.sciencedirect.com/science/article/pii/S0025556426001008), *Mathematical Biosciences* 399 (September 2026), 109710: supports treating parameter-identifiability design and model-discrimination design as separate objectives.

## Reproduction

From the repository root:

```sh
cargo test -p xtask --bin xtask enzyme_abstention_matrix
cargo run -p xtask -- enzyme-abstention-matrix --out reports/enzyme_abstention_matrix_v0.json
```

These commands are provided for independent reproduction; they are not a claim that compilation or CI has passed. The full existing `xtask` test suite and the exact-head GitHub workflow remain the source of verification status.
