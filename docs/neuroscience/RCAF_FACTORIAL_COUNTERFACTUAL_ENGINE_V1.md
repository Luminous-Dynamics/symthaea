# RCAF Four-Cell Matched Counterfactual Engine v1

**Status:** executable synthetic experiment-construction kernel; no empirical fMRI or biological qualification is claimed.  
**Base:** spatial nuisance generation and component-variance calibration.  
**Purpose:** make signal × nuisance interventions comparable across nuisance families using one matched factorial contract.

## 1. Scientific design

For clean observations X, a target intervention S and an observation nuisance A, construct the complete 2 × 2 factorial:

| Target intervention | Nuisance intervention | Cell | Observation |
|---|---|---|---|
| absent | absent | clean null | X |
| present | absent | signal only | X + S |
| absent | present | nuisance only | X + A |
| present | present | signal plus nuisance | X + S + A |

The exact same S and A arrays are reused across their matched cells. A nuisance generator is called once upstream; the factorial engine accepts the generated array and must not regenerate or redraw it.

This isolates two questions:
1. Does the nuisance create false changes when the latent target is held fixed?
2. Does the target effect remain recoverable when the same nuisance realization is added?

It is a construction protocol, not a claim that real biological and acquisition processes are necessarily additive.

## 2. Executable invariants

The Rust module `research/rcaf/src/factorial_counterfactual.rs` constructs the four matrices and reports:

- matched nuisance delta: `(Y01 - X) - (Y11 - Y10)`;
- matched signal delta: `(Y10 - X) - (Y11 - Y01)`;
- additive interaction residual: `Y11 - Y10 - Y01 + X`;
- numerical tolerance scaled to the largest cell value;
- cell RMS, maximum absolute value and FNV-1a-64 diagnostic fingerprint;
- pooled within-ROI clean variance, nuisance variance, covariance and combined variance;
- experiment, signal, nuisance-family and nuisance-instance identifiers.

The FNV fingerprint is a deterministic diagnostic, not a cryptographic hash. The focused workflow's content-addressed receipt provides the cryptographic provenance chain for source files and reports.

The numerical invariants verify faithful construction of matched arrays. They do not validate the biological signal model or identify causal origin. Every downstream estimator must still be assessed for null calibration, direction-aware recovery, uncertainty coverage and artifact sensitivity.

## 3. Four-family integration fixture

The current CLI example evaluates four independent nuisance families using the same clean observations and target intervention:

1. **Common-mode:** synthetic spatially uniform temporal fluctuation.
2. **Graph-smooth low-rank:** calibrated S1 graph Laplacian field.
3. **Localized heat-kernel:** calibrated S2 ROI-seeded diffusion field.
4. **Motion-like transient:** S3 broad and localized event-triggered exponential fields.

S1 and S2 are calibrated against the same clean observations to a 10% component-variance fraction. Common-mode and transient fields are intentionally marked as uncalibrated in this smoke fixture rather than implying that their amplitudes are comparable. Their clean–nuisance covariance is still reported.

The exact graph-based nuisance field or event realization is generated once and then passed to the factorial engine. Within each family, all four cells use that same field. Each experiment has a distinct stable nuisance-instance identifier.

## 4. Full experiment surface to build next

The current implementation constructs one small smoke fixture per family. The full research campaign should cross the factors below while preserving the same four-cell design for every cell.

### Signal factor
- exact null;
- weak, moderate and strong target changes;
- distributed and focal spatial patterns;
- positive and negative direction;
- sustained, transient and temporally drifting effects;
- effect topologies with different graph frequencies and spatial ranks.

### Nuisance factor
- common-mode amplitude and temporal spectrum;
- graph-smooth rank and diffusion time;
- localized seed placement and spatial decay;
- motion-like event onset, sign, density, overlap and decay time;
- physiological periodic/lagged processes when recorded;
- scanner drift and run-boundary discontinuities.

### Signal–nuisance relationship
- independent;
- positively aligned;
- negatively aligned;
- partially aligned;
- delayed and phase-shifted;
- exactly observationally equivalent.

Do not infer the relationship by construction and then claim a general result. It is a predeclared challenge factor. Record covariance and the actual observation geometry separately from the nominal component-variance fraction.

### Processing factor
- one-step joint regression;
- nuisance regression then temporal filter;
- temporal filter then nuisance regression;
- sequential orthogonalization;
- censoring with each declared boundary/interpolation method;
- pre-whitening and temporal-shifted physiological regressors;
- single-echo and multi-echo strategies;
- multi-band and single-band acquisition strata.

### Estimator factor
- current RCAF weighted-Jaccard modes;
- normalized edge error;
- correlation difference and rank correlation;
- intraclass/test-retest reliability;
- downstream prediction with nested, participant-grouped cross-validation.

The complete design must be preregistered and split into screening, confirmatory and external-validation stages. Do not select a winner on the same simulation or participant cohort used to estimate its performance.

## 5. Proposed qualification targets

These are proposed engineering targets for future frozen campaigns, not accepted field standards or achieved results:

| Measure | Proposed target |
|---|---:|
| Null false-positive rate | ≤5% |
| Wrong-direction acceptance | ≤1% |
| Directional recovery on moderate matched effects | ≥80% |
| Lower confidence bound on recovery | ≥75% |
| Nominal 95% interval coverage | 93–97% |
| Correct abstention on exact/declared equivalent controls | 100% |
| False abstention on clear identifiable controls | ≤5% |
| Artifact-only effect drift | ≤10% of the minimum meaningful effect |
| Synthetic deterministic replay | bit-identical matrices and report |
| Evidence completeness | every run traceable to source/config/data/artifact digests |

Run at least 1,000 seeds for screening and 10,000 Monte Carlo null replicates for confirmatory calibration where computationally feasible. Report Monte Carlo standard errors and confidence intervals. Monte Carlo replicate count M does not replace independent participant count N; inferential sample size is still based on independent participants or sites.

For a 5% false-positive rate, 10,000 independent null replicates give a Monte Carlo standard error of approximately 0.22 percentage points. This quantifies simulation precision only, not the uncertainty from participant sampling or a misspecified generator.

## 6. Industry comparator protocol

### 2026 denoising benchmark

*Benchmarking fMRI Denoising Pipelines* (Human Brain Mapping, first published 1 July 2026; DOI 10.1002/hbm.70561) benchmarks task and resting-state, single- and multi-band, and single- and multi-echo data. It incorporates one-step regression, pre-whitening and temporally shifted physiological nuisance regressors, and evaluates tSNR, retained degrees of freedom, motion correction and signal preservation.

RCAF should reproduce compatible pipeline configurations and endpoints before proposing a new pipeline. The added value is the matched signal × nuisance intervention, explicit nuisance-only false-positive controls, order-dependence measurement, uncertainty qualification and abstention boundary.

### 90-pipeline single-/multi-echo benchmark

Constable et al. (2026), *An evaluation of the efficacy of single-echo and multi-echo fMRI denoising strategies*, compares 90 pipelines in 358 healthy adults across six data-quality measures and functional-connectivity behavioural prediction. Multi-echo strategies often perform well on denoising measures, but no single pipeline optimizes both denoising and behavioural prediction.

That is a direct argument against a universal scalar denoising score. RCAF should produce endpoint-specific Pareto surfaces and expose trade-offs.

### Reproducible software comparators

- **fMRIPrep:** pin its version, container digest, BIDS input and selected confound columns. Its output documentation explicitly notes that denoising strategies trade nuisance removal against damage to neuronal fluctuations and that no universal optimal strategy exists.
- **XCP-D:** pin its release and parameters, including nuisance regressors, framewise-displacement threshold, filtering, censoring and interpolation. Its workflow documentation describes concrete choices that must be captured as an operator manifest.
- **CONN:** pin its version and every denoising/confound setting. Compare native recommended configurations as well as a matched endpoint-specific configuration.

Do not compare a synthetic Rust kernel with complete production pipelines as if they were competing denoisers. First integrate the reference as a qualification harness around frozen outputs from those systems.

## 7. How to establish a genuine improvement over SOTA

A defensible superiority claim requires all of the following:

1. Same raw data release, participant/site split, atlas, endpoint and quality strata.
2. Same acquisition strata and appropriate single-/multi-echo methods.
3. Matched tuning budget and nested cross-validation.
4. Native recommended configuration for each comparator plus declared matched configurations.
5. Lower null error without loss of signal recovery, degrees of freedom or behavioural validity.
6. Participant- and site-level uncertainty, including attrition and missingness.
7. Frozen external cohort, independent reproduction and all failed runs retained in the evidence record.

A proposed new contribution should improve a Pareto frontier, not just one average metric. A pipeline that improves tSNR by deleting task signal has not demonstrated a general improvement.

## 8. Immediate implementation sequence

1. Validate the four-cell construction kernel and four-family smoke suite.
2. Add a grid runner across nuisance family × component-variance level × signal effect × temporal coherence.
3. Feed each cell through the operator-composition variants.
4. Compute connectivity matrices with pinned estimators and measure null error, directional recovery, interval coverage and retained degrees of freedom.
5. Add exact and near observational-equivalence controls and score correct/false abstention separately.
6. Export a long-form machine-readable result table and Pareto summary.
7. Add fMRIPrep/XCP-D/CONN operator-manifest adapters.
8. Reproduce a public multi-echo/task benchmark and then run an external-site confirmation.

## References

- *Benchmarking fMRI Denoising Pipelines*. Human Brain Mapping (2026). https://doi.org/10.1002/hbm.70561
- Constable, T. et al. *An evaluation of the efficacy of single-echo and multi-echo fMRI denoising strategies*. Network Neuroscience (2026), 10(2), 444–474. https://doi.org/10.1162/NETN.a.547
- fMRIPrep confound outputs and denoising guidance: https://fmriprep.org/en/latest/outputs.html
- XCP-D processing pipeline details: https://xcp-d.readthedocs.io/en/latest/workflows.html
- Lindquist, M. A. et al. (2019). *Modular preprocessing pipelines can reintroduce artifacts into fMRI data*. Human Brain Mapping. https://doi.org/10.1002/hbm.24567

## Claim limits

This v1 engine constructs synthetic counterfactual cells and checks their matched-component arithmetic. It does not run an empirical fMRI preprocessor, estimate real motion/physiology, identify causal biological changes, or demonstrate SOTA performance. Its output is an experimental-design and evidence-provenance primitive for the next qualification layer.
