# RCAF Counterfactual Generalization Benchmark V1

**Status:** Proposed benchmark contract; design and acceptance targets only. No empirical dataset evaluation or SOTA claim is made here.  
**Date:** 2026-11-21  
**Scope:** CETB/RCAF measurement and inference qualification  
**Parent:** CETB measurement qualification stack V0 and RCAF adversarial instrument calibration V1

## 1. The paradigm shift: benchmark claims, not leaderboard scores

Most neuroimaging benchmarks optimize one stage or one endpoint: preprocessing, connectome construction, pairwise estimator quality, dataset sufficiency, classification accuracy, or test–retest reliability. These are important but not interchangeable.

CETB should be a **counterfactual qualification layer** across the complete measurement-to-claim chain:

`latent target → signal generation/acquisition → preprocessing → connectivity estimator → matrix comparator → participant-level inference → claim`

The benchmark asks two paired questions:

1. When the target changes and the observation process is held fixed, does the method recover the intended change, in the intended direction, with calibrated uncertainty?
2. When the observation process changes but the latent target is held fixed, does the method avoid inventing a target change, or explicitly qualify the result as nuisance-sensitive?

This turns "robustness" into an experimentally testable claim. It also prevents a high classification score from being mistaken for measurement validity, repeatability, or construct validity.

## 2. Formal counterfactual design

Represent an observation as:

`X_{p,s} = G(T_p, N_s, A_s, epsilon_{p,s})`

where:

- `T_p` is participant `p`'s declared latent target;
- `N_s` is the nuisance/intervention state for session or site `s`;
- `A_s` is the acquisition and preprocessing configuration;
- `epsilon` is finite-sample and measurement noise;
- `G` is the versioned observation generator or empirical acquisition process.

The benchmark requires paired interventions:

- **Signal intervention:** vary `T`, hold `N` and `A` fixed.
- **Nuisance intervention:** vary `N), hold `T` fixed.
- **Acquisition intervention:** vary `A), preserve the target and disclose the measurement transformation.
- **Interaction intervention:** cross signal and nuisance factors to test whether nuisance alters sensitivity to genuine changes.
- **Leakage intervention:** plant a known information-boundary violation, pair it with a clean negative control, and test detection plus localization.

Do not call every nuisance intervention a classical Type-I error. A nuisance shift changes the observed-data distribution; its rejection frequency is an **artifact-only operational false-alarm rate** relative to the unchanged latent target. Classical null calibration must remain a separate same-observation-distribution test.

Each intervention cell records its latent target, observed-data distribution, expected result, random seed namespace, factor levels, analysis access boundary and predeclared acceptance test. The generator must expose truth to the simulator/validator, never to the analysis function being tested.

## 3. Metamorphic qualification: invariance versus equivariance

A single invariance rule is scientifically wrong. Some transformations should leave an estimand unchanged; others should transform the result in a predictable way.

For every estimator/endpoint pair, publish a **transformation contract**:

| Transformation | Required relation | Example |
|---|---|---|
| ROI label permutation with a correspondingly permuted matrix | Invariant after label-aware reindexing | Identical result after restoring the canonical ROI order |
| Exact duplicate input / replay | Bitwise or tolerance-bounded reproducibility | Same seed, config and binary produce same receipt |
| Global gain or offset | Endpoint-specific invariant or known equivariance | Do not assume all connectivity statistics are scale invariant |
| Signal polarity inversion | Explicit sign equivariance or declared non-invariance | Signed edges must not be silently absolutized |
| Time-series resampling / duration change | No assumed invariance | Re-estimate uncertainty and effective sample size |
| Site/scanner transform | Transportability test, not automatic invariance | Train-only harmonization and locked test site |
| Missingness/censoring | Explicit missingness-mechanism sensitivity | MCAR, MAR and MNAR are separate cells |

Metamorphic failures should report the broken contract and affected claim. They must not be repaired by silently normalizing away the discrepancy.

## 4. Keep four benchmark layers separate

| Layer | Question | Candidate comparators |
|---|---|---|
| Data / measurement sufficiency | Is there enough recoverable signal for this endpoint? | NSI where its precision-fMRI input assumptions match; blinded QC |
| Connectivity construction | How is the connectivity matrix estimated? | Pearson, Spearman, shrinkage covariance, precision/partial correlation, selected pyspi statistics |
| Matrix comparison | How are two matrices compared? | Current RCAF measures plus preregistered matrix distances and graph comparisons |
| Inference / decision | Does the participant-level result support the declared contrast? | RCAF paired inference, sign/permutation procedures, bootstrap intervals, calibrated decision rules |

No comparator is credited for a capability outside its purpose. In particular, NSI is a precision-fMRI measurement-sufficiency/QC instrument, not a state-contrast inferential method. pyspi benchmarks pairwise interaction estimators, not participant-level repeatability. Pipeline classification accuracy is not a substitute for latent-target recovery.

## 5. Required multi-objective scorecard

Never collapse these outputs into a single unqualified "reliability" or "consciousness" score.

### Statistical calibration
- Same-observation-distribution null rejection rate, exact nominal alpha, Monte-Carlo interval and seed count.
- Directional true-positive recovery and wrong-direction significant rejection, separately.
- Effect bias, interval coverage, interval width and calibration of any reported probability.
- Type-I error, power and confidence intervals stratified by every predeclared critical cell.

### Counterfactual robustness
- Artifact-only operational false-alarm frequency by nuisance family and dose.
- Maximum nuisance-induced drift in the target estimate, not just its average.
- Signal × nuisance interaction: degradation in directional recovery under each nuisance dose.
- Dose-response ordering and non-monotonicity diagnostics.
- Worst-stratum performance and lower-tail performance; do not report only pooled means.

### Transport and selection integrity
- Participant holdout, unseen-site holdout, unseen-device/acquisition holdout and independent-cohort confirmation as distinct results.
- All preprocessing, harmonization, feature selection, estimator selection, calibration and thresholds fitted within training folds only.
- Planted leakage detection sensitivity, clean-control false-alarm frequency and the exact boundary crossed.
- Defensible multiverse effect/interval distribution, including sign reversals and conclusions that depend on one pipeline choice.

### Operational and evidence quality
- Runtime, peak memory, wall-clock cost and data volume for each compared pipeline.
- Exact source commit, dependencies, environment, data manifest, configuration, seeds and output digests.
- A machine-readable claim-scoped receipt that records pass, fail, inconclusive and not-assessed separately.

## 6. Proposed engineering gates

These are **stretch targets for a future preregistered qualification protocol**, not published field standards and not current RCAF results. Thresholds must be reviewed against endpoint-specific error costs and the attainable sample size before confirmatory runs.

| Gate | Proposed target | Interpretation |
|---|---|---|
| Deterministic replay | 100% exact match for deterministic fixtures | Same artifacts for same frozen inputs |
| Nominal null calibration | At `alpha=0.05`, Monte-Carlo 95% interval contains 0.05; at least 10,000 replicates for primary confirmatory null families | Must be evaluated separately across null families |
| Directional recovery | At the preregistered minimally meaningful effect, point recovery ≥0.80 and lower 95% Wilson bound ≥0.75 | Strong-power objective, not a biological truth threshold |
| Wrong-direction rejection | ≤0.01 point rate at the minimally meaningful positive effect; upper 95% bound reported | Opposite-sign significance is never successful recovery |
| Nuisance-only false alarm | ≤0.05 at each predeclared moderate nuisance challenge, with upper 95% bound reported | Operational artifact robustness, explicitly not Type-I error |
| Effect-interval coverage | 95% interval coverage within 0.93–0.97 on calibrated synthetic families | Only for generators whose ground-truth effect and interval estimand match |
| Nuisance-induced target drift | ≤0.10 of the minimally meaningful effect, unless the endpoint contract predicts equivariance | Require per-family worst-case reporting |
| Leakage controls | Detect every seeded leakage fixture; no more than 0.05 false flags on clean fixtures | Controlled software benchmark, followed by empirical audit |
| Robust direction across defensible workflows | ≥0.90 sign concordance, or classify the result as analysis-sensitive | Only compare workflows that estimate the same declared target |
| External confirmation | Frozen endpoint and decision rule reproduce direction with interval and calibration reported; no retuning on external labels | No universal accuracy threshold |

For a nominal 5% error rate, 10,000 replicates are a starting point, not a substitute for an interval or a power calculation. Small exploratory grids must not be used to claim precise tail calibration.

## 7. Competitive comparison: what the field has already achieved

The benchmark should compare against the right public baselines, not manufacture a single leaderboard.

1. **768 connectome construction pipelines / 13,824 pipeline-criterion-data evaluations.** Luppi et al. evaluated combinations of parcellation, connectivity definition, global-signal regression and graph construction. Only 9 of 768 satisfied all their stringent multi-criterion requirements across their test–retest datasets. This is a baseline for multi-criterion pipeline qualification—not a comparator for RCAF's participant-level inference statistic.
2. **239 pairwise interaction statistics.** The 2025 *Nature Methods* pyspi-based benchmark showed that estimator rankings vary by objective, including structure–function coupling, individual identification, brain–behaviour prediction and motion sensitivity. This is a construction-estimator benchmark; it argues against choosing a universal winner.
3. **256 pipelines in two-site eye-state classification.** Medvedeva et al. (2026) compared 256 fMRI pipelines under cross-site validation and few-shot adaptation and reported approximately 90% accuracy for top-performing configurations on the controlled eyes-open/eyes-closed task. This is a useful cross-site classification baseline, not a ceiling for sedation, consciousness, RCAF repeatability or construct validity.
4. **Network Similarity Index.** Buchanan et al. (2026) provide an expert-anchored precision-fMRI map-quality assessment with nested-CV usability prediction. RCAF/CETB should reuse NSI when input compatibility and its intended PFM use case match, rather than clone or relabel it.
5. **External validation.** Rosenblatt et al. (2026) emphasize independent-dataset validation as stricter evidence than unseen-sample validation within one dataset. CETB's tiers must preserve that distinction.

### Operational comparator stack: qualify the ecosystem, do not rebuild it

The competitive baseline is not just papers and leaderboards. The current research-software ecosystem already separates acquisition preprocessing, postprocessing/denoising, connectivity construction and individualized quality assessment. CETB/RCAF should be an independent qualification layer over those outputs.

| Tool or deployment | Existing role | Fair CETB comparison |
|---|---|---|
| fMRIPrep | BIDS-oriented minimal preprocessing; motion correction, susceptibility unwarping, normalization, bias correction, brain extraction and per-subject visual reports | Pin exact version and configuration; consume its derivatives and reports rather than claiming to replace preprocessing |
| XCP-D | Continues from fMRIPrep/HCP/ABCD/UK Biobank-like derivatives into denoising, QC and connectivity outputs | Run on the same derivative set; compare QC/denoising and connectivity outputs under identical input conditions |
| CONN 25b | Established SPM-based connectivity toolbox for resting/task fMRI, with transparent methods reporting and combined denoising/QC workflows | Treat as an alternative analysis implementation; compare exact denoising choices, FC matrices, QC decisions and participant-level outputs |
| ABCD Coordinating Center 4.1.0 (18 Nov 2026) | Operational pipeline deployment combining fMRIPrep 25.1.4, XCP-D 26.1.1 and ReproTM individualized network maps | Use as a pinned real-world pipeline-stack reference, not as an independent accuracy score |
| Network Similarity Index (NSI) | Precision-fMRI map-quality / usable-network assessment | Reuse only where its CIFTI/precision-fMRI input assumptions and intended endpoint match; do not substitute it for RCAF inference |

These tools are mature and complementary. The differentiator is to attach a stronger claim-scoped evidence package to their outputs: known-truth injection recovery, artifact-only false alarms, signal × artifact interaction, transformation-contract checks, calibration intervals, selection/leakage audits, transportability and content-addressed replay evidence.

An implementable industry-facing comparison should freeze the same BIDS input release, atlas, censoring rules, denoising strategy and FC estimator. Run fMRIPrep → XCP-D and an agreed CONN configuration, then evaluate each output with the same RCAF matrix-comparison and participant-level qualification harness. Also preserve each native tool's intended outputs and QC, rather than forcing all packages into one artificial API. Report runtime, memory, hardware, failed subjects, QC attrition and all exclusions.
An executable metamorphic test suite has been added separately in PR [#13163](https://github.com/Luminous-Dynamics/symthaea/pull/13163). It tests joint ROI-label permutation invariance, operand symmetry, identity, absolute-versus-signed behavior under polarity inversion, edge-transition reversal and base weighted-Jaccard scale/joint-edge-order invariance. This is a method-level contract, not evidence of upstream neuroimaging validity.

### The actual route to outperformance

A credible advantage is not "beat 90% accuracy" on a different task. It is to deliver a benchmark where the **same frozen input, target, split, and preprocessing** pass through all four layers; where signal and nuisance interventions are independently controlled; where failure modes are localized; and where external confirmation is untouched.

The proposed differentiator is a *claim-conditional evidence frontier*: for each endpoint, show the Pareto surface across directional recovery, null calibration, nuisance robustness, transportability, uncertainty calibration, runtime and memory. Publish dominated and Pareto-optimal configurations, not one composite winner. The best configuration can differ by endpoint.

## 8. Execution sequence

1. Repair and pass the existing exact-head Rust test/format/lint gate; execute the current synthetic reports and retain the content-addressed receipt.
2. Complete the currently implemented shared common-mode challenge and inspect its observed dose curve.
3. Add a second nuisance family that is mathematically distinct (spatially structured low-rank gradient or scanner/site gain-plus-offset), with matched latent random numbers and explicit invariance expectations.
4. Add a crossed signal × nuisance experiment so that sensitivity loss under artifacts can be measured directly.
5. Build a transformation-contract fixture suite (ROI permutation, polarity, gain/offset, missingness and duration) for declared invariance/equivariance.
6. Compare connectivity estimators and matrix-comparison methods as separate layers on identical inputs; include selected pyspi baselines only where dependencies, inputs and estimands match.
7. Reproduce the frozen ds006623 source analysis before using ds003171 as an independent transport cohort. Freeze preprocessing, labels, quality exclusions and estimator selection before opening external outcomes.
8. Publish raw per-cell outputs, confidence intervals, resource costs, source/configuration/environment digests and all failed cells.

## 9. Pilot execution evidence from the separate RCAF implementation branch

A first exact-head smoke run is now available from the implementation branch PR #13139. It used commit [`42582270476f110576f754de60c6a779d2fc7d6c`](https://github.com/Luminous-Dynamics/symthaea/commit/42582270476f110576f754de60c6a779d2fc7d6c), workflow run [`44949124581`](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44949124581), and archived artifact `rcaf-simulation-evidence-c1a71fbc7f40cf88011ab5626891f0e85a8468bc` (artifact SHA-256 `72579b8937dc652afa160745b5581fa125a6d5b6ce9a815a39e6db4ff1998023`). The content-addressed receipt records source-tree SHA-256 `fbf4a71ff73bb49d88e7bd43863ce5e24b5981fce95af5224744d72520cd9589`.

The focused validation run passed all 32 Rust tests, formatting, Clippy, Python report-validator tests, EIG checks, multiverse and measurement-qualification checks, schema/semantic validation, all five simulation/report stages, receipt generation and artifact upload. The workflow's broad CI, governance and syntax workflows were skipped; this is not a whole-repository pass.

### Default synthetic injection smoke case

Configuration: 24 participants, 12 regions, 160 samples per segment, 100 Monte-Carlo replicates, module size 5, global loading 0.20, module loading 0.18, loading jitter 0.06, AR(1) coefficient 0.30, expected loading increment 0.40, common-mode nuisance loading 0.20, and alpha 0.05.

| Output | Observed | 95% Wilson interval |
|---|---:|---:|
| Same-distribution null rejection | 1/100 = 0.010 | 0.0018–0.0545 |
| Nuisance-only rejection | 1/100 = 0.010 | 0.0018–0.0545 |
| Two-sided injection detection | 100/100 = 1.000 | 0.9630–1.0000 |
| Expected-direction recovery | 100/100 = 1.000 | 0.9630–1.0000 |
| Wrong-direction significant rejection | 0/100 = 0.000 | 0.0000–0.0370 |

Median participant-paired gap was -0.00269 under the same-state null, -0.00180 for the nuisance-only observation and +0.06838 under the intended injection. These results show that this strong planted signal is easy to recover in this one Gaussian-factor setting; they do not demonstrate power at a minimally meaningful effect or empirical validity.

### Matched nuisance and estimator pilot

At the common-mode nuisance dose grid `[0.00, 0.10, 0.20, 0.30]`, the absolute-weighted-Jaccard nuisance-only rejection estimates were 0.025, 0.000, 0.025 and 0.125 respectively (40 replicates per point). At dose 0.30 the Wilson interval was 0.0546–0.2611. This is a detectable concern at the high tested dose, with substantial Monte-Carlo uncertainty; the observed curve is not monotonic at every intermediate point.

In the matched-input estimator tournament at common-mode loading 0.20, the observed same-distribution null rejection was 0.025 for all three matrix comparators and expected-direction recovery was 1.000 for all three. Their nuisance-only rejection estimates differed: absolute weighted Jaccard 0.025, signed-shift weighted Jaccard 0.100, and normalized mean absolute edge error 0.200. This is an initial indication that nuisance sensitivity is estimator-dependent under the declared synthetic factor model—not a universal ranking. This estimator tournament used 40 Monte-Carlo replicates per comparator. The separate common-mode dose sweep and five-point injection dose sweep also used 40 replicates per point. Only the standalone default injection smoke report used 100 replicates. These pilot intervals are broad; confirmatory runs are still required.

The five-point effect sweep `[0.00, 0.10, 0.20, 0.30, 0.40]` (40 replicates per point) produced directional recovery estimates of `[0.025, 0.000, 0.050, 0.450, 1.000]` for absolute weighted Jaccard. The matched estimator-by-dose curves differed at intermediate doses while reaching 1.000 at the strongest dose. These are descriptive finite-Monte-Carlo pilot curves; their small non-monotonicities at weak doses must not be interpreted as a biological dose-response law.

### Interpretation

1. The strong positive injection is recovered in this synthetic world, and opposite-direction rejection is counted separately.
2. Nuisance sensitivity is not zero by construction: the high common-mode dose and comparator spread expose a real challenge that needs to be qualified.
3. The pilot intervals are too wide to certify a nominal 5% tail: nuisance/effect sweeps and the estimator tournament used 40 replicates per point/comparator, while the standalone injection report used 100. Increase primary confirmatory null families to at least 10,000 replicates, and report Wilson intervals for every critical cell.
4. The execution receipt demonstrates a content-addressed record of this run; it does not prove independent replay.
5. No biological data were downloaded or processed, and these results do not establish SOTA superiority, consciousness detection, causal inference or clinical readiness.

## 10. Crossed signal × nuisance qualification

The implementation branch now adds a four-cell counterfactual contrast:

1. Clean null: no target injection and no observation nuisance.
2. Signal only: intended target change with no observation nuisance.
3. Nuisance only: target held fixed while common-mode observation nuisance is added.
4. Signal plus nuisance: the same intended target change is present while the same nuisance family is superimposed.

The nuisance-only and signal-plus-nuisance cells use the same deterministic random innovation stream per participant and Monte-Carlo replicate. The planted module loading is the only intended difference between those two cells. At nuisance dose zero, the signal-plus-nuisance report fields must exactly equal the clean signal-only fields. Across nuisance-dose points, the clean signal and null results remain invariant; combined-cell outcomes are explicitly allowed to change.

New outputs report two-sided detection, expected-direction recovery, wrong-direction significant rejection and median paired gap for the combined cell. This creates a direct test of whether an artifact masks a genuine effect, changes its magnitude or produces a significant result in the wrong direction. The existing nuisance-only false-alarm output remains separate; it must not be collapsed into classical Type-I error.

**Implementation and execution status:** the focused RCAF workflow [44950237593](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44950237593) passed on exact source commit `470cf519cd76ab60aebb31bf3e96fb0607f9e24e`: all report-contract checks, 32 Rust tests, rustfmt, Clippy, five simulation/report stages, content-addressed receipt generation and artifact archive. The archive digest is SHA-256 `5a0d661f4016fa680eae511ee9227d9c9d52f05c9abca5b714352ab6ee781e31`; the receipt source-tree digest is `e7653f633d4b613bee7959c0cd569a14545da96655ac47e6c8fdc8dd00e3ec51`. This is a focused research-workflow pass only: broad repository CI, PR governance and workflow syntax were skipped.

### Crossed experiment results

The archived crossed signal × nuisance sweep used 40 Monte-Carlo replicates at each common-mode dose, with clean signal-only and null metrics matched across doses:

| Common-mode dose | Nuisance-only operational false alarm | Signal+ nuisance directional recovery at Δ=0.40 | Median clean-signal gap | Median combined gap |
|---:|---:|---:|---:|---:|
| 0.00 | 1/40 = 2.5% [0.4–12.9%] | 40/40 = 100% [91.2–100%] | 0.06509 | 0.06509 |
| 0.10 | 0/40 = 0% [0–8.8%] | 40/40 = 100% [91.2–100%] | 0.06509 | 0.06975 |
| 0.20 | 1/40 = 2.5% [0.4–12.9%] | 40/40 = 100% [91.2–100%] | 0.06509 | 0.07285 |
| 0.30 | 5/40 = 12.5% [5.5–26.1%] | 40/40 = 100% [91.2–100%] | 0.06509 | 0.08903 |

Intervals are 95% Wilson intervals. The nuisance-only rate is an **artifact-only operational false-alarm rate**, not classical Type-I error: the artifact changes the observed-data distribution while the latent target remains fixed. The response is non-monotonic at the lower doses, and the 40-replicate estimates are imprecise.

The key finding is not that the method becomes more robust. In this synthetic generator, common-mode nuisance **inflates** the apparent signal contrast. At dose 0.30 the combined median paired gap is about 36.8% larger than the clean signal-only gap, while the nuisance-only false-alarm estimate rises to 12.5%.

A second view comes from the 40-replicate effect sweep at common-mode dose 0.20. For absolute WJ at a moderate planted effect Δ=0.30, directional recovery was 18/40 = 45.0% without nuisance (95% Wilson 30.7–60.2%) and 34/40 = 85.0% with nuisance (70.9–92.9%). Treat this as artifact-induced amplification, not valid power improvement: the nuisance itself can create an apparent network contrast, and the combined cell inherits that confounding. At Δ=0.40 the recovery rates hit the 100% ceiling, which is not discriminating.

At common-mode loading 0.20, the three-comparator tournament (40 replicates each) produced nuisance-only rejection estimates of 1/40 = 2.5% for absolute WJ (95% Wilson 0.4–12.9%), 4/40 = 10.0% for signed-shift WJ (4.0–23.1%) and 8/40 = 20.0% for normalized mean absolute edge error (10.5–34.8%). The same-distribution null estimate was 1/40 for each. These pilots show that nuisance sensitivity depends on the comparison statistic under this generator; they do not establish a universal best estimator.

### What the result changes

1. A signal-only pass and nuisance-only pass are not enough: the crossed interaction is required.
2. “Higher power” must be judged alongside nuisance-only false alarms and effect inflation; otherwise a confounded metric can look better.
3. The current common-mode factor is one narrow synthetic artifact family, not a realistic scanner, motion or physiological model.
4. The small pilot grid is for debugging and counterexample discovery only. Increase primary null families to at least 10,000 replicates, use independent confirmation seeds, and report intervals in every stratum.
5. Next nuisance families should include site/scanner gain and offset, spatially structured low-rank artifacts, motion-like bursts, censoring, physiological waveforms, missingness, nonstationarity and positive/negative leakage controls.
## 11. Scope and safety of claims

- A passed synthetic challenge qualifies only the tested generator, intervention and analysis implementation.
- Artifact-only rejection is not classical Type-I error when the nuisance intervention changes observed-data distributions.
- A within-release held-out site is not automatically an independent cohort.
- A high eyes-open/eyes-closed classifier accuracy does not establish consciousness detection.
- Precision-fMRI data-sufficiency scores do not validate EEG/MEG or propofol-state inference.
- No result in this document establishes clinical readiness, causal identification, consciousness detection or superiority over the state of the art.

## References

- Luppi et al. (2024), *Systematic evaluation of fMRI data-processing pipelines for consistent functional connectomics*, Nature Communications, [10.1038/s41467-024-48781-5](https://doi.org/10.1038/s41467-024-48781-5).
- Benchmarking methods for mapping functional connectivity in the brain (2025), Nature Methods, [10.1038/s41592-025-02704-4](https://doi.org/10.1038/s41592-025-02704-4).
- Medvedeva et al. (2026), *Benchmarking resting state fMRI connectivity pipelines for classification: robust accuracy despite processing variability in cross-site eye state prediction*, Brain Informatics, [10.1186/s40708-026-00305-1](https://doi.org/10.1186/s40708-026-00305-1).
- Buchanan et al. (2026), *Objective quality assessment for precision functional MRI data*, Neuron, [10.1016/j.neuron.2026.05.020](https://doi.org/10.1016/j.neuron.2026.05.020).
- Rosenblatt et al. (2026), *External validation improves generalizability, replicability and reproducibility in predictive models for neuroimaging*, Nature Methods, [10.1038/s41592-026-03115-9](https://doi.org/10.1038/s41592-026-03115-9).
- CETB/RCAF, *RCAF Adversarial Instrument Calibration Protocol V1* and *CETB multidimensional measurement qualification stack V0*.
