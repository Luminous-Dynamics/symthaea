# RCAF Preprocessing Composition Assurance v1

**Status:** proposed research contract with a small executable linear-projection reference kernel; no real fMRI pipeline has been qualified by this increment.  
**Purpose:** evaluate whether a sequence of individually reasonable preprocessing operators preserves the intended nuisance-removal and signal-preservation properties of the complete workflow.

## 1. Motivation and current evidence

RCAF already evaluates the robustness of connectivity comparisons under synthetic signal and observation-nuisance interventions. That is not sufficient if preprocessing itself changes the relationship between nuisance and signal.

Lindquist et al. (2019) showed that sequential regression/filtering operations are geometric projections that generally do not commute. Later projections can reintroduce variance associated with nuisance regressors that an earlier projection removed. They discussed motion regression, scrubbing, physiological correction, global-signal regression, component correction, and temporal filtering, and proposed joint filtering or sequential orthogonalization as remedies.

The field's modern denoising benchmarks make the multi-objective nature of this problem explicit:

- The 2026 *Benchmarking fMRI Denoising Pipelines* study evaluated denoising across task and resting-state, single- and multi-band, and single- and multi-echo datasets. It included one-step regression, temporal pre-whitening, and time-shifted physiological regressors, with tSNR, retained degrees of freedom, motion correction and signal preservation among its measures.
- A 2026 single-/multi-echo comparison evaluated 90 preprocessing pipelines in 358 healthy adults. Multi-echo approaches generally performed well, but no single denoising pipeline dominated every benchmark; stronger noise reduction could come at the cost of behavioural prediction.
- A 2025 two-cohort study examining 81 behavioural variables likewise found no single strategy that consistently maximized motion mitigation and behavioural prediction across cohorts.

These results argue for evaluating complete processing operators against the intended scientific endpoint, not selecting one universal denoiser.

## 2. Core idea: preprocessing is an operator graph

Represent each temporal operation as a declared operator (A_i) and the complete workflow as an ordered composition

[
Y_{out}=A_n A_{n-1}cdots A_2 A_1 Y.
]

For a nuisance design (N) with an orthonormal basis (Q_N), ordinary least-squares residualization is

[
R_N=I-Q_NQ_N^T.
]

For a temporal subspace (H) removed by a projection-style filter,

[
R_H=I-Q_HQ_H^T.
]

In general,

[
R_HR_N 
e R_NR_H.
]

The first executable reference kernel evaluates these finite-dimensional projection operators. It is deliberately not a full temporal-frequency filter, censoring implementation, or fMRI preprocessor.

### Why re-entry is measurable

Even when (R_NY) is exactly orthogonal to the nuisance design, applying another projection (R_H) can move the result out of that orthogonal complement. The output may then contain a nonzero component in the original nuisance subspace:

[
Q_NQ_N^T R_HR_NY 
e 0.
]

The benchmark must therefore inspect the final output against *all* nuisance spaces that the pipeline claims to have removed, not merely inspect the step where each regressor was applied.

## 3. Executable reference challenge

The standalone Rust module implements:

1. A small dependency-free modified Gram–Schmidt basis builder with re-orthogonalization and explicit dependent-column accounting.
2. Orthogonal residualization of a time-by-signal matrix against a declared design matrix.
3. Sequential nuisance-then-filter and filter-then-nuisance compositions.
4. One-shot joint projection against the concatenated nuisance and removed-filter subspaces.
5. Final nuisance/filter leakage fractions, relative order difference, joint-versus-sequential difference, and effective design rank.
6. Fail-closed validation for non-finite matrices, inconsistent row counts, zero-rank designs, and invalid observations.

The joint-projection baseline is a mathematical reference for the declared projection subspaces. It is not a universal prescription to regress out every signal or every frequency component; the scientific meaning of the retained subspace must be declared by the endpoint.

## 4. Required adversarial grid beyond the reference kernel

The first module proves only that projection composition can be non-commutative. The next empirical/research grid should cross:

- Motion regressors: 6, 12, 24 parameter expansions; derivatives; censor/spike regressors; framewise-displacement and DVARS-based censoring.
- Physiological regressors: white-matter/CSF, aCompCor/tCompCor, RETROICOR/RVT/HRV where recorded, and lagged/shifted regressors.
- Global signal and ICA: with/without GSR; ICA-AROMA/ICA-FIX or equivalent, with component-admission provenance.
- Temporal operations: high-pass, low-pass and band-pass filters; run boundaries; detrending; interpolation; censoring; pre-whitening.
- Composition order: regression-before-filter, filter-before-regression, one-step joint regression/filter, and explicit sequential orthogonalization.
- Acquisition strata: task/rest; single-/multi-band; single-/multi-echo; TR and run length; site/scanner; motion burden.
- Signal controls: known task-locked responses, network perturbations, transient response, weak distributed effects, polarity controls, and known nulls.
- Nuisance-only controls: motion-linked bursts, respiratory/cardiac effects, drift, censoring and time-shifted nuisance regressors while latent target is held fixed.

Do not assume the same operation is valid for all endpoints. A temporal filter that intentionally removes task frequencies cannot be judged against a task-signal preservation target without an endpoint-specific declaration.

## 5. Qualification outputs

Report separately:

- residual energy in every nuisance design subspace after the *complete* workflow;
- intended-signal recovery, direction, attenuation and latency/phase distortion;
- retained temporal degrees of freedom and effective rank;
- QC–FC and distance-dependent residual artifact measures;
- network identifiability and test–retest reliability;
- behavioural/task prediction when the endpoint calls for it;
- participant/site attrition and quality-stratified performance;
- runtime, memory, determinism and complete operator/configuration digests.

The main result should be a multi-objective operating surface. No one scalar score may trade away a failed nuisance-removal guarantee by improving tSNR or prediction alone.

## 6. Proposed engineering gates

These are proposed gates for a frozen endpoint and are not established external standards:

- Joint-projection numerical oracle: final residual leakage into each declared design subspace no greater than (10^{-10}) on well-conditioned synthetic fixtures.
- Sequential re-entry sentinels: detect a planted post-projection nuisance leak of at least (10^{-6}) in deterministic reference fixtures.
- Exact idempotence of each orthogonal residualizer within (10^{-10}) relative Frobenius error for well-conditioned designs.
- Order-sensitivity report: never suppress a material operator-order difference by reporting only one preferred order.
- Signal preservation: endpoint-specific minimum meaningful effect and latency/phase tolerances, preregistered independently of denoising outcomes.
- Empirical calibration: confidence intervals, participant-level effective N, independent sites, and frozen external evaluation.
- Evidence: exact operator order, selected regressors, filter parameters, rank decisions, retained DoF, preprocessing versions, source data and code digests in the receipt.

A well-conditioned synthetic projection test does not qualify a noisy empirical design matrix or a frequency-selective filter. Those require additional conditioning diagnostics and endpoint-specific validation.

## 7. Industry comparison

Compare against pinned native and matched configurations of fMRIPrep, XCP-D and CONN. Treat their preprocessing and denoising operations as versioned operator graphs, not opaque names.

The 2026 denoising benchmark provides a more appropriate current baseline than a simple head-motion regression comparator because it incorporates one-step regression, pre-whitening and time-shifted physiological confounds across multiple acquisition types. The 90-pipeline multi-echo benchmark adds another important comparator family. Reproduce the relevant configurations where inputs and licenses permit; do not claim a SOTA win from the synthetic reference kernel.

Use identical source data, atlas/ROI definition, participant split, endpoint, motion/QC strata and acquisition strata. Record each tool's native recommendations separately from any harmonized comparison. Include participant attrition and degrees of freedom so aggressive censoring does not appear to win merely by removing difficult participants.

## 8. Execution order

1. Establish and test the projection-order reference kernel.
2. Add a strict smoke report and content-addressed evidence receipt.
3. Extend the model from projection-style removed subspaces to finite impulse response and frequency-selective filters with explicitly declared boundary/edge handling.
4. Add censoring/interpolation and run-boundary composition tests.
5. Import frozen fMRIPrep/XCP-D/CONN configurations and extract their actual ordered operator manifests.
6. Reproduce synthetic and then empirical comparisons on a frozen dataset release and external cohort.
7. Link the output to RCAF claim qualification: each result must identify which endpoint and which operator graph it can support.

## 9. References

- Lindquist, M. A. et al. (2019). *Modular preprocessing pipelines can reintroduce artifacts into fMRI data*. Human Brain Mapping. https://doi.org/10.1002/hbm.24567
- *Benchmarking fMRI Denoising Pipelines* (first published 1 July 2026). Human Brain Mapping. https://doi.org/10.1002/hbm.70561
- *An evaluation of the efficacy of single-echo and multi-echo fMRI denoising strategies* (2026). PubMed record: https://pubmed.ncbi.nlm.nih.gov/42039094/
- Parkes, L. et al. (2018). *An evaluation of the efficacy, reliability, and sensitivity of motion correction strategies for resting state functional MRI*. NeuroImage. https://doi.org/10.1016/j.neuroimage.2017.12.073
- fMRIPrep confounds outputs: https://fmriprep.org/en/latest/outputs.html
- CONN denoising pipeline: https://web.conn-toolbox.org/fmri-methods/denoising-pipeline

## Claim limits

The reference kernel demonstrates mathematical order effects for explicit finite-dimensional projections. It is not a reproduction of fMRIPrep, XCP-D or CONN; it does not implement general band-pass filtering, scrub interpolation, motion correction, physiological preprocessing, ICA, or empirical neuroimaging analysis. No empirical pipeline or biological claim is qualified by this increment.
