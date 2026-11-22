# RCAF Groupwise Calibration and Worst-Group Shift Guard v1

**Status:** executable synthetic stratum-conditional calibration primitive; not empirical multi-site validation.  
**Base:** estimator-level factorial response and disjoint held-out null calibration.  
**Purpose:** expose and guard against subgroup null miscalibration that can be hidden by a pooled threshold.

## 1. Paradigm shift: from pooled average to conditional measurement validity

A single pooled null threshold answers: "Does this statistic behave acceptably on the mixture of calibration observations?" It does not answer: "Does it behave acceptably at every site, echo regime, motion-burden class or acquisition stratum?"

A low-motion subgroup can dominate a pooled calibration set while a high-motion subgroup has an elevated null tail. A pooled false-positive rate can therefore look acceptable while one clinically or scientifically important subgroup is badly miscalibrated.

RCAF now supports a predeclared stratum set such as:

- site × scanner;
- single-echo versus multi-echo;
- task versus rest;
- low-, moderate- and high-motion burden;
- acquisition protocol or sequence family.

Strata must be defined before the evaluation outcomes are examined. The synthetic smoke fixture uses three such labels; these are test labels, not actual participant/acquisition strata.

## 2. Two calibration policies

For estimator e and predeclared stratum g, let q(g,e) be its conservative upper-tail threshold computed only from that stratum's calibration-null observations.

### Stratum-conditional threshold

Evaluation observations in known stratum g are tested against q(g,e). This preserves the local scale and distribution of each acquisition group, conditional on within-stratum exchangeability.

### Common worst-group guard

For each estimator, construct:

`q_guard(e) = max_g q(g,e)`

The common threshold is applied to every expected stratum. Since it is no smaller than any local threshold, it is more conservative for every calibrated group. This is a guardrail against a threshold that was calibrated only for an average mixture, at the cost of reduced power in easier groups.

The common guard is emitted only when **every expected stratum has enough calibration observations to resolve alpha**. If one expected stratum is missing or under-resolved, the common guard is undefined and the workflow fails closed rather than pretending that the remaining groups represent it.

Both policies are reported side by side. The purpose is not to declare one universally superior; it is to make the power-versus-worst-group-error trade-off visible.

## 3. Unknown and under-resolved groups

An evaluation stratum not present in the predeclared calibration set is **unseen**. It receives no threshold and is labelled `abstain_unseen_stratum`. It is not silently routed to the nearest group or assigned the pooled threshold.

An expected stratum with insufficient calibration replicates has no local threshold. Its local rate is null, not zero. It also prevents the common guard from being emitted.

An expected stratum with no evaluation observations is explicitly marked through `all_strata_evaluated = false`; a missing group is not reported as a zero false-positive rate.

This produces an operational rule:

> Known and calibrated stratum → evaluate under the declared local threshold and common guard.  
> Unknown or under-resolved stratum → abstain and obtain calibration evidence before making a thresholded claim.

## 4. Statistical assumptions and boundaries

The finite-sample rank rule is the same conservative upper-tail rule used by the preceding held-out module:

`k = floor(alpha * (n_g + 1))`

If k is zero, no threshold exists. Otherwise q(g,e) is the k-th largest calibration-null score, with strict `score > q` rejection so ties are conservative.

The groupwise interpretation is conditional on exchangeability **within each predeclared stratum** between calibration and evaluation null observations. Merely assigning different labels does not make repeated windows independent. For real data, calibration/evaluation must be separated at the independent cluster unit—typically participant, session or site—and within-participant dependence must not be disguised as additional replicates.

This v1 handles discrete known strata. It does **not** guarantee validity under arbitrary continuous covariate shift, changing conditional score distributions within a stratum, or unseen scanner/site populations. Importance-weighted conformal methods require a defensible density-ratio model and their own diagnostics; they must not be approximated by guessing weights.

The three estimator thresholds are calibrated separately at per-estimator alpha. This does not provide family-wise error control if the best estimator is selected after inspecting the held-out results. Confirmatory work must pre-register one primary estimator or add a multiple-comparison correction / joint max-statistic calibration.

Wilson intervals quantify binomial Monte Carlo uncertainty only. They do not account for site sampling, participant dependence, generator misspecification or transportability.

## 5. Executable implementation and report

The Rust module is `research/rcaf/src/groupwise_null_calibration.rs`.

It validates:
- unique, predeclared stratum IDs;
- distinct calibration/evaluation split IDs;
- unique and non-overlapping replicate IDs across all strata;
- finite non-negative statistics and valid direction cosines;
- declared calibration strata only;
- explicit unseen evaluation strata;
- calibration resolution separately for every stratum.

For all three RCAF estimators it reports:
- local stratum threshold;
- shared max-over-strata guard threshold;
- calibration and evaluation counts;
- minimum resolvable alpha;
- local and common-guard held-out false-positive rate;
- signal detection, directionally correct signal rate and wrong-direction acceptance;
- worst observed group false-positive rate for each threshold policy;
- completeness flags for expected strata.

The synthetic CLI fixture has 99 calibration-null and 100 evaluation replicates in each of three strata plus ten high-scoring observations from an unseen scanner. It checks 5% synthetic held-out null rates under local calibration, the max-quantile guard, direction-specific counts, finite-sample resolution and explicit abstention for the unseen stratum.

## 6. Current industry comparison anchors — November 2026

The benchmark should compare RCAF against complete, pinned pipeline outputs rather than describe this Rust reference kernel as a replacement preprocessor.

### fMRIPrep

The fMRIPrep 25.2.7 change log (5 October 2026) records fixes to BOLD mask generation and MCFLIRT motion-parameter reconstruction, including a nondeterministic mask-generation issue and motion estimates that differed from the original MCFLIRT representation in 25.0.0–25.2.6. This is a concrete reason to record the exact package version, container digest and motion-confound derivation when benchmarking.

- Release notes: https://fmriprep.org/en/25.2.7/changes.html
- Do not silently mix derivative versions when comparing estimator statistics.

### XCP-D

XCP-D 26.2.0 is listed as the current release on its GitHub releases page, dated 17 August 2026. It adds `--censor-between` for censoring short sets of non-outlier frames between outliers. The ABCC v4.1.0 derivative snapshot (18 November 2026) uses XCP-D 26.1.1 with the ABCD-specific mode, notch filtering of motion parameters, multiple atlases and connectivity matrices constructed from concatenated low-motion time series. That ABCC snapshot is a reproducible dataset baseline, not the newest XCP-D release.

- Releases: https://github.com/PennLINC/xcp_d/releases
- ABCC processing details: https://docs.abcdstudy.org/latest/documentation/imaging/abcc_xcpd.html

### CONN

CONN 25b (February 2026) includes a combined denoising and QC tab, with Data Validity, Data Quality and Data Sensitivity scores, plus connectome predictive modelling and a broader default atlas set. RCAF should ingest or reproduce these QC dimensions rather than invent a scalar score that ignores them.

- Release and features: https://sites.google.com/view/conn/home

### Cross-pipeline denoising benchmarks

*Benchmarking fMRI Denoising Pipelines* (Human Brain Mapping, first published 1 July 2026; DOI 10.1002/hbm.70561) covers task/rest and single-/multi-band, single-/multi-echo data. It explicitly includes one-step nuisance regression, pre-whitening and temporally shifted physiological regressors, and measures tSNR, retained DoF, motion correction and signal preservation.

A second 2026 study compares 90 single-/multi-echo pipelines in 358 adults across denoising and behavioural-prediction metrics. Its main lesson for RCAF is that denoising and downstream prediction can trade off; no universal one-number winner should be assumed.

- https://doi.org/10.1002/hbm.70561
- https://doi.org/10.1162/NETN.a.547

## 7. Proposed benchmark targets

These are proposed qualification targets, not established field standards or achieved results:

| Measure | Proposed target |
|---|---:|
| Per-stratum held-out null FPR | ≤5% |
| Worst observed stratum FPR under common guard | ≤5% |
| Wrong-direction acceptance | ≤1% |
| Moderate-effect directionally correct detection | ≥80% |
| Lower 95% bound on correct detection | ≥75% |
| Nominal 95% interval coverage | 93–97% |
| Unseen-stratum threshold borrowing | 0 |
| Under-resolved-stratum threshold borrowing | 0 |

A useful result is not just "the global FPR is 5%." It should expose the entire groupwise error/power surface, and identify where the method must abstain.

## 8. Next research frontier

1. Add acquisition-quality and motion-burden features to each stratum and audit within-stratum residual shift.
2. Test continuous-shift diagnostics and importance-weighted calibration as a separate method, with weight-estimation error and effective sample size reported.
3. Add site-held-out calibration/evaluation and participant-clustered uncertainty.
4. Add simultaneous max-statistic or pre-registered primary-estimator handling for the three estimator comparisons.
5. Repeat over independent nuisance innovation sets, not just a single deterministic synthetic fixture.
6. Reproduce fMRIPrep 25.2.7, XCP-D 26.2.0 and CONN 25b configurations on a pinned public dataset release, retaining raw configurations and outputs.
7. Evaluate tSNR, DoF retained, motion correction, signal preservation, test–retest connectivity reliability, behavioural prediction, attrition and compute alongside RCAF's null/power/direction metrics.

## Claim limits

This v1 provides group-conditional synthetic null calibration and a common worst-stratum threshold. It does not establish arbitrary distribution-shift validity, empirical site-specific FPR, participant-level statistical power, a validated biological endpoint or superiority over fMRIPrep/XCP-D/CONN. The current three-stratum fixture is a test of implementation semantics, not a study of real scanners or participants.
