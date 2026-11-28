# CMAQ-063 — Unified manifest-driven qualification contract

Status: **design/preregistration contract**. This document does not claim that any current method is statistically qualified, SOTA, or production-ready. It composes the existing CMAQ work; it does not replace it.

## 1. Objective

Make every comparator answer the same registered question on the same observations, then report statistical validity, finite-sample behavior, compute, memory, and operational behavior in one reproducible evidence bundle.

The primary estimand is **useful detection capability per unit of measured resource cost under one fixed, auditable anytime false-alarm budget**. No scalar score may trade away a null-guarantee violation.

The primary null remains direct conditional paired equality:

```
Law(X_t | F_(t-1), Z_t) = Law(Y_t | F_(t-1), Z_t)
```

where the covariate Z_t is available before the response pair and F_(t-1) includes all prior observations, policy state, optimizer state, selected actions, RNG state, monitor decisions and evidence commits.

## 2. Canonical components — reuse, do not fork

- C059 (#14207 / PR #14208): stream generator, seed schedule, SCPT reference, exact binomial intervals, paired metrics and raw outcomes.
- C060 (#14209 / PR #14210): predictable one-expert selection and sparse selected-subsequence state.
- C061 (#14211 / PR #14254): bounded previsible encoder, to be integrated with a typed parent evidence receipt.
- C062 (#14253 / PR #14255): fixed-prior localized context tree and one global mixture.
- C054/C055/C057/C058: fixed family, full-reset comparator, predictable full-information adaptation and capital-only renewal.
- C053/C056: compositional risk budget and typed null contract.

C063 must not add a second null-manifest format, PRNG, alpha allocator, SCPT implementation, or restart oracle.

## 3. Frozen experiment manifest

The manifest is immutable before confirmatory streams are generated. It binds:

1. **Protocol:** null definition, filtration timeline, covariate availability, response-pair semantics, valid dependence assumptions and registered negative controls.
2. **Grid:** every arm, scenario, horizon, nominal alpha, replicate count, change point and primary contrast. No result-driven cell removal or sample-size expansion.
3. **Generator:** scenario configuration, generator implementation/version, PRNG algorithm/version, experiment/scenario/replicate seed derivation and stream digest.
4. **Arm identity:** exact source commit, dependency lockfile digest, method/configuration, feature-map/checkpoint digests, prior weights, selector seed/policy, training/calibration split and restart semantics.
5. **Metrics:** versioned outcome definitions, independence unit, quantile convention, treatment of early alarms/misses, resource counters and multiplicity procedure.
6. **Artifacts:** raw-result schema, aggregate schema, receipt/digest rules and expected output inventory.

The canonical manifest encoding must be deterministic and versioned. Its digest is computed over the canonical manifest payload excluding the digest field itself. Every raw row references the manifest digest and exact arm/configuration digest.

For a synthetic paired replicate, stream identity is keyed by (experiment, scenario, replicate), never by algorithm arm. Every arm must consume the same stream digest. For real observations, the resampling and uncertainty unit is the source/common-cause block; time points, windows or records within one block must not be treated as independent replicates.

The validator fails closed on missing cells, duplicate IDs, duplicate arms, changed stream digests, incomplete raw outcomes, unsupported null contracts or a code/configuration digest mismatch.

## 4. Primary comparison table

Same-null primary comparators:

| Arm | Role | Required labeling |
|---|---|---|
| C054 | Fixed kernel-family mixture | Fixed inventory and prior |
| C055 | Full policy + capital reset | Full reset; correctness comparator |
| C057 | Predictable full-information adaptation | Evaluates/updates all registered experts |
| C058 | Capital-only renewal | Shared predictable policy state; not a full reset |
| C060 | Predictable sparse selection | One selected expert; bandit-feedback state |
| C062 | Localized context tree | Fixed prior over registered local processes |
| C061 | Learned predictable representation | Include only after parent evidence transaction integration |
| SCPT | Direct conditional paired baseline | Faithful past-only KNN/betting procedure; separately label fixed external bandwidth and first-50 warm-up variant |

The best fixed expert selected with hindsight is an oracle diagnostic, never an online alpha-valid competitor.

Adjacent-task protocols are reported separately:
- NeurIPS 2025 Online RFF-MMD: online unconditional two-sample change detection.
- ICML 2026 SKCI: conditional independence testing, not direct paired conditional-distribution equality.
- ICML 2026 anytime-valid kernel-family procedures: include only after checking the exact null and filtration.
- Operational monitoring systems: fixed-window KS/PSI/JS, MMD, PCA/reconstruction drift, ADWIN/Page-Hinkley, and model-performance/label-delay monitoring.

Do not collapse adjacent tasks into one leaderboard or imply that an operational platform offers an anytime-valid guarantee unless the method contract supports that claim.

## 5. Exact inferential requirements

### Null calibration

For every preregistered arm × valid-null scenario × horizon × alpha cell:
- false alarms / independent full-horizon replicates;
- two-sided exact Clopper–Pearson interval;
- one-sided simultaneous upper bound over the complete frozen null-cell family;
- explicit source/common-cause block identity for real data.

Keep pointwise intervals distinct from the familywise bound. A Monte Carlo calibration screen is not the mathematical proof of the e-process assumptions; the proof is not a substitute for finite-sample implementation checks.

### Detection and paired contrasts

- Unconditional power includes every registered alternative replicate in its denominator.
- An alarm at or before the registered change point is an early alarm, not a detection.
- A miss or early alarm contributes restricted delay H+1.
- Report restricted mean and p50/p90/p95; do not publish complete-case delay alone.
- Pair arm differences on identical replicates/blocks.
- Predeclare a small primary contrast family. Use a simultaneous paired method (e.g. a preregistered max-statistic bootstrap or conservative Bonferroni intervals) for superiority claims. Per-scenario percentile bootstrap intervals without multiplicity adjustment are exploratory.
- Report each null and alternative family separately, plus the worst-family result. Pooled metrics are supplementary only.

The bootstrap unit must be the independent replicate or source/common-cause block, not a step or overlapping segment.

## 6. Compute and memory accounting

The harness must distinguish mathematical operation counters from measured wall-clock behavior. Every procedure emits the counters it can authoritatively count; the report records unavailable counters as unavailable, never zero.

Required counters:
- response and covariate feature-coordinate evaluations;
- response-kernel evaluations;
- KNN distance-coordinate evaluations and distance comparisons/sort comparisons;
- context-tree router membership tests;
- segment-tree leaves/ancestors changed;
- expert-selection probability/RNG/weight-update work;
- representation inference, training and parameter-update work;
- bytes allocated/read/written where measurable;
- peak resident memory, persistent state growth and model/manifest storage;
- p50/p99 per-update latency under full-horizon valid-null operation;
- p50/p99 live-to-alarm latency/work under alternatives.

Important known accounting seams at the current implementation:
- C059 SCPT `neighbors()` scans all prior covariates and sorts the complete distance vector. Its current explicit kernel counter does not count these KNN distance and sorting operations. Do not present that field as total SCPT compute.
- C062's segment-tree update is (O(d\log N)), but current child routing uses a linear `find` over siblings at each visited depth. Report router comparisons separately; do not claim whole point-location is (O(d\log N)) without an indexed router.
- C060's selected feature count is not total selector cost. Include RNG, probability-vector, log-weight and sparse-state maintenance.
- C061 must report encoding plus training/update cost; do not hide post-pair learning work.

Latency on an alternative may stop when the monitor alarms. That is a valid operational measure, but it must be labelled live-to-alarm. Sustained full-horizon per-update cost must also be measured on non-alarming valid-null streams.

## 7. Industry-facing evaluation

Industry monitoring products are useful comparators for operational coverage, alert workflow and maintenance burden, but their default task and guarantees may differ from CMAQ's primary null.

Use reproducible data and locked configurations. For each baseline document:
- threshold/window calibration data and whether labels are required;
- full monitoring horizon and actual false-alert rate;
- detection/alert delay and label-arrival delay;
- missingness, data-quality and model-performance coverage;
- alerts per asset/day, operator workload and alert stability;
- latency, memory, deployment complexity and integration requirements;
- whether a formal time-uniform guarantee exists, a fixed-holdout calibration exists, or the alert is heuristic.

The differentiator CMAQ seeks to prove is not merely that it emits more sophisticated metrics. It is that a single process can offer conditional paired monitoring, localized evidence, previsible adaptation, explicit resource allocation and independently replayable evidence receipts. Product parity and production readiness require additional integration and customer-workflow evaluation.

## 8. Confirmatory benchmark grid

Retain the C059 preregistration:
- horizons 128, 256, 512 and 1,024;
- alpha in {0.01, 0.05, 0.10};
- at least 10,000 independent full-horizon valid-null paths per primary cell;
- at least 5,000 alternative paths per primary cell;
- a precision-based, arm-independent rule for expansion up to 50,000 null paths, frozen before arm results;
- a 4,096-observation SCPT stress horizon, separately labelled.

Extend the alternative grid with:
- context dimension 10, 50, 100 and a high-dimensional stress cell;
- affected support 0.5%, 1%, 5%, 10%, 25%;
- axis-aligned, oblique, conjunction, smooth-bump, narrow-manifold and moving support;
- conditional location, scale, tail, multimodality and equal-marginal pairing shifts;
- abrupt, gradual, temporary, periodic, oscillating, high-/low-frequency, delayed and repeated on/off changes;
- partition-boundary and finite-feature blind-spot alternatives.

Valid nulls include predictable covariate drift, heteroskedasticity and declared serial/common-cause dependence that preserve conditional paired equality. Current-pair feature/selector leakage, non-previsible action choice, replay, duplicate IDs, unbounded features and post-hoc threshold/prior changes are integrity negative controls, not ordinary valid-null cells.

## 9. Prospective performance gates

These are targets, not current results:

1. No simultaneous null-family upper-bound violation at the preregistered level.
2. At least 20% lower restricted mean delay on three preregistered sparse/local alternatives versus the designated global and SCPT comparators, at matched alpha.
3. No more than 2 percentage points of worst-family power loss against the full-information comparator when claiming a compute reduction.
4. At least 4× fewer feature-map evaluations for C060-style sparse allocation versus full-information C057/C058 on the registered compute-constrained cells.
5. No hidden regression on high-frequency, temporary, late, rare-context or boundary-adversarial alternatives.
6. All claims supported by paired uncertainty intervals and independently inspected raw artifacts.

The primary decision object is the **power-versus-resource Pareto frontier subject to the same statistical guarantee**. A faster method outside the false-alarm constraint is disqualified, not assigned a lower composite score.

## 10. Validation layers and stop rules

Every execution is labelled exactly one of:
- **Smoke:** tests wiring and artifact generation only.
- **Deterministic correctness:** unit/property tests and exact-head compile/format/lint.
- **Empirical qualification tranche:** preregistered independent path counts and complete raw outputs.
- **Independent review:** manifest, proofs, code, counters, and artifacts checked separately.
- **Deployment evidence:** real source blocks, operational integration and locked live-monitor evaluation.

Queued, skipped, pending, cancelled or absent workflows are not passes. A missing artifact, incomplete cell, unexplained digest mismatch, null-bound breach or failed negative control blocks promotion. No weakening of the frozen manifest or acceptance gates after outcomes are visible.

## 11. Integration sequence

1. C059: correct the compute-accounting schema for SCPT KNN work; separate exploratory paired intervals from familywise confirmatory contrasts; add sustained null-latency reporting.
2. C060: export selection/action cost and RNG receipts into the common outcome schema; keep regret claims separate from power/delay.
3. C061: integrate typed pair/evidence receipts before adding its results to the primary table.
4. C062: export region-router comparison counts and generated property-test receipts.
5. C063: implement manifest validation, common arm adapters and report assembly only after the above contracts are stable.
6. C064: evaluate predictable multi-fidelity allocations as a distinct research strategy against C060/C062 and the full-information oracle.

## 12. Research references and protocol boundaries

- Zheng & Zhang (2026), *Sequential Conditional Paired-Sample Testing with Applications to Online Monitoring*, arXiv:2611.22886 — direct paired conditional-monitoring comparator.
- Kalinke & Gavioli-Akilagun (NeurIPS 2025), *Optimal Online Change Detection via Random Fourier Features* — important computational baseline for the separate unconditional online two-sample task.
- He & Sutherland (ICML 2026), *Sequential Kernel-based Conditional Independence Testing via Adaptive Betting* — relevant adaptive betting methods, different null.
- ICML 2026 Workshop, *Anytime-Valid Two-Sample Testing over Kernel Families* — relevant kernel-family mixture construction; null/filtration must be checked before any primary comparison.

This document is a preregistration and integration contract. It is not evidence that its targets have been met.
