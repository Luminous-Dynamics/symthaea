# VIS-DVPC-014: Cluster-level paired inference v1

**Status:** Draft reference analyzer.  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8578  
**Upstream integrity gate:** https://github.com/Luminous-Dynamics/symthaea/pull/8569  
**Research anchor:** DEEP, https://arxiv.org/abs/2609.35910

## The statistical unit is the independent cluster

A thousand correlated frames from one driving sequence are not a thousand independent replications. This tool takes one already-computed, paired primary-metric delta per independent cluster and comparison. Examples of appropriate units may include sequence, scene, session, subject or episode; the adapter and domain expert must choose the unit that matches the sampling process. The values frame, image, crop, row, sample, timestamp and detection are rejected as cluster units by this reference implementation.

Research motivation:
- DEEP evaluates downstream decision value under latency and energy constraints and charges allocator overhead. Its results show why perception-level gains are not an adequate substitute for downstream decision outcomes.
- The econometrics guide to cluster-robust inference emphasizes that the cluster, not the individual observation, is the unit of inference and recommends caution where cluster counts are small.
- Benchmarking guidance recommends strong competitors, preregistered criteria, complete reporting, effect sizes and investigation of heterogeneity across datasets/criteria.

## CLI and frozen lock

Build:

    rustc --edition=2024 -O tools/vision-dvpc-cluster-inference.rs -o /tmp/vis-dvpc-014

Run:

    /tmp/vis-dvpc-014 DELTAS.tsv EXPECTED_BUNDLE_LOCK_SHA256

The expected SHA-256 argument must come from the separately frozen VIS-DVPC-012 receipt/preregistration. The input TSV's bundle_lock_sha256 metadata must match it. If the data and its metadata are edited without changing the expected lock, analysis fails. This standalone analyzer does not itself validate source-media bytes; the supplied lock must already have passed VIS-DVPC-012 verification.

The lock is a commitment to the evaluated evidence, not proof of authenticity by itself. Freeze it under independent custody before policy developers see private test outcomes.

## Input contract

All records are UTF-8, tab-separated. Empty lines are forbidden. Metadata comes before the exact DELTA_HEADER and no unknown columns are accepted.

Required metadata keys:
- format: exactly VIS-DVPC-CLUSTER-DELTAS-001-v1
- experiment_id, dataset_id, dataset_revision
- split: test or private_test; validation data is not a final confirmatory test
- cluster_unit: a defensible independent sampling unit, not an individual frame
- primary_metric: the preregistered primary endpoint
- positive_is_better: literal true; each delta is candidate minus baseline on that metric
- bundle_lock_sha256: digest of the upstream independently frozen bundle lock, checked against the separate CLI argument
- comparison_family_size: number of predeclared candidate-vs-baseline comparisons and must equal the unique comparison count in the file
- bootstrap_seed, bootstrap_replicates: frozen deterministic resampling controls
- familywise_confidence: in [0.5, 1); adjustment uses Bonferroni across the predeclared comparison family

Exact table header:

    DELTA_HEADER<TAB>comparison_id<TAB>candidate_id<TAB>baseline_id<TAB>cluster_id<TAB>delta

Each cluster appears once in each comparison. All comparisons must use the exact same set of cluster IDs. Duplicate rows, changing candidate/baseline identities mid-comparison, missing/extra clusters, nonfinite values, metadata drift or an incorrect family size fail closed.

Rows are already group-level paired deltas created by the post-commit evaluator. Do not average per-frame observations here and then call them independent clusters. The upstream evaluator must preserve grouping and score the same committed policy on the same paired traces. Policy choices and outcome labels remain separate.

## Statistics emitted

For every predeclared comparison the receipt reports:

- equal-weighted mean and median of the independent-cluster deltas;
- minimum and maximum cluster deltas;
- counts of positive, negative and tied clusters;
- a deterministic Bonferroni-adjusted percentile cluster-bootstrap interval;
- a sign-flip diagnostic p-value and Bonferroni-adjusted p-value.

The bootstrap is a percentile interval approximation, not an exact finite-sample distribution-free confidence interval. It resamples whole cluster deltas with replacement and adjusts each two-sided interval across the frozen family. The seed, replicate count, family size, cluster count and exact bundle lock appear in the receipt. Exact replay and row-order invariance are tested.

For up to 16 clusters, the sign-flip calculation enumerates all sign configurations. For larger samples, it uses deterministic Monte Carlo sign flips with a plus-one correction. Its interpretation requires independent cluster deltas and a symmetric-under-the-null sign-flip assumption; it is a diagnostic, not a universal randomization test and not a substitute for a design-appropriate estimator or confirmatory interval.

## Interpretation / no-claims

- CLUSTER_LEVEL_INFERENCE_ONLY means the analyzer computed stated summaries from admitted inputs.
- A positive mean and interval above zero support improvement for the declared primary metric and tested comparison, within this analysis' assumptions; they do not prove general SOTA superiority.
- performance_claim=NONE is literal. Users must interpret the results, report failed comparisons and inspect preregistered subgroup/domain results.
- This tool does not compute harm certificates, CVaR95, latency/energy budgets or legality. Those metrics must be independently reported from the evaluator and safety tools.
- Five clusters is a low executable-mechanics floor, not an adequate scientific sample-size recommendation. For a real claim, preregister a power/minimum-detectable-effect calculation, aim for dozens or more genuinely independent clusters where feasible, report uncertainty, and replicate on a second held-out domain.
- Do not rank methods on one headline mean alone. Compare against all-cheap, all-full, uniform, same-budget random, confidence/entropy routing, perception-gain routing, a simple lightweight router and the strongest available published baseline. Measure allocator overhead and charge it against separate latency and energy constraints before interpreting utility deltas.

No third-party dataset, frames or evaluator labels are included in this PR. Synthetic unit tests test arithmetic/state mechanics only.
