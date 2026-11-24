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

    /tmp/vis-dvpc-014 DELTAS.tsv EXPECTED_BUNDLE_LOCK_SHA256 EXPECTED_CLUSTER_ASSIGNMENT_MANIFEST_SHA256

Both expected SHA-256 values must come from separately frozen receipts/preregistration and must be supplied independently of the TSV. The first locks the evaluated source/evaluator evidence. The second locks the canonical episode-to-cluster assignment manifest. The TSV metadata must match both external values. Changing cluster membership after inspecting outcomes therefore requires changing the manifest digest and fails against the preregistered value.

The expected bundle lock must already have passed VIS-DVPC-012 verification. The cluster assignment digest must be computed from the exact canonical manifest bytes. This analyzer validates the two digests and the delta table; it does not parse source media or independently recompute the manifest hash. Freeze both under independent custody before policy developers see private test outcomes.

## Input contract

All records are UTF-8, tab-separated. Empty lines are forbidden. Metadata comes before the exact DELTA_HEADER and no unknown columns are accepted.

Required metadata keys:
- format: exactly VIS-DVPC-CLUSTER-DELTAS-001-v1
- experiment_id, dataset_id, dataset_revision
- split: test or private_test; validation data is not a final confirmatory test
- cluster_unit: a defensible independent sampling unit, not an individual frame
- primary_metric: the preregistered primary endpoint
- positive_is_better: literal true; each delta is candidate minus baseline on that metric
- bundle_lock_sha256: digest of the upstream independently frozen bundle lock, checked against the first separate CLI argument
- cluster_assignment_manifest_sha256: digest of the canonical episode-to-cluster assignment manifest, checked against the second separate CLI argument
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


## Compute × observation interaction handoff

The dataset-agnostic analyzer is now explicitly reusable by the paired compute × observation evaluator in PR #13281. The evaluator-side adapter at `research/decision-value-interaction-001/src/cluster_adapter.rs` exports the exact `VIS-DVPC-CLUSTER-DELTAS-001-v1` input contract consumed by this tool.

The adapter performs a hierarchy-preserving reduction before inference:

1. Recompute and validate the certificate's four signed effects from its raw four-arm regret values.
2. Average hidden-world counterfactual effects within each episode.
3. Average episode means within each preregistered independent cluster.
4. Emit exactly one delta per cluster for each of four contrasts: compute benefit, observation benefit, joint benefit, and compute×observation interaction.

It refuses duplicate (cluster, episode, hidden-world) rows, episode-to-multiple-cluster assignment, changes to task/loss/evaluator/environment/action/resource/model/policy context, changes to an episode's initial-state or seed identity, and a changed hidden-world cohort within a cluster. It does not calculate confidence intervals or p-values; this analyzer remains the owner of those statistics.

The synthetic interoperability fixtures are:
- `fixtures/vis-dvpc-compute-observation-cluster-deltas-v0.1.tsv`: six scene clusters and four predeclared comparisons (24 cluster-level deltas);
- `fixtures/vis-dvpc-compute-observation-cluster-assignments-v0.1.tsv`: twelve episode-to-scene assignments, each bound to an initial-state digest.

The synthetic assignment manifest SHA-256 is `8257074ad9d509a144c807fe1eeca8c0f29fb9c5101589a3e220d207236d0aab`. The all-`a` bundle-lock value is a placeholder for this synthetic fixture only. CI computes the assignment file hash, checks it against the pinned value, and passes it independently to the analyzer. A real campaign must freeze its assignment manifest and source-bundle lock before evaluation. The fixtures are not benchmark results.

The current workflow pins Rust 1.96 and:
- runs the analyzer's adversarial unit tests;
- builds the standalone binary;
- runs strict Clippy via `clippy-driver`;
- executes the synthetic compute×observation handoff fixture;
- checks deterministic receipt replay;
- verifies that a mismatched external bundle-lock digest is rejected.

### Important interpretation limits

The cluster bootstrap interval is an approximate percentile interval, not distribution-free finite-sample coverage. The sign-flip p-value relies on independent cluster deltas and its declared symmetry-under-the-null assumption. Bonferroni adjustment covers the four preregistered contrasts; it does not fix a biased sampling design, outcome leakage, post-hoc selection, or an incorrectly chosen cluster unit.

Five clusters are only a minimum mechanics threshold. Real confirmatory work still requires a preregistered power/minimum-detectable-effect analysis, substantially more independent clusters where feasible, appropriate hierarchical sampling, and a replication on a second held-out domain. A positive cluster-level interval is evidence only for the declared metric, population, comparator family and design—not general SOTA superiority.


## Cluster-assignment manifest format

The assignment manifest is a separate UTF-8 TSV with exact header:

    ASSIGNMENT_HEADER<TAB>episode_id<TAB>cluster_id<TAB>initial_state_sha256

It has one row per evaluated episode. Duplicate episode IDs, missing/extra episodes, cluster reassignment, or a mismatch with the episode's initial-state commitment must fail before cluster deltas are emitted. The adapter in #13281 receives the parsed typed assignment list and verifies those conditions against every paired per-world certificate. The analyzer receives only the separately supplied expected manifest digest and rejects TSV metadata carrying a different value.

This split keeps the boundaries explicit:
- the evaluator adapter validates the assignment mapping against the episode certificates;
- the external integrity/preregistration process freezes and hashes the canonical manifest;
- the downstream analyzer checks that the delta table carries the independently supplied expected digest;
- the broader evidence-bundle lock independently binds the source/evaluator evidence.

A digest alone does not prove that an arbitrary assignment map is scientifically appropriate. Cluster units must be selected from the sampling design before outcomes are inspected, with sequence/scene/session/subject definitions justified for the domain.
