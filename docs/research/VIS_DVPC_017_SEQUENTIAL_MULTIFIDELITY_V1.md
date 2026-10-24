# VIS-DVPC-017 — Sequential multi-fidelity perception replay

Status: research/evaluation tool; not a production router, risk certifier, or benchmark result.

## Why this experiment

Binary cheap/full routing hides an important question: after an intermediate perception pass, is another pass worth its incremental cost for this case? Treat the problem as sequential stop/continue decisions rather than one static confidence threshold.

The evaluator implements three tiers:
1. **Cheap** always runs.
2. **Medium** runs only if the medium policy's upstream risk state is CERTIFIED, its decision-time medium_gain_lcb meets the preregistered floor, and the upper-bound latency/energy reservation fits the remaining budgets.
3. **Full** is considered only after medium finishes and has complete measured costs within budget. Its policy risk state must be CERTIFIED; its decision-time full_gain_lcb must meet the same floor; and the full-stage upper-cost reservation must fit the remaining budget. If the medium stage exceeds its reservation estimate, escalation stops.

Latency and energy are separate hard constraints; the tool does not combine them into a scalar. The min_gain_lcb is in the utility units declared by the frozen utility specification; it is a policy threshold, not a learned or universally meaningful constant.

## Important boundary

The routing function only reads:
- decision-time gain lower bounds;
- globally supplied per-tier risk-certificate states;
- actual measured resource cost for already-executed stages;
- upper-cost reservations for the next stage; and
- the frozen latency/energy ceilings.

It must not use utility_cheap, utility_medium, utility_full, or any harm_* field to choose a tier. These are hindsight outcomes for scoring only. The regression test routing_does_not_depend_on_realized_utility_or_harm protects this separation.

This tool **does not independently verify** the upstream risk-certificate bundle, calibration, meter accuracy, sampling design, data licenses, or statistical independence. A string CERTIFIED in the input is an upstream assertion; it is not a certificate verified by this tool. Empirical harm rates in the receipt are descriptive and not finite-sample safety guarantees. Cluster-aware inference and preregistration are still required before making claims.

## Build and run

~~~sh
rustc --edition=2024 --test tools/vision-dvpc-sequential-fidelity.rs -o /tmp/vis-dvpc-017-tests
/tmp/vis-dvpc-017-tests
rustc --edition=2024 -O tools/vision-dvpc-sequential-fidelity.rs -o /tmp/vis-dvpc-017
sha256sum frozen-input.tsv
/tmp/vis-dvpc-017 frozen-input.tsv EXPECTED_SHA256_FROM_SHA256SUM
~~~

The program rejects any input-byte SHA-256 mismatch. Use an immutable, held-out file and preserve both the file and stdout receipt. The repository includes a five-trace synthetic smoke bundle at tests/fixtures/vision-dvpc-017-smoke.tsv; it is fixture data only, not empirical evidence.

## Strict TSV contract

Every line is nonblank. META rows must precede the exact TRACE header; metadata keys may not be duplicated, omitted, or added. Unknown/reordered columns fail closed. Numeric fields must be finite/non-negative where integer costs are used. The split must be test or private_test.

Required metadata keys (each line uses META<TAB>key<TAB>value):

| Key | Meaning |
| --- | --- |
| format | VIS-DVPC-SEQUENTIAL-001-v1 |
| experiment_id, dataset_id, dataset_revision | Stable identifier tokens |
| split | test or private_test |
| utility_spec_sha256 | Digest of the frozen utility definition |
| risk_certificate_bundle_sha256 | Digest naming the upstream evidence bundle; not verified by this tool |
| paired_unit_set_sha256 | Digest of the canonical trace/cluster/group identity set, computed and checked by the tool |
| expected_trace_count | Exact number of trace rows |
| expected_cluster_count | Exact number of unique cluster IDs; implementation rejects fewer than five, which is not a power recommendation |
| latency_budget_us, energy_budget_uj | Per-trace hard ceilings |
| min_gain_lcb | Non-negative minimum incremental utility lower confidence bound |
| medium_risk_certificate_state, full_risk_certificate_state | Upstream status; only CERTIFIED permits that tier |
| harm_rate_limit | Descriptive comparison threshold only, not a safety certificate |

The exact data header is:

~~~text
TRACE<TAB>trace_id<TAB>cluster_id<TAB>group_id<TAB>utility_cheap<TAB>utility_medium<TAB>utility_full<TAB>harm_cheap<TAB>harm_medium<TAB>harm_full<TAB>medium_gain_lcb<TAB>full_gain_lcb<TAB>medium_cost_latency_upper_us<TAB>medium_cost_energy_upper_uj<TAB>full_cost_latency_upper_us<TAB>full_cost_energy_upper_uj<TAB>allocator_latency_us<TAB>allocator_energy_uj<TAB>cheap_latency_us<TAB>cheap_energy_uj<TAB>medium_latency_us<TAB>medium_energy_uj<TAB>full_latency_us<TAB>full_energy_uj<TAB>allocator_cost_complete<TAB>cheap_cost_complete<TAB>medium_cost_complete<TAB>full_cost_complete
~~~

All costs are incremental for the named component. *_upper_* values are the decision-time upper reservations for an optional next stage. allocator_* is paid by the sequential policy; the fixed-tier baselines are standalone fixed-model costs. Booleans are exactly true or false. Utilities are any finite values where higher is better; harm is a preregistered binary outcome.

To create paired_unit_set_sha256, sort rows lexicographically by (trace_id, cluster_id, group_id), emit each identity as trace_id<TAB>cluster_id<TAB>group_id<LF> using UTF-8 bytes, and take SHA-256. The tool recomputes and requires an exact match; duplicate trace IDs fail.

## Receipt interpretation

The output reports per-policy utility mean, empirical harm rate, all-in latency/energy, budget violations, incomplete cost measurements, per-group utility/harm summaries, and the sequential route/reason per trace. It includes fixed cheap, medium, and full baselines. The sequential policy's comparison against cheap is paired by construction because all four policies are scored on the same trace rows.

A budget violation or incomplete measurement is not hidden. Do not drop bad rows or treat incomplete cost measurements as a pass. The output explicitly declares EVALUATION_ONLY_NOT_A_CERTIFICATE, statistical_independence=NOT_ASSESSED, and SOTA_CLAIM=NONE.

## Benchmark plan and acceptance criteria

Freeze these before private-test access:
- target domain and a second held-out confirmation domain;
- utility/harm definitions, groups, cluster IDs, independent sampling unit, and split manifests;
- calibration method for the stage-wise gain lower bounds;
- per-stage cost estimator, its upper-bound coverage test, and the allocator's measured overhead;
- risk certificate bundle and exactly which population/group/shift claims it supports;
- policy thresholds, baseline family, latency/energy ceilings, primary endpoint, and cluster-aware inference plan.

Compare at minimum against standalone cheap, standalone medium, standalone full, binary cheap/full routing, fixed confidence routing, random routing at the same escalation rate, and a calibrated continuation-value policy. Report quality, harm, per-group outcomes, tail loss, actual all-in latency, actual all-in energy, and budget violations separately. Do not infer a win from a utility-only average or a single budget point.

A useful research target, not a claim about current performance, is positive paired utility gain over the strongest budget-matched deployable baseline with a cluster-aware 95% interval excluding zero, no hard-budget violations, no unacceptable pooled or group-level harm, and replication in a second domain. Any practical target such as a 5% quality improvement must be defined in a task-specific utility scale before seeing the held-out outcomes.

## Limits and next work

- The replay assumes all tiers' potential outcomes are recorded for each trace. Real deployments do not observe counterfactual outcomes; causal claims require an appropriate experimental design.
- Current aggregate intervals and confidence calculations are deliberately outside this tool; use the cluster-inference tool and a signed preregistration artifact for formal comparisons.
- Risk states and cost predictions are supplied by upstream systems; this tool does not audit their provenance or statistical validity.
- Extend to separately calibrated tier-specific continuation values, shift-aware revocation, prospectively collected online traces, and matched-budget ablations only after the core replay passes exact-head validation.
