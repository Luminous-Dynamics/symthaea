# CMAQ-012 measured resource-vector and D1 qualification contract

**Contract version:** v0.3  
**Status:** executable paired resource-vector audit and exact profile/receipt content verifier; not a real hardware measurement, profiler attestation or verified D1 result.

## Why the scalar budget is insufficient

The v0.1 field `planner_compute_budget_units` is retained as a shared declared cap, but equal opaque units are not proof of equal work. A direct value network, a transition model, a CEM allocator and a simulator can perform different amounts of work under the same nominal "budget 100000".

CMAQ-012 v0.2 records a resource vector per task/capability/split/seed/arm:

- `compute_work_units`: a normalized operation-work count defined by a content-addressed resource profile. It is not labelled FLOPs unless the profile explicitly defines and validates that mapping.
- `latency_ms`: measured end-to-end decision latency. The profile declares the clock, timing boundaries, synchronization and warmup policy.
- `peak_memory_bytes`: peak memory under the profile's declared scope (host, device or both).
- `energy_millijoules`: optional, only populated when directly metered or obtained from a named calibrated counter. It must never be inferred from latency.
- `direct_value_calls`, `world_model_forward_calls`, `world_model_transition_calls`, `oracle_transition_calls`, `environment_transition_calls`, and `optimizer_iterations`: workload counts that explain where measured compute is spent.
- `resource_profile_sha256`: reference to the same hardware/runtime/instrumentation profile across every arm and episode in one comparison group.
- `profiling_receipt_sha256`: unique per-arm/episode receipt digest.

The profile binds the host and device, driver, runtime/compiler, executable digest, profiler, normalized work-unit definition, timing boundaries, memory scope, energy source and preregistered tolerances. A comparison across different profile hashes is not matched.

## D1 is a role, not merely an arm label

The auditor checks the following declared workload semantics:

| Arm | Required workload semantics |
|---|---|
| D0 | Direct policy; its resource usage is recorded, not used as the D1 compute target. |
| D1 | At least one direct-value call; zero learned-world-model forward calls and zero learned-world-model transition calls. |
| W1 | At least one learned-world-model forward call and at least one learned-world-model transition call. |
| W2 | Same direct-value calls, learned-model forward/transition call counts and optimizer iterations as W1, while its candidate-to-future correspondence is shuffled by the adapter. |
| W3 | At least one evaluator-oracle transition call and zero learned-world-model calls. It remains an evaluator-only upper bound. |

These checks prevent a D1 ledger from passing after relabelling a model-predictive controller as a direct controller. They validate the reported call structure; the call counters themselves still require independent receipt verification and source/adapter review.

## Matching rules

The v0.2 defaults are fixed in the auditor and repeated in the emitted report. For every resource dimension, the auditor first computes a relative difference for each *paired episode* (same task history and seed), then takes p50/p95 of those per-episode differences. It does not compare only independent arm-level marginals, which could hide crossed cost profiles where the arms have equal marginal distributions but different per-seed costs:

- Compute-work p50 and p95: relative difference no greater than 5%.
- End-to-end latency p50 and p95: relative difference no greater than 5%.
- Peak-memory p95: relative difference no greater than 10%.
- Energy, if completely measured for D1/W1/W2: mean relative difference no greater than 10%.
- W1 and W2 must have identical direct-value calls, learned-model forward calls, learned-model transition calls and optimizer iterations per paired episode.

Both comparisons must pass: D1 versus W1 (compute-only control) and W1 versus W2 (sham workload parity). Comparing only mean latency or candidate count is insufficient. The report publishes the separate relative deltas; one good dimension cannot hide a failure in another.

The measurement method must be `operator_trace` or `hardware_counter`. `static_estimate` and `unavailable` are reported but never qualify a compute match. Missing compute or memory values also produces `resource_unmeasured`.

## Status and claim ceiling

- `budget_matched`: the reported resource vector and D1/W1/W2 role checks fall within the declared tolerances.
- `budget_unmatched`: a resource dimension is out of tolerance, D1's role is invalid, W1/W2/W3 workload semantics are invalid, or measured energy differs beyond tolerance.
- `resource_unmeasured`: accepted measurements are absent or only static/unavailable resource methods are supplied.

A group with at least 20 episodes and `budget_matched` is labelled `budget_match_unverified` without artifact inputs. With exact profile and receipt bytes successfully validated, it is labelled `budget_match_receipts_verified`. Fixture-only verification is labelled `synthetic_fixture_only`. None of these statuses is `qualification_candidate`: content hashes prove which bytes were analyzed, not that the claimed profiler ran, that the recorder is authentic, or that hardware meters were calibrated. These remain explicit attestation gates.

This is deliberate. Equal self-reported counters are not independent evidence of equal compute.

## Artifact schemas

- `resource-profile.schema.json`: the content-addressed host/runtime/profiler definition.
- `profiling-receipt.schema.json`: a per-arm/episode record containing the measured vector and recorder identity.
- `report.schema.json`: v0.3 report contract including artifact-verification status, resource-match decision and per-arm resource summaries.

The profile bytes are SHA-256 hashed exactly as stored and named `profiles/<digest>.json` under the artifact directory. Each receipt is similarly named `receipts/<digest>.json`. The v0.3 CLI recomputes those hashes, parses the artifacts, rejects duplicate JSON keys, binds profile method/units/tolerances and compares each receipt's run identity, method, measurements and workload counters to its exact TSV row. Fixture-only files require explicit `--allow-fixtures`. A valid digest is content integrity—not proof of producer identity or trustworthy instrumentation.

## Statistical and systems discipline

1. Use the same physical host, device, driver, runtime, power/clock policy and profiler profile for all five arms in a paired group.
2. Warm up identically and measure the full decision path, including model dispatch, candidate generation, scoring, selection, synchronization and required post-processing. Do not omit allocator cost.
3. Keep training cost separate from inference cost. Do not count training FLOPs as inference FLOPs or vice versa.
4. Report query counts and transition evaluations separately from normalized compute work. They describe different cost dimensions.
5. Use paired histories and seeds. Report per-episode rows; calculate p50/p95 across the paired group. Do not treat model forwards or action candidates as independent evaluation episodes.
6. Record failed and timed-out runs, including their compute and latency. A timeout is not a free or absent computation.
7. If energy cannot be measured, report it as null/unmeasured. Do not manufacture joules from GPU utilisation, TDP or elapsed time.
8. Keep W3 oracle work separate. Its runtime can be reported, but it is not a deployable contender and does not define the D1 budget.

## Remaining qualification work

1. Independently attest the recorder and profiler binary, measurement scope, synchronization and calibration. Content hashes alone do not authenticate hardware origin.
2. Generate real per-arm receipts from pinned Gymnasium/MuJoCo and DeepMind Control adapters; do not reuse synthetic fixture receipts.
3. Implement a real direct-value D1 policy and preserve D1's zero learned-model calls.
4. Reproduce the exact task, seeds, hardware/runtime profile and all five arms, including failed/timed-out episodes.

Only after independent origin attestation and a reproducible real-environment run should the resource match be considered qualified.
