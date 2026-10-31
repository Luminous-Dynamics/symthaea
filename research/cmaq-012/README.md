# CMAQ-012: Paired Advantage-of-Prediction Result Auditor

**Status:** executable Rust auditor v0.3; draft  
**Tracking issue:** [#9122](https://github.com/Luminous-Dynamics/symthaea/issues/9122)  
**Claim ceiling:** paired ledger/resource audit plus exact profile/receipt content-integrity verification; profiler/hardware origin remains unattested; no real environment result, competitive claim or SOTA claim

## Why this is a paradigm shift

A world model can appear to outperform a direct policy because it received more compute, a better action proposer, a different candidate set, more favorable initial states, or selective episode reporting. A benchmark must isolate the contribution of predictive rollouts themselves.

This auditor requires five paired arms on the same initial state, proposer, action-candidate set, environment configuration, adapter, rollout horizon and declared planner compute budget:

| Arm | Meaning |
|---|---|
| D0 | Direct policy without predictive rollouts |
| D1 | Compute-matched direct/value policy without learned future rollouts |
| W1 | Learned action-conditioned world-model planner |
| W2 | Shuffled-future negative control that breaks candidate-to-prediction correspondence |
| W3 | Oracle-dynamics upper bound |

The key quantity is **advantage of prediction (AoP) = W1 - D1**. D1 controls for extra computation. W1-D0 is deployment gain; W1-W2 tests whether candidate-aligned prediction matters rather than merely adding a model-shaped computation path; W3-W1 measures remaining oracle gap; D1-D0 estimates compute-only gain.

This is stricter than reporting world-model return beside a direct-policy paper result. Those unpaired numbers do not isolate the value of prediction.

## Connection to current literature

- [Do World Models Make Better Robots?](https://arxiv.org/abs/2609.29669) (2026 survey) reports that only a small fraction of its 160-benchmark corpus explicitly compares VLA and world-model approaches, and only four benchmarks in its taxonomy bridge prediction to executed action. This motivates a paired estimator of predictive advantage, rather than another isolated prediction score.
- [How Should World Models Be Evaluated?](https://arxiv.org/abs/2606.15032) argues for counterfactual action fidelity, closed-loop rollout validity, policy ranking, optimization lift, exploitability and uncertainty calibration.
- [MRS.Q](https://proceedings.mlr.press/v306/chang26a.html) shows that search can hurt despite high model accuracy and that overestimation control matters. Its Gymnasium/MuJoCo and DMC tasks are a relevant future source of environment adapters, not a baseline result produced by this auditor.

## Input contract

The dependency-free Rust CLI consumes strict UTF-8 TSV. It rejects:
- a missing, reordered or extra header field;
- malformed SHA-256 digests, identifiers, numbers or statuses;
- duplicate rows, missing arms or unpaired seeds;
- mismatched initial-state, proposer, candidate-set, environment-config or adapter digests;
- unequal candidate counts, rollout horizons, declared planner-compute budgets or resource-profile identities;
- reused per-run profiling receipt digests or measurement methods that do not match within an episode.

The exact header is:

    task    capability    split    seed    arm    initial_state_sha256    proposer_sha256    candidate_set_sha256    environment_config_sha256    adapter_sha256    source_sha256    checkpoint_sha256    candidate_count    rollout_horizon    planner_compute_budget_units    status    return    task_success    latency_ms    resource_profile_sha256    profiling_receipt_sha256    resource_measurement_method    compute_work_units    peak_memory_bytes    energy_millijoules    direct_value_calls    world_model_forward_calls    world_model_transition_calls    oracle_transition_calls    environment_transition_calls    optimizer_iterations

Each task × capability × split × seed must have exactly one D0, D1, W1, W2 and W3 row. Each row binds its own source and checkpoint digests. The shared state/proposer/candidate/environment/adapter digests and budget fields must match within a paired episode.

Statuses are success, failed and timed_out. Every row—including failed and timed-out rows—must carry an explicit finite adapter-assigned return. Failed/timed-out rows must have task_success=0. They remain in the paired denominator; the auditor never drops them or silently replaces their return.

A qualification group requires at least 20 paired episodes. Smaller groups are labelled smoke_only; 20 is a minimum eligibility gate, not a claim that all statistical power concerns have been solved.

## Outputs and uncertainty

For return and binary task success, each task/capability/split gets five paired contrasts with deterministic paired bootstrap 95% percentile intervals:
- AoP: W1 − D1
- deployment gain: W1 − D0
- prediction mediation: W1 − W2
- oracle gap: W3 − W1
- compute-only gain: D1 − D0

Per-arm return, success rate, status counts and latency p50/p95 are emitted. The report includes the SHA-256 digest of the exact input bytes, bootstrap seed and replicate count.

No results are pooled across tasks or capabilities. Do not use a single headline score to hide a failing capability.

## Run

    cargo test --locked
    cargo run --quiet --locked -- --input fixtures/qualification-smoke.tsv

    # Explicitly allow only the checked-in synthetic artifact fixtures:
    cargo run --quiet --locked -- --input fixtures/qualification-smoke.tsv --artifacts-dir fixtures/artifacts --allow-fixtures

The checked-in fixture contains four paired episodes and is intentionally labelled smoke_only. It validates mechanics; it is not a qualification result.

## Next

1. Add adapters that emit this ledger from pinned Gymnasium/MuJoCo HalfCheetah-v4 and DeepMind Control quadruped-walk runs.
2. Implement D0/D1/W1/W2/W3 using an identical action proposer, frozen seed manifest and compute accounting.
3. Validate the shuffled-future control: W2 must preserve compute and candidate count while breaking candidate-to-prediction alignment.
4. Add held-out parameter families, calibration/development seeds and a sealed qualification manifest.
5. Independently reproduce the baseline and candidate, preserving raw episode ledgers and all failed runs.

A parser pass is not a benchmark pass. A smoke fixture is not an environment result. An AoP claim requires matched real-environment runs and independent artifact verification.


## v0.2 measured resource-vector qualification

The v0.1 opaque `planner_compute_budget_units` remains as a declared cap, but equality of that number alone is no longer sufficient to report a compute match. Each row now also carries a resource-profile digest, unique profiling-receipt digest, measurement method, normalized `compute_work_units`, peak memory, optional energy, direct-value/world-model/oracle/environment call counts, and optimizer iterations.

The auditor accepts `operator_trace` and `hardware_counter` as measurement methods for matching. `static_estimate` and `unavailable` remain visible but yield `resource_unmeasured`. Matching requires D1/W1 and W1/W2 agreement for compute-work p50/p95 (5%), latency p50/p95 (5%) and peak-memory p95 (10%). If energy is measured for all three arms, its mean must also agree within 10%.

**Role checks:** D1 must have direct-value calls and zero learned-model calls; W1/W2 must have learned-model forward and transition calls; W3 must have oracle transition calls and zero learned-model calls; W2 must preserve W1's direct/model calls and optimizer iterations.

Report group eligibility now distinguishes `budget_unmatched`, `resource_unmeasured` and `budget_match_unverified`. The latter is intentional: the current CLI verifies receipt digest syntax and uniqueness but does not yet ingest the receipt bytes. Therefore a resource vector that passes its declared tolerances is not yet authenticated hardware evidence and is not a qualification candidate. See [COMPUTE_MATCHING.md](COMPUTE_MATCHING.md), [resource-profile.schema.json](resource-profile.schema.json) and [profiling-receipt.schema.json](profiling-receipt.schema.json).

No static FLOP estimate, latency-derived energy estimate or dummy work is accepted as proof of compute parity. The normalized work-unit definition, hardware/runtime, profiler and measurement scope must be bound by the content-addressed resource profile. Failed and timed-out runs remain in the resource denominator.


## v0.3 independent artifact-content verification

The CLI now accepts `--artifacts-dir DIR`. It resolves each row's profile and receipt by content digest:

- `DIR/profiles/<resource_profile_sha256>.json`
- `DIR/receipts/<profiling_receipt_sha256>.json`

The Rust verifier reads the exact bytes, rejects symlinks/non-regular files/oversized artifacts, recomputes SHA-256, parses JSON with a strict dependency-free parser (duplicate keys and trailing data rejected), checks the profile contract, and binds every receipt field to the corresponding paired TSV row: task/capability/split/seed/arm, resource-profile identity, measurement method, latency, compute, memory, energy, direct/model/oracle/environment call counts and optimizer iterations. Receipt files may not contain a self-referential receipt-digest field.

Synthetic fixtures are rejected unless `--allow-fixtures` is explicitly supplied. The checked-in smoke fixture is intentionally synthetic and exercises 1 profile plus 20 unique per-arm/per-seed receipt files. It must report `synthetic_fixture_content_verified_origin_unattested`, never a hardware qualification. Without `--artifacts-dir`, the report remains `digest_only_not_independently_verified`.

For non-fixture artifacts, successful byte/field checks report `content_verified_origin_unattested`. This proves exact artifact content and row binding—not the truthfulness of the counter, profiler identity at runtime, trusted execution, hardware meter calibration or physical provenance. Therefore a 20+ pair group can become `budget_match_receipts_verified`, but not `qualification_candidate` until recorder/profiler origin and hardware provenance are independently attested. The report carries this claim ceiling explicitly.

The verifier tests include JSON duplicate-key/trailing-data rejection; successful fixture verification; rejecting fixture artifacts without opt-in; receipt-byte tampering; receipt-field tampering with a recomputed digest; and profile-byte tampering. The focused workflow validates report/profile/receipt schemas and runs the same artifact-bound smoke fixture.
