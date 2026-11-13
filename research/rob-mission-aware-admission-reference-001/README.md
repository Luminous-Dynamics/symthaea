# ROB-MISSION-AWARE-FLEET-SERVING-001 — deterministic admission reference

Parent workstream: Symthaea #11627 and protocol PR #11630.

This standalone reference evaluates **optional remote inference request admission** for a heterogeneous robot fleet against one declared multi-resource budget. It does not schedule local control, RTA, actuator interlocks or safe-stop, and it never authorizes applying a remote result.

## Decision-sufficiency horizon composition

This reference also consumes the calibrated decision-sufficiency output from #11650, calibrated by the simultaneous episode-level coverage owner #11659. It does not recompute expected losses, regret, calibration or robust admissibility.

Every request binds:
- the decision and selected-action identities;
- the robot clock/decision epoch and current control tick;
- task-loss scale, loss-model digest and candidate action-domain digest;
- the exact decision-sufficiency receipt digest;
- the conservative remaining ticks before the selected action ceases to be decision-sufficient.

The receipt is checked against the request's exact decision/action/epoch/tick/model/domain/loss-scale identities. Missing or Unknown evidence, an already-insufficient decision, a rejected receipt, or any mismatch blocks remote admission.

The conservative p99 return latency plus local projection/dispatch guard must fit **all three** independent time windows:
- remaining decision-sufficiency horizon;
- local action-buffer/fallback slack;
- task/actuation deadline slack.

The effective bound is their minimum. Exact-boundary equality is inclusive. EVSI remains a separate positive-value requirement: a long decision horizon does not make a zero-EVSI request useful, and positive EVSI cannot authorize a result that arrives after the selected decision expires.

This is not an action-authority extension. Local control, RTA, physical interlocks, fallback and safe-stop remain independent.

## Hard eligibility precedes policy ranking

A request is eligible only when the caller supplies:
- valid endpoint, session, model and request identity evidence;
- a complete, digest-matched source/decision receipt;
- a valid clock relation and finite conservative AoI upper bound;
- p99 return latency plus a separately declared local projection/dispatch guard within both the action-buffer and task-deadline slack;
- empirical PoMA upper bound no greater than the declared per-request limit;
- local fallback readiness and remote-data privacy permission;
- positive EVSI in a declared task-loss scale.

Unknown clock, age, latency, miss-risk or EVSI is not an optimistic default. Zero EVSI means that optional remote computation is declined, even when a newer result would reduce AoI.

The caller must preserve per-modality AoI source vectors. This initial admission kernel consumes a separately qualified conservative age upper bound for the task's required sources; it does not average camera, IMU, contact, encoder or thermal ages.

## Compared policies

- Round Robin: deterministic robot-order rotation from a declared cursor.
- EDF: smallest remaining action/deadline slack first.
- EVSI-per-compute: greatest EVSI per GPU-microsecond, permitted only when all compared requests share one task-loss scale.
- Luminous constrained admission: service-floor urgency, starvation debt, remaining slack and miss-risk urgency, followed by a within-scale EVSI/GPU-microsecond tie-break only when both GPU demands are nonzero. It does not add EVSI, physical joules and fairness into one score.

Armory Lookahead and RED remain external matched-scheduler adapters, not claimed implementations in this crate. The reference is a small deterministic admission kernel intended to anchor trace replay and known-answer semantics.

## Resource constraints

A selected batch must remain within all declared budgets:
- GPU service microseconds;
- memory-bandwidth service units;
- network bytes.

The KV-cache growth budget is a hard incremental reservation, not a detailed runtime cache model. Long-lived brain-context residency, phase-aware eviction, re-prefill cost and foreground sensor memory-bandwidth deadlines still require measured or frozen trace inputs. GPU-time density is only a tie-breaker; memory bandwidth and network remain hard independent budgets.

## Output and limits

The result identifies admitted requests, request-level rejections, consumed GPU, memory-bandwidth, network and KV-growth resources, the next Round Robin cursor, service-floor obligations not met by the batch, and maximum-starvation breaches that predate the current admission decision. An already-overdue robot remains eligible and is prioritized for remediation; the prior breach remains visible rather than being erased by a late service event. It carries no actuation authority. A service-floor breach makes the schedule unacceptable even if aggregate throughput increases.

This is deterministic policy/reference logic only. It does not validate network timing, PoMA calibration, EVSI estimation, model identity cryptographically, physical state, hardware resource estimates or hard-real-time guarantees.

## Run

    cargo fmt --manifest-path research/rob-mission-aware-admission-reference-001/Cargo.toml -- --check
    cargo test --locked --manifest-path research/rob-mission-aware-admission-reference-001/Cargo.toml
    cargo clippy --locked --manifest-path research/rob-mission-aware-admission-reference-001/Cargo.toml --all-targets -- -D warnings
