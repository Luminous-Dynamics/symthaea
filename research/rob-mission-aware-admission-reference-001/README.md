# ROB-MISSION-AWARE-FLEET-SERVING-001 — deterministic admission reference

Parent workstream: Symthaea #11627 and protocol PR #11630.

This standalone reference evaluates **optional remote inference request admission** for a heterogeneous robot fleet against one declared multi-resource budget. It does not schedule local control, RTA, actuator interlocks or safe-stop, and it never authorizes applying a remote result.

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

This is deliberately not a detailed GPU/KV-cache runtime model. The next extension must add long-lived brain-context/KV residency, short-deadline VLA chunks, KV eviction/re-prefill cost and foreground sensor memory-bandwidth deadlines, using measured or frozen trace inputs. GPU-time density is only a tie-breaker; memory bandwidth and network remain hard independent budgets.

## Output and limits

The result identifies admitted requests, request-level rejections, consumed resources, the next Round Robin cursor and any service-floor obligations not met. It carries no actuation authority. A service-floor breach makes the schedule unacceptable even if aggregate throughput increases.

This is deterministic policy/reference logic only. It does not validate network timing, PoMA calibration, EVSI estimation, model identity cryptographically, physical state, hardware resource estimates or hard-real-time guarantees.

## Run

    cargo fmt --manifest-path research/rob-mission-aware-admission-reference-001/Cargo.toml -- --check
    cargo test --locked --manifest-path research/rob-mission-aware-admission-reference-001/Cargo.toml
    cargo clippy --locked --manifest-path research/rob-mission-aware-admission-reference-001/Cargo.toml --all-targets -- -D warnings
