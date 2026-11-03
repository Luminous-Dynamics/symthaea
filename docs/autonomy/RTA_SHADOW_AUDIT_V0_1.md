# Runtime-assurance shadow audit v0.1

**Status:** implemented audit adapter; no monitor-performance results reported  
**Code:** `symthaea_vehicle::rta_shadow_audit`  
**Monitor:** `symthaea_vehicle::runtime_assurance`

## Purpose

This adapter records the output of the existing stateful Simplex-style vehicle RTA monitor while making it impossible for the adapter to return the monitor's selected command. It contains no simulator, embodiment or actuator reference. Every decision is reduced to:
- monitor mode;
- reason codes;
- advanced blending weight;
- whether the monitor produced a command proposal;
- a constant marker that the proposal was discarded;
- monitor-profile and observation/decision fingerprints.

The original selected command is not copied into the audit frame and is not returned by the adapter. A report may state that a command proposal existed, but cannot be used as the proposed command itself.

## Evidence inputs remain caller-owned

The underlying RTA observation still requires:
- baseline admissibility;
- baseline input/controller/control-path health;
- estimator, real-time and actuator health;
- model applicability;
- envelope and controllability margins;
- controller identifiers;
- a baseline-independence evidence reference.

These are explicit assertions. This adapter does not authenticate them, derive them from simulator truth, verify controller independence or establish that a fallback is physically safe. A non-empty evidence ID is not evidence content.

Therefore, the audit wrapper measures how the monitor processes the supplied observation contract; it does not establish how accurately the inputs describe a real system.

## Output and integrity

The audit report contains frame-indexed decisions, counts by mode, counts by reason, number of frames with a command proposal (and discarded proposal), the monitor profile fingerprint, observation and decision fingerprints, and an output fingerprint.

BLAKE3 digests are integrity/replay identifiers only. They are not signatures, authenticated records, cross-language canonical identities or binary attestations.

## Controls

Tests are authored for:
- nominal clean observation entering recovery dwell;
- seeded state-envelope violation selecting the baseline mode;
- baseline control-path fault producing no admissible fallback;
- monitor epoch reset while preserving previous audit frames;
- aggregate-count inconsistency rejection even after the report digest is recomputed;
- missing reason-vocabulary rejection even after the report digest is recomputed.

These controls test the adapter and named RTA behavior only. They have not been compiled or executed in this session.

## Integration boundary

The adapter is not yet wired into `run_paired_lane_keep_rollout`. That integration should require an explicit caller-provided RTA observation source containing independently justified margins, model applicability and component-health evidence. The current bicycle-model lane-keeping runner must not silently treat its simulator state as an independent estimator, sensor-health monitor or controllability proof.

Once integrated, record RTA decisions alongside both paired trajectories without using the shadow decision to alter either plant's command. Compare RTA-trigger precision/recall only against preregistered monitor fault injections, not a post-hoc heuristic label.

## Claim ceiling

This establishes only that a named monitor build processed the supplied observations and emitted the recorded state/reason/command-availability decisions. It does not establish false-safe rate outside the explicit fixture, safe fallback execution, controller superiority, physical safety, roadworthiness, legal compliance or SAE Level 4/5 capability.
