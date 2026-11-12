# ROB-OPTION-LOSS-REPLAY-INTEROP-001

Status: stacked Rust reference adapter and synthetic identity-join tests. This module extends the existing causal mission replay reference; it does not create a second mission trace, viability solver, physical model, recovery-topology owner, or actuation authority.

Parent issue: [Symthaea #11810](https://github.com/Luminous-Dynamics/symthaea/issues/11810). This branch is stacked on [ROB-CAUSAL-MISSION-REPLAY-001 PR #11664](https://github.com/Luminous-Dynamics/symthaea/pull/11664), because the authoritative MissionTrace types live on that feature branch.

Upstream owners:
- Symtropy #4588: finite observation-constrained common-continuation sets.
- Symtropy #4908 / PR #4918: sequential option-loss/restoration ledger.
- Symtropy #4889 / PR #4907: physical joint evidence receipt contract.

## Join contract

The adapter consumes a versioned receipt containing protocol/schema/package/commit/payload identities; mission/robot/scenario; exact decision, ordered source observations and action-domain; delivered-observation partition and complete physical-branch manifest; exact physical-transition ID/sequence and reserve predecessor/result; predecessor/result option-set digests; loss/restoration counts; a reason-coded evidence disposition; and literal actuation authority false.

The adapter runs the existing causal mission replay and joins against the exact DecisionRecord, PhysicalTransition and corresponding ActuatorApplication. A producer disposition can only be preserved or downgraded. Incomplete branch/action manifests remain Unknown. A Qualified receipt without both option-set digests remains Unknown. NotEvaluated remains NotEvaluated only when no option-set result or loss/restoration count is claimed.

## Tests and qualification

The module contains 13 authored deterministic tests for exact joins, wrong mission/action/observation/physical identities, sequence order, missing physical transition, upstream disposition downgrade, NotEvaluated handling, incomplete manifests, missing option-set digests, actuation-authority rejection, base replay rejection and unsupported integrity identity.

Exact-head Rust 1.96 format, locked tests and strict Clippy are enforced by .github/workflows/rob-option-loss-replay-interop.yml. Authored tests are not execution evidence until that workflow succeeds.

## Claim ceiling

A passing adapter proves only deterministic identity binding and retrospective receipt joining over supplied records. It does not prove upstream option sets or branch manifests are complete, recovery routes are physical, restoration costs are accurate, or a robot preserves future capability on hardware.
