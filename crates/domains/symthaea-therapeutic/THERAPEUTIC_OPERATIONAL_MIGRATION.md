# Therapeutic Operational Hardening Migration

This document covers Series III patches 0019–0027. Series III assumes the
fail-closed safety and governance boundaries from Series I and II are already
present.

## New guarantees

1. Consent, jurisdiction, retention, and evidence policy activate as one atomic,
   monotonic policy bundle.
2. Therapeutic models are identified by immutable artifact digest, explicit
   output semantics, evidence claim, data class, owner, and review horizon.
3. Crisis regression evaluation uses a versioned synthetic corpus and reports
   exact counts rather than clinical-performance claims.
4. Operational metrics use fixed dimensions and counters only; no therapeutic
   content or subject identifier is accepted by the metrics API.
5. Safety-control failures move the runtime into explicit degraded or locked
   modes. Crisis triage remains available; ordinary therapeutic operations do
   not.
6. Runtime checkpoints are keyed, audit-chain verified, configuration bound,
   and reject policy, generation, and cycle rollback.
7. Release promotion requires an authenticated evidence envelope in which every
   required production gate is explicitly passed.
8. Safety incidents are content-free, hash chained, and cannot skip required
   lifecycle transitions or close without remediation evidence.

## Recommended activation order

### 1. Build a policy bundle

Create `TherapeuticPolicyBundle` from the reviewed consent, jurisdiction,
retention, and evidence registries. Bootstrap `TherapeuticPolicyRuntime` once,
then use `activate` for later updates. Never mutate individual active policies
in place.

Store each `PolicyActivationReceipt` in release evidence. Reject candidates
whose revision is not greater than the active revision.

### 2. Register every model and ruleset

Register every crisis detector, ranking rule, proxy, estimator, and generated
text transformer in `TherapeuticModelRegistry`. The artifact digest must bind
the exact immutable implementation or data file used at runtime.

A successful registry lookup does not authorize execution by itself. Both the
model boundary and the linked `EvidenceRegistry` claim must authorize the
requested `EvidenceUse`.

### 3. Establish a regression baseline

Run `CrisisRedTeamSuite::canonical_v1()` and retain the complete
`CrisisEvaluationReport`. The suite is synthetic. Report fixture counts as
fixture counts; do not convert them into claims about clinical sensitivity,
false-negative rate, or population performance.

Add new cases for every relevant incident and bug. Do not silently edit old
corpus versions after release; issue a new corpus version.

### 4. Connect privacy-safe observability

Export `TherapeuticMetricsSnapshot` only to approved metrics infrastructure.
The snapshot contains fixed enum dimensions and counters. Do not attach user,
session, prompt, geography, diagnosis, model text, or free-form error labels.

Treat any non-zero `audit_chain_failures` counter as an incident and release
blocker.

### 5. Wire the circuit breaker

Before normal operations, report runtime health signals to
`TherapeuticCircuitBreaker`. In degraded or locked-down mode, only crisis
triage is authorized. The application must use `fail_safe_message()` rather
than falling back to an unguarded model or generic chat path.

Operator holds require an explicit clear action. Do not auto-clear them after a
restart.

### 6. Capture and recover operational state

Capture `RuntimeCheckpoint` with a deployment-secret checkpoint key and the
active configuration fingerprint, policy revision, generation, and last cycle.
Keep checkpoint keys outside snapshots and source control.

On restore, set minimum accepted revision, generation, and cycle from an
external monotonic store. A checkpoint stored beside itself cannot provide
rollback resistance.

Raw client state, therapeutic text, and contact details are outside the
operational checkpoint format and require their own encrypted lifecycle.

### 7. Gate release promotion

Create `TherapeuticReleaseEvidence` for the exact source tree and deployment
configuration. Record all required checks:

- format
- compile-default
- test-default
- clippy-default
- feature-matrix
- crisis-red-team
- deployment-readiness
- audit-chain
- artifact-replay

`NotRun`, missing, failed, or unauthenticated evidence blocks production
promotion. Release keys must be provisioned outside the evidence artifact.

### 8. Operate the incident ledger

Open a `SafetyIncident` when safety assurance is reduced. Reference only
configuration, policy, audit, and evidence digests. Never paste user messages,
generated responses, safety plans, or contact details into the incident ledger.

Allowed progression is open → contained → remediated → closed, with an explicit
fast path from open → remediated. Closure requires evidence. A new recurrence
after closure is a new incident rather than silent reopening.

## Recovery drill

Before production, demonstrate all of the following:

1. A stale policy revision is rejected.
2. A broken audit chain causes degraded operation.
3. Ordinary response release is blocked while crisis triage remains available.
4. A checkpoint signed with the wrong key is rejected.
5. A lower policy generation or cycle is rejected.
6. A release with one `NotRun` gate is denied.
7. A critical incident cannot close without remediation evidence.
8. The complete patch series replays to the documented source tree hash.

## Remaining validation boundary

These patches add architectural controls and internal regression instruments.
They do not establish clinical efficacy, medical-device compliance, legal
sufficiency in any jurisdiction, or real-world crisis-detection performance.
Those claims require independent study, governance, and review.
