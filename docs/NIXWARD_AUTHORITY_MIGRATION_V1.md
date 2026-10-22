# Nixward Authority Migration v1

**Status:** architecture/migration contract only. This document does not change runtime execution behavior.

**Tracks:** #4892, #669, `Luminous-Dynamics/xenia-peer#351`, draft `Luminous-Dynamics/rfcs#2` / RFC 0183 authority separation.

## Purpose

Nixward currently combines useful action-risk classification, Phi/confidence context, explicit user confirmation paths, command execution, rollback, and post-action learning. The newer Luminous authority model requires a sharper distinction:

```text
cognitive confidence / Phi
!= execution authorization

proposal
!= risk assessment
!= authorization
!= execution
!= verified post-state
```

This migration preserves the useful Nixward machinery while moving privileged execution authority to an explicit, bounded, action-bound authorization prerequisite.

## Current implementation boundary

As of the branch base, the relevant code behaves as follows:

- `action/executor.rs::SafetyLevel::required_phi()` maps safety classes to `ConsciousnessThresholds`;
- `NixOSExecutor::execute(command, phi)` returns `PendingConfirmation` below that threshold and otherwise executes;
- the function documentation already warns that caller-supplied or hard-coded Phi can become a rubber stamp;
- `NixOSExecutor::execute_confirmed(command, phi)` bypasses the threshold after a real upstream confirmation;
- the daemon active-healing path already uses an explicit human `Approved` verdict that is one-shot-consumed before `execute_confirmed`;
- `action/phi_gate.rs` presently provides command-risk/destructiveness classification and rollback lookup despite its historical authority-oriented name;
- `PlanExecutor` stores one Phi value for the whole plan, applies it to each step, and uses `execute_confirmed` for rollback.

The migration should build on these existing strengths rather than deleting them.

## Target authority theorem

For governed modifying operations:

```text
Observation / cognition
        ↓
Action proposal
        ↓
Risk + uncertainty assessment
        ↓
Exact action intent
        ↓
Explicit bounded authorization
        ↓
Pre-execution revalidation
        ↓
Execution
        ↓
Post-state verification / attestation
```

Phi, confidence, expected free energy, model quality, past success, or recommendation quality may affect review policy. They do not mint privilege.

## Roles

### Nixward may own

- Nix/NixOS action modeling;
- typed host/machine action intent;
- command/action safety classification;
- reversibility and rollback discovery;
- state/currentness observations needed for execution preconditions;
- plan construction and deterministic validation;
- local machine execution after authority is established;
- execution receipts and post-state observation;
- advisory Phi/confidence/risk context.

### Nixward must not infer

- that high Phi authorizes root/system mutation;
- that a recommendation is a permit;
- that a successful command establishes desired postconditions;
- that a persisted approval record recreates live authority;
- that one approval covers materially different commands, targets, state, or parameters.

### External / explicit authority may come from

- an exact local human approval under a declared local policy;
- a delegated local policy/capability;
- a Xenia state-bound authorization permit;
- an explicitly defined emergency recovery policy.

The authority source must be represented rather than inferred from cognition.

## Proposed types

Names are provisional; semantics are the important part.

### `NixActionIntentV1`

```text
NixActionIntentV1 {
  intent_schema_identity,
  subject_host_identity,
  current_state_or_generation_ref,
  action_kind,
  exact_parameter_or_artifact_digest,
  maximum_scope,
  preconditions[],
  required_postconditions[],
  rollback_or_recovery_ref?,
  validity_context?,
}
```

Where a typed operation exists, shell command text alone should not be the authorization identity.

### `NixActionRiskAssessmentV1`

```text
NixActionRiskAssessmentV1 {
  action_intent_identity,
  safety_level,
  phi?,
  confidence?,
  reversibility,
  rollback_available,
  predicted_side_effect_refs[],
  currentness_or_evidence_refs[],
  recommended_review_class,
}
```

This is advisory evidence for policy/review. It cannot be converted into a permit by itself.

### `NixExecutionAuthorizationRefV1`

A narrow reference/envelope proving that the selected authority profile approved the exact action intent.

Required semantics depend on the profile but should support:

```text
exact action-intent identity
exact subject/target scope
current-state binding or state predicate
validity window
nonce / replay domain where required
maximum permitted action scope
required recovery/rollback preconditions
policy / delegation identity
```

For remote/high-impact actions, this may point to a Xenia permit rather than duplicating Xenia authorization semantics inside Nixward.

### `NixExecutionReceiptV1`

```text
NixExecutionReceiptV1 {
  action_intent_identity,
  authorization_profile_and_ref,
  executor_identity,
  actual_pre_state_ref,
  actual_action_digest,
  started_at,
  finished_at,
  mechanical_result,
  actual_post_state_ref?,
  postcondition_verification,
  rollback_attempt_and_result?,
  risk_assessment_ref?,
}
```

`mechanical_result = success` is not equivalent to `postcondition_verification = satisfied`.

## Migration phases

### A. Correct the semantic boundary without breaking callers

1. Treat Phi thresholds as legacy confirmation/review heuristics, not authorization.
2. Update docs/UI wording so Phi does not appear to grant privilege.
3. Keep `SafetyLevel`, rollback lookup, dry-run, risk scoring, and history.
4. Document the daemon's existing one-shot human approval path as a genuine authority input with a limited local profile.
5. Mark `execute` and `execute_confirmed` authority ceilings explicitly.

No behavior change is required merely to complete this documentation phase.

### B. Add typed action intent and authorization context

Introduce a new API conceptually equivalent to:

```text
execute_authorized(intent, authorization_context)
```

Before dispatch it must:

1. validate the action-intent identity;
2. validate authorization binding to that exact intent;
3. re-observe/revalidate required machine state or generation;
4. verify scope, expiry, nonce/replay and policy requirements applicable to the profile;
5. execute only the exact authorized operation;
6. record the authorization reference in the execution receipt;
7. evaluate declared postconditions separately.

### C. Move multi-step plans to step/plan-bound authority

A single floating-point Phi value must not stand in for authorization of an arbitrary multi-step plan.

Plans should bind:

```text
exact plan identity
exact ordered step intents
which steps share one authorization envelope
which require distinct authority
rollback policy and rollback authority
state transitions that invalidate later-step authorization
```

Rollback deserves explicit semantics. Starting a plan does not automatically imply unlimited rollback authority; however, a policy may pre-authorize a bounded rollback set as part of the original action permit.

### D. Close privileged legacy bypasses

After callers migrate:

- governed remote/system-critical/destructive paths require the typed boundary;
- legacy Phi-only execution receives an explicit low-authority/local ceiling or is deprecated;
- `execute_confirmed` cannot be a generic bypass for high-impact paths;
- read-only/local diagnostic paths may retain low-friction policy where appropriate.

## Human confirmation

Human approval is a legitimate authority input when the policy permits it. Bind it to the exact thing approved.

Prefer:

```text
Approved:
  action intent X
  host H
  current generation/state G
  exact parameters P
  validity window W
```

rather than:

```text
Approved once
-> arbitrary later modifying command
```

The daemon's current one-shot `Approved` verdict is a useful starting point; the next step is binding it to an immutable action intent and state context.

## Phi and cognition after migration

Phi/confidence remain useful for:

- deciding whether a proposal should be shown at all;
- requiring manual review under uncertainty;
- escalating the required evidence/review class;
- preferring reversible candidates;
- requesting additional diagnostics before a change;
- surfacing uncertainty in the approval UI;
- learning from outcomes after independently authorized execution.

They must not mean:

```text
Phi >= threshold
=> privilege granted
```

## Network Twin composition

For Network Twin driven host/network-policy changes:

```text
Network Twin
  exact current/candidate snapshot
  + semantic diff
  + realization artifact identity
  + verification/evidence
        ↓
Nixward
  host/process-side action intent
  + machine-state prerequisites
  + advisory risk assessment
        ↓
Xenia / local authority policy
  state-bound action permit
        ↓
Nixward executor
  exact authorized host-side transition
        ↓
Nixward + Network Twin observers
  execution and post-state evidence
```

No layer may reinterpret verification evidence as authorization.

## Compatibility and non-goals

This migration does not require:

- removing HDC, active inference, or Phi;
- removing automatic diagnosis/proposal generation;
- making every read-only query require remote authorization;
- making Nixward depend directly on Network Twin or Xenia implementation crates;
- rewriting rollback support;
- claiming that cryptographic authorization proves safety or correctness.

Prefer interface/receipt boundaries rather than cross-project dependency cycles.

## Adversarial qualification

At minimum freeze regressions for:

1. high Phi + no required authorization cannot execute governed destructive action;
2. low Phi cannot itself nullify a separately valid permit unless the selected policy explicitly makes risk assessment a prerequisite;
3. exact action changes after approval -> reject;
4. target/host changes -> reject;
5. state/generation drifts -> revalidate/refuse according to policy;
6. copied/replayed one-use permit -> reject;
7. stale persisted approval cannot recreate live authority;
8. multi-step plan cannot smuggle an unapproved later step;
9. rollback cannot exceed the pre-authorized rollback envelope;
10. command exit zero + failed postcondition -> not verified success;
11. Network Twin verification receipt cannot decode as an authorization;
12. Xenia lab permit cannot authorize a production target;
13. hard-coded Phi constants cannot satisfy authority tests;
14. read-only low-friction paths remain usable under their declared policy.

## Exit criteria

This migration is complete when:

- cognition and execution authority are structurally distinct;
- privileged execution consumes an exact bounded authorization input;
- action/state/target/scope binding is explicit;
- Phi remains advisory rather than privilege-bearing;
- plan and rollback authorization are explicit;
- post-state verification remains distinct from process exit status;
- existing one-shot human confirmation has an exact action-bound representation;
- Network Twin/Xenia can compose through typed references without ownership duplication;
- legacy APIs have a clear compatibility ceiling and retirement path.
