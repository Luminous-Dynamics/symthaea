# Human Agency Kernel — Reset & Restart Continuity Audit v1

Status: HAK-002 supporting audit / documentation only

Parent: `HUMAN_AGENCY_KERNEL_AUTHORITY_AUGMENTATION_CONTRACT_V1.md`

## 1. Why this audit changed

This audit began from a valid HAK question:

```text
Can reset/restart silently widen authority?
```

The first source pass found several subterranean `reset_runtime()` methods that clear restrictive state. That looked like an operational-recovery bypass.

A broader call-site audit changed the conclusion.

The shared `EmbodimentBridge` trait documents `reset()` as:

```text
Reset body to default state.
```

Current call sites also use it as full scenario/test reinitialization: robotics bridge reset resets both the embodiment and FEP state, and platform tests call `reset()` expecting counters/body state to return to defaults.

Therefore the repository does **not** currently establish that `SubterraneanEmbodiment::reset()` is an operational restart primitive.

The corrected finding is:

```text
Current reset is evidence of new-scenario/default-body semantics.
It is NOT evidence of operational recovery.
```

This distinction matters because a correct authority theorem applied to the wrong lifecycle operation would itself create a bug: preserving an emergency stop from simulation A into a deliberately new simulation B is not necessarily conservative; it may simply violate reset semantics.

## 2. Evidence hierarchy

### Established

1. `EmbodimentBridge::reset()` is documented as resetting the body to default state.
2. Existing call sites exercise it as reinitialization/test-reset behavior.
3. `SubterraneanOperationalCheckpoint` separately persists authority/recovery state, including operator authority, degraded supervision, partition recovery, and temporal assurance.
4. Checkpoint restoration tests already demonstrate continuity of an operator hold and a latched temporal hold.
5. The generic subterranean reset sets several simulated/runtime fields to configured nominal defaults.

### Not established

1. That the generic `reset()` API is used for live physical operational recovery.
2. That clearing operator/degraded/partition state during a *new simulation lineage* is a security bypass.
3. That the current physical deployment lifecycle reuses `reset()` after a process/power failure.

### Architectural risk

The names `reset()` and `reset_runtime()` are broad enough that future callers could mistake scenario reset for operational recovery. The right repair is therefore to make the lifecycle distinction explicit before changing local reset semantics.

## 3. Four lifecycle operations

### 3.1 New simulation/scenario lineage

Purpose: deterministic scenario reinitialization.

```text
old simulated lineage terminates
new simulated lineage begins
```

Configured nominal state is legitimate because it is part of the new scenario fixture.

A new simulation need not inherit prior simulated stop/hold/recovery state unless the test explicitly models continuity.

### 3.2 Ephemeral computation reset

Purpose: discard partial computation while preserving the same operational lineage.

Examples:

- temporary buffers;
- partial quorum accumulation;
- incomplete negotiations;
- non-authoritative caches.

Rule:

```text
may discard provisional positive progress
must not manufacture positive evidence
must not silently remove durable same-lineage restrictions
```

### 3.3 Operational restart/recovery

Purpose: continue a real authority lineage after process restart, watchdog restart, power interruption, update, or similar event.

For this operation the authority-monotonicity theorem applies:

```text
Authority(after recovery initialization)
    ⊆
Authority(before interruption)
```

until fresh evidence or an explicitly authorized transition widens it.

Operational recovery must define continuity for, where applicable:

- operator constraints;
- replay barriers;
- revocation/refusal generations;
- degraded/recovery latches;
- partition/reconciliation truth;
- temporal/causal review latches;
- physical capability/failure state;
- evidence freshness;
- audit/update lineage.

### 3.4 Administrative factory reset

Purpose: intentionally destroy a local lineage/configuration.

This is a privileged administrative/destructive operation, not an implicit recovery shortcut. A deployed post-factory-reset system should reacquire the evidence and authority required for operation.

## 4. Current subterranean lifecycle: two real paths

### 4.1 Scenario reset path

`SubterraneanEmbodiment::reset()` resets the simulator plus cognitive/runtime state and writes nominal fixture values including:

```text
operator_link_fresh = true
control_loop_healthy = true
checkpoint_valid = true
reboot_count_in_window = 0
```

Under *new-scenario* semantics these are fixture initialization values, not claims that a previous operational failure recovered.

They become unsafe only if this method is later reused as same-lineage operational recovery.

### 4.2 Operational checkpoint path

`SubterraneanOperationalCheckpoint` explicitly contains:

- controller state;
- mission state;
- operator authority;
- degraded supervisor;
- update manager;
- sensor fusion;
- actuator isolation;
- field envelope;
- partition recovery;
- temporal assurance.

`load_operational_checkpoint()` restores those objects rather than reconstructing them from nominal defaults.

Existing tests already verify at least:

```text
operator HoldPosition survives checkpoint restore
latched temporal HoldForReview survives checkpoint restore
```

That is evidence that the repository already has the beginnings of a distinct operational-continuity model.

## 5. Corrected status of the earlier reset findings

### RST-001 — operator authority reset

Observation:

`OperatorAuthority::reset_runtime()` clears the operator constraint when called by the generic embodiment reset.

Initial interpretation:

```text
reset bypasses ResumeNominal quorum
```

Corrected interpretation:

```text
NOT DEMONSTRATED as an operational bypass,
because the only established caller is the new-scenario reset path.
```

A draft code patch that preserved operator restrictions across the existing reset was opened during the first audit and then **closed unmerged** after this semantic correction.

### RST-002 — degraded RecoveryRequired reset

Observation:

The generic reset clears `RecoveryRequired` even though same-lineage recovery normally needs explicit authorization and healthy dwell.

Corrected interpretation:

This is appropriate for a deliberately new scenario unless `reset()` is reused as operational recovery.

The first preservation patch was therefore **closed unmerged**.

Future operational recovery must preserve/revalidate the latch according to domain rules.

### RST-003 — partition reconciliation reset

Observation:

The generic reset initializes partition state as `Connected` with authoritative team state.

Corrected interpretation:

This is a valid new-scenario fixture. It is not valid evidence of reconnection in the prior lineage.

The first preservation patch was **closed unmerged**.

### RST-004 — temporal hold reset

Observation:

The scenario reset creates a fresh default temporal supervisor.

Corrected interpretation:

That is valid for a new scenario. Operational checkpoint restoration already preserves the temporal latch, with a regression test demonstrating continuity.

No temporal reset hardening PR should be opened merely against scenario-reset semantics.

### RST-005 — field/capability defaults

Observation:

Scenario reset restores nominal field/capability fixtures.

Corrected interpretation:

Valid for a freshly reset simulated plant. A future physical operational-recovery path must instead establish capability from physical/durable evidence.

### RST-006 — optimistic health booleans

Observation:

Scenario reset assigns positive health booleans.

Corrected interpretation:

These are simulation fixtures today. They become an architectural hazard only if the same API is treated as operational restart/recovery.

A future operational API should use explicit unknown/revalidated health or a typed recovery-evidence input rather than inheriting these fixture defaults.

## 6. The actual open problem: lifecycle typing

The repository currently has a narrow trait operation named simply:

```text
reset()
```

while the subterranean domain also has an explicit checkpoint continuity path.

The design opportunity is to make those meanings impossible to confuse.

Candidate conceptual split:

```text
reset_simulation_lineage(...)
reset_ephemeral_runtime(...)
recover_operational_lineage(RecoveryEvidence)
factory_reset(AdminAuthority)
```

Names are illustrative; the semantic separation is the requirement.

This is tracked by the dedicated architecture issue:

```text
arch(embodiment): split simulation reset from operational recovery
```

## 7. Candidate OperationalRecoveryContextV1

A future *domain-owned* recovery input could expose the premises that nominal operation depends on:

```text
OperationalRecoveryContextV1 {
    lineage_id
    checkpoint_identity
    checkpoint_validity
    boot_generation

    operator_authority_snapshot
    revocation_state

    physical_state_evidence
    capability_evidence

    network_state
    peer_revision_state

    temporal_state
    audit_head

    recovery_authorization
}
```

This is not yet a proposed common Rust type.

The point is the theorem:

```text
operational recovery requires evidence about continuity
```

rather than:

```text
object construction/default values == recovery evidence
```

## 8. Property tests for a future operational-recovery API

### RST-P1 — no same-lineage authority widening

```text
Authority(recover(state, evidence))
    ⊆
Authority(state)
```

until evidence or explicit authority transitions justify widening.

### RST-P2 — no positive-evidence manufacture

Memory loss alone cannot convert stale/false/unknown health into verified-positive health.

### RST-P3 — durable restriction continuity

Where the domain defines them as same-lineage durable, restrictions such as revocations, emergency stops, maintenance locks, recovery-required latches, and causal review holds survive interruption or require explicit revalidation.

### RST-P4 — provisional-positive progress may be lost

Partial quorum, reconciliation dwell, clean-health dwell, or unsigned plans may be conservatively discarded if their continuity cannot be established.

### RST-P5 — simulation reset is allowed to start clean

A scenario reset may clear simulated restrictions when it clearly starts a new scenario lineage.

The test should verify *lineage separation*, not monotonicity across two unrelated simulated worlds.

### RST-P6 — API non-confusion

Code that performs operational recovery must not compile or route through the scenario-reset API by accident once the lifecycle split is introduced.

## 9. Revised implementation sequence

Do **not** land the first three reset-preservation patches against the existing scenario-reset API.

Instead:

```text
RST-LIFE-001  define scenario-reset vs operational-recovery contract
RST-LIFE-002  inventory EmbodimentBridge reset call sites across platforms
RST-LIFE-003  define operational recovery evidence/provenance requirements
RST-LIFE-004  add/strengthen checkpoint continuity property tests
RST-LIFE-005  introduce typed lifecycle API with migration path
RST-LIFE-006  only then patch domain-specific operational recovery gaps
```

The previously opened operator/degraded/partition preservation drafts were closed unmerged after the call-site evidence changed the interpretation.

## 10. Meta-lesson for HAK

This correction is itself an important HAK result.

A safety rule is not sufficient; the *semantic object to which it applies* must also be correct.

```text
Authority monotonicity across one lineage: required.
Authority monotonicity across unrelated new lineages: not generally meaningful.
```

So HAK review should always establish:

```text
identity of subject
identity of resource
identity of authority lineage
identity of lifecycle operation
```

before proving attenuation/monotonicity.

That is stronger than simply adding more fail-closed code.

## 11. Non-claims

This audit does not claim:

- the current subterranean embodiment is physically deployed;
- generic `reset()` is presently used for live recovery;
- deterministic scenario reset must preserve previous scenario authority;
- current operational checkpoint semantics are complete for physical deployment;
- positive simulation fixture defaults are valid recovery evidence;
- every platform needs identical recovery semantics;
- a generic HAK lifecycle crate should be introduced now.

The corrected conclusion is narrower:

```text
A new scenario may start clean.
A continuing operational lineage may not use scenario reinitialization
as proof that its prior restrictions, faults, or uncertainty disappeared.
```
