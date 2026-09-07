# Human Agency Kernel — Reset & Restart Continuity Audit v1

Status: HAK-002 supporting audit / documentation only

Parent: `HUMAN_AGENCY_KERNEL_AUTHORITY_AUGMENTATION_CONTRACT_V1.md`

## 1. Question

When a system resets, restarts, restores a checkpoint, migrates state, or begins a new simulation, which facts are being forgotten and which facts are being *asserted*?

The HAK authority-monotonicity principle requires special care here because resetting an implementation field to a convenient default can have authority semantics.

```text
zeroing a counter                 may be cleanup
setting Unknown -> Healthy        is a claim
setting Hold -> Nominal           widens authority
setting Disconnected -> Connected is a claim
clearing Revoked                  widens authority
clearing partial positive quorum  may be conservative
```

The core theorem is:

```text
OperationalReset != NewWorld
```

and:

```text
OperationalReset cannot manufacture positive evidence or remove a durable restriction.
```

## 2. Four reset classes

### 2.1 New simulation lineage

A deterministic simulation reset may intentionally construct an entirely new simulated world.

Semantics:

```text
old simulation lineage terminates
new simulation lineage begins
```

It is legitimate for the new world to begin with configured nominal fixtures if those fixtures are explicitly part of the scenario initialization.

Required safeguard:

```text
new simulation state MUST NOT be represented as continuity of a deployed operational authority lineage
```

### 2.2 Ephemeral computation reset

Purpose: discard partial work that must not survive interruption.

Examples:

- partial quorum accumulation;
- in-progress negotiation state;
- temporary buffers;
- incomplete calculations;
- non-authoritative caches.

Semantics:

```text
provisional positive state may disappear
negative/restrictive authority state must not disappear
positive evidence must not be invented
```

### 2.3 Operational restart/recovery

Purpose: continue a real authority lineage after process restart, watchdog restart, update, power interruption, or similar event.

Required continuity domains include, where applicable:

- operator stop/hold/maintenance constraints;
- revocation/refusal generations;
- replay barriers;
- degraded/recovery latches;
- partition/reconciliation state;
- causal/temporal review latches;
- physical capability/failure state;
- checkpoint validity;
- evidence freshness;
- update authority;
- audit continuity.

A restart can make evidence unavailable. It cannot safely make unavailable evidence become positive evidence.

### 2.4 Administrative factory reset

A factory reset intentionally destroys prior local state/lineage.

That is a privileged administrative action, not an implicit operational recovery path.

A post-factory-reset deployed system should normally begin unqualified/restricted until deployment identity, configuration, physical state, policy, and required authority are re-established.

## 3. Current subterranean reset composition

The current `SubterraneanEmbodiment::reset()` mixes all four reset classes.

It resets the simulated plant and cognitive/runtime state, which is reasonable for a new simulation lineage, but the same method also resets authority-relevant operational state and is exposed through `EmbodimentBridge::reset()`.

This means its semantics are ambiguous:

```text
Is this:
  new deterministic scenario?
  ephemeral process restart?
  live operational recovery?
  factory reset?
```

Until those meanings are split, the safest architectural rule is:

```text
SubterraneanEmbodiment::reset() is not evidence of operational recovery.
```

## 4. Finding RST-001 — operator authority

Current authority recovery requires `ResumeNominal` quorum plus a clear physical-hazard check.

The historical `OperatorAuthority::reset_runtime()` cleared the active operator constraint directly.

That allowed:

```text
EmergencyStop / MaintenanceLock / Hold
        ↓
reset
        ↓
None
```

without the normal authority-widening transition.

This finding has been isolated into an independent code tranche (`fix(subterranean): preserve operator authority across reset`).

Repair theorem:

```text
reset may discard partial resume quorum
reset must preserve active operator constraint and replay history
```

## 5. Finding RST-002 — degraded recovery latch

`DegradedMode::RecoveryRequired` is intentionally sticky.

Normal `update()` does not clear it merely because the operator link returns. The explicit clear path additionally requires:

- external recovery authorization;
- safe surface/service-bay location;
- fresh operator link;
- healthy control loop;
- valid checkpoint;
- reboot count below the policy limit;
- configured healthy dwell.

However, `DegradedOperationsSupervisor::reset_runtime()` currently performs:

```text
mode = Normal
operator_link_loss_steps = 0
consecutive_watchdog_failures = 0
healthy_recovery_steps = 0
```

Therefore:

```text
RecoveryRequired
   ↓ reset_runtime
Normal
```

bypasses the domain's explicit recovery authorization and dwell theorem.

Candidate repair:

```text
reset_ephemeral():
  preserve mode
  clear only provisional recovery progress/counters that are safe to forget

clear_recovery():
  remains the only RecoveryRequired -> Normal transition
```

Exact details remain domain-owned and need a focused code review.

## 6. Finding RST-003 — manufactured health assertions

`SubterraneanEmbodiment::reset()` currently assigns:

```text
operator_link_fresh = true
control_loop_healthy = true
checkpoint_valid = true
reboot_count_in_window = 0
```

These are not all ordinary caches.

In an operational lineage they are positive assertions used by degraded-operation policy.

A restart cannot establish these facts by resetting memory.

Candidate operational-restart state should instead use one of two patterns:

### Pattern A — explicit unknown state

```text
operator_link = Unknown
control_loop_health = Unknown
checkpoint_validity = Unknown
reboot_window = RestoredFromDurableEvidence | Unknown
```

and fail closed until observations establish current health.

### Pattern B — typed recovery evidence

Construct a `RestartEvidence` / `RecoveryContext` from durable or freshly verified sources and initialize the supervisor from that evidence.

Do not use boolean defaults as authority-bearing recovery evidence.

## 7. Finding RST-004 — partition reconciliation bypass

The partition-recovery module explicitly states:

```text
restored radio link != restored operational truth
```

After a partition it requires reconciliation dwell before team state becomes authoritative.

Current `PartitionRecoverySupervisor::reset_runtime()` sets:

```text
mode = Connected
partition_steps = 0
reconciliation_steps = 0
last_assessment = connected()
```

and `connected()` means:

```text
motion_permitted = true
team_state_authoritative = true
```

Thus reset can manufacture the exact state that reconciliation is designed to establish.

Candidate repair theorem:

```text
reset cannot produce Connected/team_state_authoritative
unless this is explicitly a new simulation lineage
```

For an operational reset, conservative behavior is likely:

- preserve current partition/reconciliation mode and authority;
- discard partial *positive* reconciliation dwell if continuity cannot be proven;
- require fresh connectivity and revision observations before `Connected` is regained.

## 8. Finding RST-005 — temporal/causal hold bypass

Temporal assurance can latch `HoldForReview` after invalid control timing, clock rejection, causal contradiction, or related uncertainty.

The latch clears only after clean nominal temporal evidence for a configured dwell at a safe service location.

The embodiment reset currently replaces the temporal supervisor with `TemporalAssuranceSupervisor::default()`.

Default begins with:

```text
authority = Nominal
hold_latched = false
```

Therefore an operational interpretation of reset could perform:

```text
HoldForReview
   ↓ reset
Nominal
```

without the clean-evidence dwell.

Candidate repair theorem:

```text
temporal/causal uncertainty survives operational restart
until continuity is restored or a domain-qualified recovery path clears it
```

## 9. Finding RST-006 — field-envelope/capability optimism

The embodiment reset also resets field-envelope and capability state to nominal values.

This may be perfectly correct for a fresh simulated plant because the simulator itself is reset.

It is not a valid operational restart assumption for physical hardware.

A physical restart should derive capability from current sensors, component health, maintenance state, and durable failure evidence before granting nominal work authority.

Candidate rule:

```text
NewSimulation -> configured nominal fixture is allowed
OperationalRestart -> nominal capability must be re-established from evidence
```

## 10. Why one generic reset cannot safely serve all roles

The problem is not that `reset()` is intrinsically unsafe.

The problem is semantic overloading.

A single API currently spans:

```text
new simulated universe
runtime cache cleanup
fault recovery
potential embodiment lifecycle reset
```

Those operations have different authority rules.

HAK therefore recommends API names that reveal the lineage semantics, for example:

```text
reset_simulation_lineage(...)
reset_ephemeral_runtime(...)
recover_operational_lineage(RecoveryEvidence)
factory_reset(AdminAuthority)
```

The exact names are illustrative.

The important part is that call sites cannot accidentally choose an authority-widening operation because all of them are spelled `reset()`.

## 11. Candidate OperationalRecoveryContextV1

A future domain-owned recovery input could resemble:

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

This is not proposed as a generic HAK struct yet.

The purpose is to make the missing premise visible:

```text
recovery requires evidence about continuity
```

rather than treating process construction/default values as that evidence.

## 12. Property tests for reset boundaries

### RST-P1 — no authority widening

For operational reset `R`:

```text
Authority(R(state)) ⊆ Authority(state)
```

until explicit fresh recovery evidence is processed.

### RST-P2 — no positive evidence manufacture

If a fact is `false`, stale, degraded, or unknown before restart, resetting memory alone cannot make it verified-positive.

### RST-P3 — restriction persistence

Durable restrictions survive reset:

- revocations;
- refusals where domain semantics require continuity;
- emergency stops;
- maintenance locks;
- recovery-required latches;
- causal review holds.

### RST-P4 — provisional-positive loss is safe

Partial authority-accruing state may be discarded conservatively:

- one of two quorum approvals;
- partial reconciliation dwell;
- partial clean-health dwell;
- unsigned/uncommitted action plans.

### RST-P5 — simulation reset begins a new lineage

A simulation reset that intentionally clears restrictions must produce a new lineage/scenario identity rather than masquerading as recovery of the previous one.

## 13. Proposed code tranches

Keep fixes independent so each theorem receives focused evidence.

```text
RST-CODE-001  operator constraint reset continuity     [opened]
RST-CODE-002  degraded RecoveryRequired continuity
RST-CODE-003  partition reconciliation continuity
RST-CODE-004  temporal review-latch continuity
RST-CODE-005  split simulation reset from operational recovery
RST-CODE-006  replace optimistic operational health defaults with typed evidence
```

Ordering recommendation:

```text
001 -> 002 -> 003 -> 004 -> 005/006
```

The first four repair concrete state-machine escapes. The final two redesign the lifecycle boundary once the local invariants are explicit.

## 14. Non-claims

This audit does not claim:

- the current subterranean embodiment is deployed on physical machinery;
- deterministic simulation reset should preserve the previous scenario's authority state;
- every cached diagnostic value is authority-bearing;
- all supervisor state must persist forever;
- operational recovery should be impossible after a fault;
- the proposed `OperationalRecoveryContextV1` is ready for implementation.

The narrower finding is sufficient:

```text
A reset API must not be allowed to decide, accidentally and implicitly,
that the conditions which previously restricted authority are now healthy.
```
