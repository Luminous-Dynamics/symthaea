# Spore Golden Parity Corpus v1

Status: **Pre-extraction behavioral contract**

Parent boundary: `SPORE_EXTRACTION_BOUNDARY_V1.md`

This document defines the minimum behavior that the future independent `Luminous-Dynamics/spore` repository must reproduce before migrated recovery code may be called parity-preserving.

It does **not** transfer qualification from the source repositories. A source test establishes a migration obligation; only fresh execution against exact destination bytes can establish destination parity.

## 1. Evidence rule

The migration relationship is:

```text
source test / source invariant
        |
        v
parity obligation
        |
        v
migrated exact destination bytes
        |
        v
fresh destination execution
        |
        v
PARITY_PROVEN
```

The following implication is forbidden:

```text
source test exists or historically passed
        -X->
destination parity proven
```

This corpus therefore records **expected semantics**, not inherited PASS claims.

## 2. Exact source roots

The host-side recovery obligations in this corpus are derived from the exact inspected lineage:

```text
repository: Tristan-Stoltz-ERC/nixos-config
branch:     spore/runtime-expendability-v1.3.2-proof
commit:     5d80360768ee329c50756e71fbce4692ac3a8e45
tree:       51c04910b3a97586ecf88a46699b7de22e3e1b0b
```

Primary exact source fixtures:

```text
tests/spore-boot-fail-open.nix
4fcc1618b2e993ca38f0c7a988a7afae65dd9ede

tests/spore-boot-helper-expendability.nix
0fc089e9677d0398d79028baf65ddfe0682a4493

tests/spore-boot-ovmf-recovery.nix
b2f645694c53c579d43b2b8f8781b0e6b8a4fbd8

tests/test_spore_systemd_authority.py
4db5d9af7f942a978c2c786ffcbb8f6b1ad35e26
```

The source lineage is classified `transformed-candidate` by the parent migration manifest. That classification remains in force. This corpus does not strengthen it.

## 3. Parity classes

A destination obligation may be in exactly one migration state:

```text
DEFINED
MIGRATED_UNEXECUTED
EXECUTED_FAILED
EXECUTED_PASS
SUPERSEDED_WITH_JUSTIFICATION
```

`EXECUTED_PASS` is not product qualification. It means only that the named parity obligation passed against a recorded exact destination lineage.

A later qualification layer may consume parity evidence, but it must independently define its subject, environment, and evidence tier.

## 4. Boot availability obligations

### GP-BOOT-001 — Presentation failure cannot hold graphical boot

Renderer failure must not prevent `multi-user.target`, `graphical.target`, or a stable display manager from becoming available when the underlying machine is otherwise bootable.

Source fixture: `tests/spore-boot-fail-open.nix`.

### GP-BOOT-002 — Stubborn presentation is bounded

A renderer that ignores normal termination must still be bounded by the host handoff/stop policy. Presentation failure may remove presentation; it may not hold the desktop indefinitely.

Source fixture: `tests/spore-boot-fail-open.nix`.

### GP-BOOT-003 — Partial preparation evidence cannot become truth

If state preparation crashes after leaving partial/corrupt runtime files, those files must not make presentation or qualification proceed as though preparation succeeded.

A failed preparation must not create LKG.

Source fixture: `tests/spore-boot-fail-open.nix`.

### GP-BOOT-004 — Bless failure is recovery-local

Failure of the state blessing/promotion operation must not make an otherwise healthy graphical session unavailable. The failed candidate must not be promoted to LKG merely because the boot itself reached the desktop.

Source fixture: `tests/spore-boot-fail-open.nix`.

## 5. Physical identity and LKG obligations

### GP-ID-001 — Physical boot identity is authoritative

Recovery state must follow the generation identified as actually booted, not a userspace generation activated later.

Equivalent host semantics must preserve:

```text
physical boot identity != live activation identity
```

### GP-ID-002 — Live activation never promotes

If the current userspace generation differs from the actual booted generation, qualification must not promote the live-activated generation as LKG.

### GP-ID-003 — Stable health precedes promotion

A display/service merely becoming active is insufficient. Promotion must require the configured non-zero stability observation interval before advancing LKG.

### GP-ID-004 — LKG promotion is idempotent

Repeated evaluation of an already-promoted exact boot must not repeatedly mint new blessing side effects or reinterpret the generation as a distinct successful boot.

### GP-ID-005 — Current/Previous are boot-history semantics

`current` must represent the exact booted generation. `previous` must represent the previous distinct booted generation rather than an arbitrary live activation.

Source fixture for GP-ID-001 through GP-ID-005: `tests/spore-boot-fail-open.nix`.

## 6. Helper expendability obligations

### GP-HELPER-001 — Hung preparation is bounded

Inject a `prepare` helper that never returns.

Required result:

```text
graphical target reachable
prepare service terminally failed/bounded
explicit timeout evidence present
no valid boot-state receipt
no valid lineage receipt
no LKG promotion
promotion timer not left active
```

### GP-HELPER-002 — Hung qualification is bounded and terminal

Inject a `qualify` helper that never returns.

Required result:

```text
desktop reachable
qualification service bounded/failed
promotion timer terminated
no LKG promotion
decision = failed
reason = state-qualification-timeout
bounded helper result records timeout/forced-kill outcome
```

### GP-HELPER-003 — Hung lifecycle evidence cannot hold lifecycle

Inject a shutdown/lifecycle helper that never returns.

Required result:

```text
reboot marker service bounded
sleep hook bounded
machine shutdown still completes
```

### GP-HELPER-004 — Privileged helper is path-confined

Inject a state helper that attempts to write an unauthorized path such as `/etc/spore-boot-unauthorized-write`.

Required result:

```text
unauthorized write absent
preparation fails
no valid runtime evidence
no LKG promotion
```

Exact source fixture for GP-HELPER-001 through GP-HELPER-004:

```text
tests/spore-boot-helper-expendability.nix
blob 0fc089e9677d0398d79028baf65ddfe0682a4493
```

## 7. Effective systemd authority obligations

The authority checker must reason over the effective unit graph rather than source-text naming conventions.

### GP-AUTH-001 — Safe subordinate activation is accepted

A graph in which graphical state softly activates Spore but critical progress does not wait on it must remain clean.

### GP-AUTH-002 — Hard authority is detected

Direct or transitive critical dependency through any of:

```text
Requires
Requisite
BindsTo
```

must produce `HARD_AUTHORITY` when the critical path depends on Spore.

### GP-AUTH-003 — Temporal authority is detected

Activation plus ordering that forces a critical unit to wait for Spore must produce `TEMPORAL_AUTHORITY`, including transitive ordering paths.

### GP-AUTH-004 — Ordering without activation is not authority

A mere ordering edge that cannot pull Spore into the transaction must not be mislabeled as temporal authority.

### GP-AUTH-005 — Stop authority is detected

A critical unit made `PartOf` Spore must produce `STOP_AUTHORITY`.

### GP-AUTH-006 — Runtime conflict authority is detected

A Spore conflict against critical runtime must produce `CONFLICT_AUTHORITY`.

A normal shutdown-target conflict must remain accepted as shutdown semantics rather than a runtime authority violation.

### GP-AUTH-007 — Incomplete evidence fails closed

A snapshot referencing an absent unit, or a malformed snapshot missing required relationship data, must be rejected rather than producing a green authority proof.

Exact source regression corpus:

```text
tests/test_spore_systemd_authority.py
blob 4db5d9af7f942a978c2c786ffcbb8f6b1ad35e26
```

## 8. Firmware/boot-attempt recovery obligations

These obligations preserve the strongest current distinction between bootloader attempt authority and userspace recovery observation.

### GP-FW-001 — Known-good A becomes LKG for local evidence

Boot generation A through real OVMF/systemd-boot/NixOS test machinery. After stable health, Spore current and LKG both identify A.

The bootloader's own success/blessing mechanism and Spore's LKG mechanism remain distinct authorities even when they agree.

### GP-FW-002 — Candidate B consumes its boot attempt below userspace

Register B with one bootloader try. The bootloader consumes the try before B's userspace can determine its own success.

B genuinely boots, but if its graphical health remains unstable:

```text
booted = B
current = B
previous = A
last-known-good = A
```

B must not become Spore LKG.

### GP-FW-003 — Exhausted bad B falls back to A after abrupt loss

After B consumes its attempt and remains unqualified, crash the VM without a clean Spore shutdown marker or successful bootloader blessing.

On the next firmware boot, exhausted B must be skipped and A selected.

### GP-FW-004 — Recovery receipt explains the rollback

After fallback, the runtime recovery receipt must describe the semantic transition:

```text
RolledBack.attempted = B
RolledBack.restored = A
generation_health = Recovery
```

### GP-FW-005 — Failed B stays failed after A recovers

A may be blessed again for its new physical boot, but B's exhausted/bad state must not be erased merely because A recovered successfully.

Exact source fixture:

```text
tests/spore-boot-ovmf-recovery.nix
blob b2f645694c53c579d43b2b8f8781b0e6b8a4fbd8
```

The existence of this fixture is not a claim that every historical OVMF execution was valid or green. Historical infrastructure failures remain separate evidence. The destination must execute its own controlled OVMF qualification/parity lane.

## 9. Destination parity receipt

Each migrated parity execution should emit a receipt containing at least:

```text
schema = spore-parity-receipt-v1
obligation_id
source_repository
source_commit
source_blob
destination_repository
destination_commit
destination_tree
destination_artifact_digest
runner_identity
toolchain_identity
result
failure_classification
```

For obligations requiring a VM or firmware boot, also record the exact Nix derivation/closure identity used by the experiment.

A `PASS` without exact destination identity is insufficient for migration parity.

## 10. Extraction gate

Generic recovery product code must not be removed from its source owner merely because it has been copied into the destination.

The preferred sequence is:

```text
source remains canonical
        |
        v
copy/extract into destination
        |
        v
fresh parity execution
        |
        v
exact-source destination qualification
        |
        v
consumer cutover
        |
        v
source compatibility shim/removal
```

This keeps rollback available for the migration itself.

## 11. Parity is a floor, not the final architecture

After this corpus is green in the independent repository, Spore may intentionally strengthen behavior through later PRs such as Recovery Capsule v1, boot-attempt authority, typed health profiles, persistent-state compatibility, and crash-oracle qualification.

Those improvements should receive new invariants and tests rather than silently changing what `PARITY_PROVEN` meant for the extraction.
