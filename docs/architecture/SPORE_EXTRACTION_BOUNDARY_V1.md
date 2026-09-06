# Spore Extraction Boundary v1

Status: **Pre-extraction architecture contract**

This document freezes the authority and ownership boundary for extracting the boot/recovery Spore system from Symthaea and host-specific NixOS configuration into an independent `Luminous-Dynamics/spore` repository.

It is intentionally a migration contract, not a claim that the extraction is complete or qualified.

## 1. Scope

The Spore being extracted here is the boot/recovery and trustworthy system-transition architecture.

It is **not** `crates/domains/symthaea-spore`, which is an existing Symthaea domain/consciousness component. That crate remains under Symthaea ownership during this migration and MUST NOT be silently reinterpreted as the recovery product.

The target ownership split is:

- **Spore**: admission, actual boot identity, attempt accounting, qualification protocol, recovery state, LKG semantics, recovery selection interfaces, generic NixOS integration, and qualification tooling.
- **Nixward**: proposal/evaluation of candidate system changes. It does not bless machine health.
- **Symthaea**: optional boot presentation, Boot Ecology, rendering, semantic animation, and telemetry consumption. It does not select, promote, or recover machine state.
- **nixos-config**: host-specific policy, enablement, hardware integration, and exact pinning of a qualified Spore revision. It must not remain the canonical home for generic Spore recovery semantics after cutover.

## 2. Architectural rule

Spore is a recovery and trustworthy system-transition architecture. Nix supplies immutable system materialization. Nixward may propose change. Symthaea may present evidence and state. Spore owns the narrow machine-state trust/recovery boundary.

The intended dependency direction is:

```text
Nixward proposal/evaluation
          |
          v
       Spore
 admission -> boot identity -> health evidence -> qualification -> recovery
          |
          v
 read-only presentation state
          |
          v
      Symthaea

nixos-config = host policy + exact pinned Spore consumer
```

The following dependency edges are forbidden:

```text
Spore recovery        -X-> Symthaea
Spore qualification   -X-> presentation authority
Symthaea presentation -X-> recovery actions
nixos-config          -X-> define generic Spore semantics
fleet policy          -X-> mint local health
observer              -X-> manufacture qualification
```

## 3. Constitutional invariants

These invariants govern the extraction and all subsequent Spore product work.

### SPORE-001 — Presentation cannot hold boot

Failure, absence, crash, timeout, or sandbox denial of the presentation layer MUST NOT make an otherwise bootable system unavailable.

### SPORE-002 — Observation cannot create authority

Observers and witnesses report facts. They do not select arbitrary generations, bless candidates, promote LKG, reset attempt budgets, or grant themselves recovery authority.

### SPORE-003 — Unknown is not healthy

`UNKNOWN` MUST remain semantically distinct from `SATISFIED`. Missing, stale, malformed, duplicated, or untrusted evidence MUST NOT be normalized into success.

### SPORE-004 — Live activation is not physical boot

`/run/booted-system` or an equivalently authoritative physical-boot identity MUST remain distinct from `/run/current-system` or other live userspace activation state.

A generation that was merely activated live MUST NOT become LKG as though it had physically booted.

### SPORE-005 — Unbooted never becomes LKG

A generation MUST NOT be promoted to LKG without evidence bound to the exact physically booted generation and the applicable qualification policy.

### SPORE-006 — Candidate failure preserves recovery

Failure of a candidate, qualification helper, renderer, observer, or lifecycle helper MUST NOT destroy the previous valid recovery path.

### SPORE-007 — Recovery truth is reconstructible

Durable recovery state MUST have an explicit authoritative representation whose validity can be checked after crash/restart without trusting stale convenience aliases.

### SPORE-008 — Helpers are expendable

Spore helper processes MUST be bounded and failure-contained. They MUST NOT be able to hold critical boot or lifecycle progress indefinitely.

### SPORE-009 — Boot fails open; promotion fails closed

Optional Spore functionality MUST not gate ordinary system boot merely because Spore failed. However, promotion, blessing, or recovery-state advancement MUST fail closed when required evidence is absent, invalid, stale, or ambiguous.

### SPORE-010 — External policy cannot mint local health

Fleet, repository, update, AI, or remote coordination systems may determine eligibility to attempt a candidate. They MUST NOT manufacture evidence that the local machine successfully booted and remained healthy.

### SPORE-011 — AI may propose, not bless

AI-assisted systems may recommend or synthesize candidate changes and diagnostic interpretations. They MUST NOT independently promote LKG, replenish boot attempts, or bypass recovery policy.

### SPORE-012 — Core recovery has no Symthaea dependency

The extracted Spore recovery/qualification core MUST build, test, and qualify headlessly without Symthaea.

## 4. Migration evidence rules

Repository extraction is an authority migration, not a file move.

No source is considered qualified in the new repository merely because its origin was previously qualified elsewhere.

For every migrated artifact, record at minimum:

```text
source_repository
source_commit
source_branch_or_pr
source_path
source_digest
destination_path
transformation
qualification_status
```

The required evidence flow is:

```text
known source lineage
        |
        v
deterministic migration
        |
        v
committed destination bytes
        |
        v
fresh exact-source qualification
```

Historical qualification MAY be cited as provenance evidence, but MUST NOT be relabeled as destination-repository product qualification.

## 5. Exact-source qualification rule

The canonical independent Spore repository MUST converge on:

```text
git commit
    |
    v
exact committed source bytes
    |
    v
Nix materialization
    |
    v
exact derivation/closure
    |
    v
qualification
```

Qualification workflows MUST NOT silently mutate product source between checkout and qualification.

If a deterministic transformation is temporarily unavoidable, the qualification record MUST explicitly bind:

- triggering source digest,
- transformation digest/version,
- pre-transform tree digest,
- post-transform tree digest,
- produced artifact/closure identity,
- and the exact qualification result.

The preferred steady state is still to materialize the stabilized implementation as committed source and qualify those exact bytes.

## 6. Extraction phases

The migration is intentionally divided into parity work and new architecture work.

### Phase A — Boundary and provenance

1. Freeze this ownership/authority contract.
2. Create an independent `Luminous-Dynamics/spore` repository.
3. Establish a migration provenance ledger.
4. Preserve frozen historical qualification heads; do not rewrite them.

### Phase B — Behavioral parity

5. Capture a golden parity corpus for receipts, boot identity, qualification, LKG behavior, helper expendability, lifecycle behavior, and effective systemd authority.
6. Extract generic protocol types without recovery authority in presentation-facing APIs.
7. Extract generic Linux recovery implementation.
8. Extract generic NixOS module/tests.

### Phase C — Fresh destination qualification

9. Materialize stabilized product source without CI source mutation.
10. Qualify exact committed destination bytes.
11. Re-run effective-systemd authority proof in the destination repository.
12. Keep physical Spore disabled until the new qualification lineage is green.

### Phase D — Consumer cutover

13. Convert `nixos-config` into a host-policy consumer of an exact qualified Spore revision.
14. Convert Symthaea into a read-only presentation consumer.
15. Prove that removing Symthaea removes presentation only, not recovery.

Only after these phases are green is the independent repository canonical.

## 7. Extraction-complete gate

The extraction MUST NOT be called complete until all of the following are true:

- [ ] Independent Spore repository exists and has no core Symthaea dependency.
- [ ] Spore builds and qualifies headlessly.
- [ ] Removing Symthaea removes presentation only.
- [ ] `nixos-config` contains host policy but not canonical generic recovery implementation.
- [ ] Existing fail-open and helper-expendability tests pass in the destination repository.
- [ ] Existing effective-systemd authority proof passes in the destination repository.
- [ ] Physical boot identity remains authoritative over live activation.
- [ ] No migration PR claims inherited qualification without fresh evidence.
- [ ] CI qualifies exact committed product bytes.
- [ ] Physical Spore remains disabled until the destination qualification lineage is green.
- [ ] Old frozen qualification branches remain untouched and attributable.
- [ ] Presentation cannot acquire recovery authority through the destination API.

## 8. First post-extraction product work

New product architecture MUST begin only after the extraction/parity gate is sealed.

The preferred first post-extraction sequence is:

1. Recovery Capsule v1 — one durable recovery-truth commit point.
2. Formal recovery model — small TLA+/PlusCal model of recovery state transitions.
3. Boot-attempt authority — attempt budget owned below ordinary userspace.
4. Typed health profiles — witnesses report facts; evaluator alone produces qualification state.
5. Persistent-state compatibility — automatic rollback only when state compatibility is proven.
6. Crash/power-loss qualification — old-valid or new-valid after every injected failure, never ambiguous.

## 9. Non-goals during extraction

The extraction MUST NOT be used as justification to simultaneously add:

- fleet orchestration,
- TUF replacement,
- TPM/Secure Boot redesign,
- remote recovery control,
- speculative new recovery daemons,
- broad API cleanup unrelated to parity,
- or presentation features.

Those may be valuable later, but mixing them with extraction would weaken causal attribution and qualification clarity.

## 10. Review rule

Every extraction PR should answer four questions explicitly:

1. **What authority moved?**
2. **What authority did not move?**
3. **What exact source lineage was migrated?**
4. **What evidence proves destination behavior rather than assuming source qualification transferred?**

If a PR cannot answer those questions precisely, it is not ready to advance the migration boundary.
