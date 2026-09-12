# ADR-018 — RSK Resource Schema v0.2 Runtime-ID Binding

- **Status:** Accepted for reference/prototype architecture; production admission denied
- **Change Class:** A
- **Scope:** Replicator Safety Kernel semantic integrity
- **Related:** #1679, #1867, #1936, #1966, #1969, #1973

## Context

RSK resource accounting already binds numeric amounts to one `ResourceAccountingSchemeId`, but static review found that the v0.1 golden resource schema committed canonical semantic dimension names while the Rust reference layer separately chose numeric runtime dimension IDs.

That left a semantic-identity hole:

```text
same scheme digest + different runtime numeric mapping
```

could change the meaning of a `ResourceVector` without changing its advertised scheme identity.

## Decision

Introduce a new production-target resource schema profile:

```text
symthaea.rsk.resource-accounting-schema.v2
```

Each canonical dimension now includes an explicit `numeric_id` that is part of the exact digested schema bytes.

The runtime numeric ID is therefore identity, not mutable metadata.

The v0.1 profile remains interpretable historical/reference evidence but is not eligible for production-target runtime-ID binding.

## Required invariants

1. `ResourceAccountingSchemeId` commits semantic and runtime numeric dimension identities.
2. Numeric dimension IDs are unique `u16` values.
3. Semantic IDs remain canonical and unique.
4. Changing only a numeric mapping changes the scheme digest.
5. The numeric mapping cannot be overridden by registry side metadata.
6. Runtime vectors accept only numeric IDs committed by the exact verified schema.
7. Unknown, duplicate or missing-required numeric IDs fail closed.
8. v0.1 cannot be retroactively reinterpreted as v0.2.
9. v0.1 -> v0.2 requires an explicit cross-schema conservative transition.
10. Migration cannot reset consumption or widen remaining authority.

## Reference evidence

The abstract v0.2 reference profile commits:

```text
budget.compute -> 0
budget.energy  -> 1
```

with scheme digest:

```text
8386b56b11818273612cc9f15e6c6fa8cd19bbf9aa2c7a3476022d2f7ef0f1e1
```

Remapping only the numeric IDs produces:

```text
04f5f4cce1d7f819becf84fdff236ff6db695e0590c37c665994487d2b5ff024
```

These values are abstract golden vectors, not production physical-resource semantics.

## Compatibility strategy

The Python reference validator supports both resource schema tags so historical v0.1 evidence can still be checked.

A separate API, `require_runtime_bound_resource_schema`, rejects v0.1 and accepts only the v0.2 runtime-ID-bound profile.

This prevents compatibility from silently becoming production authority.

## Migration

No grant, budget, authorization, checkpoint or ledger state may be relabeled from v0.1 to v0.2.

Any carryover requires separately verified translation evidence proving that target remaining authority is no larger than a conservative image of source remaining authority.

## Consequences

### Positive

- runtime numeric identity is cryptographically committed by the scheme digest;
- Python/Rust mapping disagreement becomes detectable;
- mutable registry metadata cannot reinterpret numeric vectors;
- historical evidence remains readable without becoming current authority;
- schema migration remains explicit and fail-closed.

### Costs

- v0.1 and v0.2 are distinct scheme identities;
- existing reference vectors need a second golden corpus;
- future Rust integration must consume the v0.2 mapping rather than hard-coded side assumptions;
- durable evidence and replay must carry the new scheme identity.

## Rejected alternatives

### Keep numeric IDs in registry metadata

Rejected because metadata outside the digested scheme could change runtime meaning without changing `ResourceAccountingSchemeId`.

### Infer numeric IDs from array position

Rejected because reordering, insertion or canonicalization changes could silently change meaning.

### Retroactively declare v0.1 to imply `0/1`

Rejected because those bytes did not encode that fact. Evidence must not acquire stronger semantics after the fact.

### Automatically migrate v0.1 authority to v0.2

Rejected because schema migration is an authority-sensitive transition and must prove non-widening remaining authority.

## Validation status

The v0.2 Python/reference implementation and golden vectors are authored in the stacked candidate branch.

Exact-head GitHub execution remains pending because repository Actions are still queued. Rust/Cargo execution remains blocked by #1926 until the workspace lockfile is regenerated and reviewed under the pinned toolchain.

No production admission is established by this ADR.

## Production status

**DENIED / NOT YET ELIGIBLE**.
