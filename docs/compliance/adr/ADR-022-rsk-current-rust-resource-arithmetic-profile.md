# ADR-022 — RSK Current Rust Resource Representation / Arithmetic Profile

- **Status:** Accepted for current reference/runtime target; production admission denied
- **Change Class:** A
- **Scope:** Replicator Safety Kernel resource semantic representation and arithmetic
- **Related:** #1679, #1682, #1936, #1966, #1973, #2035, #2038, #2057, #2064, #2078

## Context

Cross-language audit found that the generic resource schema is more expressive than the current Rust semantic implementation:

- generic integer bounds are not restricted to `u64`;
- generic aggregation permits `sum` or `max`;
- generic rounding permits policies beyond `exact`;
- Rust stores resource quantities as `u64` and implements checked additive/subtractive arithmetic.

Without an explicit runtime arithmetic profile, a correctly hashed schema could declare semantics different from those executed by Rust.

## Decision

Define the current resource representation/arithmetic profile:

```text
symthaea.rsk.resource-representation.rust-u64-sum-exact.v1
```

Current eligibility requires:

- resource schema v0.2 runtime-ID binding;
- all bounds representable by `u64`;
- `aggregation = sum`;
- `rounding = exact`.

Generic schema parsing remains more expressive for future/reference purposes.

The deterministic resource adapter is the stricter boundary and refuses unsupported schemas before emitting a current-Rust rule table.

## Adapter identity

The representation/arithmetic profile is included in the derived resource validator table.

For the abstract v0.2 golden resource schema, the resulting adapter-output SHA-256 is:

```text
baf0022df5f556743032e0ab1c1105838d120461649167b34a433679cd810b0a
```

This supersedes the earlier resource adapter-output digest recorded by ADR-020, which did not yet bind the runtime arithmetic profile.

The source resource scheme digest remains unchanged because schema identity and runtime execution-profile identity are distinct.

## Required fail-closed boundaries

- `u64::MAX` is representable;
- `u64::MAX + 1` is not current-profile eligible;
- `aggregation=max` is not interpreted as `sum`;
- non-exact rounding is not silently ignored;
- no truncation, wrapping, saturation or coercion;
- schema v0.1 remains ineligible for runtime numeric-ID binding.

## Future profiles

Wider quantities, max aggregation, conversion-aware arithmetic or other rounding semantics require a new explicitly admitted profile and qualification.

Existing authority cannot silently move between execution profiles.

## Consequences

### Positive

- schema semantics and executed arithmetic cannot silently diverge;
- Python/reference and Rust representation limits are explicit;
- future arithmetic expansion remains possible through new profiles;
- build/runtime evidence can bind the exact execution semantics.

### Costs

- the current runtime intentionally rejects otherwise valid generic schemas;
- resource adapter golden identity changes;
- future profiles require explicit migration/qualification work.

## Rejected alternatives

### Let Rust ignore unsupported schema fields

Rejected because a signed/verified schema would then advertise semantics the runtime does not execute.

### Clamp wider integer ranges to u64

Rejected because that changes authority semantics and can alter remaining-budget interpretation.

### Treat aggregation/rounding as documentation only

Rejected because they are authority-relevant accounting semantics.

## Validation status

Boundary tests for `u64` width, aggregation and rounding plus the updated golden adapter identity are authored in the candidate branch. Exact-head execution remains pending GitHub runner availability.

Rust/Cargo qualification remains additionally blocked by #1926.

## Production status

**DENIED / NOT YET ELIGIBLE**.
