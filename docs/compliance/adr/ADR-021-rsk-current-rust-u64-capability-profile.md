# ADR-021 — RSK Current Rust u64 Capability Representation Profile

- **Status:** Accepted for current reference/runtime target; production admission denied
- **Change Class:** A
- **Scope:** Replicator Safety Kernel capability semantic representation
- **Related:** #1678, #1682, #1936, #1966, #1969, #2038, #2057, #2064, #2071

## Context

Cross-language audit found a concrete mismatch:

- generic Python/reference capability schema validation allowed widths up to 4096;
- the current Rust `BoundCapabilitySet` stores a `u64`;
- the current Rust structural validator accepts widths only through 64.

Without an explicit representation profile, reference tooling could approve a schema that the admitted Rust implementation cannot represent exactly.

## Decision

Define the current representation profile:

```text
symthaea.rsk.capability-representation.rust-u64.v1
```

with:

```text
1 <= bit_width <= 64
```

Generic schema parsing remains wider for future/historical use. The current Rust deterministic adapter is stricter and refuses schemas wider than 64 bits.

The representation profile is included in the derived capability validator table and therefore in its canonical adapter-output identity.

## Golden identity change

Adding the representation profile changes the abstract capability adapter-output digest to:

```text
d118f3777c4d78ac05293e9228706afa31a57ba69c7f1bd65dde9aca96c8c460
```

This supersedes the earlier `a6fec0f3...` adapter-output digest recorded by ADR-020, which did not yet bind the runtime representation profile.

The source capability schema digest itself is unchanged because schema meaning and runtime representation are separate identities.

## Required behavior

- width 64 is eligible for the current Rust profile;
- width 65 remains generically parseable where appropriate but cannot produce a current-Rust validator;
- no truncation or masking;
- no low-bit-only bypass for a wider schema;
- no modulo/remapping of high bits;
- build/runtime admission eventually binds the profile identity.

## Future upgrade rule

A wider Rust capability representation requires a new explicit profile and qualification evidence. It is not a transparent implementation detail.

Existing authority cannot silently move between representation profiles.

## Consequences

### Positive

- Python/Rust representability is explicit;
- prevents silent truncation;
- keeps the abstract schema format future-extensible;
- representation becomes part of validator evidence;
- runtime/build identity can bind the exact semantic representation.

### Costs

- current-Rust adapter output identity changes;
- wider future schemas need a new admitted runtime profile;
- production grants/ledger/replay must eventually carry or transitively bind the profile identity.

## Rejected alternatives

### Reduce the generic schema format to 64 bits permanently

Rejected because it unnecessarily prevents future wider representations and conflates schema validity with one runtime implementation.

### Accept wider schemas when only low bits are set

Rejected because later values or replays could activate unrepresentable bits under the same schema identity.

### Truncate to u64

Rejected because truncation silently changes authority meaning.

## Validation status

Boundary tests for widths 64/65 and the updated adapter golden identity are authored in the candidate branch. Exact-head execution remains pending GitHub runner availability.

Rust/Cargo qualification remains additionally blocked by #1926.

## Production status

**DENIED / NOT YET ELIGIBLE**.
