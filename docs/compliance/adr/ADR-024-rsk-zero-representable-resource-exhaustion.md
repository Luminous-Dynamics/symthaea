# ADR-024: RSK zero-representable resource exhaustion

**Status**: Proposed

**Change Class**: A

**Date**: 2026-09-13

## Context

RSK resource accounting derives remaining authority by subtracting consumed resource from an exact bound limit under one exact accounting scheme.

A cross-language audit found a semantic mismatch around the schema `minimum` field:

- Rust `ResolvedResourceAccountingScheme::checked_remaining()` computes the remainder and then validates the derived `ResourceVector` under the same scheme;
- Python `remaining_vector()` previously validated only the input vectors and returned the subtraction result without validating that result;
- the generic schema permits `minimum > 0`.

Therefore valid input vectors could produce a remainder accepted by Python but rejected by Rust.

Example:

```text
minimum = 10
limit = 20
consumed = 15
remaining = 5
```

The result is below the same scheme's declared representable minimum.

For consumptive authority budgets, a second problem follows: if `minimum > 0`, complete exhaustion (`remaining = 0`) is not representable.

This ADR contains no physical resource definition, fabrication recipe, biological/molecular design, or physical replication mechanism.

## Decision

Make two Class A semantic changes.

### 1. Validate derived remaining vectors

The generic Python reference `remaining_vector()` must validate the derived remainder under the exact same resource scheme before returning it.

The required shape is:

```text
validate(limit)
validate(consumed)
require identical dimension set
checked subtraction
validate(remaining)
return remaining
```

Valid operands no longer imply a valid derived authority value automatically.

### 2. Require zero-representable exhaustion in the current budget profile

The current Rust resource execution profile:

```text
symthaea.rsk.resource-representation.rust-u64-sum-exact.v1
```

now additionally requires:

```text
minimum == 0
```

for every admitted resource dimension.

This guarantees that full consumption can produce an ordinary valid zero remainder.

Generic schemas with positive minima remain structurally parseable/reference evidence but are not eligible for this current consumptive authority profile.

## Rationale

Zero remaining authority must be representable without special semantics.

Alternatives such as upward clamping, implicit sentinel values, dropping exhausted dimensions, or treating an invalid vector as zero would create multiple arithmetic meanings and could accidentally widen authority.

Rejecting positive minima in the current budget profile is simpler and fail-closed.

## Semantic execution profile relationship

The semantic execution profile derives only after current runtime profile eligibility succeeds.

Therefore a resource schema with `minimum > 0` cannot produce the current `SemanticExecutionProfileId`.

The committed abstract golden corpus already uses `minimum = 0`, so this change does not alter the existing current-profile adapter or semantic-execution golden identities.

## Restart and recovery

An exhausted zero budget remains exhausted across restart unless separately governed authority explicitly changes the relevant grant/policy/ledger state.

A restart, cache restore, grant-generation rollover, or recovery procedure must not interpret zero as missing/uninitialized budget and refill it implicitly.

## Alternatives considered

### Keep Python behavior and stop re-validating Rust remainders

Rejected. A derived value that violates its own declared semantic type should not become authority evidence.

### Clamp below-minimum results to the minimum

Rejected. For authority budgets this can recreate positive remaining authority after it has been consumed.

### Use a sentinel for exhausted state

Rejected for the current profile. This introduces a second arithmetic representation and complicates durable/replay semantics.

### Require only `minimum <= 0`

The generic schema already requires nonnegative minima. For the current consumptive profile, exact zero is the unambiguous requirement.

## Tests

Reference tests are extended to cover:

- valid inputs producing an invalid below-minimum remainder;
- full consumption yielding valid zero when the scheme admits zero;
- current runtime profile rejection of positive minima;
- semantic execution profile rejection of positive-minimum resource schemas.

The existing zero-minimum golden corpora remain unchanged.

## Evidence status

At authoring time:

- the generic reference arithmetic change is authored;
- current-profile zero-minimum gating is authored;
- parity/adversarial tests are authored;
- exact-head GitHub workflow execution is not yet claimed because repository runners remain backlogged;
- admissible Rust/Cargo qualification remains blocked by #1926;
- production admission remains denied.

No production gate is checked off by this ADR.

## Related work

- #1679 — typed/versioned resource accounting
- #1926 — controlled workspace-lockfile qualification blocker
- #1966 — Rust structural resource validation
- #2035 — resource schema v0.2 runtime numeric-ID binding
- #2078/#2080 — current Rust resource representation/arithmetic profile
- #2143 — semantic execution profile identity
- #2149 — zero-representable exhaustion blocker

## Production admission

```text
DENIED / NOT YET ELIGIBLE
```
