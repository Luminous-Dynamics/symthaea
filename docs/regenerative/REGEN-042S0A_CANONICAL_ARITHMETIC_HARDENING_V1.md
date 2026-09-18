# REGEN-042S0A — Canonical Arithmetic Hardening v1

Status: preregistration hardening only. This document freezes the primitive-layer conditions that must hold before REGEN-042S1 immutable model state may depend on the S0 quantity/time substrate. It creates no resilience, hazard, policy, emergency-authority, or physical-action claim.

## 1. Purpose

REGEN-042S0 established a deliberately small canonical substrate: opaque identifiers, exact fixed-decimal quantities, non-negative stock values, exact ratios, and half-open campaign time.

Before those primitives become inputs to immutable state commitments and shock transitions, four representation hazards must be closed explicitly:

1. logically unreachable panic paths in constructors;
2. ad-hoc numeric comparison across exact quantities;
3. hidden information loss during rational scaling;
4. unchecked time arithmetic or ambiguous interval composition.

Core theorem:

```text
canonical representation
+ panic-free construction
+ checked total comparison
+ explicit rational-scaling residuals
+ checked time composition
= model-state-safe primitive substrate
```

not:

```text
primitive substrate is sound
= resilience model is valid
= upstream observations are true
= service policy is adopted
```

## 2. Exact predecessor

This contract is a child of REGEN-042S0 exact head:

`0f71c0e11f34e31ce734ec8927ed87ec773947a7`

S0's exact-head CI remains an independent gate. This document does not relabel a queued or failed run as a PASS.

## 3. Panic-free public construction

The primitive crate is intended to be usable under adversarial fixtures without caller-triggerable panics.

The S0 implementation currently contains a logically unreachable `expect(...)` while rebuilding the reduced ratio denominator. The mathematical argument is valid, but the executable contract should not rely on a panic assertion for a value that can be represented as a checked result.

S0A therefore freezes:

```text
all public constructors and arithmetic helpers
-> Result / explicit infallible proof path
-> no unwrap/expect/panic/unreachable in production primitive code
```

This is stronger than `#![forbid(unsafe_code)]`: memory safety and panic freedom are distinct propositions.

A later implementation should permit an AST-aware or lint-assisted audit of production code while not falsely applying the same restriction to test-only assertions.

## 4. Checked quantity ordering

REGEN-042S1 needs invariants such as:

```text
committed <= usable <= nominal
protected + committed <= total
```

Those checks MUST NOT be reconstructed independently by callers through mantissa/scale manipulation.

The primitive layer should expose one checked comparison operation conceptually equivalent to:

```text
checked_cmp(a, b) -> Result<Ordering, QuantityError>
```

It must:

- reject unit mismatch;
- reject basis mismatch;
- compare mathematical values rather than raw mantissa/scale fields;
- use checked scale alignment;
- return overflow rather than wrap or silently approximate.

Structural equality remains valid only because construction canonicalizes decimal representation; ordering is still a distinct operation.

## 5. Rational scaling must expose information loss

S0 currently provides an explicit floor operation at the quantity's existing decimal scale. That is acceptable as a named rounding operation, but it is not an exact rational result when the quotient is not representable at that scale.

The later model MUST NOT silently convert:

```text
exact ratio x exact quantity
```

into:

```text
rounded quantity
```

without preserving the rounding fact.

S0A freezes two separate semantics:

```text
ExactScaleResult
```

for cases representable exactly under the adopted quantity grammar, and

```text
RoundedScaleResult {
    quantity,
    rounding_mode,
    discarded_remainder,
}
```

for deliberate rounding.

The exact concrete API may differ, but a caller must be able to distinguish mathematically exact scaling from rounded scaling without recomputing hidden internals.

## 6. Residual identity

For non-negative scaling by `n / d`, a floor result should preserve enough information to verify:

```text
input_mantissa * n
= output_mantissa * d + remainder
```

with:

```text
0 <= remainder < d
```

at the declared decimal scale.

The residual is model evidence about arithmetic transformation. It is not a physical residual, material loss, ecological loss, or measurement uncertainty.

## 7. No implicit precision expansion

A caller MUST NOT silently increase decimal scale merely to make a rational quotient appear exact.

Any precision expansion must be explicit and bounded by `MAX_DECIMAL_SCALE`.

If an exact decimal representation is impossible within the adopted scale bound, the exact API should return a declared non-representable result/error rather than secretly round.

This keeps model precision policy reviewable.

## 8. Ratio composition

If later stages compose multiplicative effects, ratio composition must be checked.

Conceptually:

```text
(a / b) * (c / d)
```

must reduce deterministically while rejecting numerator/denominator overflow.

No floating-point intermediate belongs in the canonical path.

S0A does not require compound shock semantics; it only freezes how exact ratios may compose if/when later stages use them.

## 9. Tick arithmetic

`Tick(u64)` is a canonical coordinate, not permission to perform unchecked `+` or `-` in model code.

The primitive layer should own checked operations for at least:

```text
Tick + duration
Tick - earlier_tick
```

where applicable.

Overflow or reversed temporal relationships must fail explicitly.

No wrapping-time semantics exist in REGEN-042S v1.

## 10. Interval algebra

REGEN-042A requires explicit overlap/order behavior. S0A therefore freezes deterministic interval helpers for half-open intervals, including at least:

```text
overlaps(a, b)
intersection(a, b)
is_before(a, b)
```

with the law:

```text
[a,b) and [b,c) do not overlap
```

Adjacency is not overlap.

An empty intersection is represented explicitly rather than by constructing an invalid zero-length `TimeInterval`.

## 11. Instant vs interval remains distinct

S0 correctly separates:

```text
EventTiming::Instant(t)
```

from:

```text
EventTiming::Interval([start,end))
```

S0A preserves this distinction.

A zero-duration interval is not introduced as an alias for an instant event.

## 12. Canonical arithmetic does not infer domain meaning

Primitive operations know units and bases only by exact identity equality.

They do not perform:

- unit conversion;
- wet/dry basis conversion;
- energy/mass equivalence;
- currency conversion;
- quality conversion;
- physical-system inference.

Any conversion belongs to a separately evidenced transformation outside the primitive kernel.

## 13. Comparison is not conversion

If two quantities differ by unit or basis, comparison fails.

The model must not treat a known external conversion factor as if it were part of canonical arithmetic unless that conversion has first been represented as an explicit transformation with its own provenance.

## 14. Signed vs non-negative domains

`CanonicalQuantity` may remain signed for deltas/deficits.

Physical stocks/capacities that are semantically non-negative should continue to use `NonNegativeQuantity` or later stronger newtypes.

No later model-state constructor should weaken a non-negative field back to unrestricted signed quantity merely for convenience.

## 15. Zero semantics

Canonical zero remains scale-normalized.

However:

```text
zero quantity
!= missing quantity
!= unresolved quantity
!= unavailable dependency
```

S1 must preserve those distinctions through explicit state/resolution types.

## 16. Error taxonomy

Primitive arithmetic errors should remain stable and inspectable rather than collapsing into strings.

S0A expects later implementation to distinguish at least:

- scale bound exceeded;
- unit mismatch;
- basis mismatch;
- arithmetic overflow;
- non-negative-domain violation;
- zero ratio denominator;
- non-representable exact scaling where applicable;
- time overflow;
- reversed/invalid temporal relation.

Exact enum naming may evolve during implementation review, but callers must not need string parsing to classify failure.

## 17. First regression campaign

A later executable S0A implementation should add regressions for at least:

1. ratio denominator reduction without panic-capable production code;
2. equal quantities with different input decimal forms compare equal after canonicalization;
3. cross-scale less/greater comparison;
4. unit mismatch comparison rejection;
5. basis mismatch comparison rejection;
6. scale-alignment overflow rejection;
7. exact rational scaling case;
8. non-exact scaling with explicit floor residual;
9. residual conservation identity;
10. precision-expansion bound rejection;
11. checked ratio composition and overflow;
12. checked tick addition;
13. checked tick overflow;
14. interval overlap symmetry;
15. adjacent half-open intervals do not overlap;
16. intersection identity;
17. disjoint intersection returns explicit absence;
18. instant/interval distinction unchanged;
19. negative physical-stock construction remains rejected;
20. no production `unwrap/expect/panic/unreachable` regression.

## 18. Metamorphic laws

Where applicable the first executable campaign should include laws such as:

```text
cmp(a,b) == reverse(cmp(b,a))

intersection(a,b) == intersection(b,a)

scale_floor(q,r).quantity <= exact mathematical q*r

scaled*d + remainder == original*n
```

subject to compatible unit/basis and successful checked arithmetic.

## 19. Qualification boundary

S0A should be implemented only after S0's exact-head gate is classified.

If S0 fails, the failure is preserved and S0A must not be used to retroactively convert the failed subject into a PASS.

The preferred sequence is:

```text
S0 exact-head result
-> S0A primitive hardening implementation
-> S0A exact-head qualification
-> S1 immutable state implementation
```

S1 documentation may continue to exist as preregistration, but executable S1 should consume the hardened primitive theorem rather than bypassing it.

## 20. Cross-repo boundary

This primitive hardening changes no ownership rule from Mycelix REGEN Phase E.

Mycelix remains authoritative for adopted service/profile/evidence/rights/ecology/quality/authority identity. Symthaea's primitive arithmetic only provides deterministic representation and computation over frozen inputs.

No arithmetic result can create an adopted requirement, evidence truth, right, emergency power, recommendation authority, or physical execution authority.

## 21. Deliberate non-claims

REGEN-042S0A establishes no S0 PASS, no model-state PASS, no shock-model validity, no disaster probability, no service sufficiency, no real-world conservation law beyond the explicitly modeled arithmetic object, no hazard threshold, no emergency policy, and no physical-control authority.

Its proposition is narrow:

> before immutable resilience state depends on the S0 primitive layer, exact arithmetic and time composition should be panic-free, comparison-safe, overflow-checked, and explicit about every rational-scaling remainder or rounding decision.
