# Validity-memory phase-fidelity boundary

This note separates two distinct numerical claims in the historical validity-memory work.

## Exact causal-coordinate representation

A discrete checkpoint `n` is queried at the half-integer coordinate `n + 1/2`.
Binary64 represents that half-integer exactly for

`0 <= n < 2^52`.

The production validity-memory API therefore fails closed at `2^52` rather than
silently collapsing distinct checkpoint centers.

This is a representation theorem only.

## Phase fidelity is a separate claim

Production temporal roles currently evaluate a channel phase as

`omega * (n + 1/2)`

before applying `sin_cos`.

Even when `n + 1/2` itself is exactly representable, multiplication by an
arbitrary binary64 `omega` rounds. As the absolute checkpoint grows, the spacing
between representable products grows too. Therefore the exact-coordinate domain
must not be described as an equally strong phase-accuracy domain.

The independent direct-sum oracle deliberately avoids this shared large-argument
path. It constructs

`T(n + 1/2) = exp(i*omega/2) * exp(i*omega)^n`

using small-angle trigonometric primitives and normalized integer complex
exponentiation by squaring.

This gives the qualification suite two numerically distinct routes to the same
temporal group element.

## Current qualified claim surface

The frozen `research_v0` capacity experiments use horizons no larger than 512,
which are far from the representational boundary. The oracle directly checks its
group-power phase path against the direct formula in this small-offset regime.

A separate production/oracle score comparison is retained at checkpoint
1,000,000 with a fixed tolerance; the tolerance is not loosened as a function of
offset.

No claim is made that direct `omega * t` phase evaluation remains equally accurate
all the way to `2^52`. Extending the operationally qualified phase domain requires
either additional evidence at preregistered offsets or an implementation change
such as temporal-origin rebasing / locally referenced phase evaluation.

## Non-claims

This note does not establish unlimited-duration memory, exact transcendental
arithmetic, a capacity law, or historical-HLS superiority. It exists to prevent
an exact-coordinate theorem from being accidentally promoted into a stronger
numerical phase-fidelity claim.
