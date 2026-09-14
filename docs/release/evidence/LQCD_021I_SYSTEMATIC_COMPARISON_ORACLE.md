# LQCD-021I — independent systematic-error and sealed-comparison oracle

Independent standard-library execution for the final comparison semantics tracked by #2718. This tranche stacks directly on #2840 and consumes its already-sealed synthetic result identity.

Exact executed subject SHA-256:

`3206ecd5a23bdcc2c7111db5f31efa23fc0294cfee236ec0a028b8a49d617181`

Canonical result SHA-256:

`7dbaf43e1e874b9a7c2e2ce406b3b8d8480bf9ad3d1ed4e6d3ea63eae1ae7042`

Consumed sealed-analysis digest:

`a6e9235bd001fe4b90d736ec6e502c0d92eb7e89dbae5b57618077c8c94fe3b7`

## Comparison boundary

The comparator consumes only:

- an immutable sealed result;
- a benchmark object;
- a frozen systematic-evidence ledger.

The benchmark cannot alter the sealed estimate. Mutating the benchmark changes the downstream comparison receipt while preserving the exact sealed-analysis digest.

The direct EHK source coordinates frozen in the fixture are:

- `a*sqrt(sigma)=0.2189(9)`;
- `r0/a=5.369(9)`;
- `r4/a=8.831(21)`;
- `r6/a=10.89(3)`.

This fixture is a comparison-semantics exercise only. #2840 is synthetic data and this comparison therefore makes no beta=6.0 physics claim.

## Explicit uncertainty authority

The v1 standardized-discrepancy denominator contains only:

`sealed statistical SE` and `benchmark source SE`

combined in quadrature under an explicit synthetic independence declaration.

Systematic components are preserved separately. In particular, the oracle carries numeric synthetic plateau/excited-state and fit/r-range sensitivity values but marks them `NotAuthorizedUnknownCorrelation`. Attempting to add either to the denominator is rejected.

Thus:

`numeric systematic estimate != permission to quadrature it`

A future real campaign may authorize covariance-aware combination only through an explicitly frozen semantics/profile. Unknown correlation never defaults to independence.

## Missing uncertainty fails closed

#2840 did not seal a statistical uncertainty for `a*sqrt(sigma)`. The 021I oracle therefore emits no standardized discrepancy for that coordinate instead of inventing, deriving post hoc, or borrowing an uncertainty.

The three Sommer-coordinate standardized discrepancies in the #2840-to-EHK synthetic fixture are approximately:

- `r0/a: -22.67977623`;
- `r4/a: -16.97640746`;
- `r6/a: -13.03075684`.

The synthetic fixture therefore exercises `ExactVolumeNumericalReproductionNotSupported`; that disposition is not a statement about a real beta=6.0 Symthaea ensemble.

## Claim-ladder fixtures

Separate synthetic fixtures exercise all three typed outcomes:

- `ExactVolumeNumericalReproductionSupported`;
- `ExactVolumeNumericalReproductionNotSupported`;
- `Inconclusive`.

An unresolved operator-convention systematic forces `Inconclusive` even when a numerical comparator could otherwise be evaluated.

No benchmark-provided pass threshold exists. The claim policy is frozen in the ledger before comparison.

## Scientific boundary

This oracle qualifies systematic-ledger separation, denominator authority, missing-uncertainty behavior, benchmark/estimator separation, and typed comparison outcomes only. It does not establish a real final sample, real EHK agreement/disagreement, production Rust parity, finite-volume control, or continuum physics.
