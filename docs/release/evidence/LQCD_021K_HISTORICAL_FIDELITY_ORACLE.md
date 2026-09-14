# LQCD-021K — executed historical-fidelity ledger

This subject turns the EHK beta=6.0 historical-fidelity policy into a machine-validated artifact rather than prose.

Exact executed validator subject SHA-256:

`f3d3fa9bdafe090c1d0ddc4e473a7bbd292b1c72d00bb2f07f55c964d73b0279`

Checked-in ledger file SHA-256:

`1018b651fb5943d20141f6c98e9d3e1225be0f33463c807cc12969c6a80360e3`

Canonical authoritative ledger SHA-256 (entries sorted by `field_id`):

`1dd5a6db9cc68290ea80cffb3ecbc5a47fcaf01e2a1663a6e7950463224e8fc1`

Canonical validator result SHA-256:

`66f8ecfdef78f24e3d7d0b69783848c3a7eefa84478aac677fe3ddb8a4c268bd`

## Executed classification census

45 entries total:

- 15 `ExactlySpecified`;
- 2 `ApproximatelySpecified`;
- 14 `HistoricallyUnderdetermined`;
- 11 `DeclaredReproductionConvention`;
- 2 `DerivedFromSpecifiedQuantity`;
- 1 `Unavailable`;
- 0 `IndependentlyReconstructed` / `NotApplicable` in this initial ledger.

The direct published coordinates `a*sqrt(sigma)`, `r0/a`, `r4/a`, and `r6/a` are required to remain `ExactlySpecified`.

The validator also requires key historical gaps—burn-in, stride, RNG/seeds, exact off-axis path, exact smearing projection/alpha/iteration count, fit weighting and related fields—to remain underdetermined unless later source evidence explicitly upgrades them.

## Claim gate

The current ledger mechanically rejects an `exact historical implementation` claim because implementation-significant fields remain historically underdetermined or unavailable.

It permits only the *wording template* for a future successful numerical result:

> numerical reproduction of the published EHK beta=6.0 16^3x32 benchmark under declared Symthaea conventions for historically underdetermined implementation details

The validator explicitly records that this wording still requires an actual future numerical result. This artifact does not claim the reproduction has succeeded.

## New convention bindings

The ledger now includes the executed-oracle/pending-Rust status of:

- #2784/#2789 phase/purpose-scoped ChaCha8 replay as a Symthaea convention candidate;
- #2798/#2805 `wilson_gauge_field_be_f64_v1` persistence as a Symthaea convention candidate.

These entries are not retroactively attributed to EHK.

## Scientific boundary

Historical fidelity and claim discipline only. No equilibrium, production execution, final-analysis, or EHK numerical agreement follows from this ledger.
