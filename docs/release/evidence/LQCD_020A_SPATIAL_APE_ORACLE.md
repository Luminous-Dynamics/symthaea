# LQCD-020A — independent spatial APE smearing / SU(3) projection oracle

## Scope

This is an independent standard-library Python qualification subject for **operator construction**, not Markov-chain evolution and not a physical static potential.

It imports no Symthaea/Rust code.

The frozen convention ID is:

`spatial_ape_ehk_polar_v1`

For spatial link direction `k in {0,1,2}` the unsmeared field is updated synchronously as

`M_k(x) = alpha U_k(x) + sum(spatial staples transverse to k)`

with four spatial staples total. `M_k(x)` is then projected to SU(3) by the unitary polar factor, followed by removal of the determinant phase. Temporal (`mu=3`) links are copied unchanged.

The numerical `alpha=0.7` used here is a qualification fixture only. It is not claimed to be an optimal or universal APE parameter.

## Exact executed subject

- subject: `scripts/lqcd-020a-spatial-ape-oracle.py`
- subject SHA-256: `074614e6c2bdba1949c9c6a3bf40458b00f6502c7dd3029ee201ae8eacb1d10c`
- subject Git blob: `4cb0b23408c1be6f36c9899b58a62f4bcb51ecc1`
- stdout SHA-256: `4a75f06332252a6b70fd0070b5130460ad090fb0c5db6711e0c0ea2cced0e912`
- canonical result SHA-256: `809cafe3e8036dc7a2e814f27f39b69b4ef2e2658038f07584e0a7670758b9c4`

Fixture dimensions: `3x3x3x2`.

## Qualified invariants

### Polar projection

A positive scalar multiple of a known SU(3) matrix projects back to the original matrix within `8e-16` maximum element error.

### Identity fixed point

One spatial APE step leaves the identity gauge field unchanged:

`identity_fixed_point_error = 0.0`

### Gauge covariance

Smearing then gauge-transforming is numerically identical to gauge-transforming then smearing:

`gauge_covariance_max_error = 4.449557262054371e-16`

### SU(3) preservation

Across the full smeared fixture:

- max unitarity error: `5.551115123125783e-16`
- max determinant error: `3.333883600879643e-16`

### Temporal-link preservation

Spatial smearing does not modify temporal links:

`temporal_link_max_error = 0.0`

### Cross-language probe targets

The subject freezes three one-step smeared spatial link matrices at:

- `(0,0,0,0), mu=0`
- `(1,0,1,0), mu=1`
- `(2,1,0,1), mu=2`

The exact matrices are preserved in the result receipt and asserted by the subject. A production implementation should reproduce these values within a declared floating tolerance rather than relying only on generic unitary/determinant checks.

## Scientific / authority boundary

This oracle establishes a deterministic spatial-link construction and SU(3) projection convention only.

It does **not** establish:
- that APE smearing improves a particular ensemble or operator;
- an optimal `alpha` or iteration count;
- a static-potential plateau;
- a string tension;
- glueball-state overlap;
- equivalence between different smearing schemes;
- permission to replace the unsmeared Markov-chain configuration with a smeared field for action/update dynamics.

The smeared links are intended for **measurement/operator construction only**. The underlying ensemble/action authority remains the unsmeared gauge field unless a separate transition algorithm is explicitly qualified.

## Motivation / later comparison

The coarse Wilson-action static-potential literature, including Edwards, Heller and Klassen (`hep-lat/9711003`), uses spatial-link APE smearing before Wilson-loop/static-potential extraction to improve ground-state overlap while leaving the temporal transport structure distinct. LQCD-020A freezes the local construction semantics needed for Symthaea to reproduce that kind of operator path without conflating it with ensemble generation.
