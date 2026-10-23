# LQCD-020L — shortest-path symmetrized off-axis Wilson-loop oracle

## Scope

Independent standard-library Python qualification subject for off-axis static-potential operator geometry.

Stable convention ID:

`shortest_path_symmetrized_ape_spatial_unsmeared_temporal_wilson_v1`

Spatial transport is the arithmetic average of every unique shortest Manhattan path transporter for a declared integer displacement vector. Each path uses the APE-smeared spatial operator field. Temporal transport uses the original unsmeared ensemble field only.

The arithmetic average is not asserted to be an SU(3) link; it is a gauge-covariant endpoint transporter used only inside a gauge-invariant Wilson loop.

## Exact executed subject

- subject SHA-256: `9715b6ebd193da51358f6e4331fbbcdbd8be5b4746ceb0f00c21952ce3c4e4ad`
- stdout SHA-256: `472f3ac69c506652b3f9aff71d4bcbf120c09f4d3fe20b1871fb737d7ccfeed5`
- canonical result SHA-256: `20458685d99bcd4d67d373dae12cded11665aa2784a1a8dd9965787628030906`
- fixture dimensions: `3x3x3x2`
- APE qualification weight: `alpha=0.7`

## Frozen path multiplicities

- `(1,0,0)`: `1`
- `(2,0,0)`: `1`
- `(1,1,0)`: `2`
- `(1,1,1)`: `6`
- `(2,1,0)`: `3`

## Frozen loop means

At `T=1` on the deterministic fixture:

- `(1,0,0)`: `0.9993772194252611`
- `(2,0,0)`: `0.9990068485378498`
- `(1,1,0)`: `0.9988687110881265`
- `(1,1,1)`: `0.9986193091927639`
- `(2,1,0)`: `0.9984136341025711`

Identity-field values are exactly `1.0` for all five vectors.

## Qualification invariants

- local gauge invariance max disagreement: `1.1102230246251565e-16`;
- temporal-link tamper in the derived operator field: exactly `0.0` effect;
- axis-aligned unique-path collapse error: exactly `0.0`;
- all unique shortest paths for one displacement terminate at the same endpoint.

The one-path axis-aligned cases reproduce the same measurement semantics as the earlier mixed smeared-spatial / unsmeared-temporal operator.

## Scientific boundary

This oracle qualifies one explicit off-axis operator construction only. It does not establish that shortest-path symmetrization is optimal, equivalent to a generalized Bresenham path, or superior to Laplacian/static-force operator constructions. It does not establish a potential plateau, string tension, rotational restoration, finite-volume adequacy, or continuum physics.

Its purpose is narrower: production static-potential fits may now use off-axis separation vectors only after their spatial transporter semantics are made explicit and independently testable, rather than silently feeding axis-only measurement code into an off-axis Cornell fit.
