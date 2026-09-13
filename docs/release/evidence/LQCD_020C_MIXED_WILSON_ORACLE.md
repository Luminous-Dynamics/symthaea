# LQCD-020C — mixed smeared-spatial / unsmeared-temporal Wilson-loop oracle

## Scope

Independent standard-library Python measurement oracle composed on the exact LQCD-020A spatial APE subject.

The subject verifies the SHA-256 of `lqcd-020a-spatial-ape-oracle.py` before importing it, so the mixed-link result cannot silently drift to a neighboring smearing convention.

Stable convention ID:

`mixed_spatial_ape_temporal_unsmeared_wilson_v1`

## Operator semantics

For a rectangular spatial-temporal Wilson loop:

- positive/negative spatial edges use the APE-smeared spatial field;
- positive/negative temporal edges use the original unsmeared ensemble field;
- backward edges use the dagger of the corresponding link at the previous site;
- the path must close exactly;
- `R` and `T` must both be positive and strictly smaller than the corresponding periodic extent, so this API cannot silently produce a winding observable.

The operator field is therefore not an alternate ensemble. Temporal transport remains explicitly tied to the original configuration.

## Exact executed subject

- subject SHA-256: `23a750eb915780f479dd406de8b020fbd7ab4ded242046633c10bfd7710072b3`
- subject Git blob: `b520aee93cd36e442e2b42a51c9e2436165ffc9a`
- result stdout SHA-256: `c70067c83e28db8e0bad60c9731a1416a96e700a4fbff0b64c2e2c5e9117afce`
- canonical result SHA-256: `164c89ca9d3d2e6a4e28bfc6ab15c31c0f84ce32eb7860e8a7ed0fe6d29a777e`
- exact APE dependency SHA-256: `074614e6c2bdba1949c9c6a3bf40458b00f6502c7dd3029ee201ae8eacb1d10c`

Fixture: LQCD-020A `3x3x3x2`, `alpha=0.7`, one APE iteration.

## Frozen values

Identity-field mixed loops:

- `R=1,T=1`: `1.0`
- `R=2,T=1`: `1.0`

Nontrivial fixture means:

- `mu0,R1,T1 = 0.9993772194252611`
- `mu0,R2,T1 = 0.9990068485378498`
- `mu1,R1,T1 = 0.999415081324206`
- `mu1,R2,T1 = 0.999233962373143`
- `mu2,R1,T1 = 0.9994521375611396`
- `mu2,R2,T1 = 0.9993316815423768`

Gauge-transforming both the original ensemble field and its derived smeared operator field leaves these loop means invariant to:

`1.1102230246251565e-16`.

## Temporal-source negative control

The subject deliberately changes one temporal link in a clone of the **smeared/operator** field only. The mixed Wilson result is unchanged exactly:

`temporal_source_is_original_error = 0.0`.

This proves the measurement convention does not accidentally consume temporal links from the derived operator field.

## Scientific boundary

This oracle establishes measurement semantics only. It does not establish:
- an optimal APE parameter or iteration count;
- improved overlap on a physical ensemble;
- a static-potential plateau;
- a preferred spatial orientation;
- a string tension;
- finite-volume or continuum control.

It exists to make later static-potential and glueball measurements auditable: the ensemble field remains authoritative, while spatial smearing is an explicitly derived operator construction.
