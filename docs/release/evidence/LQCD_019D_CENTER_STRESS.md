# LQCD-019D — four-sector pure-SU(3) center-mobility stress campaign

## Authority scope

Independent standard-library Python follow-up to LQCD-019B. This branch stacks on the exact 019B subject and imports only that Python qualification subject; it imports no Symthaea/Rust implementation code.

The purpose is narrow: challenge `Z3` center-sector mobility from deliberately different symmetry-related initial conditions while retaining center-invariant gauge observables. This is not a physical benchmark, deconfinement result, topology result, scale setting or glueball calculation.

## Frozen subject

- lattice `2^4`, beta `5.7`;
- HB + one overrelaxation sweep per cycle;
- direct six-staple force with the 019B finite-probe parity pre-check;
- ensemble slot `32`, independent ChaCha8 production streams;
- four starts: exact zero-phase center cold field, exact `+2pi/3` center transform, exact `-2pi/3` center transform, and an independently disordered field;
- each center transform multiplies temporal links on one timeslice by a center element and asserts unchanged identity plaquette plus the expected rotated Polyakov loop before production;
- burn-in `30`, stride `2`, `12` retained samples per chain / `48` total;
- retained plaquette, full complex temporal Polyakov loop, `|P|`, center-aligned real Polyakov, categorical nearest-`Z3` sector and temporal-spatial `1x1` Wilson mean.

Exact executed subject SHA-256:

`1695e80fd4f01228449132d8628470da6badc39ad1f974ae8db6f6517252c510`

Canonical result SHA-256:

`33c66dfe20d41eb16fd263c3b23db3c4644371bd3472d6f4daeb4b38a6339ca3`

Retained-history CSV SHA-256:

`2445ff6020fbf029afa6f9ccb1f3a39355d47550c00e72ae5156b68e6e4643f4`

Frozen stdout SHA-256:

`97283b7a4dc32bc0b4b78fb4cecd5aa6d92008ac616ee4c3872c63f44c5431fc`

Pre-run direct-staple / finite-probe force disagreement:

`6.661338147750939e-16`

## Center-invariant diagnostics

Across all four chains:

| observable | max rank/folded R-hat |
|---|---:|
| plaquette | `0.9915626877` |
| `|P|` | `1.0384545238` |
| center-aligned `Re(P)` | `1.0358999181` |
| temporal-spatial `1x1` Wilson mean | `1.0497148374` |

Raw center-sensitive components remain much less compatible:

- raw `Re(P)`: `1.3300233804`;
- raw `Im(P)`: `1.4029458477`.

No universal R-hat threshold is asserted here.

## Categorical center histories

- `center0`: counts `{0:6, 1:4, 2:2}`, `4` retained-sample transitions, maximum dwell `3`;
- `center_plus`: counts `{0:10, 1:1, 2:1}`, `2` transitions, maximum dwell `10`;
- `center_minus`: counts `{0:3, 1:5, 2:4}`, `2` transitions, maximum dwell `5`;
- `disordered`: counts `{0:0, 1:12, 2:0}`, **`0` transitions**, maximum dwell `12`.

Thus identical sampler duration does not imply comparable categorical mobility even when center-invariant scalar observables look broadly compatible. This is the empirical motivation for LQCD-019C's separate center-sector mobility contract.

## Interpretation boundary

This short campaign does not establish a mixing time or a universal minimum transition count. Exact center-related starting fields can leave their initial sectors on this tiny lattice, while another independently initialized chain remains entirely within one retained sector. More chains and longer histories are required before making a quantitative center-mobility claim.

The correct lesson is methodological: **center-invariant convergence and center-sector mobility are independent evidence dimensions**. Neither should be substituted for the other.

## Files

- `scripts/lqcd-019d-center-stress.py`
- `docs/release/evidence/LQCD_019D_CENTER_STRESS_HISTORY.csv`
- `docs/release/evidence/LQCD_019D_CENTER_STRESS_RESULT.txt`
