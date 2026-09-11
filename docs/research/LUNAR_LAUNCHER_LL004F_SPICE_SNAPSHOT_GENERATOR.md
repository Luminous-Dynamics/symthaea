# LL-004F — reproducible DE440/SPICE snapshot generator

Status: **Phase-0 research tooling. Not navigation, guidance, rendezvous, or launch qualification.**

This tranche implements the offline evidence producer defined by #1602. Runtime orbital/launcher code does not fetch Horizons, download kernels, or load arbitrary SPICE data. The intended flow is:

```text
NAIF/JPL kernels on disk
        ↓
checked-in LL-004F config
        ↓
offline generator
        ↓
normalized JSON snapshot
        ├── kernel byte sizes + SHA-256
        ├── config + generator hashes
        ├── body-fixed -> inertial rotations
        ├── finite-difference angular velocity
        └── requested target state tracks
        ↓
review / independent parity
        ↓
checked-in evidence lineage
        ↓
production LL-004/LL-004E consumers
```

## Reference lineage

`configs/lunar_transport/ll004f_de440_reference.json` currently names:

- `naif0012.tls` — leap-seconds kernel;
- `de440s.bsp` — short-form DE440 planetary ephemeris;
- `moon_pa_de440_200625.bpc` — high-accuracy DE440 lunar principal-axis orientation;
- `moon_de440_250416.tf` — DE440 lunar frame definitions.

The high-fidelity body-fixed frame is explicitly `MOON_ME_DE440_ME421`, not the mutable generic `MOON_ME` alias. The inertial frame is `J2000`.

NAIF's lunar-frame documentation treats the Moon as a special high-accuracy case and defines DE-specific Principal-Axes and Mean-Earth frames. The 2025-04-16 frame kernel is the current generic lunar DE440 FK at the time this tranche was prepared. The current NAIF leap-second kernel remains `naif0012.tls` through at least 2027-07-01 per NAIF's 2026 announcement.

## Usage

The generator requires Python plus `spiceypy`/CSPICE only for real snapshot generation. Its self-test is standard-library-only:

```bash
python3 scripts/generate_ll004f_spice_snapshots.py --self-test
```

Place the exact kernels named by the config in a local directory. Before any SPICE calculation, verify and display the hashes actually present:

```bash
python3 scripts/generate_ll004f_spice_snapshots.py \
  --config configs/lunar_transport/ll004f_de440_reference.json \
  --kernel-dir /path/to/naif-kernels \
  --verify-kernels
```

Generate the normalized evidence fixture:

```bash
python3 scripts/generate_ll004f_spice_snapshots.py \
  --config configs/lunar_transport/ll004f_de440_reference.json \
  --kernel-dir /path/to/naif-kernels \
  --output-dir docs/research/evidence/ll004f
```

The tool refuses to overwrite a differing existing fixture by default. A changed kernel, frame, config, generator, or study definition should normally receive a **new lineage ID and output filename**, preserving earlier evidence instead of mutating it.

## Orientation convention

The stored 3×3 matrix maps lunar body-fixed vectors into the declared inertial frame:

```text
r_inertial = R_body_fixed_to_inertial · r_body_fixed
```

Angular velocity is independently derived by centered finite difference rather than relying on an opaque sign convention from a convenience API. For `R(t)`:

```text
[ω]_x = Rdot · Rᵀ
```

and the resulting vector is documented as the angular velocity of the body-fixed frame relative to inertial, expressed in inertial coordinates. The generator checks rotation orthonormality and determinant before accepting each sample.

## Target states

The reference config requests geometric (`NONE`) Earth and Sun states relative to the Moon in `J2000`. This is deliberately modest. It establishes reproducible moving-body state fixtures and frame/time lineage before we introduce a real catcher, L1/L2, or Gateway/NRHO target track.

Real catcher tracks remain a separate evidence step. A valid ephemeris/orientation snapshot is not evidence that a pod can rendezvous with or be captured by that target.

## Reproducibility requirements

A promoted fixture must preserve:

1. exact kernel filenames and source references;
2. hashes computed from the actual kernel bytes used;
3. exact config hash;
4. exact generator hash;
5. explicit frames, epochs, and timescale conversion;
6. rotation orthonormality/determinant diagnostics;
7. finite-difference angular-velocity convention;
8. independent state/orientation parity where an equivalent Horizons/SPICE reference is available.

Runtime code consumes checked-in, reviewed evidence. It does not silently follow upstream kernel aliases or "latest" URLs.

## Current non-claims

This tranche does **not** establish:

- an operational South-Pole launcher site;
- LOLA elevation/site binding;
- a real catcher or Gateway trajectory;
- navigation or pointing uncertainty;
- launch windows;
- acceptable miss distance or relative speed;
- terminal guidance;
- capture authority;
- physical release authority.

The next evidence step is to execute this generator in a controlled SPICE environment, check the resulting rotations/states against an independent source where possible, and then feed the resulting immutable snapshot into LL-004E → LL-004.
