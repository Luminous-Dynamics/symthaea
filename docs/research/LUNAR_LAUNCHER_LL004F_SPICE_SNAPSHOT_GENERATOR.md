# LL-004F — reproducible DE440/DE421 SPICE snapshot generator

Status: **Phase-0 research tooling. Not navigation, guidance, rendezvous, capture, or launch qualification.**

This tranche implements the offline evidence producer defined by #1602. Runtime orbital/launcher code does not fetch Horizons, download kernels, or load arbitrary SPICE data. The intended flow is:

```text
NAIF/JPL kernels on disk
        ↓
upstream checksum locks
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

## Two-stage kernel trust

Kernel identity is intentionally checked in two different ways.

1. `scripts/verify_ll004f_kernel_locks.py` compares required source artifacts against checked-in authority-published checksum locks before generation. The initial lock set covers the planetary ephemerides:
   - `de440s.bsp`: NASA/JPL NAIF published MD5 `3917ee56769db332790c751e2168843d`;
   - `de421.bsp`: NASA/JPL NAIF archived published MD5 `25814bb3622904d0c444c52a4051eaf7`.
2. `scripts/generate_ll004f_spice_snapshots.py` independently records SHA-256 and byte size over every local kernel actually consumed.

These answer different questions. The upstream lock asks whether a known artifact matches the authority-published object. The generated snapshot preserves exactly which bytes entered the evidence lineage.

Absence of a published lock for a non-required kernel does not imply trust. Those files remain identified by their actual-byte SHA-256 in the generated manifest, and additional authority locks can be added when stable upstream checksum sources are available.

## DE440 reference lineage

`configs/lunar_transport/ll004f_de440_reference.json` names:

- `naif0012.tls` — leap-seconds kernel;
- `de440s.bsp` — short-form DE440 planetary ephemeris;
- `moon_pa_de440_200625.bpc` — high-accuracy DE440 lunar principal-axis orientation;
- `moon_de440_250416.tf` — DE440 lunar frame definitions.

The high-fidelity body-fixed frame is explicitly `MOON_ME_DE440_ME421`, not the mutable generic `MOON_ME` alias. The inertial frame is `J2000`.

## DE421 / PGDA companion lineage

`configs/lunar_transport/ll004f_de421_pgda_bridge.json` names:

- `naif0012.tls`;
- `de421.bsp`;
- `moon_pa_de421_1900-2050.bpc`;
- `moon_080317.tf`;
- explicit frame `MOON_ME_DE421` -> `J2000`.

Its epochs intentionally match the DE440 reference configuration. The paired snapshots are intended to feed LL-009A2's independent frame-reconciliation oracle so Phase 0 measures the actual DE421↔DE440 South-Pole displacement instead of assuming the two Mean-Earth realizations are identical.

## Usage

Both scripts have standard-library-only self-tests:

```bash
python3 scripts/verify_ll004f_kernel_locks.py --self-test
python3 scripts/generate_ll004f_spice_snapshots.py --self-test
```

Real SPICE generation additionally requires `spiceypy`/CSPICE. With the exact kernels in a local directory, first verify locked upstream identity:

```bash
python3 scripts/verify_ll004f_kernel_locks.py \
  --config configs/lunar_transport/ll004f_de440_reference.json \
  --kernel-dir /path/to/naif-kernels
```

Then display the actual kernel-byte manifest or generate the snapshot:

```bash
python3 scripts/generate_ll004f_spice_snapshots.py \
  --config configs/lunar_transport/ll004f_de440_reference.json \
  --kernel-dir /path/to/naif-kernels \
  --verify-kernels

python3 scripts/generate_ll004f_spice_snapshots.py \
  --config configs/lunar_transport/ll004f_de440_reference.json \
  --kernel-dir /path/to/naif-kernels \
  --output-dir docs/research/evidence/ll004f
```

Repeat with the DE421 companion configuration. The generator refuses to overwrite a differing existing fixture by default. A changed kernel, frame, config, generator, or study definition should normally receive a **new lineage ID and output filename** rather than mutate earlier evidence.

## Orientation convention

The stored 3×3 matrix maps lunar body-fixed vectors into the declared inertial frame:

```text
r_inertial = R_body_fixed_to_inertial · r_body_fixed
```

Angular velocity is independently derived by centered finite difference rather than relying on an opaque sign convention from a convenience API. For `R(t)`:

```text
[ω]_x = Rdot · Rᵀ
```

The reported vector is the angular velocity of the body-fixed frame relative to inertial, expressed in inertial coordinates. The generator checks rotation orthonormality and determinant before accepting each sample.

## Target states

The reference configurations request geometric (`NONE`) Earth and Sun states relative to the Moon in `J2000`. This establishes reproducible moving-body state fixtures and frame/time lineage before introducing a real catcher, L1/L2, or Gateway/NRHO target track.

A valid ephemeris/orientation snapshot is not evidence that a pod can rendezvous with or be captured by a target.

## Promotion requirements

A promoted fixture must preserve:

1. authority-published checksum locks for required source roles;
2. actual-byte SHA-256 for every kernel consumed;
3. exact config hash;
4. exact generator hash;
5. explicit frames, epochs, and timescale conversion;
6. rotation orthonormality/determinant diagnostics;
7. finite-difference angular-velocity convention;
8. matching epochs across paired DE421/DE440 frame studies;
9. independent state/orientation parity where an equivalent Horizons/SPICE reference is available;
10. immutable evidence lineage when source bytes/configuration/tooling change.

Runtime code consumes checked-in, reviewed evidence. It does not silently follow upstream kernel aliases or `latest` URLs.

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

The next evidence step is to execute both paired configurations in a controlled SPICE environment, verify their upstream locks, check the resulting states/orientations independently where possible, and feed the immutable orientation samples into LL-009A2 to quantify the DE421↔DE440 surface displacement.
