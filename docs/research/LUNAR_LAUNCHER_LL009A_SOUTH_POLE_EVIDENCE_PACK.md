# LL-009A — South-Pole terrain/site evidence pack

Status: **Phase-0 evidence tooling. Not site selection, land-use approval, route approval, or construction qualification.**

This tranche makes the data contract for the South-Pole grand transport trade explicit before any architecture is allowed to score terrain, route length, or ballistic clearance.

## Why this exists

NASA Goddard's high-resolution LOLA South-Pole products provide 5 m/pixel elevation, slope, elevation-uncertainty, slope-uncertainty, point-count, clone, and region products for multiple candidate landing-study regions. The products are published in south-polar stereographic coordinates in the `MOON_ME` reference frame of **DE421**.

Our LL-004E high-fidelity launcher release work uses the explicit DE440 frame `MOON_ME_DE440_ME421`. Those frames are intentionally **not treated as identical**. The source manifest therefore requires an explicit frame bridge before terrain coordinates enter the launcher/corridor study.

## Tooling

`validate_ll009a_south_pole_evidence.py` is dependency-free and has two jobs:

1. validate source metadata, frame closure, uncertainty products, no-data policy, and candidate-node status;
2. materialize a local artifact receipt with SHA-256 and byte counts once the declared GeoTIFFs/archives are actually present.

Validate the checked-in source manifest:

```bash
python3 scripts/validate_ll009a_south_pole_evidence.py \
  --manifest configs/lunar_transport/ll009a_south_pole_pgda_v1.json
```

Run its dependency-free synthetic guard:

```bash
python3 scripts/validate_ll009a_south_pole_evidence.py --self-test
```

After acquiring the exact declared source files, produce an immutable local receipt:

```bash
python3 scripts/validate_ll009a_south_pole_evidence.py \
  --manifest configs/lunar_transport/ll009a_south_pole_pgda_v1.json \
  --artifact-root /path/to/pgda/files \
  --output docs/research/evidence/ll009a/pgda_v1_receipt.json
```

Missing required files fail closed. The validator never converts missing terrain to zero elevation and never invents a candidate site coordinate.

## Initial source manifest

The v1 manifest starts deliberately small with two well-described PGDA regions:

- Site01 — Connecting ridge;
- Site04 — Shackleton rim.

For each, the declared required products are:

- surface elevation;
- slope;
- total elevation RMS uncertainty;
- slope RMS uncertainty.

ROI geometry archives are optional in the first materialization pass. More South-Pole regions can be added without changing the evidence schema.

These names and products come from NASA Goddard PGDA Product 78, which also provides 100 clone realizations for uncertainty studies. Product 81 provides a broader 87–90°S 5 m/pixel mosaic for later regional route studies, but that multi-gigabyte mosaic is intentionally not a hidden dependency of this first evidence pack.

## Candidate-site semantics

A node in this pack has one of only two statuses:

- `study_hypothesis`;
- `authoritative_selected_site`.

The second status requires an explicit authoritative selection reference. Merely appearing in a NASA terrain product or a trade study does **not** make a location a NASA-selected future mine, settlement, launcher, receiver, or elevator anchor.

The v1 manifest therefore names **study regions**, not precise infrastructure coordinates.

## Frame closure

The data path is intentionally explicit:

```text
PGDA raster
MOON_ME / DE421
      ↓
provenance-bound frame bridge
      ↓
MOON_ME_DE440_ME421
      ↓
LL-004E site/release state
      ↓
LL-003 / LL-005 corridor physics
```

NAIF documents that the DE440 Mean-Earth realization is closely aligned with DE421, but not exactly identical. That difference can be of order sub-meter to meter depending on epoch. At 5 m raster resolution it may or may not matter for a particular trade, but Phase 0 must **measure that sensitivity**, not erase the distinction by naming both frames `MOON_ME`.

## No-data and uncertainty rules

The pack uses `no_data_policy = fail_closed`.

A corridor cannot gain favorable terrain clearance because:

- a raster tile is absent;
- an uncertainty raster is missing;
- a projection/frame is unknown;
- a source region does not cover part of the route.

The conservative corridor layer should use the elevation and its uncertainty jointly. Increasing declared uncertainty must never improve conservative terrain clearance.

## Next integration

After the first real artifacts are hashed and frame reconciliation is validated, the next derived products should be reproducible corridor inputs for:

- 50 m–5 km heavy-haul routes;
- 5–100 km fixed rail/FLOAT/cable/ballistic corridors;
- 100–1,000+ km regional surface transport;
- ballistic launch/receiver node pairs;
- candidate feeder corridors toward an elevator-anchor hypothesis.

Every architecture should consume the same terrain/source lineage where applicable, preventing the launcher from receiving a flat synthetic Moon while rail is charged for real crater topography—or vice versa.

## Non-claims

This tranche does not choose a launcher site, receiver site, mine, settlement, elevator anchor, protected-zone policy, corridor, or winning transport architecture. It creates the evidence discipline needed for LL-009 to compare those hypotheses fairly.
