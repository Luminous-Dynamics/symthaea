# LL-009J — terrain-horizon solar and communications visibility evidence

Status: **Phase-0 geometric evidence. Not an RF link budget, solar-power model, thermal model, site qualification, or launch authority.**

LL-009J converts a provenance-bound 3-D terrain horizon plus time-sampled Sun/Earth/relay directions into conservative illumination and geometric line-of-sight evidence for LL-009I.

## Why

At the lunar South Pole, Sun and Earth remain close to the local horizon. Local and distant relief can therefore turn small terrain changes into long solar-darkness or communications-outage intervals. A real-site trade study should consume those derived geometric conditions rather than hand-entered `sunny = true` or `earth_visible = true` assumptions.

V1 deliberately stops at geometry:

```text
3-D terrain + target direction history
              ↓
conservative local horizon
              ↓
Sun/Earth/relay visibility history
              ↓
time-sampling uncertainty intervals
              ↓
LL-009I illumination/communications metrics
```

Power conversion, battery/storage sizing, thermal response and RF link performance remain separate downstream models.

## Input schema

`ll009j.horizon-visibility-input.v1`

The input binds:

- `study_id`, `frame_contract_id`, `epoch_contract_id`, `site_ref`;
- one declared lunar body-fixed frame;
- Moon-centered site position and pole vector;
- hash-verified source artifacts;
- 3-D Moon-centered terrain samples in the same frame;
- per-terrain-sample angular uncertainty margin;
- time-ordered Sun/Earth/relay direction vectors in the same frame;
- Sun apparent angular radius for full-disc visibility;
- maximum accepted horizon azimuth gap;
- maximum accepted target time gap;
- additional LOS/elevation margin.

The first real adapter is expected to derive terrain samples from the native LOLA/PGDA evidence chain, with any DE421→DE440 reconciliation explicitly represented by the existing frame-bridge lineage.

## Local frame

The generator forms local up from the site radius vector. Local north is the declared pole vector projected into the tangent plane; east completes an orthonormal frame.

At an exact/near-degenerate pole where geographic north is undefined, the implementation chooses a deterministic projected global axis and records `fallback_used = true`. This preserves numerical determinism without pretending the fallback azimuth origin is geodetically privileged.

## Terrain horizon

For each 3-D terrain point:

```text
d = terrain_position - site_position
```

The direction `d` is converted into local azimuth/elevation. The conservative horizon sample is:

```text
terrain_elevation_angle + declared_angular_uncertainty
```

The circular azimuth profile is piecewise-linearly interpolated only after its largest gap has passed the declared maximum-gap policy.

Missing sectors are therefore not interpreted as a zero-degree horizon.

Moon curvature is implicit in the Moon-centered 3-D positions; V1 does not use a flat-surface range formula.

## Target visibility

At each target sample the target center elevation is compared with:

```text
interpolated_conservative_horizon + los_margin
```

Center visibility is:

```text
target_center_elevation > conservative_horizon
```

For the Sun, full-disc visibility is stricter:

```text
target_center_elevation - apparent_angular_radius > conservative_horizon
```

This explicitly distinguishes a center-visible Sun from one whose lower limb remains terrain-obscured.

Earth and relay visibility are geometric LOS only. No antenna gain, link margin, diffraction, multipath, interference or data-rate claim is inferred.

## Time-sampling uncertainty

V1 refuses to assign a precise transition time between samples.

For each adjacent sample interval:

- visible → visible: the whole interval is counted visible;
- blocked → blocked: none of the interval is counted visible;
- a state change: visible duration is bracketed from 0 to the whole interval, with the midpoint used only as the reported central value.

This produces low/central/high visibility fractions that reflect sampling resolution.

Contiguous darkness/outage duration is also reported as a bracket. A run's lower bound spans confirmed blocked samples; its upper bound includes adjacent transition intervals where the exact state-change time is unknown.

Epochs must be strictly increasing. A gap above `max_time_gap_s` fails the promoted time-series calculation rather than silently bridging missing data.

## LL-009I-compatible metrics

V1 emits available metrics when corresponding targets exist:

- `solar_center_visibility_fraction`;
- `solar_full_disc_visibility_fraction`;
- `longest_full_solar_occlusion_h`;
- `dte_los_availability_fraction`;
- `max_contiguous_dte_outage_h`;
- `<relay>_los_availability_fraction`;
- `max_contiguous_<relay>_outage_h`.

Each metric carries low/central/high, unit, evidence class and exact source references suitable for inclusion in an LL-009I site evidence pack.

## Source and lineage binding

Every declared source artifact has:

- stable source ID;
- safe relative path beneath the artifact root;
- exact SHA-256.

Terrain samples and target histories must reference those source IDs. Path traversal, missing files or hash drift fail closed.

The output also records the input SHA-256, study/frame/epoch/site lineage, local basis, horizon coverage/gap statistics and source hashes.

## Executed synthetic gates

`--self-test` covers:

- a flat synthetic horizon;
- a 15° raised ridge that blocks a 10° target;
- exact/near-pole local-basis orthonormality and deterministic fallback;
- Sun center at 1.0° with a 0.5° apparent radius behind a 0.75° horizon: center visible, full disc hidden;
- sampled visibility transitions producing interval-valued availability rather than point certainty;
- missing horizon-sector rejection;
- excessive temporal-gap rejection.

Run with:

```bash
python3 scripts/generate_ll009j_horizon_visibility.py --self-test
```

A materialized study run is:

```bash
python3 scripts/generate_ll009j_horizon_visibility.py \
  --input docs/research/evidence/<study>/visibility/horizon-input.json \
  --artifact-root docs/research/evidence/<study>/visibility \
  --output docs/research/evidence/<study>/visibility/horizon-visibility-receipt.json
```

Differing existing output is immutable and is not overwritten.

## Non-claims

A passing LL-009J receipt does not establish:

- delivered solar-array power;
- battery or survival-power closure;
- thermal qualification;
- an RF link budget or communications service level;
- navigation accuracy;
- site suitability;
- corridor safety;
- architecture superiority;
- construction or launch authorization.

Those claims require separate evidence/model layers.

Tracks #1603, #1615, #1716/#1717, #1719 and master #1542.
