# symthaea-living-watershed-witness

A deterministic **mechanism witness** for the Living Watershed / Wetland Watch research program.

This crate is intentionally not presented as a real-world wetland predictor. Its purpose is to prove that the existing Symthaea science infrastructure can compose into one auditable loop before real Sentinel fixtures and learned/HDC models are introduced.

## What v0 composes

The witness reuses existing subsystems instead of inventing parallel ones:

- `symthaea-earth-system::HydrologyBucket` generates a reduced-order, mass-conserving water trajectory;
- `symthaea-ecology::SoilMoistureResponse` maps bounded soil moisture into a bounded ecological moisture-response signal;
- `symthaea-futures-core` owns the neutral forecast/abstention representation;
- `symthaea-futures-calibration` owns proper scoring;
- `symthaea-research-protocol` owns frozen preregistration and run lineage;
- `symthaea-research-result` owns immutable result manifests and primary-metric retention;
- `symthaea-research-replication` owns reproduction/replication distinctions.

## Observation firewall

The central v0 invariant is:

> a predictor receives history, never the held-out next outcome.

`SealedWatershedFixture` contains:

- a content-addressed deterministic fixture specification;
- predictor-visible `WatershedHistory`;
- one private held-out `WetlandObservation`;
- the dataset-manifest digest.

The two baseline generators implement:

```text
TrajectoryGenerator<Observation = WatershedHistory>
```

The runner first obtains both forecast outputs. Only afterward does the sealed fixture supply the actual next-state outcome to the canonical Brier scorer.

This is a code-level observation firewall, not a cryptographic or organizational blinding claim. Stronger future campaigns should separate fixture custody, execution, and verification when independent blinding matters.

## Neutral baselines

v0 contains only two deliberately simple baselines:

1. **Persistence** — assigns 0.8 probability to the last observed stress state continuing one more step.
2. **Empirical climatology** — assigns probability equal to the fraction of predictor-visible history in stress; it abstains with `InsufficientObservationHistory` below its declared minimum history.

They are not claimed to be good wetland models. They are contest anchors that exercise validated probability construction, typed abstention, canonical scoring, and result lineage.

## Canonical scoring

The witness does not implement a Brier formula itself.

It uses `symthaea-futures-calibration::BrierScore`, whose Boolean/discrete convention is the Futures Laboratory multiclass Brier score. The result metric unit is therefore explicitly named:

```text
brier_multiclass
```

so it is not silently confused with the half-sized single-probability binary Brier convention often used in weather forecasting.

## Primary metrics cannot disappear

The preregistered primary metrics are:

- `persistence-brier`;
- `climatology-brier`.

If a baseline emits a distribution, the canonical finite score is stored.

If it abstains, the corresponding primary metric becomes:

```text
NotComputed { reason: "forecaster abstained: ..." }
```

rather than `0`, `NaN`, omission, or another numeric sentinel.

## Replication boundary

The witness can build a `ReplicationAssessment`, but it does **not** infer replication success from similar scores.

The caller must explicitly provide:

```text
Concordant
Discordant
Mixed
Inconclusive
NotComparable
```

and the existing research-replication layer separately records whether protocol, source, dataset, environment capsule, and seeds were the same or different.

A same-fixture rerun can be exact reproduction evidence. It cannot be mislabeled direct replication.

A direct replication requires a different dataset-manifest lineage under the same frozen protocol.

## What v0 does not establish

This crate does not establish that:

- the synthetic bucket is an adequate model of a real wetland;
- the moisture-response threshold is ecologically calibrated;
- Sentinel-1 or Sentinel-2 can infer this synthetic state accurately;
- either baseline has useful predictive skill;
- HDC improves wetland forecasting;
- a forecast recommends or authorizes intervention;
- score similarity proves replication;
- code-level held-out separation proves independent organizational blinding.

## Upgrade path

### v1 — frozen real Sentinel fixtures

After Planetary Perception PRs #136–139 qualify, add immutable Sentinel-1/2 fixture adapters behind the same observation boundary. Do not replace the synthetic witness; keep it as the deterministic mechanism control.

### v2 — deterministic EO feature witness

Add explicit calibrated/masked features from the Earth-observation stack, initially NDVI / named NDWI variants / NBR and reviewed SAR features. Preserve provider and processing lineage.

### v3 — preregistered Wetland Watch forecast contest

Freeze:

- real held-out regions and dates;
- target definition;
- temporal split;
- baselines;
- primary proper-scoring metric;
- calibration/coverage metrics;
- exclusions;
- stopping rule.

Compare simple baselines before any Symthaea/HDC lane receives credit.

### v4 — semantic observation/downlink experiment

Test whether semantic prioritization improves mission-relevant information delivered per byte/joule over conventional codec + simple ROI/change baselines.

### v5 — direct replication across new regions/times

Use new Sentinel lineage under the same frozen protocol. Do not call same-scenes reanalysis a direct replication.

### v6 — conceptual replication across regimes

Deliberately change biome, hydrological regime, sensor mix, or model family and retain those differences explicitly.

## Required gates

After the parent research-integrity stack is green:

```bash
cargo fmt --all -- --check
cargo check -p symthaea-living-watershed-witness --all-targets
cargo test -p symthaea-living-watershed-witness
cargo clippy -p symthaea-living-watershed-witness --all-targets -- -D warnings
```

The v0 witness remains a research/engineering mechanism test until those gates execute on the exact branch head.
