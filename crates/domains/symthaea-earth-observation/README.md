# symthaea-earth-observation

Provider-neutral evidence contracts for Symthaea planetary perception.

This crate is intentionally **not** a Sentinel client, raster engine, image codec, geological inversion package, or Mycelix bridge. It defines the facts and epistemic boundaries those systems exchange.

## Why this crate exists

Remote sensing becomes dangerous when four different things are collapsed into one value:

1. what an instrument measured;
2. what processing derived from that measurement;
3. what a model inferred about hidden reality;
4. what later evidence verified or refuted.

Planetary Perception keeps those stages explicit.

```text
provider / instrument
        |
        v
ObservationEvidence
        |
        +--> Measurement
        |
        +--> DerivedFeature
                  |
                  v
              Hypothesis
                  |
                  v
               Inference
                  |
                  v
              Prediction
                  |
                  v
             Verification
```

Contradictory evidence is represented explicitly with `EvidenceConflict`; it is not silently averaged away.

## Load-bearing subsurface rule

`ObservationSensitivity` distinguishes:

- `SurfaceOnly`
- `IndirectSubsurface`
- `DirectPenetrating { max_validated_depth_m }`

A `Hypothesis` labelled `ClaimMode::DirectObservation` for a subsurface domain cannot validate unless at least one referenced observation is explicitly `DirectPenetrating`. If a depth is claimed, that depth must be no greater than the observation's validated penetration bound.

This prevents language such as "the satellite saw a cavity at 5 m" when the real evidence is only surface deformation, thermal response, vegetation stress, gravity anomaly, or another indirect signature.

`DirectPenetrating` is acquisition-specific. The crate deliberately does **not** encode folklore such as a universal penetration depth for C-, L-, or P-band radar. Medium, moisture, wavelength, incidence geometry, processing, calibration, and validation all matter.

## Provider boundary

Provider-specific I/O should live in bridge crates, for example:

```text
crates/bridges/symthaea-sentinel/
```

A provider bridge may discover/fetch Sentinel-1/2 products, decode metadata, calibrate measurements, and then construct `ObservationEvidence`.

The domain crate must not know Copernicus URLs, credentials, HTTP semantics, or STAC endpoint versions.

## First target integrations

1. Sentinel-2 L2A optical/multispectral observations.
2. Sentinel-1 calibrated SAR observations and repeat-pass deformation evidence.
3. NISAR/other L-band observations through the same provider-neutral contract.
4. P-band, gravity, magnetics, EM, GPR, seismic, resistivity, lidar, and in-situ observations as independent modalities.
5. Morphos `BioregionSteward-v0` feature adapter.
6. `symthaea-visual-compression-probe` semantic/ROI sidecar.
7. Mycelix Climate evidence claims that reference content-addressed payloads rather than embedding large imagery.
8. DTN progressive transmission profiles.

## Evidence hierarchy

The intended semantics are:

```text
Measurement
    raw/calibrated quantity

DerivedFeature
    deterministic or model-derived feature with explicit support

Hypothesis
    possible explanation of observations

Inference
    evaluated conclusion with alternatives and uncertainty

Prediction
    future/testable consequence

Verification
    later evidence confirms, refutes, or leaves the claim inconclusive
```

Never upgrade one stage merely because the result is plausible.

## Spectral-index naming

Ambiguous aliases are avoided where they hide different equations. For example the contract distinguishes `McFeetersNdwi` from `GaoNdwi` rather than exposing a single unspecified `NDWI` value.

## Planned Planetary Perception tranches

### EO-1 — contracts (this tranche)

- provider-neutral acquisition identity
- footprints and modality metadata
- uncertainty and processing lineage
- explicit evidence stages
- subsurface direct-vs-indirect claim discipline
- first-class contradiction

### EO-2 — deterministic feature math

- NDVI
- explicitly formulated NDWI variants
- NBR
- calibrated SAR transforms
- VV/VH features
- quality/mask propagation
- uncertainty propagation

### EO-3 — Sentinel bridge

- offline fixture provider
- Copernicus/STAC discovery adapter
- Sentinel-1/2 product metadata mapping
- reproducible AOI/window extraction
- no network dependency in CI

### EO-4 — Bioregion Steward

- map derived EO features into Morphos channels
- preserve cloud/SAR/coverage uncertainty
- retain local-verification requirements

### EO-5 — Planetary Perception / subsurface

- gravity and magnetic observations
- EM/resistivity/seismic/GPR evidence
- surface-subsurface hypothesis graph
- multimodal contradiction detection
- next-best-observation planning
- synthetic hidden-ground-truth benchmarks before real-world claims

### EO-6 — semantic downlink

- multiband visual-memory adapter
- progressive synopsis / ROI / full-product classes
- conventional codec baselines
- mission-relevant information per byte/joule evaluation

## Claim boundary

This crate establishes data contracts and validation invariants only.

It does **not** establish that:

- Symthaea can reconstruct arbitrary underground structures;
- a radar wavelength has a universal penetration depth;
- HDC improves geophysical inversion;
- satellite-only observations are sufficient for subsurface confirmation;
- a derived feature is causal;
- a high-confidence model output is equivalent to direct observation.

Those are empirical questions for controlled experiments and independent validation.

## Verification

From the complete pinned Symthaea workspace:

```bash
cargo fmt --all -- --check
cargo check -p symthaea-earth-observation --all-targets
cargo test -p symthaea-earth-observation
cargo clippy -p symthaea-earth-observation --all-targets -- -D warnings
```
