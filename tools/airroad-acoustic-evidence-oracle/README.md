# AIRROAD Acoustic Evidence Oracle

A dependency-free Rust evidence-contract validator for synchronized electric-aircraft rotor aeroacoustic measurement bundles.

This crate validates metadata and evidence lineage. It does not read or transform waveform samples, calculate SPL, compute PSD, perform beamforming, predict aeroacoustics or establish regulatory compliance.

## Bundle contents

A run binds:

- aircraft, rotor and geometry configuration identities;
- exact analysis time window and reference clock;
- calibrated raw channels with immutable SHA-256 digests;
- microphone and array coordinates, boresight and position uncertainty;
- rotor phase, RPM, thrust and torque channel references for each exact rotor;
- per-rotor thrust, torque, RPM, pitch and tilt observations;
- environmental conditions with a separate evidence digest and declared test mode.

The validator checks that microphones resolve to calibrated raw channels in the declared coordinate frame, each array has at least two positioned microphones, rotor declarations resolve to phase and speed channels for the same rotor identity, and all channels cover the analysis window within declared clock/skew budgets.

Different channels may have different sample rates. This version does not resample or interpolate them. The caller must preserve a common clock/time reference and provide already synchronized acquisition metadata.

## Verification

```bash
cargo fmt --manifest-path tools/airroad-acoustic-evidence-oracle/Cargo.toml -- --check
cargo test --locked --manifest-path tools/airroad-acoustic-evidence-oracle/Cargo.toml
cargo clippy --locked --manifest-path tools/airroad-acoustic-evidence-oracle/Cargo.toml --all-targets -- -D warnings
```

The tests use synthetic metadata only. A valid bundle proves internal consistency of declared identities and time/geometry metadata; it does not prove that sensors were physically calibrated, that the raw samples are authentic, or that an acoustic model matches a physical aircraft.

## Research references

- Melo et al., JASA 160(1), 725–736 (2026), DOI 10.1121/10.0044421: https://doi.org/10.1121/10.0044421
- Colli et al., 2026 rotor-rotor / rotor-wing interaction campaign: https://research-information.bris.ac.uk/en/publications/aerodynamics-and-aeroacoustics-of-rotor-rotor-and-rotor-wing-inte/
- NASA full-scale eVTOL aerodynamic/acoustic validation project 154389: https://techport.nasa.gov/projects/154389

These references motivate the measurement contract; they are not claims about Luminous physical measurements.
