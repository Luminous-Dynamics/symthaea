# symthaea-owl

A pure-Rust starting point for owl-inspired embodied robotics research.

## Current modules

- **Geometry:** a bounded two-microphone far-field TDOA-to-azimuth contract, with explicit front/back and elevation limitations.
- **Spatial likelihood:** a 3-D discrete candidate map scored from already-estimated pairwise TDOAs and timing uncertainty. Weights are relative likelihoods, not calibrated probabilities.
- **Hybrid evidence:** combine candidate-aligned modality log-likelihood fields with explicit tempering weights and missing-sensor states. Assumes conditional independence; not a calibrated fusion model.
- **Active sensing:** choose a candidate action using expected information gain, hard energy/latency budgets, a caller-supplied safety eligibility flag and a minimum useful-gain threshold; can return no action.
- **Robust active sensing:** rank candidate actions by worst-case expected information gain over the same finite scenario set, to reduce dependence on one nominal model.

The crate does not yet implement waveform TDOA extraction, room/reverberation modeling, source tracking, calibrated uncertainty, sensor-fault detection from raw signals, a physical safety-envelope checker, actuator control, or flight control. Model likelihoods must be validated on held-out evidence. Robustness is only relative to declared scenarios; none of the supplied normalized weights are calibrated probabilities.

The crate is a workspace member through crates/domains/*.

Run scoped tests from the repository root:

    cargo test -p symthaea-owl

See the OWL-inspired robotics research plan at ../../../docs/robotics/OWL_INSPIRED_ROBOTICS_2026-10-17.md for staged architecture, benchmarks, LOCATA/DCASE validation, hybrid regimes, safety/privacy constraints and evidence gates.
