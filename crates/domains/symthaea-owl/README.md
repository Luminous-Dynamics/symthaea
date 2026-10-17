# symthaea-owl

A small, pure-Rust starting point for owl-inspired embodied robotics research.

Current scope has two intentionally bounded layers:

- A signed two-microphone far-field TDOA-to-azimuth geometry contract, which explicitly documents the front/back and elevation limitations.
- A 3-D sampled spatial-likelihood map that scores candidate source positions against already-estimated pairwise TDOAs, given calibrated microphone locations and caller-supplied timing uncertainty. The map preserves competing hypotheses but its normalized weights are not calibrated probabilities.

The crate does not yet implement waveform TDOA extraction, room/reverberation modeling, tracking, calibrated uncertainty, active-head control, or flight control. Pairwise TDOAs may be correlated; the map's first version assumes independent Gaussian timing errors.

The crate is a workspace member through crates/domains/*.

Run scoped tests from the repository root:

    cargo test -p symthaea-owl

See the OWL-inspired robotics research plan at ../../../docs/robotics/OWL_INSPIRED_ROBOTICS_2026-10-17.md for the staged system architecture, benchmark protocol, LOCATA/DCASE validation path, safety/privacy requirements and evidence gates.
