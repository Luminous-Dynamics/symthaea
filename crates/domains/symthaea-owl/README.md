# symthaea-owl

A small, pure-Rust starting point for owl-inspired embodied robotics research.

Current scope is intentionally narrow: a signed two-microphone TDOA-to-azimuth geometry contract, with finite-input checks, physical-bound rejection, known-answer tests and explicit ambiguity limits. It does not implement audio DSP, a calibrated 3-D auditory map, flight control, or biological owl hearing.

The crate is a workspace member through crates/domains/*.

Run the scoped tests from the repository root:

    cargo test -p symthaea-owl

See the OWL-inspired robotics research plan at ../../../docs/robotics/OWL_INSPIRED_ROBOTICS_2026-10-17.md for the staged system architecture, benchmark protocol, safety/privacy requirements and evidence gates.
