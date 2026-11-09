# symthaea-marine-structural-health

A small, dependency-free Rust crate containing evidence-gated primitives for marine structural inspection decisions.

## Scope

Implemented:
- exact region and structure-revision binding;
- method qualification, data-quality and calibration gates;
- required-coverage gate;
- conservative interval comparison for a numeric measurement against supplied acceptance bounds;
- explicit no-indication-within-declared-scope result;
- value-of-information arithmetic when the caller declares its decision models qualified;
- advisory recommendation mapping where mandatory inspection requirements dominate optional negative VoI.

Not implemented:
- fatigue crack-growth or corrosion-rate physics;
- structural loads, stress, buckling, stability or scantlings;
- acoustic-emission or EIS/EN signal interpretation;
- UT/PAUT/NDT image or waveform interpretation;
- material-specific acceptance criteria;
- inspection interval setting, RUL prediction, maintenance release or statutory approval.

The crate does not accept a component or vessel. It evaluates only the narrow evidence contract passed to it. A no-indication result applies only to the exact region/revision, method, criterion, coverage and evidence supplied.

## Run

From the workspace root, when Rust/Cargo are available:

```sh
cargo test -p symthaea-marine-structural-health
cargo clippy -p symthaea-marine-structural-health --all-targets -- -D warnings
cargo fmt --check
```

The crate has no external dependencies. The marine structural-health branch includes 14 unit tests. Compilation and execution have not yet been performed in the current environment because Rust/Cargo are not installed.

## Evidence boundary

Acceptance bounds, method qualification, required coverage, calibration status and model qualification are supplied by the responsible domain owner. This crate must not invent structural limits or treat synthetic data as measured vessel evidence.
