# open-display-platform-verify

Standalone Rust 2024 / Rust 1.96 reference verifier for the open display reference platform and run-record contract.

From the repository root:

    cargo fmt --manifest-path tools/open-display-platform-verify/Cargo.toml -- --check
    cargo clippy --manifest-path tools/open-display-platform-verify/Cargo.toml --all-targets -- -D warnings
    cargo test --manifest-path tools/open-display-platform-verify/Cargo.toml
    cargo run --manifest-path tools/open-display-platform-verify/Cargo.toml

To validate the checked-in synthetic control-drift plan and print its screening report, run the CLI with `--drift-plan docs/engineering/display-drift-control-plan-v0.1.json --drift-schema docs/engineering/display-drift-control-plan.schema.json`. The plan must remain `SYNTHETIC_TEST_ONLY`; no physical result is asserted. The fixed-guard-band algorithm marks missing, invalid or out-of-band controls and conservatively quarantines observations between acceptable anchors. It requires a preregistered maximum scheduled control gap including run edges, rejects schedules that exceed it, and reports the actual schedule maximum. This bounds the plan's spacing; it does not establish that a selected cadence can detect real drift. It does not fit or correct a drift curve.

To validate the checked-in synthetic power trace and print energy per successful task, run:
    
    cargo run --manifest-path tools/open-display-platform-verify/Cargo.toml -- \
      --energy-trace docs/engineering/display-energy-trace-synthetic-v0.1.json \
      --energy-schema docs/engineering/display-energy-trace.schema.json

The sample trace is synthetic and must remain SYNTHETIC_TEST_ONLY. Its known-answer output is 35 J total and 17.5 J per successful task (3 attempts, 2 successes); it does not establish instrument accuracy or physical monitor power.

To screen whether an instrument's ideal rectangular field-of-view footprint fits inside a uniform test patch, use:

    cargo run --manifest-path tools/open-display-platform-verify/Cargo.toml -- \
      --geometry-record docs/engineering/display-measurement-geometry-synthetic-v0.1.json \
      --geometry-schema docs/engineering/display-measurement-geometry.schema.json

This uses an ideal planar-target geometry model, propagates first-order distance/FOV uncertainty, then makes the fit decision with a conservative upper-envelope plus centering and edge guards. Its report includes residual margin and patch-utilization fractions on each axis. The checked-in inputs are synthetic: it is not a lens/distortion, detector-response, calibration or physical-fit qualification.

To validate a run-specific geometry receipt after the OpenSCAD checks:

    cargo run --manifest-path tools/open-display-platform-verify/Cargo.toml -- \
      --cad-receipt target/odrh-validation/odrh-cad-runtime-receipt.json \
      --cad-receipt-schema hardware/open-display-reference/validation/odrh-cad-runtime-receipt.schema.json

The verifier validates these contracts and includes a testable energy-integration arithmetic core:

- The platform manifest and its Draft 2020-12 schema: component openness, module/source closure, benchmark claim ceilings, and manifest-to-native-CAD default parity.
- The run-record template and its Draft 2020-12 schema: unique identities, cross-record closure, calibration identity/expiry, raw-artifact linkage, observation status semantics and uncertainty-versus-minimum-decision-effect consistency.
- An optional run-generated CAD receipt: schema validity, native-source SHA-256, repository/run URL linkage, expected default/alternate mesh extents, all required negative controls, and the physical-validation claim ceiling.
- An optional synthetic drift-control plan and its Draft 2020-12 schema: strictly ordered controls, unique IDs, fixed guard-band classification, missing/invalid/out-of-band dispositions, and conservative quarantine windows. It refuses plans marked as acquired physical measurements.
- The Rust energy core integrates a timestamped power trace over a bounded task window using trapezoids and declared linear interpolation at covered window boundaries. The optional CLI plan/schema validator emits source hashes and a synthetic-only claim boundary. It rejects non-monotonic/invalid samples, insufficient time coverage and gaps beyond the configured cap; energy-per-successful-task is undefined when the success count is zero. Unit tests are arithmetic fixtures only, not physical wattmeter validation.

Negative tests cover missing/duplicate IDs, dangling references, unsupported openness claims, self-promoted physical validation, benchmark/gate self-promotion, CAD-source drift, measurement data smuggled into a plan-only record, inconsistent decision-capability claims, malformed RFC 3339 acquisition/review timestamps, impossible calibration expiry dates, missing calibration/capability artifacts, wrong artifact roles, digest-to-artifact mismatch, wrong measurement units, CAD-receipt source tampering, bad run linkage, altered expected bounds and omitted/relabelled negative controls, plus inclusive guard-band boundaries, immediately out-of-band values, missing and invalid controls, duplicate/out-of-order control slots, run-edge quarantine and overlapping excursions.

Success means the schemas/records conform to their contracts. It does not mean the design is fit-tested or build-ready, any selected part is available or licensed for redistribution, any instrument is calibrated, or physical display performance has passed. The checked-in run record is deliberately PLAN_ONLY: it contains four NOT_EVALUATED observation slots, no instruments, and no raw artifacts.
