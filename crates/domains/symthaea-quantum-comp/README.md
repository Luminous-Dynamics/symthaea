# symthaea-quantum-comp

Experimental quantum and quantum-inspired substrate probes for Symthaea HDC cognition research.

## Claim boundary

This crate does **not** claim quantum consciousness, quantum advantage, physical entanglement execution, or hardware backend validation.

It is a small Rust research scaffold for comparing classical HDC binding, phase-HDC simulation, correlation sketches, entanglement-proxy sketches, noise behavior, negative controls, topology summaries, replicated comparison reports, reproducibility metadata, and future circuit export artifacts.

The intended posture is:

- the AI or theory may propose hypotheses;
- the probe produces reproducible measurements;
- reports record substrate assumptions and caveats;
- local audits reject or warn on over-strong interpretations;
- external quantum tooling remains the source of physical backend execution.

## Current version

`0.1.0-alpha.10`

Alpha.10 is a beta-transition hardening release. It adds release-readiness surfaces rather than new speculative math:

- `verification_matrix` module for smoke, local-research, pilot, and external validation checklists.
- `migration` module for alpha.9 → alpha.10 migration guidance.
- `beta_readiness` module with conservative beta-transition status.
- `validation_snapshot` module that joins inventory, manifest, matrix, migration, and beta-readiness into one release summary.
- CLI commands: `verify-matrix`, `migration`, `beta`, and `snapshot`.
- New examples: `verification_matrix`, `migration_guide`, `beta_readiness`, and `validation_snapshot`.
- New docs: `ALPHA10_UPGRADE_NOTES.md`, `VERIFICATION_MATRIX_ALPHA10.md`, `BETA_TRANSITION.md`, `VALIDATION_SNAPSHOT.md`, and `ALPHA10_RELEASE_CHECKLIST.md`.
- Schema labels now end in `alpha10`.

The crate remains conservative: local reports are not publishable proof, hardware validation, quantum advantage evidence, or Mycelix attestation.

## Modules

- `classical_hdc` — packed binary HDC baseline.
- `phase_hdc` — quantum-inspired phase hypervectors.
- `correlation_hdc` — explicit parity/correlation binding sketch.
- `entanglement_proxy` — classical proxy for entanglement-mediated binding hypotheses.
- `controls` — negative controls for research sanity checks.
- `noise_sweep` — controlled degradation sweeps.
- `robustness` — noise-sweep robustness summaries.
- `statistics` — small dependency-free report statistics.
- `comparative` — replicated comparison reports. **Uncalibrated**: compares
  classical and phase-HDC at the same literal `noise` parameter value, which
  does not mean the same perturbation magnitude in each channel — see
  `docs/RESEARCH_NOTES.md` ("First independent run and a real finding"). Use
  `calibrated_comparison` for cross-representation noise claims.
- `calibrated_comparison` — cross-representation comparison at a matched,
  calibrated bit-error-rate, with both arms scored by the same final metric.
  The fair replacement for `comparative`/`noise_sweep`'s cross-representation
  columns; see the module doc comment and
  `docs/RESEARCH_NOTES.md` for the full result.
- `capacity_comparison` — calibrated bundling-capacity comparison (how many
  superposed items survive reliable two-alternative forced-choice recall),
  the more theoretically distinctive claim for phase/holographic
  representations than noise-robustness; see `docs/RESEARCH_NOTES.md` for
  the full result.
- `continuous_value_comparison` — calibrated comparison of storing a
  continuous scalar (not a bit) directly as a phase angle vs. classical
  thermometer coding, including (a) a debiasing correction for thermometer
  decode's noise-induced bias, and (b) a shrinkage-factor sweep testing
  whether a *partial* correction beats full debiasing. **Not a flat win for
  either side**: phase wins at zero noise (a real quantization-floor gap)
  and at higher noise (target BER ≳ 0.10) even at classical's best available
  decoder; debiased classical wins at low-to-moderate noise. Shrinkage
  doesn't meaningfully change this picture except very close to the noise
  ceiling. See `docs/RESEARCH_NOTES.md` for the full crossover result and
  the honest caveats.
- `matrix` — dimension-by-noise replicated experiment grids.
- `stability` — alpha surface stability annotations.
- `api_inventory` — dependency-free API inventory and surface catalog.
- `release_manifest` — blocked claims and recommended verification manifest.
- `verification_matrix` — smoke/local/pilot/external verification checklist.
- `migration` — alpha release migration guides.
- `beta_readiness` — conservative beta-transition status reports.
- `validation_snapshot` — combined release-readiness snapshot.
- `provenance` — local environment and reproducibility metadata.
- `receipts` — local research artifact receipt scaffolding for future Mycelix integration.
- `reporting` — CSV and Markdown report exports.
- `significance` — paired comparisons and exact sign-test helpers.
- `audit` — conservative local claim-audit helpers.
- `preflight` — local configuration checks before running experiments.
- `presets` — stable local run presets for examples and CLI use.
- `bundle` — local research bundle packaging for lab notes.
- `fixtures` — stable named local run fixtures.
- `replay` — operator-facing replay plans.
- `release_gate` — local preflight/audit/fixture/replay release summaries.
- `interop` — explicit future Symthaea/Mycelix/backend boundary declarations.
- `schema` — stable report schema labels for downstream scripts.
- `topology` — lightweight graph/topology proxy summaries.
- `substrate` — backend metadata and confidence labels.
- `experiment` — claim boundaries and experiment manifests.
- `qasm` — optional OpenQASM helpers for external tooling tests.

## Examples

Run the baseline binding probe:

`cargo run --example binding_probe`

Run the noise sweep:

`cargo run --example noise_sweep`

Run the robustness summary:

`cargo run --example robustness_summary`

Run the replicated comparison report (uncalibrated — see `docs/RESEARCH_NOTES.md`):

`cargo run --example comparative_report`

Run the calibrated cross-representation comparison (the fair replacement for the above):

`cargo run --example calibrated_comparison`

Run the calibrated capacity comparison (bundling capacity, not noise-robustness):

`cargo run --example capacity_comparison`

Run the calibrated continuous-value comparison (a real crossover, not a flat win for either side):

`cargo run --example continuous_value_comparison`

Run the shrinkage probe (does a partial bias correction beat full debiasing at high noise?):

`cargo run --example shrinkage_probe`

Run a dimension-by-noise experiment matrix:

`cargo run --example experiment_matrix`

Run a paired significance probe:

`cargo run --example significance_probe`

Generate a local research receipt:

`cargo run --example research_receipt`

Run the negative control:

`cargo run --example negative_control`

Run the entanglement proxy probe:

`cargo run --example entanglement_proxy`

Run report exports:

`cargo run --example report_exports`

Run local audit helpers:

`cargo run --example audit_controls`

Run preset preflight summaries:

`cargo run --example preflight_presets`

Generate a local research bundle:

`cargo run --example research_bundle`

List alpha.10 fixtures:

`cargo run --example fixture_catalog`

Print a replay plan:

`cargo run --example replay_plan`

Run a local release gate:

`cargo run --example release_gate`

Reference smoke gate artifact:

`docs/artifacts/SMOKE_BINDING_GATE_ALPHA10.md`

Show integration boundaries:

`cargo run --example interop_boundary`

Print the API inventory:

`cargo run --example api_inventory`

Print the release manifest:

`cargo run --example release_manifest`

Print the alpha.10 verification matrix:

`cargo run --example verification_matrix`

Print the alpha.9 to alpha.10 migration guide:

`cargo run --example migration_guide`

Print conservative beta-readiness status:

`cargo run --example beta_readiness`

Print the validation snapshot:

`cargo run --example validation_snapshot`

Use the alpha CLI:

`cargo run --bin symthaea-quantum-comp -- binding smoke`

`cargo run --bin symthaea-quantum-comp -- gate smoke-binding`

`cargo run --bin symthaea-quantum-comp -- inventory`

`cargo run --bin symthaea-quantum-comp -- manifest`

`cargo run --bin symthaea-quantum-comp -- verify-matrix`

`cargo run --bin symthaea-quantum-comp -- beta`

`cargo run --bin symthaea-quantum-comp -- snapshot`

Use QASM helpers:

`cargo test --features qasm-export`

Run the local verification script:

`./scripts/verify-local.sh`

## Research rule

Treat every result as a controlled probe, not a proof of consciousness or quantum advantage.

The phrase to keep in mind:

**Reproducible substrate experiments first. Interpretation second. Claims last.**
