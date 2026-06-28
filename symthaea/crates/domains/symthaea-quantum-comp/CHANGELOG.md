# Changelog

## 0.1.0-alpha.10

### Added

- `verification_matrix` module for smoke, local-research, pilot, and external validation checklists.
- `migration` module with alpha.9 to alpha.10 migration guidance.
- `beta_readiness` module with conservative beta-transition status.
- `validation_snapshot` module combining inventory, release manifest, verification matrix, migration, and beta-readiness.
- CLI commands: `verify-matrix`, `migration`, `beta`, and `snapshot`.
- Examples: `verification_matrix`, `migration_guide`, `beta_readiness`, and `validation_snapshot`.
- Tests: `tests/alpha10_tests.rs`.
- Docs: `ALPHA10_UPGRADE_NOTES.md`, `VERIFICATION_MATRIX_ALPHA10.md`, `BETA_TRANSITION.md`, `VALIDATION_SNAPSHOT.md`, and `ALPHA10_RELEASE_CHECKLIST.md`.

### Changed

- Crate version is now `0.1.0-alpha.10`.
- Schema labels now end in `alpha10`.
- README now frames alpha.10 as a beta-transition hardening release rather than a new probe-math release.
- Local verification script now exercises the new examples and CLI commands.

### Non-claims preserved

- No quantum consciousness claim.
- No quantum advantage claim.
- No physical backend execution claim.
- No Mycelix attestation claim from local metadata or FNV fingerprints.

## 0.1.0-alpha.9

- Added `stability` module with alpha public-surface annotations.
- Added `api_inventory` module for dependency-free API inventory reports.
- Added `release_manifest` module for blocked claims and recommended local verification commands.
- Expanded CLI with `inventory` and `manifest` commands.
- Added examples: `api_inventory`, `release_manifest`.
- Added docs: `API_INVENTORY.md`, `ALPHA9_RELEASE_MANIFEST.md`, `ALPHA9_UPGRADE_NOTES.md`, `ALPHA9_RELEASE_CHECKLIST.md`.
- Updated schema labels to alpha9.
- Added `tests/alpha9_tests.rs`.
- Preserved conservative claim posture: no quantum consciousness, quantum advantage, hardware execution, physical entanglement, production safety, or Mycelix attestation claim.

## 0.1.0-alpha.8

- Added `fixtures` module with named smoke/demo/pilot local fixtures.
- Added `replay` module for operator-facing replay plans.
- Added `release_gate` module for local preflight/audit/fixture/replay gate summaries.
- Added `interop` module with explicit Symthaea, Mycelix, local lab, and external backend integration-boundary declarations.
- Updated schema labels to alpha8 and added labels for fixtures, replay plans, release gates, and integration declarations.
- Expanded CLI with `presets`, `schemas`, `fixtures`, `replay`, and `gate` commands.
- Added examples: `fixture_catalog`, `replay_plan`, `release_gate`, `interop_boundary`.
- Added docs: `FIXTURES_AND_REPLAY.md`, `RELEASE_GATES.md`, `INTEROP_BOUNDARIES.md`, `ALPHA8_UPGRADE_NOTES.md`, `ALPHA8_RELEASE_CHECKLIST.md`.
- Added `tests/alpha8_tests.rs`.

## 0.1.0-alpha.7

- Added `preflight` module for local configuration checks.
- Added `presets` module with stable `smoke`, `local-research`, and `pilot-matrix` profiles.
- Added `bundle` module for local research bundle packaging.
- Added `schema` module with stable alpha.7 report labels.
- Added a minimal dependency-free CLI: `symthaea-quantum-comp`.
- Added examples: `preflight_presets`, `research_bundle`.
- Added docs: `CLI_USAGE.md`, `PREFLIGHT_AND_PRESETS.md`, `RESEARCH_BUNDLES.md`, `ALPHA7_UPGRADE_NOTES.md`.
- Added `tests/alpha7_tests.rs`.
- Updated local verification script to exercise alpha.7 examples and CLI smoke runs.

## 0.1.0-alpha.6

### Added

- `matrix` module with dimension-by-noise replicated experiment grids.
- `significance` module with paired difference summaries and exact two-sided sign-test helpers.
- `receipts` module with local non-cryptographic research artifact receipts shaped for future Mycelix integration.
- `examples/experiment_matrix.rs`.
- `examples/significance_probe.rs`.
- `examples/research_receipt.rs`.
- `tests/alpha6_tests.rs`.
- `docs/EXPERIMENT_MATRIX.md`.
- `docs/STATISTICAL_CAUTIONS.md`.
- `docs/MYCELIX_INTEGRATION.md`.
- `docs/ALPHA6_UPGRADE_NOTES.md`.

### Changed

- README now describes alpha.6 as a matrix, significance, and receipt hardening release.
- Research artifact receipts explicitly state that they are not cryptographic signatures or Mycelix source-chain entries.
- Experiments can now be explored across several dimensions and noise settings before interpretation.

### Non-claims preserved

- No quantum consciousness claim.
- No quantum advantage claim.
- No hardware backend execution claim.
- No physical entanglement claim.
- No cryptographic provenance claim from local receipts.

## 0.1.0-alpha.5

### Added

- `provenance` module with `RunEnvironment`, `ReproducibilityRecord`, and a dependency-free FNV-1a helper for non-cryptographic report fingerprints.
- `reporting` module with CSV and Markdown exports for binding, noise-sweep, comparative, and robustness reports.
- `audit` module with conservative claim-boundary, negative-control, and robustness guardrails.
- `examples/report_exports.rs`.
- `examples/audit_controls.rs`.
- `scripts/verify-local.sh` for local validation.
- `tests/alpha5_tests.rs`.
- `docs/LOCAL_VERIFICATION.md`.
- `docs/REPORT_FORMATS.md`.
- `docs/CLAIM_BOUNDARIES.md`.

### Changed

- README now describes alpha.5 as a provenance, reporting, and local-audit release.
- Reports are easier to paste into lab notes without adding serialization dependencies.
- The crate now distinguishes reproducibility fingerprints from real security receipts more explicitly.

### Non-claims preserved

- No quantum consciousness claim.
- No quantum advantage claim.
- No hardware backend execution claim.
- No physical entanglement claim.

## 0.1.0-alpha.4

### Added

- `statistics` module with dependency-free sample summaries, approximate 95% CI helper, AUC, slopes, threshold crossings, monotonicity checks, and paired effect sizes.
- `robustness` module for deriving robustness summaries from noise sweeps.
- `comparative` module with replicated binding comparison reports.
- `examples/comparative_report.rs`.
- `examples/robustness_summary.rs`.
- Alpha.4 tests for statistics, robustness, and replicated comparisons.

### Changed

- README now describes alpha.4 as a more research-usable reporting release.
- Reports now support stronger interpretation discipline through replicated summaries rather than relying only on single-run aggregates.

### Non-claims preserved

- No quantum consciousness claim.
- No quantum advantage claim.
- No hardware backend execution claim.
- No physical entanglement claim.

## 0.1.0-alpha.3

### Added

- `experiment` module with `ExperimentManifest`, `ExperimentProtocol`, and `ClaimBoundary`.
- `controls` module with `NegativeControlRunner`.
- `entanglement_proxy` module with `EntanglementProxySketch` and `EntanglementProxyRunner`.
- `examples/negative_control.rs`.
- `examples/entanglement_proxy.rs`.
- Bell-pair register QASM export helper behind `qasm-export`.
- Alpha.3 reproducibility tests.

### Changed

- README now describes alpha.3 modules and claim boundaries.
- Research notes now distinguish implementation checks, local simulation, circuit export, and external backend observations.

### Non-claims preserved

- No quantum consciousness claim.
- No quantum advantage claim.
- No hardware backend execution claim.
- No physical entanglement claim.

## 0.1.0-alpha.2

### Added

- Correlation-style binding sketch.
- Noise sweep runner.
- Additional topology summary fields.
- Reproducibility fingerprints.
- Optional toy QASM export.

## 0.1.0-alpha.1

### Added

- Initial classical binary HDC baseline.
- Phase-HDC simulation baseline.
- Binding probe runner.
- Basic topology proxy.
- Substrate profiles and caveats.
