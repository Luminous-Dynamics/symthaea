# symthaea-semiconductor

Rust-native semiconductor engineering contracts and evidence models.

This crate is deliberately separate from the general additive-fabrication kernel. It owns semiconductor/device-specific subjects and benchmark contracts; it does not duplicate generic fabrication, material, process, calibration, provenance, or execution-authority ontologies.

## Current scope

V1 provides a Rust-native split-manifest validator for semiconductor benchmark datasets:

- group-disjoint train/validation/test assignments;
- strict forward-time windows;
- offset-aware timestamps;
- source-local naive timestamps with an explicit clock-domain ID;
- canonical SHA-256 identity compatible with the independent Python reference oracle;
- split support counts and explicit claim limits.

The validator does not download data, train models, or claim process generalization. A source-local clock establishes ordering only within its declared clock domain; it does not imply UTC or cross-site comparability.

## Cross-language reference

Tests load the frozen SEMI-BENCH-001 Python known-answer fixture and the row-level SECOM source-clock split manifest. Python remains a test/reference oracle; production contracts are implemented in Rust.

## Next increments

1. Binary probability metrics and calibration.
2. Validation-only threshold policy and selective prediction.
3. Budgeted inspection decision model.
4. Rights-aware source acquisition and deterministic SECOM parser.
5. Matched model baselines after split/source admission.
