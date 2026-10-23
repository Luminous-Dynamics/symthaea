# Clinical Inference Envelope v2 Qualification Checklist

This checklist qualifies the typed-evidence semantic contract only. It does not qualify a clinical model, detector, treatment, or presentation workflow.

## Build / static gates

- [ ] `cargo fmt -- --check`
- [ ] focused `cargo clippy -p symthaea-clinical --all-targets -- -D warnings`
- [ ] focused `cargo test -p symthaea-clinical`
- [ ] workspace checks required by repository policy
- [ ] exact Rust/toolchain and lockfile lineage recorded

## Schema gates

- [ ] only envelope schema version 2 accepted
- [ ] only evidence identity version 1 accepted
- [ ] unknown fields rejected for all new v2 structs
- [ ] evidence namespace non-empty, trimmed, whitespace-free, bounded
- [ ] artifact ID non-empty, trimmed, control-free, bounded
- [ ] zero digest rejected
- [ ] zero execution nonce rejected
- [ ] execution inputs non-empty and unique

## Typed evidence gates

- [ ] same raw digest under different namespaces is not equal identity
- [ ] evidence namespace substitution breaks execution binding
- [ ] evidence artifact-ID substitution breaks execution binding
- [ ] evidence digest substitution breaks execution binding
- [ ] subject-binding namespace substitution breaks execution binding
- [ ] subject-binding artifact substitution breaks execution binding
- [ ] duplicate `(namespace, artifact_id)` evidence is rejected even if digest differs
- [ ] all claim evidence is exactly execution-bound
- [ ] subject-binding evidence is exactly execution-bound

## Model lineage / calibration gates

- [ ] training lineage uses typed evidence identity
- [ ] evaluation lineage uses typed evidence identity
- [ ] model calibration evidence uses typed evidence identity
- [ ] calibrated inference requires probability + typed calibration evidence
- [ ] calibrated inference requires model calibration evidence
- [ ] inference/model calibration identity must match namespace + artifact ID + digest
- [ ] calibration namespace substitution fails closed

## Distribution gates

- [ ] `Unknown` may omit detector evidence
- [ ] `InDistribution` requires typed detector evidence
- [ ] `OutOfDistribution` requires typed detector evidence
- [ ] detector evidence namespace/artifact/digest validate independently
- [ ] no producer distribution state grants downstream clinical authority

## Clinical semantic gates

- [ ] clinical-decision-support intended use requires subject binding
- [ ] missing critical evidence remains explicit
- [ ] critical missing evidence does not prevent artifact representation but must remain visible to downstream policy
- [ ] v2 contains no diagnostic/treatment/prescribing/dispensing/administration authority field

## Compatibility / migration gates

- [ ] no silent reinterpretation of v1 bare digest as v2 typed identity
- [ ] no function converts arbitrary `[u8; 32]` to evidence identity without namespace + artifact ID
- [ ] v1 remains explicitly bounded to internal/preflight use unless separately crosswalk-qualified
- [ ] v2 namespace examples are not treated as trusted merely because strings match

## Required before external interoperability

- [ ] canonical v2 wire/framing contract
- [ ] domain-separated v2 wire identity
- [ ] frozen v2 conformance fixture(s)
- [ ] independent Mycelix v2 parser/verifier
- [ ] accepted namespace policy
- [ ] exact Mycelix fact/source crosswalk verification
- [ ] external qualification records exact head/run/job IDs

## Non-claims

A PASS establishes the software semantics of typed evidence identity and execution binding only. It does not establish evidence truth, scientific validity, clinical effectiveness, OOD detector validity, regulatory clearance, presentation authority, or treatment authority.
