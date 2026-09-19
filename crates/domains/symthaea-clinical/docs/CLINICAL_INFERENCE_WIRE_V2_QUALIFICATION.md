# Clinical Inference Wire v2 Qualification Checklist

This checklist qualifies the binary framing and exact-wire identity only. It does not qualify a clinical model or workflow.

## Build / static gates

- [ ] `cargo fmt -- --check`
- [ ] `cargo clippy -p symthaea-clinical --all-targets -- -D warnings`
- [ ] `cargo test -p symthaea-clinical`
- [ ] repository-required workspace/governance checks
- [ ] no unsafe code

## Canonical encoding gates

- [ ] magic bytes fixed and documented
- [ ] wire version encoded separately from envelope schema
- [ ] integers big-endian
- [ ] floats use exact IEEE-754 bits
- [ ] strings length-prefixed UTF-8
- [ ] vectors length-prefixed and order-preserving
- [ ] options use only tags 0/1
- [ ] all enum tags frozen/documented
- [ ] typed evidence namespace + artifact ID + digest all encoded
- [ ] model training/evaluation/calibration identities encoded
- [ ] inference calibration identity encoded
- [ ] OOD detector identity encoded
- [ ] execution input evidence encoded
- [ ] execution nonce encoded

## Decoder adversarial gates

- [ ] wrong magic rejected
- [ ] unknown wire version rejected
- [ ] invalid enum tag rejected
- [ ] invalid option tag rejected
- [ ] invalid UTF-8 rejected
- [ ] truncated input rejected
- [ ] trailing bytes rejected
- [ ] over-limit complete message rejected before nested decode
- [ ] over-limit strings rejected
- [ ] over-limit vector counts rejected
- [ ] decoded object re-runs semantic v2 validation

## Identity gates

- [ ] identical object -> identical canonical bytes
- [ ] identical object -> identical wire digest
- [ ] namespace substitution changes wire bytes/digest
- [ ] artifact-ID substitution changes wire bytes/digest
- [ ] evidence digest substitution changes wire bytes/digest
- [ ] model/runtime/config/nonce substitution changes wire identity
- [ ] exact float-bit substitution changes wire identity
- [ ] digest-from-bytes requires canonical valid wire

## Frozen conformance vector

- [ ] `fixtures/clinical_inference_wire_v2.hex` decodes to exactly 1260 bytes
- [ ] encoder output equals frozen vector byte-for-byte
- [ ] decoder produces the frozen typed fixture object
- [ ] re-encoding decoded fixture reproduces frozen bytes exactly
- [ ] fixture obtains nonzero domain-separated v2 wire digest

## Cross-repository qualification still required

Before clinical interoperability promotion:

- [ ] copy/freeze the same v2 fixture in Mycelix
- [ ] independently implement Mycelix binary decoder (no Symthaea dependency)
- [ ] independently reproduce enum/framing semantics
- [ ] independently reproduce v2 BLAKE3 wire identity
- [ ] verify exact cross-repository fixture bytes
- [ ] apply deployment namespace/engine/model/admission policy
- [ ] bind accepted `mycelix/clinical-fact-snapshot/v1` evidence to exact validated fact snapshots

## Non-claims

A PASS establishes software-level canonical representation and exact wire identity only. It does not establish evidence truth, model validity, clinical effectiveness, OOD validity, patient applicability, regulatory clearance, clinical presentation authority, or treatment authority.
