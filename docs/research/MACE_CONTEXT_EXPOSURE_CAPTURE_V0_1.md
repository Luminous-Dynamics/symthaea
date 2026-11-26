# MACE model-context exposure capture and fake transport v0.1

**Status:** provider-neutral request serializer/source-map, artifact-writer interface, transport interface, receipt producer and deterministic tests are authored on a public research branch. Rust compilation, rustfmt, tests and CI have not been executed. The included test artifact writer is explicitly a fake fixture; it does not encrypt data or establish custody.

## Purpose

This is the producer-side complement to [#13721](https://github.com/Luminous-Dynamics/symthaea/pull/13721), which verifies signed receipts. It closes the gap between testing a verifier with hand-authored JSON and exercising the real request serialization path.

Source:
`research/broca-mace-transfer-bench-v0-1/aml-adapter/src/mace_context_exposure_capture.rs`

Crate export:
`aml_adapter::mace_context_exposure_capture`

## What is implemented

- A provider-neutral request model with a pinned model selector, deterministic non-memory request parameters, ordered messages and typed content segments. The literal request model selector must equal the independently pinned model revision; parameter keys are deterministic and reserved message/model keys are rejected.
- Each segment is either non-memory content or memory content with canonical evidence identity, payload digest, exposure kind, parent IDs, dependency closure, transformation, source, authority, admissibility and current-lifecycle commitments.
- A deterministic JSON serializer that writes the request exactly once.
- UTF-8 byte offsets are captured while serializing each typed memory-origin segment. It does not search the final prompt for repeated strings or use pre-serialization character offsets.
- JSON quotes, backslashes, newline/tab/control escaping and non-ASCII Unicode are accounted for in the emitted byte ranges.
- A provider-neutral immutable encrypted-artifact writer interface.
- A transport interface that accepts the exact serialized byte slice; the fake transport echoes its request-body digest.
- The response ID/body and candidate decision trace are bound into the receipt after transport completion, then the receipt is signed through the signer interface in #13721.
- Artifact commitment, profile, immutable-object and plaintext digest/length bindings are checked before transport.
- A true no-memory arm rejects typed memory-origin segments.

The returned request body is sent through the transport without a second serialization. This is central: a receipt for a different serialization than the bytes actually sent is not exposure evidence.

## Authored tests (not executed)

Seven deterministic tests cover:
1. UTF-8/Unicode plus quote, backslash and newline escaping with exact offsets.
2. Repeated identical memory strings from distinct evidence IDs retaining distinct offsets.
3. Full fake-transport → artifact commitment → observer signing → consumer verifier round trip, including same-request-byte digest binding.
4. Typed memory content rejected in the true no-memory arm.
5. Artifact storage-profile substitution rejected before transport.

Tests use a fake artifact writer whose ciphertext digest is a test commitment over the body, not encryption. The fake resolver returns the test body. These fixtures only test the interface and binding logic.

## Critical boundary

The source-map records memory origin only when the renderer supplies a typed Memory segment. It cannot infer that an arbitrary NonMemory string was secretly copied from memory by an untrusted upstream renderer. Production qualification therefore needs:
- the source-aware renderer to construct typed segments from canonical admitted evidence;
- instrumentation of every memory-derived path, including summaries, caches and tool outputs;
- an independent capture observer at the serialized-request boundary;
- a production artifact writer that verifies encryption, immutability, custody and retention;
- an observer key resolver and signing service whose trust profile is pinned independently;
- a fake-transport round trip under all deterministic/metamorphic controls before live provider deployment.

Version 0.1 emits disjoint spans. If a renderer cannot preserve a complete disjoint byte map through a transformation, the invocation must fail closed and become non-qualifiable.

A verified exposure receipt proves which memory-derived bytes were transmitted under the pinned capture boundary. It does not prove model attention, understanding, reliance or causal influence. The paired randomized intention-to-treat effect in #13141 remains the primary causal estimate.

## Next validation

Run, on the exact PR head:
- `cargo fmt --check` for the adapter crate;
- focused `cargo test` for the capture and verifier modules;
- Draft 2020-12 schema meta-validation and emitted JSON schema validation;
- exact-head CI.

Report compilation, unit tests, schema validation, and CI independently. Skipped checks are not passes.
