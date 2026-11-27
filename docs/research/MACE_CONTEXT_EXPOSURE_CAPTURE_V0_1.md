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

## Use-time admission before signed exposure

The capture adapter now accepts an injected `MaceExposureAdmissionVerifierV1` and invokes it for every serialized memory-origin span **before** immutable artifact storage and provider transport. A rejection stops the invocation before either side effect.

The new `MaceIngressUseTimeAdmissionVerifierV1` composes three independently supplied dependencies:

- A state resolver retrieves the canonical raw event and its previously admitted ingress view.
- The ingress `EvidenceVerifier` revalidates raw source, authority, trust, lineage, calibration, causal assessment, outcome and lifecycle using the independently pinned `valid_at_ms` / `known_at_ms` snapshot.
- A separate purpose/rights verifier checks current consent/rights, purpose code, scope and consumer policy. Source trust is not treated as a substitute for user permission.

The adapter checks the verified use receipt's process-local custody and integrity, exact evidence/event/payload/source/authority/transformation/lineage bindings, bitemporal snapshot, scope, purpose and consumer-policy identity. Its canonical receipt digest must exactly match the span's `lifecycle_revalidation_receipt_sha256`.

The existing observer Ed25519 signature covers the span and this digest. Receipt consumers invoke the same admission-verifier interface again, so a signed exposure receipt alone cannot bypass current use-time revalidation. A consumer must provide the same independent run binding and current resolver/verifier implementations.

The new pre-send gate proves only the configured verifiers' decisions at the pinned snapshot. It does not make the SHA-256 use receipt a signature, does not implement a rights authority, and does not guarantee a resolver or rights verifier is honest. Those are deployment trust roots and must be independently qualified. The capture still serializes the exact request once and sends the same byte buffer that was committed to immutable artifact storage.
