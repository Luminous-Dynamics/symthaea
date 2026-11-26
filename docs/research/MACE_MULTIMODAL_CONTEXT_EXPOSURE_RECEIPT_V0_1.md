# MACE multimodal model-context exposure receipt v0.1

**Status:** research contract and JSON Schema proposal authored on a public branch stacked on the text capture producer. The schema and document have not yet been meta-validated or implemented in Rust. No provider-side ingress, media decoding, model attention, causal effect or benchmark result is claimed.

## Purpose

The text-only MACE capture path records typed memory-origin segments as exact UTF-8 byte spans in a serialized provider request. Multimodal model requests add a second identity problem: the bytes visible in the client request may be encoded media, an opaque provider asset reference, a selected video frame, an audio window, or a transformed derivative. These are not interchangeable.

This extension preserves the current text receipt semantics and adds a separately versioned, per-invocation signed media receipt. It does not add a memory store, source registry, authority system, lifecycle controller, artifact store, evaluator or provider abstraction.

## Core claim ladder

The receipt keeps four claims separate:

1. **Inline bytes transmitted.** The exact request body contains a media payload. The capture binds the serialized JSON content-part span and separately binds the decoded media payload digest and byte length.
2. **Asset reference transmitted.** The exact request body contains a provider file ID, URL or opaque reference. The request-span digest proves only that the reference bytes were sent. It is not counted as media-byte exposure.
3. **Upload acknowledged.** A separate, signed client-boundary transfer receipt binds the exact upload request body, media payload digest/length, provider response digest and returned asset-ID digest. This proves what the client sent and what identifier the provider returned; it does not independently prove the provider retained or decoded the payload.
4. **Provider ingress attested.** A provider-signed or independently attested receipt binds the provider asset identity, payload digest and target invocation. Without that external attestation, the strongest status remains upload acknowledged or reference transmitted.

The client observer cannot elevate a claim by signing its own assertion that the provider ingested or attended to media. Provider acknowledgement and provider ingress attestation are different evidence classes.

## Identity model

Each media exposure binds:

- run, pair block, world cluster, query, arm, invocation sequence and pinned model revision;
- exact request-body digest, byte length and immutable encrypted artifact;
- message role/index, content part index and RFC 6901 content path;
- exact request-body byte span (inclusive start, exclusive end) and span digest;
- canonical source evidence ID, media payload SHA-256 and payload byte length;
- modality-specific coordinates and/or time identity;
- parent evidence IDs, dependency closure, transformation receipt, source, authority, admissibility and current-lifecycle receipts;
- transfer mode and provider-ingress status.

A content path is a structural locator; it is not a substitute for byte verification. The verifier must resolve the path in the exact captured body and check that it denotes the same content part covered by the byte span. All numeric offsets are half-open. Every video time is represented as presentation timestamp ticks multiplied by an explicit rational timebase; no implicit milliseconds or frame-rate assumptions are allowed. Audio ranges use sample frames, not scalar channel samples: start is inclusive and end is exclusive.

## Media-specific lineage

### Images

Bind the original media payload digest and dimensions. A selected region uses an explicit coordinate space (source pixels or normalized 0..1) and a region rectangle. Crops, rotations, resize, EXIF normalization, tiling and other transforms require their own transformation receipt and parent binding. A crop does not become a new independent source.

### Video

Represent each exposed frame as a separate event with source-video evidence ID, frame index, presentation timestamp ticks, rational timebase and decoded-frame digest/dimensions. The consumer verifies ordering, duplicate indices, timestamp uniqueness where the source semantics require it, and declared coverage of selected intervals. Missing frames or unknown extraction policy produce an incomplete/indeterminate source map, not a silently complete video exposure.

A provider-native video object may instead be represented by a provider asset reference. The receipt then proves that the reference was sent; it does not imply that all frames or the complete video were decoded by the provider.

### Audio

Bind sample rate, channel count, codec/channel layout where known, and half-open sample-frame ranges. The media payload digest covers the exact encoded asset or canonical segment artifact as declared by the transfer profile. Transcript, subtitle, speaker label, caption and audio-summary derivatives are separate derived-text exposures with all required parent audio IDs and a transformation receipt. Time alignment must be explicit; a transcript without a verified source interval is not treated as a time-exact quotation.

### Derived text

OCR, transcripts, captions, descriptions and subtitles are first-class derivatives. They carry their own text digest/length, derivative kind, parent media IDs and transformation receipt. The ordinary text receipt captures their request-body spans; this extension links those spans back to the source media. A model-generated description is not source evidence and cannot inherit the source's authority.

## Separate upload and later reference

A provider upload is a distinct transfer event, not merely a field in the later inference request. Its receipt binds the exact upload body digest/length and encrypted body artifact, the canonical media payload digest/length, provider response ID/body digest, provider asset-ID digest, observer profile and signature.

A subsequent inference request must bind the exact asset-reference span to the prior transfer ID and provider asset-ID digest. Transfer receipts carry an explicit transfer sequence, monotonic capture time and clock-profile digest; ordering is only compared inside the same pinned clock profile. Transfer receipts are checked for run/tenant scope, sequence, replay, asset identity and lifecycle revalidation at the later use time. If the upload event cannot be independently resolved or the returned provider ID cannot be bound, the inference exposure is only a reference transmission.

The upload receipt's signature is domain-separated from the invocation receipt. The transfer signature covers SHA-256("MACE-MULTIMODAL-ASSET-TRANSFER-V1\\0" || JCS(transfer_receipt_without_signature)); the invocation receipt covers SHA-256("MACE-MULTIMODAL-CONTEXT-EXPOSURE-V1\\0" || JCS(receipt_without_signature)). Both observer signatures resolve through the independently pinned observer profile. A provider-ingress attestation, when present, is separately verified under a distinct provider trust profile and is never inferred from either observer signature.

The provider asset ID is hashed in public receipts. Raw IDs, media bytes, transcripts and image contents must not be emitted into public JSONL.

## No-memory arm

The true no-memory arm must contain zero memory-derived text exposures, zero memory-derived media parts, zero memory-derived asset references and zero memory-derived uploads. This includes prior-session images, video frames, audio windows, cached crops, OCR/transcripts, tool outputs and model-generated summaries with memory ancestry.

The schema includes explicit counts and a structural zero-item constraint for the no-memory arm. The verifier must independently recompute the counts from the captured request and upload receipts; candidate-supplied zero counters are not sufficient.

## Verifier requirements

1. Verify the top-level signature and independently pinned observer profile.
2. Retrieve the immutable request artifact and verify custody, ciphertext, plaintext digest and byte length.
3. Resolve every content path into the exact serialized request body; verify the request-span bounds and digest.
4. For inline payloads, decode the declared wire encoding with strict canonical rules and verify media payload digest, byte length and MIME/media-type consistency.
5. For provider asset references, verify the reference span, linked transfer receipt, upload digest, provider asset-ID digest and monotonic sequence. Never count the reference string as media bytes.
6. Verify any provider-ingress attestation under a distinct, independently pinned provider key/profile; do not accept an observer's self-assertion as provider evidence.
7. Verify modality/details consistency, image dimensions and region bounds, rational video timebase and frame ordering, audio sample-rate/channel declarations, non-empty half-open sample-frame ranges and media-part coverage.
8. Verify all evidence IDs, source/authority/rights/scope/admissibility/currentness and transitive parent closure through existing MACE owners.
9. Verify each derivative's transformation receipt and all required parent IDs; reject cycles, dangling edges, ambiguous aliases and revoked ancestry.
10. Recompute the no-memory counts and reject any memory-derived content or reference in the true no-memory arm.

JSON Schema cannot establish these cross-record, cryptographic, media-decoding, lineage or lifecycle invariants by itself. It also cannot prove that a content path resolves to the declared span, that an audio end sample exceeds its start sample, or that a video frame belongs to the declared source; these are verifier and independent-fixture responsibilities.

## Deterministic and adversarial tests

- PNG/JPEG/WebP inline bytes; invalid base64, non-canonical padding, MIME mismatch, payload digest/length mismatch and truncated bodies.
- Same media bytes attributed to different evidence IDs; identity must not collapse through digest equality.
- Changed/replayed provider asset IDs and transfer receipts; upload acknowledgement without ingress attestation.
- Crop/resize/rotation/EXIF normalization with wrong transform digest or missing parent.
- Video frame reorder, duplicate index, missing frame, invalid rational timebase, repeated PTS and incomplete interval coverage.
- Audio wrong sample rate/channel layout, overlapping/gapped windows, off-by-one frame boundary and transcript time-alignment drift.
- Content path points to a different request part than the recorded byte span.
- Request span valid but decoded payload corrupted; payload digest valid but request span changed.
- Text source map complete while a multimodal content part is untracked.
- Media part appears through a tool message, cache, previous output or derivative but is mislabeled non-memory.
- Correction/revocation after asset upload but before model invocation.
- True no-memory arm with any memory-derived media, asset reference, upload or derivative.
- Provider returns an asset ID but no independently verifiable ingress statement; status must remain upload acknowledged.
- Provider-side crop/transcode/frame extraction without an attested transformation downgrades source-exact claims.

## Benchmark implications

For the official AML multimodal track, preserve the native leaderboard score as a separate result. Add MACE-specific diagnostics rather than altering the official metric:

- media evidence recall and retrieval precision by modality;
- temporal and spatial localization accuracy;
- source/derivative lineage completeness;
- stale-media exposure after correction/retraction;
- fraction of reference-only versus byte-captured versus provider-ingress-attested exposures;
- no-memory contamination rate;
- image/video/audio byte and token budgets;
- p50/p95/p99 upload and inference latency;
- storage, verification, decode and transform cost;
- paired downstream task utility, regret and harmful memory dependence.

The official AML Cycle 2 published on 2026-11-19 reports Khipu (open-source multimodal) at 43.26 overall, including 34.23 temporal reasoning, 38.45 continual learning, 11.04 personalization and 31.29 evidence governance; MemoraX (commercial/closed-source) reports 53.68 overall. These are native track scores and must not be compared to textual/coding scores or MACE causal utility. They motivate the modality-specific work; they are not MACE results.

This extension is tracked by [issue #13744](https://github.com/Luminous-Dynamics/symthaea/issues/13744) and is stacked on [text producer PR #13725](https://github.com/Luminous-Dynamics/symthaea/pull/13725).

## Claim boundary

This contract can establish client-side bytes/reference transmission and bind them to canonical evidence lineage under a qualified capture observer. It cannot, by itself, prove provider-side decoding, provider retention, model routing, neural attention, semantic understanding or causal influence. Provider ingress needs provider-signed/independently attested evidence; causal influence remains owned by randomized intervention work in #13141.