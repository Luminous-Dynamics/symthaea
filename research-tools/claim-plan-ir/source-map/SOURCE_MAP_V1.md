# CPIR Source Coordinate and Transformation Map v1

**Status:** research contract proposal; not a production parser, HTTP decoder, TLS verifier or proof circuit  
**Owner:** [SYM-CPIR-SOURCE-MAP-001 / #12466](https://github.com/Luminous-Dynamics/symthaea/issues/12466)  
**Related:** #12301 (authenticated transcript openings), #12328 (field semantics), #12373 (TLSNotary verified commitment metadata), #12447 (independent parser oracle), #12423 (matched performance benchmark)

## 1. Security proposition

A private claim is qualified only when the verifier can reproduce or verify every semantic transformation between the authenticated source transcript and the predicate input.

The required relation is:

```text
TLSNotary-authenticated transcript segment(s)
  -> HTTP response/message extraction
  -> transfer-decoded body
  -> content-decoded entity body
  -> strict UTF-8 JSON bytes
  -> duplicate-aware JSON structure
  -> verifier-authorized JSON Pointer and raw token span
  -> typed conversion
  -> canonical semantic value / CPIR plan
  -> predicate program and proof public inputs
```

The chain must not collapse these representations into one undifferentiated "response bytes" field. A hash identifies a representation; it does not establish how that representation was derived from another one.

## 2. Coordinate spaces are types

A byte span is meaningful only together with:
- a stable coordinate-space identifier;
- the digest and byte length of the exact representation being indexed;
- an explicit offset convention (v1: zero-based, half-open `[start,end)`);
- the producing transformation/profile and its implementation identity.

The v1 vocabulary is:

| Coordinate space | Meaning | What it is not |
|---|---|---|
| `tlsn_transcript_plaintext_bytes` | Byte positions in the exact owner-defined TLSNotary transcript representation for a named direction/response segment | TCP packet offsets, TLS record ciphertext offsets, or an assumed offset system not exposed by the verified TLSN API |
| `http_message_bytes` | Byte positions in the exact HTTP message representation extracted from the transcript | A decoded entity body |
| `http_transfer_decoded_body_bytes` | Entity bytes after transfer framing/coding is removed | Bytes after content decompression |
| `http_content_decoded_body_bytes` | Bytes after the declared Content-Encoding decoder(s) | Original compressed representation |
| `json_source_utf8_bytes` | Exact UTF-8 JSON document supplied to duplicate-aware admission | Re-serialized JSON or JCS output |
| `json_raw_value_token_bytes` | Exact source token range selected in the JSON source representation | A span into canonical JSON |
| `jcs_canonical_json_bytes` | RFC 8785 canonical representation of an admitted I-JSON value | Original source bytes or a source byte range |
| `typed_value_encoding_bytes` | Versioned canonical encoding of the converted typed value | A JSON token unless explicitly defined as one |
| `predicate_input_bytes` | Exact public/private input encoding consumed by the pinned predicate program | A human-readable value without an encoding contract |

An implementation must not pass an unqualified integer offset between these spaces. If upstream TLSNotary exposes only a bounded commitment opening and not a global transcript offset, record the upstream-defined commitment/range identifier; do not invent a global offset.

## 3. Representation records

Every produced representation is recorded as a node in a directed acyclic derivation graph.

Required fields:
- `representation_id` (unique within one evidence bundle);
- `coordinate_space`;
- `sha256` and `byte_length`;
- `media_type` and `content_encoding` where applicable;
- `producer_stage_id`;
- `profile_id` and `profile_digest`;
- `implementation_id` (name, version, source revision/build digest);
- `parent_representation_ids`.

The digest is over the exact bytes in that node's coordinate space. Never reuse a digest field to mean both an encoded body and a decoded body. If a no-op transform produces identical bytes, retain separate stage identity and equal digest/length rather than silently skipping the declared stage.

## 3.1 Cross-field invariants (not expressible by shape alone)

The JSON Schema is an envelope/shape check, not a complete semantic validator. A verifier must additionally enforce:

- representation IDs are unique; all parent, transformation and mapping references resolve exactly once;
- the derivation graph is acyclic, stage ordinals are contiguous, and each stage input is the preceding authorized output;
- a transformation's input/output digests and lengths equal the corresponding representation records;
- an identity mapping has identical input/output bytes and digest; range-map and semantic-node-map stages reference a map artifact with matching input/output representation IDs and digests;
- a `field_selection.source_span` is measured in `json_source_utf8_bytes`, references `source_representation_id`, is within its byte length, and has `start_byte < end_byte`;
- `token_representation_id` identifies the exact extracted `json_raw_value_token_bytes`; its digest equals `raw_token_sha256`, and the original source slice equals those token bytes;
- a `semantic_node_map` links one unique source node and JSON Pointer to one unique canonical node; its source span must equal the raw field selection;
- the typed conversion's `input_raw_token_sha256` equals the selected token digest and the output digest equals its typed-value representation;
- the predicate public-input digest equals the final predicate-input representation; audience, purpose, request challenge and uniqueness scope are separately bound;
- the session response ordinal, direction, server identity and commitment ID match the upstream verified opening evidence.

The schema's `required` fields and enums cannot replace these relational checks. Unknown, duplicated, dangling or mismatched edges are reject/indeterminate, never inferred from names.

## 4. Transformation records and mapping modes

Each transformation records:
- stable stage ordinal and operation identifier;
- input and output representation IDs/digests/lengths;
- decoder/parser/canonicalizer profile ID and digest;
- implementation/build identity;
- parameters and resource limits;
- mapping mode and (when applicable) a source-map artifact digest;
- deterministic outcome and failure disposition.

Allowed mapping modes:

### 4.1 `identity`

The bytes are unchanged. The map is the identity relation over the entire input range. This still receives a stage record and an output representation ID.

### 4.2 `range_map`

The transform declares a piecewise mapping between output ranges and input ranges. The map must be complete for every selected output span, ordered, bounds-checked, and bound to both representation digests. Ambiguous, overlapping or missing mappings are indeterminate.

### 4.3 `derived_whole_input`

There is no byte-for-byte inverse mapping (for example, decompression). The output is a deterministic function of the entire authenticated input representation under a pinned transform profile. Qualification requires:
- the complete input representation is authenticated;
- the decoder and parameters are pinned;
- input/output digests and lengths are bound;
- resource limits and decoder failure modes are explicit;
- a verifier can independently replay the transform or verify an independently qualified transform proof.

This mode proves derivation from the committed input only when replay/proof requirements are met. It does not claim a selected decoded span is a direct transcript opening.

### 4.4 `semantic_node_map`

A semantic transformation such as JCS canonicalization changes byte layout, member order and possibly token spelling. It therefore cannot supply a general byte-offset map.

Instead it may provide a node-origin map:
- stable node identifier in the admitted source parse tree;
- canonical node identifier / canonical output range;
- original JSON token byte span and raw token digest;
- source and canonical representation digests;
- parser and canonicalizer profile digests.

A selected canonical node is acceptable only when the verifier can resolve that exact node back to one unique admitted source node and its raw token span. If this mapping is absent or ambiguous, the claim is indeterminate. A canonical byte offset is never a transcript byte offset.

### 4.5 `none`

No provenance-preserving mapping exists. This is not sufficient for a source-derived claim. The downstream result may be used only as an untrusted hint, not as qualified evidence.

## 5. Recommended CPIR ordering

For JSON source claims, the simplest auditable path is:

1. Verify the TLSNotary commitment opening for the requested response direction/ordinal and the exact HTTP bytes or owner-defined committed segment.
2. Parse the HTTP message under an explicit protocol/framing profile and bind the selected response entity body.
3. Apply transfer decoding and Content-Encoding decoding as separate, pinned transformation stages. Enforce maximum body size, maximum expansion ratio, nesting/time/memory limits and trailing-data policy.
4. Validate UTF-8 and feed the exact decoded JSON bytes to duplicate-aware admission. Preserve raw bytes; do not construct an ordinary map before duplicate detection.
5. Resolve the verifier-authorized RFC 6901 pointer against the admitted tree. Record the original JSON UTF-8 byte span, raw token digest and node identity.
6. Apply a versioned typed conversion (currency scale, decimal arithmetic, time-zone/offset state, units, freshness) without locale or floating-point ambiguity.
7. Canonicalize the CPIR plan and typed value for deterministic identity. Keep this semantic digest distinct from source-byte digests and field-span digests.
8. Bind predicate program, audience, purpose, request challenge, uniqueness scope and verifier key into the proof/receipt.

A different order may be specified, including canonicalize-before-extraction as in Scribe, but then it must include the semantic node-origin map in §4.4. Canonicalizing first does not remove the need to prove which original source field was used.

### 5.1 Stage identity and HTTP extraction split

The operation registry distinguishes `http_message_extract` (selecting the response/message from the authenticated transcript representation) from `http_body_extract` (selecting the entity-body range from that message). These are not interchangeable: the former binds response ordinal/direction; the latter binds framing/header interpretation and body coordinates. A transformation must not silently combine them into one uninspectable parser call.

## 6. HTTP transformation requirements

HTTP framing and content decoding are separate stages. A profile must state its supported protocol/version and reject unsupported ambiguity. At minimum, tests must cover:
- HTTP/1.1 fixed Content-Length and chunked transfer coding;
- malformed/truncated chunks, duplicate/conflicting length/transfer metadata and illegal trailing bytes;
- content encoding identity and gzip (additional encodings only when explicitly pinned);
- concatenated gzip members, trailing compressed bytes, corrupt checksums and decompression-bomb limits;
- representation headers and response ordinal/direction binding;
- multi-response, interim response and redirect handling where present;
- CRLF and permitted whitespace behavior at the exact HTTP/JSON boundary.

Do not infer that the TLS plaintext segment is a complete HTTP response without an explicit parser result and response identity. Do not treat a decoded body as if it were the committed transcript bytes.

## 7. Canonicalization and standards boundary

JCS (RFC 8785) requires I-JSON-compatible input, including no duplicate property names, valid Unicode string data and numbers representable as IEEE-754 binary64. Property sorting is by UTF-16 code units. It preserves Unicode strings without normalization.

CPIR v1 may deliberately enforce a stricter integer-only claim profile, but must call it a restricted CPIR profile—not full RFC 8785 conformance. Decimal money and higher-precision values should be represented as strings and converted under an explicit typed conversion profile. Keep the raw number lexeme and source span until admission is complete.

JSON Pointer resolution follows RFC 6901. Escapes are decoded by replacing `~1` with `/` first, then `~0` with `~`; therefore `~01` becomes `~1`, not `/`. No Unicode normalization is performed when comparing member names.

References:
- RFC 8259: https://www.rfc-editor.org/rfc/rfc8259.html
- RFC 7493 (I-JSON): https://www.rfc-editor.org/rfc/rfc7493.html
- RFC 8785 (JCS): https://www.rfc-editor.org/rfc/rfc8785.html
- RFC 6901 (JSON Pointer): https://www.rfc-editor.org/rfc/rfc6901.html
- RFC 9557 (updates RFC 3339 offset semantics): https://www.rfc-editor.org/rfc/rfc9557.html
- Scribe §3.2.2 and §4.2.2 (canonical extraction and stated attestation-node trust): https://link.springer.com/article/10.1007/s12083-026-02246-2

## 8. Adversarial and metamorphic test matrix

| Test class | Positive control | Negative control / expected result |
|---|---|---|
| Identity mapping | Identity transform preserves digest and offsets | Declared identity with changed output bytes → reject |
| HTTP framing | Valid Content-Length and chunked body produce expected entity bytes | Conflicting lengths, truncated chunk or forbidden trailer → reject |
| Content decoding | Valid gzip yields frozen decoded body digest | Corrupt checksum, trailing data or expansion limit → reject |
| Response binding | Correct response ordinal/direction | Valid opening from another response/direction → reject |
| Field span | Pointer selects exact raw JSON value token | Same pointer with wrong representation digest or shifted range → reject |
| Unicode | Multi-byte and supplementary-plane string spans are UTF-8 byte-correct | Unpaired surrogate, invalid UTF-8 or duplicate decoded name → reject |
| Pointer | `/a~1b/~0key` and `/~01` resolve exactly | `~2` or dangling `~` → reject pointer |
| JCS relation | Member-order/whitespace variants have equal JCS bytes | Source digests must remain distinct; canonical range cannot be used as source range |
| Node-origin map | Canonical node maps to one raw source token | Missing, ambiguous or substituted node mapping → indeterminate |
| Typed conversion | ZAR decimal string maps to minor units under pinned ISO 4217 snapshot | Excess precision, changed exponent table or float round-trip → reject |
| Timestamp | Offset-equivalent timestamps preserve instant and offset-state semantics | Unknown/known offset state silently collapsed → reject |
| Dependency substitution | Same pinned decoder/parser/schema/conversion digests replay | Any selected dependency digest changed → requalification required |

Every failure must preserve the original raw bytes, minimized reproducer, all stage records, expected/actual dispositions and adjudication. Do not normalize outputs before comparing them.

## 9. Performance protocol

Extend #12423, which owns source-session and end-to-end workload comparisons. This document owns per-transform attribution.

Measure separately:
- transcript opening verification;
- HTTP message extraction;
- transfer decoder;
- content decoder;
- UTF-8 validation and duplicate-aware parse;
- pointer resolution and raw-span production;
- JCS and node-origin-map generation;
- typed conversion;
- full time from authenticated bytes to qualified predicate input.

Report p50/p95/p99, throughput, peak RSS, allocation count, bytes copied, input/output sizes, expansion ratio, CPU, energy and stage critical path. Include identity and gzip; chunked and non-chunked; 1 KiB, 4 KiB, 16 KiB, 64 KiB, 256 KiB and 1 MiB. Use at least 1,000 measured samples per primary cell before a p95 superiority claim. Do not compare MPC, proxy, delegated attestation or TEE results as trust-equivalent.

## 10. Exit criteria and claim ceiling

A source-map conformance claim requires:
1. Versioned coordinate-space registry and schema.
2. Pinned HTTP/framing/decoder/parser/conversion profiles.
3. Positive and negative controls for every accepted mapping mode.
4. At least two independent implementations agree on the transform chain and selected raw-source node/span.
5. Independent replay or proof of every non-identity derived stage.
6. Wrong-response, wrong-coordinate, wrong-digest, wrong-parser and wrong-node substitution controls all rejected.
7. Exact raw fixtures, expected digests, minimized counterexamples and reproducible execution receipts.

Passing this contract would establish only the tested transform-chain and source-field relation under pinned dependencies. It would not establish provider truth, TLS witness honesty beyond the chosen protocol assumptions, ZK soundness, privacy against all side channels, subject uniqueness, consent or institutional action authority.
