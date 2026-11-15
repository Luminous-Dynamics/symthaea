# Symthaea ZKP JSON Field Binding V1

**Status:** Host-side research prototype, source authored on a research branch. Not executed. Not a TLSNotary verifier, not a transcript-opening verifier, and not a ZK parser.

## Purpose

Select an exact raw JSON value token using a verifier-supplied RFC 6901 JSON Pointer and retain its byte span without normalizing its JSON lexeme. This provides a deterministic host-side parser/range primitive for the field-semantic binding investigation in Symthaea #12328.

## Contract

- Uses serde_json borrowed RawValue support; it does not implement a new JSON grammar.
- Rejects duplicate decoded object member names anywhere in the document, including escaped spellings that decode to the same key.
- Enforces body size, nesting, node count, container count, key length and scalar token bounds.
- Supports RFC 6901 tilde-zero and tilde-one pointer escapes and canonical array indices.
- Returns the exact raw JSON token byte range within the supplied body, including string quotes and preserving number lexemes such as 1.00 versus 1e2.
- Does not return or hash the private field value in a public receipt.

## Partial structural disclosure and HTTP mapping

The implementation now supports one hidden string-content range represented as exactly one contiguous run of None bytes. The verifier parses the body after removing that run; the requested JSON Pointer must resolve to an empty string token, and the removed run must sit exactly between that token's two quote bytes. This binds the sole structural hole to the verifier-selected field under the declared serde_json profile. Duplicate decoded keys anywhere in the document reject.

The strict HTTP/1.1 helper admits only one status-200 response with exactly one Content-Length and Content-Type equal to application/json, no Transfer-Encoding, no Content-Encoding, no Trailer/Upgrade/Content-Range, and no trailing bytes. It maps body-relative ranges into offsets within the received application-data stream. It intentionally rejects chunked framing, compressed content, parameterized media types, partial responses, pipelining and duplicate headers.

A hidden string's actual contents are not validated by the structural skeleton. The downstream private predicate relation must constrain the exact hidden raw bytes to the claim schema (for example, ten ASCII bytes matching YYYY-MM-DD for a DOB), including any required JSON lexical restrictions. Generic hidden strings need a corresponding grammar proof. The parser profile/path result is not enough by itself.

## Crucial boundary

The returned range is relative to the exact JSON body byte slice supplied to the selector. The crate does not:
- parse HTTP framing;
- decompress gzip or other content encodings;
- convert charsets;
- map a decoded body back to its byte positions in the TLS transcript;
- verify a TLSNotary transcript commitment or proof opening;
- prove that the verifier selected the correct parser profile/pointer;
- prove business meaning, issuer authority, source truth or currentness.

A separate HTTP capture/framing layer must prove that the body slice is the exact representation in the authenticated TLS transcript and derive the body-start mapping. This function's range is only a parser selection candidate until that mapping and TLSN opening are independently verified.

## Tests authored

- exact nested field range;
- one hidden string field bound to the exact JSON Pointer;
- wrong-field hidden value rejection;
- rejection of multiple/no hidden ranges and quote-boundary removal;
- HTTP/1.1 response framing and exact body-to-received-stream range mapping;
- rejection of duplicate Content-Length, Transfer-Encoding, Content-Encoding, parameterized Content-Type, non-200 status, trailing bytes, malformed lengths, folded headers and oversized bodies;
- similar sibling field selection;
- duplicate keys and escaped duplicate keys;
- RFC 6901 slash/tilde escapes;
- array indexing and noncanonical indices;
- raw numeric lexeme preservation;
- malformed pointer rejection;
- trailing junk rejection;
- document resource bound;
- duplicates in unselected nested subtrees.

All tests are authored but not executed. No compile or runtime result is claimed. The Rust source currently includes 19 selector/partial-hole tests and 13 HTTP framing/composition tests; check the branch head for the exact authored count.


## Reproducibility

This standalone research tool is intentionally outside the root Symthaea workspace glob. Its direct dependencies are pinned to versions already present in the Symthaea Cargo.lock at the branch base:
- serde 1.0.228
- serde_json 1.0.150 with raw_value
- thiserror 2.0.18

The profile compares decoded JSON object keys by exact Unicode scalar sequence; it performs no Unicode normalization. Escaped JSON spellings are decoded by serde_json before duplicate detection, so two spellings that decode to the same key are rejected.

The tool does not claim compatibility with every parser or provider schema. Its parser identity is explicit and must be bound in the verifier request. The raw body supplied to this API must be the exact representation whose offsets are being referenced; compressed/chunked/decoded derivatives require a separately qualified mapping step.
