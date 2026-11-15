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
- similar sibling field selection;
- duplicate keys and escaped duplicate keys;
- RFC 6901 slash/tilde escapes;
- array indexing and noncanonical indices;
- raw numeric lexeme preservation;
- malformed pointer rejection;
- trailing junk rejection;
- document resource bound;
- duplicates in unselected nested subtrees.

All tests are authored but not executed. No compile or runtime result is claimed.
