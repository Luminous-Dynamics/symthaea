# CPIR source-map v1 synthetic execution

**Claim ceiling:** synthetic structural composition only. This is not a TLSNotary session, authenticated transcript opening, HTTP implementation, JSON Schema validation, proof or production qualification.

## Reproduce

From the repository root, with Node.js 20 or later:

```sh
node research-tools/claim-plan-ir/source-map/verify_source_map_v1.mjs
```

The checker consumes the synthetic HTTP message, body/token range maps, node-origin map, source-map bundle and expected-vector manifest. It checks representation IDs and parent edges, stage ordering and digests, mapping artifact identity, raw body extraction, exact JSON token span, JCS node-origin uniqueness, typed-value input binding and predicate input digest. It then mutates the bundle and requires rejection of wrong coordinate, wrong span, wrong token digest, wrong response ordinal, missing maps, wrong profile digest, wrong pointer, ambiguous node mapping, broken parent edge and wrong source digest.

The positive control intentionally uses an identity content decoder and a simple JSON value whose JCS output is byte-identical to its raw input. It does not exercise gzip, chunked transfer decoding, non-identity canonicalization, Unicode source-map expansion, or a real TLSNotary commitment. Those require separate fixtures and actual decoder/upstream implementations.

## Known limitations

- The synthetic transcript is merely a byte fixture with a placeholder session/commitment identity; no cryptographic source witness is present.
- The synthetic HTTP extraction map is a hand-authored range fixture, not the result of an HTTP parser.
- The first end-to-end bundle's JCS output happens to be byte-identical to its source. A separate non-identity fixture now covers member reordering and whitespace: source `{"name":"A", "age":30}` (22 bytes) canonicalizes to `{"age":30,"name":"A"}` (21 bytes); `/name` maps from source `[8,11)` to canonical `[17,20)`. Its node-origin map preserves the same raw token digest across the two coordinate spaces.
- The bundle's type/predicate digests are synthetic binding values; this does not execute a ZK program.
- The JSON Schema was checked for JSON parseability, required-property shape coverage and local $ref resolution. A full Draft 2020-12 schema engine was not run.
- The source-map checker itself still needs exact-branch runtime execution and independent review. Syntax inspection alone is not an execution result.

## Non-identity JCS vector

The separate `JCS_NONIDENTITY_V1.json` vector and `verify_nonidentity_jcs_map.mjs` exercise the case where canonicalization changes both total length and selected-node offset. The local independent smoke model recomputed the source, canonical, token and node-map SHA-256 values; checked that both byte ranges contain the same raw token; checked the restricted known-answer JCS subset; and rejected all eight mutation controls. The committed Node runner has only been syntax-checked, not runtime-executed from the exact branch.

## Next conformance increments

1. Non-identity JCS source-map fixture with whitespace, member reordering and escaped strings.
2. HTTP chunked and gzip vectors; replay each transform independently with bounded expansion.
3. Multi-response/direction substitution controls.
4. A Rust implementation of the envelope's relational checks, compared with this Node research checker.
5. Binding an actual TLSNotary verified commitment/opening through the owner API in #12373 / #12301.
