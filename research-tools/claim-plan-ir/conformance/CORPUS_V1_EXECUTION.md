# CPIR parser-oracle corpus execution notes

**Status:** research-only fixture-integrity and second-parser tooling; no production-conformance claim.

## Run the checks

From the repository root:

```sh
node research-tools/claim-plan-ir/conformance/verify_corpus_manifest.mjs
node research-tools/claim-plan-ir/conformance/cpir_js_oracle.mjs
```

Both tools use Node.js built-ins only.

- `verify_corpus_manifest.mjs` checks manifest identity, unique IDs, safe paths, exact byte lengths/SHA-256, declared byte spans and metamorphic-relation metadata.
- `cpir_js_oracle.mjs` implements a research-only recursive-descent parser over raw bytes. It checks duplicate decoded member names before constructing object entry lists, preserves numeric lexemes and UTF-8 byte ranges, resolves JSON Pointer, checks a restricted integer-only canonicalization subset, and exercises the currency/time conversion vectors.

Neither script is a complete RFC 8785 implementation. The JS oracle does not establish agreement with the Rust CPIR path, a second independent Rust parser, the TLS transcript commitment, a proof system or a production verifier. A successful local run is one implementation's research result, not multi-implementation conformance.

## Corpus coverage

The corpus contains 20 vectors: top-level, nested and array-contained duplicate names; normalization-distinct Unicode names; JCS UTF-16 ordering; safe-integer boundaries, unsafe values, negative zero and decimal/exponent tokens; unpaired surrogates and malformed escapes; JSON Pointer `~0`, `~1` and `~01`; exact UTF-8 byte spans; absent/null/empty/zero/empty-array states; trailing top-level data; ZAR minor-unit conversion as a pinned-table obligation; RFC 3339/RFC 9557 timestamp semantics; and a whitespace/member-order metamorphic pair.

Raw line fixtures in `fixtures/ADVERSARIAL_RAW_V1.raw-lines` are LF-delimited. The LF delimiter is excluded from each vector's digest and byte length. Each manifest offset is zero-based, half-open and measured in the exact UTF-8 source fixture, never canonical JSON.

## Timestamp interpretation

RFC 9557 updates RFC 3339 section 4.3. It aligns `Z` with legacy `-00:00`: both express known UTC time with unknown local offset. `+00:00` means UTC is the preferred reference point. The three forms can denote the same instant while preserving different offset-state metadata. The corpus tests that distinction explicitly.

## Still not qualified

- Rust CPIR preflight/build/tests;
- independent Rust parser/canonicalizer execution;
- full RFC 8785 conformance;
- multi-implementation differential agreement;
- differential fuzzing, minimization or long-running campaigns;
- parser throughput, memory or latency benchmarks;
- HTTP content decoding/source-coordinate continuity;
- TLS session, transcript opening, proof generation or verifier qualification.
