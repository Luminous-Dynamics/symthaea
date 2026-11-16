# CPIR parser-oracle corpus execution notes

**Status:** research-only fixture-integrity tooling; no parser-conformance claim.

## Verify the frozen corpus

From the repository root:

```sh
node research-tools/claim-plan-ir/conformance/verify_corpus_manifest.mjs
```

The checker uses only Node.js built-ins. It verifies:
- unique vector IDs and valid manifest/profile identity;
- safe fixture paths constrained to the conformance directory;
- exact raw fixture byte lengths and SHA-256 digests;
- declared zero-based, half-open byte spans against the original fixture bytes;
- the JCS metamorphic relation declares identical expected canonical output and distinct raw-source digests.

The checker does **not** parse JSON semantics, detect duplicate keys, implement RFC 8785, evaluate JSON Pointer, perform currency/time conversion, compare independent implementations, fuzz, or qualify source provenance. A successful run is corpus-integrity evidence only.

## Coverage in this tranche

The original three seeds are retained. New raw controls cover nested and array-contained duplicate names; safe-integer boundaries and unsafe/negative-zero/decimal/exponent lexical cases; unpaired surrogates and malformed escapes; JSON Pointer `~0`, `~1`, and `~01` behavior with exact UTF-8 byte spans; absent/null/empty/zero/empty-array distinctions; trailing top-level data; exact currency conversion requiring a pinned ISO 4217 snapshot; RFC 3339 offset equivalence and the unknown `-00:00` offset; and a whitespace/member-order pair whose expected JCS output is identical while raw source digests differ.

Raw fixtures in `ADVERSARIAL_RAW_V1.raw-lines` are one-line source byte strings. The line delimiter is excluded from each fixture's byte length and digest.

Currency and timestamp expected values are test vectors for future conversion adapters. They are not qualified until the corresponding conversion profile, table/runtime version and digest are pinned.

## Still not run

- Rust CPIR preflight/build/tests;
- independent ECMAScript parser/canonicalizer execution;
- independent Rust parser/canonicalizer execution;
- differential fuzzing or counterexample minimization;
- parser throughput, memory or latency benchmarking;
- TLS session, transcript opening, proof generation or verifier qualification.
