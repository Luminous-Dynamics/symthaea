# CPIR Parser Differential Oracle v1 — Research Profile

**Status:** research proposal and seed corpus; not a production parser contract  
**Owning issue:** [#12447 — SYM-CPIR-PARSER-ORACLE-001](https://github.com/Luminous-Dynamics/symthaea/issues/12447)  
**Related implementation:** [CPIR PR #12395](https://github.com/Luminous-Dynamics/symthaea/pull/12395)  
**Profile identifier:** `luminous.cpir.parser-oracle.v1`

This document defines the first executable comparison boundary for CPIR. It is intentionally narrower than a general JSON interoperability profile. The target is to show that distinct implementations derive the same verifier-authorized claim from the same authenticated source bytes—or reject the input before qualification.

No Rust test, differential run, fuzz campaign, benchmark, TLS session, proof, or production qualification is claimed by the presence of this document or its seed files.

## 1. The security proposition

The system must preserve this chain without substituting a value between stages:

```text
authenticated source bytes
  -> exact HTTP body/range
  -> duplicate-aware JSON admission
  -> structural JSON value
  -> verifier-selected JSON Pointer + raw byte span
  -> declared typed conversion
  -> predicate input
  -> proof statement / receipt
```

Each edge needs its own binding. A JSON Pointer string alone does not prove which bytes were opened. A parsed number alone does not preserve its original decimal spelling. A canonical digest alone does not prove that the input had no duplicate keys before parsing.

The oracle therefore compares at least four layers:

1. **Wire:** raw input bytes, source digest, byte offsets, HTTP framing/content-encoding metadata.
2. **Structure:** parser disposition, decoded member names, arrays, strings and JSON Pointer resolution.
3. **Meaning:** raw numeric lexemes, typed values, conversion profile and predicate operands.
4. **Canonical identity:** JCS bytes, plan digest, field-set digest and dependency-manifest digest.

Do not merge these digests or describe them as interchangeable.

## 2. Proposed profile rules to adjudicate

The following are proposed CPIR v1 rules, aligned with the current CPIR prototype's narrow admission intent. Each must be checked against the exact normative references and then frozen in the machine-readable profile before a compatibility claim.

### 2.1 Raw input before ordinary object construction

- Bound input size and nesting before accepting a document for expensive work.
- Require valid UTF-8 for the JSON text. Reject a UTF-8 BOM rather than relying on parser-specific BOM tolerance.
- Reject malformed escapes and unpaired UTF-16 surrogate escape sequences; do not replace them with U+FFFD.
- Detect duplicate object member names **after JSON string escape decoding** and before conversion into an ordinary map/object.
- Compare decoded names by exact Unicode scalar sequence. Do not apply Unicode normalization. Thus `"a"` and `"\u0061"` are duplicates; precomposed `"é"` and decomposed `"é"` remain distinct names.
- Reject trailing non-whitespace data after the one JSON value.
- Preserve the original bytes; never rewrite the authenticated source before the byte range/commitment relation is checked.

The duplicate-key fixture is a raw byte artifact. It must never be loaded and saved through a normal map/object serializer because that may erase the very condition under test.

### 2.2 Number admission and lexical provenance

- The CPIR v1 plan profile admits only integer numeric literals within the exact IEEE-754 safe-integer interval `[-(2^53-1), +(2^53-1)]`.
- Reject decimal-point and exponent-form numeric literals in the CPIR v1 plan and claim-value profiles. Decimal quantities must use an explicit string-plus-conversion profile.
- Retain the exact source lexeme and byte span for every selected claim number until the lexical admission decision is complete.
- Distinguish integer tokens, decimal/exponent tokens, numeric strings, and ordinary text. Do not reconstruct a raw lexeme from a binary floating-point value.
- Treat negative zero and exponent edge cases as explicit test vectors; the adjudicated profile must state whether each is rejected or accepted rather than inheriting a language runtime's default.

This section builds on the existing raw-number audit work in [#10951](https://github.com/Luminous-Dynamics/symthaea/issues/10951); it does not change REA Evidence identity semantics.

### 2.3 Canonicalization

- Use the exact RFC 8785 JSON Canonicalization Scheme implementation contract selected by CPIR.
- Duplicate names must be rejected before canonicalization. JCS is not a duplicate-key repair algorithm.
- Object member sorting must follow UTF-16 code-unit ordering, not UTF-8 byte order or Unicode scalar-value order.
- Preserve Unicode strings exactly as decoded; do not normalize.
- Reject values outside the selected I-JSON/JCS subset and reject unsupported floating-point claim values under the stricter CPIR profile.
- Preserve known-answer vectors and expected canonical bytes as immutable fixtures.

Reference: [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785.html).

### 2.4 JSON Pointer and byte spans

- Use RFC 6901 token escaping: `~` becomes `~0`, then `/` becomes `~1`.
- Reject malformed escape sequences; never treat `~2` or dangling `~` as a literal.
- Resolve pointers against the admitted structural tree, but separately retain the exact source token byte range.
- Byte offsets are UTF-8 byte offsets into the exact authenticated body, half-open `[start,end)`; they are not Unicode code-point or UTF-16 character indexes.
- Preserve absent, null, empty string, zero and empty array as distinct states. The schema validator must not silently inject defaults into the parsed source value.

### 2.5 HTTP-to-JSON boundary

The HTTP response body committed by the source protocol and the JSON text consumed by the claim parser must be connected by an explicit transformation record.

- Bind response ordinal, direction, content type, content encoding, transfer framing and selected byte range.
- If a content-decoding step is required, name and version the decoder and bind its output digest to the original committed bytes and response metadata.
- Never silently treat a decompressed body, normalized line ending, repaired UTF-8 string or reserialized JSON value as the originally committed transcript bytes.
- Unsupported, ambiguous or unbound transformations are reject/indeterminate outcomes.

Exact HTTP framing and content-decoding semantics remain coordinated with [#12328](https://github.com/Luminous-Dynamics/symthaea/issues/12328) / [PR #12336](https://github.com/Luminous-Dynamics/symthaea/pull/12336).

## 3. Independent implementation matrix

Initial candidate matrix:

| Implementation | Purpose | Qualification condition |
|---|---|---|
| Current CPIR Rust preflight | Reference candidate | Pin exact commit, Cargo.lock, compiler and feature flags; compile and execute tests |
| ECMAScript JSON runtime + independent JCS implementation | Cross-language comparator | Pin Node/runtime and canonicalizer independently; preserve pre-parse raw bytes |
| Second Rust parser path (candidate: simd-json) | Parser implementation diversity | Verify parser and canonicalizer dependency ancestry is materially independent |
| Python standard-library parser (optional) | Research-only comparator | Never a production dependency; do not claim independence from shared components without inspection |

A different language or wrapper is not automatically an independent oracle. Record parser, canonicalizer, runtime, dependency versions, source revision and shared implementation ancestry.

A majority vote is not an adjudicator. When outputs differ, preserve all raw outputs, minimize the input, and decide against the frozen CPIR profile and its normative references. Until resolved, the vector is indeterminate and cannot support production qualification.

## 4. Differential result record

Every implementation/vector pair should emit a deterministic record with these logical fields:

- `vector_id`
- `profile_id` and `profile_digest`
- `implementation_id`, source revision, runtime and dependency lock digest
- `raw_input_sha256` and byte length
- parser disposition and exact error stage/class
- duplicate-key findings, including decoded member name and object path
- structural output digest (for comparison only; never a replacement for raw bytes)
- per-field JSON Pointer, raw byte range, raw lexeme digest and typed conversion result
- JCS canonical bytes/digest where admission succeeds
- predicate input digest and dependency-manifest digest
- execution environment and measured resource usage
- expected disposition, adjudication state and reviewer/reference

Do not upload private source payloads or secrets into public fixtures. Use synthetic data; for real traces, preserve only suitably redacted data and a reproducible local procedure.

## 5. Initial seed vectors

| Vector | Raw input / characteristic | Proposed disposition or property |
|---|---|---|
| `dup-decoded-name-001` | `{"a":1,"\\u0061":2}` | Reject duplicate decoded member name before object construction |
| `unicode-normalization-distinct-001` | `{"é":1,"é":2}` | Keep the two code-point sequences distinct; do not normalize |
| `jcs-utf16-order-001` | Keys U+E000 and U+1F600 | Astral key sorts before U+E000 under UTF-16 code-unit ordering |
| `unsafe-int-positive-001` | `9007199254740992` | Reject under CPIR v1 safe-integer rule |
| `decimal-exponent-001` | `1e0` | Reject as a numeric token under CPIR v1 claim profile |
| `pointer-escape-001` | `{"a/b":{"~key":7}}` | Pointer `/a~1b/~0key` resolves to integer 7 |
| `pointer-invalid-001` | Pointer token containing `~2` | Reject invalid RFC 6901 escape |
| `absent-null-empty-001` | Absent member vs `null` vs `""` | Preserve three distinct source states |
| `utf8-span-001` | Selected string contains a multi-byte UTF-8 character | Byte offsets point to original UTF-8 bytes, not character indexes |
| `trailing-data-001` | One JSON value followed by a second token | Reject trailing non-whitespace data |

The duplicate-key raw fixture is stored separately at `fixtures/duplicate_decoded_name_001.raw.json`. It must remain raw source text.

## 6. Differential and metamorphic properties

For every supported implementation:

- Same exact raw input and profile must produce the same accept/reject/indeterminate disposition.
- Every admitted input must have identical decoded semantic structure.
- Every admitted JCS input must produce byte-identical canonical JSON and digest.
- Every selected field must resolve to the same pointer, raw byte range, raw lexeme and typed value.
- Changing a selected value, parser/schema/conversion profile, source response binding or predicate must change the relevant binding.
- Reordering object members or changing insignificant JSON whitespace must preserve JCS identity where RFC 8785 requires it, while the raw source digest must still change.
- Escaped-equivalent duplicate names must reject even if the underlying host parser would otherwise apply first-wins or last-wins behavior.
- Unicode normalization must not be silently applied.
- Reject-all is not a passing conformance result: each corpus slice needs valid positive controls, invalid negative controls and explicit indeterminate cases.

## 7. Fuzzing and performance protocol

Initial proposed campaign gates (not measured results):

- At least 10,000 deterministic mutations of frozen seeds per CI run.
- At least 1,000,000 generated inputs per pinned offline campaign.
- Grammar-aware generation plus mutation of escapes, key equivalence, numbers, arrays, nesting, pointers, framing and conversion metadata.
- Keep every mismatch as a fixture; minimize it without changing its semantic failure class.
- Track distinct mismatch classes, false acceptances, false rejections, crashes, timeouts and resource-bound violations separately.

Measure each stage independently across 1 KiB, 4 KiB, 16 KiB, 64 KiB, 256 KiB and 1 MiB inputs:

- raw admission and duplicate scan;
- parse and structural representation;
- JSON Schema validation;
- semantic validation and JSON Pointer selection;
- numeric lexeme and typed conversion;
- JCS canonicalization/digest;
- total CPIR compilation latency.

Report throughput, p50/p95/p99, peak RSS, allocations, bytes copied and environment. Establish the baseline before setting a throughput improvement target. Security gates must never be weakened to win latency.

## 8. Exit criteria

A research conformance claim requires:

1. Pinned, independently implemented parsers/canonicalizers, with dependency ancestry disclosed.
2. All adjudicated frozen vectors passing expected dispositions, including positive controls.
3. Zero unresolved critical/high semantic mismatches.
4. Byte-identical JCS output and plan digest across independent implementations for the admitted common subset.
5. Exact byte-span and typed-value parity for selected fields.
6. Reproducible fuzz corpus, minimized findings, raw outputs and environment receipts.
7. Actual execution evidence. Authored tests, schema inspection and skipped CI are not execution.

Even a full pass establishes conformance only for the pinned profile and tested corpus. It does not establish TLS source authenticity, proof soundness, notary honesty, credential currentness, subject uniqueness, privacy against every observer, consent, institutional authority or truth of the provider's statement.
