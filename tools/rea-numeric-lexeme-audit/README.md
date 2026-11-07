# REA raw numeric-lexeme audit

A standalone, read-only diagnostic for exact JSON integer tokens before JCS/ECMAScript binary64 conversion. It is not a production dependency and does not change REA Evidence identity or claim dispositions.

## Why this exists

RFC 7493 section 2.2 says receivers cannot be expected to treat integer magnitudes outside plus/minus 9,007,199,254,740,991 as exact. RFC 8785 uses ECMAScript binary64 number serialization for canonical JSON. This tool inspects original numeric token bytes before conversion, preserving the exact integer lexeme and RFC 6901 pointer.

It uses serde_json's raw_value feature to retain the byte range of each nested JSON value, and decodes object keys for correct pointer escaping and duplicate detection. It never reconstructs a source lexeme from a floating-point value.

## Output

- schema: luminous.rea.numeric-lexeme-audit.v0
- source_raw_sha256: hash of the exact input JSON bytes.
- visited_values: number of raw JSON values traversed.
- numeric_precision_warnings: stable pointer-sorted records with exact integer_lexeme and magnitude_class.

Integer-form tokens only are audited. Decimal and exponent tokens remain separate JCS floating-point cases. Decimal strings are not interpreted as numbers.

## Limits

- Input: 64 MiB.
- Nesting depth: 128.
- Visited values: 1,000,000.
- Warnings: 100,000.
- Duplicate object names are rejected recursively, after JSON string escape decoding.
- No filesystem writes, extraction, network access or execution.

The CLI reads one JSON file and writes the audit report to stdout:

~~~sh
cargo run --manifest-path tools/rea-numeric-lexeme-audit/Cargo.toml -- input.json
~~~

## Tests

The finite suite covers safe-integer boundaries, signed unsafe values, u64::MAX, 120-digit values beyond u64, nested and escaped JSON Pointers, decimal/exponent/string controls, escaped duplicate names, malformed/trailing input, root scalar pointers, deterministic warning order and all resource limits.

A successful audit establishes only that the raw JSON document contains the reported lexical integer tokens at the reported paths. It does not establish numeric truth, domain validity, provider authenticity or exact semantics after downstream JCS processing. Integration with ClaimEvaluation is a separate future step and must disclose whether raw-token coverage is complete or unavailable.
