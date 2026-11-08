# REA schema-bound numeric-intent validator v0

Standalone Rust validation for schema-bound numeric intent. This is a read-only qualification utility, not a production dependency and not a change to REA Evidence v1 identity or claim dispositions.

## Why

JSON numbers have a source lexeme, a JCS/ECMAScript binary64 interpretation, and a domain intent. Those are separate contracts. A field can mean binary64, a safe exact integer token, an exact integer string, a fixed-scale decimal string, or an opaque numeric token. None of those meanings may be inferred from magnitude or spelling.

This crate composes the parent rea-numeric-lexeme-audit crate rather than replacing it. Every profile, value schema and document is audited for duplicate JSON keys and bounded traversal before parsing. The profile schema is Draft 2020-12 validated; the profile and domain schema are bound by JCS SHA-256 commitments.

## Run

```bash
cargo test --manifest-path tools/rea-numeric-intent-validator/Cargo.toml --locked
cargo run --manifest-path tools/rea-numeric-intent-validator/Cargo.toml -- \
  profile.json value-schema.json document.json
```

The embedded profile schema is stored under schema/rea-numeric-intent-profile-v0.schema.json and is copied from the research contract in PR #11047. Keep it byte-for-byte aligned with docs/research/rea-numeric-intent-profile-v0.schema.json until the research PR is merged and the canonical file can become the only copy.

## Validation rules

- The profile must pass the embedded Draft 2020-12 schema and v0 semantic checks.
- Duplicate json_pointer entries are rejected.
- The profile digest is SHA-256 over JCS canonical bytes of the profile object with profile_sha256 removed.
- The domain schema digest is SHA-256 over JCS canonical bytes of the complete schema.
- Both schemas must declare Draft 2020-12. External and file-based references are forbidden in v0; callers provide a self-contained schema.
- safe_integer requires an integer-form JSON token, rejects decimal/exponent spellings, and enforces bounds inside plus/minus (2^53-1).
- exact_integer_string and fixed_decimal_string operate on decoded JSON string values without binary64 conversion.
- binary64 reports the JCS canonical numeric representation.
- opaque_numeric confirms numeric token presence but grants no arithmetic, ordering or equivalence authority.
- Quantity-kind and unit references are preserved but reported as unresolved. This crate does not invent or mirror the canonical unit registry.

## Output and claim ceiling

Each declared path receives an independent status and precision-warning flag. The report binds the profile digest, domain schema digest and exact document raw SHA-256.

A conformant result establishes only path-specific representation, grammar, range and scale conformance to the supplied profile and schema. It does not establish value truth, calibration, source authenticity, suitability of a unit mapping, or downstream security/engineering correctness. A conformant report is not a signature, producer-authentication proof, or Evidence v1 identity.

## Qualification status

The focused workflow applies rustfmt and persists formatting-only source changes to the pull-request branch before running the format check, tests and warnings-denied Clippy. The resulting branch head is re-qualified after the formatting commit.

The included tests are authored for schema meta-validation, profile/digest checks, exact integer/decimal behavior, binary64 canonicalization, opaque no-arithmetic semantics, raw duplicate-key rejection and fixture structure. They require exact-head CI compile/test/Clippy qualification before any pass is claimed.