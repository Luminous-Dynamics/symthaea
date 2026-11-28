# Scientific artifact delimited-text intake v0

Status: first bounded CSV/TSV parser implementation. It is a parsing adapter only; it does not infer units, establish source authenticity, or qualify scientific claims.

## Contract alignment

This crate implements the CSV_CELL locator from [#13900](https://github.com/Luminous-Dynamics/symthaea/issues/13900) and the manifest proposal in [PR #13901](https://github.com/Luminous-Dynamics/symthaea/pull/13901).

- Input bytes are checked against the configured byte cap before hashing, UTF-8 decoding or scanning.
- Artifact and acquisition receipt SHA-256 values must both match the exact supplied bytes.
- Header names, data-record widths, quoting and UTF-8 are validated strictly.
- Each observation records a zero-based data-record/column index and an exact inclusive/exclusive source byte span.
- Raw lexemes are preserved byte-for-byte as UTF-8 strings; quoted/escaped values also expose a separately decoded value.
- Numeric-looking values retain their original spelling (for example, 21.50); no float conversion is performed.
- Empty cells are UNKNOWN, not NULL. Tokens such as NA and null are not silently coerced.
- Unit inference is intentionally absent. unit_lexeme remains null.
- Any malformed row or resource-limit breach fails the whole artifact with zero observations; partial output is not admitted.
- The derived observation payload has deterministic, length-prefixed SHA-256 canonicalization.

## Supported subset

- CSV with comma delimiter and RFC4180-style double-quoted fields.
- TSV with tab delimiter and the same quoting rules.
- LF and CRLF record separators.
- UTF-8 only.

The adapter rejects bare carriage returns, malformed quoting, duplicate/empty headers, inconsistent row widths, invalid UTF-8 and unsupported input sizes.

## Run

```sh
cargo test --manifest-path tools/scientific-artifact-delimited/Cargo.toml
cargo fmt --manifest-path tools/scientific-artifact-delimited/Cargo.toml -- --check
cargo clippy --manifest-path tools/scientific-artifact-delimited/Cargo.toml --all-targets -- -D warnings
```

## Claim boundary

Tests verify parser determinism and contract behavior on synthetic byte strings. They are not a scientific benchmark, a source-authenticity test, a unit parser, or a PDF/XLSX/Parquet/FASTA/PDB adapter.
