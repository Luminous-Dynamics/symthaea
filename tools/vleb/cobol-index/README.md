# VLEB COBOL indexer: read-only pilot

This crate is the first narrow COBOL capability increment. It indexes **fixed-format source text only** and emits source-located anchors for program IDs, divisions, COPY references, FD/SD declarations, data levels, PIC strings, selected USAGE forms, REDEFINES, and EXEC SQL/CICS blocks. Its JSON output is validated against `index-output.schema.json` and the synthetic `settlement.cbl` fixture by `tools/vleb/test_cobol_index_output.py`.

It is deliberately not a COBOL parser, compiler, copybook expander, control-flow analyzer, business-rule interpreter, or execution backend. A hit in the index means only that a textual anchor was found at the stated physical source location.

## Run

```sh
cargo test --locked --manifest-path tools/vleb/cobol-index/Cargo.toml
cargo run --locked --manifest-path tools/vleb/cobol-index/Cargo.toml -- \
  --format fixed tools/vleb/cobol-index/fixtures/settlement.cbl
```

The source format must be explicitly declared. It rejects free-format input, UTF-8 non-ASCII source records, tabs, unknown indicator-column bytes, and continuation records that could cause literal contents to be mistaken for syntax. It does not automatically decode EBCDIC source. Decode using a declared encoding before using this indexer, and preserve the original bytes and encoding metadata in the task evidence.

## Safety boundary

- Source columns are one-based and follow fixed-format physical columns 8–72; columns 73+ are ignored.
- Comment records are not indexed. Debug records are skipped unless `--include-debug` is specified.
- Continued literals are treated as opaque, with diagnostics; the tool does not attempt to reconstruct COBOL literal semantics.
- It does not resolve COPY books or REPLACING, determine actual PIC/USAGE storage layout, emulate packed decimal rounding, prove data flow, or establish runtime behavior.
- Any diagnostic makes the CLI return exit code 2 after emitting the JSON report. A clean indexing run returns 0. CLI usage/format errors return 64.
- Use only authorized source and fixtures. Keep source offsets and hashes so an auditor can verify exactly which bytes were indexed.

## Proposed next validation

Before extending toward grammar-level parsing, test index precision/recall on an independently reviewed, licensed or synthetic corpus: COPY names, PIC clauses, USAGE/COMP-3, REDEFINES, FD/SD, division boundaries, SQL/CICS anchors, comments, literals, continuation, identification columns and malformed records. Measure exact-anchor precision and recall per anchor type, diagnostic recall for ambiguous/unsupported records, and false positives from comments/literals. A target of 99% precision and 99% recall is a proposed gate for this *indexing task only*, not a target for full COBOL parsing or code generation.

A later stage must add a separate grammar-level parser and a compiler/runtime-backed verifier. Do not mark COBOL parsing, generation, execution or held-out benchmark performance supported based on this source indexer alone.
