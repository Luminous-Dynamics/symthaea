# symthaea-rea-artifact-reader

A standard-library-only, bounded, read-only ZIP metadata inventory prototype for REA-compatible artifact work.

## Scope of this slice

- Preflight the EOCD / ZIP64 end records before allocating entry metadata.
- Bound archive bytes, central-directory bytes, entry count, names, extra fields, and declared uncompressed sizes.
- Preserve each central-directory occurrence independently, including duplicate raw names.
- Preserve raw filename bytes and decoded path separately.
- Decode UTF-8 names and CP437 fallback names; honor a CRC-valid Unicode Path extra field.
- Validate every listed local header against its central-directory record.
- Parse and compare data-descriptor CRC and compressed/uncompressed sizes (with or without the optional signature; 32-bit and ZIP64 widths).
- Reject overlapping local-header/compressed-data/data-descriptor extents.
- Capture file identity for both the open descriptor and selected path, and revalidate it.
- Report leading stubs, central-directory extent, trailing comment and opaque pre-central-directory bytes.

## Non-goals and claim ceiling

This is metadata inventory, not a full REA ArtifactReader replacement. It does not extract, decrypt, decompress, hash member bytes, or verify payload CRCs against decompressed bytes. It does parse data-descriptor metadata and checks its declared CRC/sizes against the central directory; that does not verify the member payload. It does not scan for unlisted local headers, validate Android APK Signing Blocks, inspect package signatures, or interpret native executable formats. It does not execute archive contents. It records encrypted members but never decrypts them.

Traversal is reported as partial until the omitted checks above are implemented or explicitly excluded by a reviewed policy. A central-directory CRC or size is a producer declaration, not an observation over payload bytes.

The parser rejects multi-disk archives and ambiguous EOCD candidates. It also rejects malformed local/central header pairs, ZIP64 arithmetic overflow, out-of-bounds extents and overlapping listed member extents.

## Build and test

The isolated harness compiles this exact source file without relying on unrelated root-workspace metadata:

```bash
cargo fmt --manifest-path tools/rea-artifact-reader-harness/Cargo.toml -- --check
cargo test --manifest-path tools/rea-artifact-reader-harness/Cargo.toml --locked
```

## Upstream contract

Pinned reference: [REA ArtifactReader](https://github.com/morluto/rea/blob/8250bf2320ea57aab4e664db4b4d44b8abd2004e/src/artifacts/ArtifactReader.ts), [REA ZipArtifactReader](https://github.com/morluto/rea/blob/8250bf2320ea57aab4e664db4b4d44b8abd2004e/src/artifacts/ZipArtifactReader.ts).

The next qualification stage must differential-test this prototype against upstream and a pinned mature ZIP parser. The mature parser is a second implementation/oracle, not the policy authority. Its documented duplicate-name limitation must not be used to collapse archive occurrences.
