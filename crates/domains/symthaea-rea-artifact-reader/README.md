# symthaea-rea-artifact-reader

A bounded, read-only ZIP inventory and per-occurrence content-verification prototype for REA-compatible artifact work.

## Metadata inventory

The custom Rust parser is the policy authority for:

- Preflight of EOCD / ZIP64 end records and archive-size limits.
- Central-directory byte, entry-count, name, extra-field, and declared-size budgets.
- Per-occurrence identity in central-directory order, including duplicate raw names.
- Raw filename-byte preservation plus UTF-8, CP437 and CRC-valid Unicode Path decoding.
- Local/central header consistency and data-descriptor metadata checks.
- Checked member extents, overlap rejection, and held-descriptor plus selected-path source identity.
- Opaque-region accounting, including prepended stubs and bytes not mapped to listed entries.

## Per-member content verification

`ZipArtifactReader::verify_entry_content(occurrence_index, limits)` verifies one exact central-directory occurrence without extracting or writing it.

- The secondary decoder is pinned to `zipcore = 0.10.2` with default optional codecs disabled. SHA-256 uses `sha2 = 0.10.9`.
- The mature `zip = 9.3.0` parser is a test-harness-only differential oracle; it is not a production dependency. The initial differential fixture covers central-directory order, duplicate raw names, raw local-header offsets, sizes, CRCs, header consistency, overlap detection and Stored payload bytes. Do not infer codec independence from parser diversity; record dependency ancestry by method.
- The generated harness lockfile currently resolves both `zipcore` and `zip` to the same `crc32fast = 1.5.4` checksum implementation. They are distinct container parsers but not independent CRC oracles. Symthaea's bitwise CRC routine is separate and has a standard `123456789` known-answer test; this still does not replace a second independently implemented decompressor for Deflate.
- The custom inventory remains authoritative. Decoder entry count, central-directory position, ZIP64 status, occurrence index, raw name, offset, flags, method, sizes, CRC, data-descriptor bit, encryption bit and file kind are compared before streaming.
- The first tranche supports Stored (method 0) and Deflate (method 8) only.
- Encrypted, directory and symlink entries are explicitly unsupported. No password or decryption path is exposed.
- Decompressed bytes are read through a fixed-size buffer, hashed incrementally, and bounded by both caller budget and declared uncompressed size.
- A result is returned only after EOF and decoder CRC/size verification. A partial read is not a successful verification.
- The held descriptor and selected path are revalidated after the attempt; source change discards the result.
- Source-stability qualification is currently platform-gated: content verification is enabled only on Unix, where the admitted descriptor and selected path are checked using device, inode, size, mode, mtime and ctime. On non-Unix, metadata inventory remains available using a weaker best-effort identity, but content verification returns explicit `Unsupported` until handle-bound identity is qualified under [#10452](https://github.com/Luminous-Dynamics/symthaea/issues/10452).
- Each verification call opens independent decoder and identity-witness descriptors instead of sharing the inventory descriptor's file cursor; concurrent calls do not race through a shared seek position.
- A deterministic 256-sample, three-bit-flip mutation smoke test exercises admission and, when admitted, content verification under panic capture; this is not a substitute for coverage-guided fuzzing.
- `declared_compressed_bytes` is metadata, not a measured compressed-byte count.
- Per-entry verification does not make archive traversal complete. Unlisted local-header scanning, APK Signing Block interpretation, package-signature checks and full REA ArtifactReader parity remain separate work.

The selected decoder is a second implementation, not proof of correctness or a substitute for an independent differential oracle. Its own tests and security claims are not independent qualification. Keep a separately pinned mature parser and the upstream REA implementation as distinct oracles; preserve disagreements.

## Claim ceiling

A successful result supports only this statement: this named member occurrence was streamed to EOF from the revalidated admitted source, its observed decompressed byte count matched the declared size, the selected decoder's CRC check passed, and the observed decompressed bytes have the returned SHA-256.

It does not establish APK signing validity, IPA code-signing validity, package installability, runtime behavior, producer authenticity, target truth, whole-archive completeness or security compliance.

## Build and test

The isolated harness compiles the exact source file without relying on unrelated root-workspace metadata:

```bash
cargo fmt --manifest-path tools/rea-artifact-reader-harness/Cargo.toml -- --check
cargo test --manifest-path tools/rea-artifact-reader-harness/Cargo.toml --locked
cargo clippy --manifest-path tools/rea-artifact-reader-harness/Cargo.toml --locked --all-targets -- -D warnings
```

## Upstream contract

Pinned reference: [REA ArtifactReader](https://github.com/morluto/rea/blob/8250bf2320ea57aab4e664db4b4d44b8abd2004e/src/artifacts/ArtifactReader.ts), [REA ZipArtifactReader](https://github.com/morluto/rea/blob/8250bf2320ea57aab4e664db4b4d44b8abd2004e/src/artifacts/ZipArtifactReader.ts).
