# REA member-verification receipt replay — research contract v0

**Owner:** #10492  
**Parents:** #10368, #10477, #10451  
**Status:** research contract; no replay implementation or parity result claimed.

## Objective

Treat a member-verification receipt as a falsifiable claim, not as evidence of its own correctness. A replay verifier consumes the original archive bytes and the raw receipt, independently rebuilds the relevant source and member observations, and reports whether the receipt is reproducible.

The verifier must not trust any receipt assertion about source hash, occurrence index, raw filename, offsets, sizes, CRC, payload digest, decoder result, source identity, toolchain or outcome until it has recomputed the corresponding fact.

This is a separate assurance plane from the receipt schema/JCS validator. A valid canonical receipt is not proof that the receipt describes the supplied archive.

## Independent parser candidate

Evaluate rawzip 0.5.2 as the primary low-level replay parser:
- ZipArchive::from_file reads a file-backed archive.
- entries iterates central-directory records in order and exposes an expected entry-count hint.
- Each central record yields a wayfinder; get_entry resolves the corresponding local record.
- The central record exposes raw file path, method, flags, CRC, size hints, local-header offset and central-directory offset.
- The local record exposes its own raw path, flags, method, CRC, size hints and extra fields.
- The entry exposes compressed data range, reader and verifying-reader abstractions.

rawzip is low-level and has no required runtime dependencies of its own. It intentionally assigns consumer policy to the caller: maximum compressed/decompressed sizes and ratios, overlapping extents, symlinks, duplicate filenames and exact central-directory count reconciliation. These are explicit replay responsibilities, never implicit guarantees.

For Deflate, pair rawzip's structure reader with a separately pinned flate2 using zlib-rs. Do not conflate rawzip parser independence, Deflate codec independence, ZIP CRC independence and source-content binding.

## Replay sequence

1. **Raw receipt intake.** Parse original receipt bytes with duplicate-key rejection at every JSON nesting level. Validate JSON Schema Draft 2020-12 and cross-field semantics. Validate RFC 8785 JCS and recompute the receipt statement digest if the caller supplied one.
2. **Archive binding.** Stream SHA-256 over the original archive and compare against source.archive_sha256. Record file size and independent file identity. A file identity tuple is not a content digest or immutable snapshot.
3. **Archive admission.** Open the archive through rawzip's file-backed API. Apply archive-size, central-directory, entry-count, per-entry, aggregate-output, compression-ratio, buffer and time limits before any decompression.
4. **Directory reconciliation.** Iterate central records in their raw directory order. Count each record—including duplicate names—and reconcile iterator completion with the EOCD/ZIP64 entry-count hint. Early termination, parse errors and unconsumed records must become rejection or abstention, never success.
5. **Exact occurrence selection.** Select the receipt's central-directory occurrence index. Compare exact raw path bytes, flags, method, declared sizes, CRC, local-header offset, central-directory position and duplicate-name context. Never select by decoded path string alone.
6. **Local-record binding.** Resolve the selected wayfinder, inspect the local header, compare local/central name/flags/method and the fields that ZIP rules require to agree. Explicitly model data descriptors and zero/placeholder local CRC/sizes permitted under streaming bit 3.
7. **Extent policy.** Record compressed-data ranges for every entry; check boundaries, duplicate/aliased extents and overlap. Enforce archive and ratio/output budgets. The replay tool never extracts or writes members.
8. **Streaming verification.** Support only unencrypted Stored/Deflate regular-file entries in the first lane. Read bounded buffers, compute SHA-256 and an independently implemented incremental CRC-32, enforce caller and declared member budgets, require decoder EOF and size/CRC acceptance, then compare recomputed values to the receipt.
9. **Source revalidation.** Recheck held-handle identity and selected path after replay. Non-Unix source identity remains unsupported until #10452 qualifies handle-bound identity. The pre-replay archive SHA-256 remains the byte-level replay binding.
10. **Decision.** Return one explicit outcome: replayed_match, replayed_mismatch, unsupported, source_changed, limit_exceeded or abstained. No unknown parser behavior, missing fields or unsupported format becomes success.

## Independence is a vector, not a boolean

Record separate lineage for:
- archive/container parser;
- compression decoder and backend;
- CRC implementation;
- SHA-256 implementation;
- receipt JSON parser and JCS canonicalizer;
- source-binding mechanism.

Rawzip uses a caller-provided decompressor. The replay build should use a distinct Deflate implementation (zlib-rs) from the original candidate decoder (miniz_oxide), and a separately implemented incremental CRC. Record versions, feature flags, dependency graph, lockfile digest, compiler and target. A different top-level crate does not automatically mean independent codec or checksum logic.

## Duplicate-name semantics

The receipt binds central-directory index plus exact raw filename bytes. Two entries may have identical raw names and distinct contents. The replay implementation must preserve those as separate occurrences. A unique-name map, path lookup, or HashMap is not an acceptable occurrence selector.

## Adversarial corpus

- duplicate raw names with different and identical payload bytes;
- decoded-name aliases, invalid UTF-8, CP437, Unicode extra field CRC mismatches and normalization collisions;
- central-directory count greater than, less than or inconsistent with the EOCD hint;
- malformed and truncated central records, ZIP64 end records and locator offsets, prepended stubs;
- local/central name, flags, method, size and CRC disagreement;
- data descriptors with and without the signature, truncated descriptors and zero local placeholders;
- duplicate local offsets, overlapping data extents, data ranges into central-directory bytes;
- Stored and Deflate truncation, malformed streams, false CRC, understated/overstated sizes;
- high compression ratio, aggregate expansion, huge entry count and large central-directory budgets;
- source replacement, same-size rewrite, timestamp restore and path race;
- receipt mutations to source digest, occurrence, raw name, offsets, method, sizes, CRC, payload SHA, verifier lineage, limits, outcome, claim scope and attestation reference;
- duplicate receipt properties, invalid base64, noncanonical decimal strings, unsafe numeric values and invalid JCS.

Every failing case must produce a minimized frozen archive + raw receipt pair and a permanent regression.

## Benchmarks

| Family | Target |
|---|---|
| Supported-subset receipt replay | 100% match on frozen successful fixtures |
| Receipt tamper detection | Zero false accepts across every one-field receipt mutation |
| Archive corruption | Zero false accepts for CRC, size, truncation, overlap, local/central mismatch and source mutation |
| Occurrence fidelity | 100% exact raw occurrence mapping, including duplicate names |
| Directory completeness | Zero success when the actual record count does not reconcile |
| Resource safety | Zero output beyond caller/member/aggregate limits; no overlap or high-ratio bypass |
| Independence | Report parser, codec, CRC and source binding separately |
| Reproducibility | 100% run receipts bind archive digest, receipt JCS digest, exact locks, toolchain and platform |
| Performance | p50/p95/p99 replay time, archive hash cost, parser cost, decode cost, peak RSS and energy per verified MiB |

Keep unsupported, mismatch, refused, skipped and abstained results in denominators. Do not collapse them into one composite score.

## Claim ceiling

A replay match establishes that a separately implemented verifier reproduced the receipt's scoped member-byte integrity claim for the exact supplied archive and occurrence. It does not establish producer authenticity, APK/IPA signatures, package installability, runtime behavior, archive semantic safety, benignness or MASVS compliance.

The signature on a future receipt or DSSE envelope authenticates the issuer's statement under a trust policy; it does not convert byte integrity into semantic truth.
