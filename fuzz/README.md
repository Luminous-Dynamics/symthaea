# REA ZIP differential fuzz target

This is a standalone Cargo-fuzz workspace, deliberately outside the large root Cargo workspace.

The initial target fuzzes the real ZipArtifactReader::open admission path. When a mutated archive is admitted, it also attempts bounded verification of up to eight eligible Stored/Deflate member occurrences. The target writes only its input bytes to a unique temporary file because the production reader binds source identity to an opened file; it never extracts archive members. A guard removes each temporary file.

Resource bounds are part of the fuzz contract:

- Input archive: at most 1 MiB.
- Central directory: at most 256 KiB.
- Entries: at most 256.
- Member name: at most 4 KiB.
- Extra fields: at most 64 KiB.
- Aggregate declared output: at most 4 MiB.
- Individual verified member: at most 256 KiB.
- Entries verified per input: at most eight.
- Read buffer: 4 KiB.

Run on Linux x86-64 with a nightly Rust toolchain and cargo-fuzz 0.13.2. Install with: cargo install cargo-fuzz --version 0.13.2 --locked.

Prepare the initial corpus with:
mkdir -p fuzz/corpus/zip_inventory_and_content fuzz/artifacts
base64 --decode fuzz/seeds/empty-stored.zip.b64 > fuzz/corpus/zip_inventory_and_content/empty-stored.zip

Run:
cargo +nightly fuzz run zip_inventory_and_content -- -max_total_time=180 -max_len=1048576 -rss_limit_mb=512 -print_final_stats=1

The weekly/manual workflow runs a bounded 180-second smoke campaign. It is not a security certification. Every crash must be minimized, checked into the permanent regression corpus and assigned an exact source/toolchain/lockfile receipt. Fuzz-clean means only that no crash was found in the exercised build, seed corpus and time budget.

Follow-on targets should separate archive admission, local/central mismatch handling, occurrence mapping, decompression/CRC verification and cross-parser differential behavior. Keep zipcore, zip, rawzip and pinned REA provenance separate; do not treat parser-name diversity as independent codec or checksum evidence.
