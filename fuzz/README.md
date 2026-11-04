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

Run on Linux x86-64 with Rust nightly dated 2026-11-03 and cargo-fuzz 0.13.2. The repository root rust-toolchain.toml pins Rust 1.96.0, so the fuzz build must explicitly override it with `+nightly-2026-11-03`. Install cargo-fuzz with: `cargo install cargo-fuzz --version 0.13.2 --locked`.

Prepare the initial corpus with:
mkdir -p fuzz/corpus/zip_inventory_and_content fuzz/artifacts
base64 --decode fuzz/seeds/minimal-stored-entry.zip.b64 > fuzz/corpus/zip_inventory_and_content/minimal-stored-entry.zip

Run:
cargo +nightly-2026-11-03 fuzz run zip_inventory_and_content -- -max_total_time=180 -max_len=1048576 -rss_limit_mb=512 -print_final_stats=1

The starter ZIP payload SHA-256 is `dd8a2bd8c16d808363a1d1e7560a4cc86494393c1a202a478db00f00ae53d099`. The workflow verifies this decoded digest with `sha256sum --check` and writes a `run-context.txt` artifact containing the exact source commit, pinned toolchain version, cargo-fuzz version, runner OS, Cargo.lock digest and seed digests.
The fuzz step captures combined stdout/stderr in `fuzz/artifacts/fuzz-output.txt`, extracts final execution/throughput/RSS counters into `fuzz/artifacts/fuzz-summary.txt`, hashes the raw log into the run context, and uploads those files with the corpus and crash artifacts. This preserves the campaign summary beyond transient GitHub Actions console logs.

The workflow uses the committed `fuzz/Cargo.lock` with `--locked`, records the lock SHA-256, and runs a bounded 180-second smoke campaign. It is not a security certification. Every crash must be minimized, checked into the permanent regression corpus and assigned an exact source/toolchain/lockfile receipt. Fuzz-clean means only that no crash was found in the exercised build, seed corpus and time budget.

Follow-on targets should separate archive admission, local/central mismatch handling, occurrence mapping, decompression/CRC verification and cross-parser differential behavior. Keep zipcore, zip, rawzip and pinned REA provenance separate; do not treat parser-name diversity as independent codec or checksum evidence.
