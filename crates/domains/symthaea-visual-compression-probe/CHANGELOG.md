# Changelog

## 0.1.0-alpha.8

- Fixed the alpha.7 regression test scope bug: `packet_hash_survives_text_roundtrip` now uses the existing in-module `tiny_image()` helper instead of a missing `fixture_image()` helper.
- Added `docs/ALPHA8_SCOPE_FIX.md` documenting the compile failure and the intended local verification command.
- No packet format, CLI behavior, or codec behavior changed.

## 0.1.0-alpha.7

- Fixed `VisualMemoryPacket::stable_hash64()` so packet hashes survive `.svmp` write/read roundtrips.
- Changed packet hashes to hash the canonical persisted packet text rather than raw in-memory `f32` bits.
- Added `packet_hash_survives_text_roundtrip` regression coverage.
- Removed an unused import from `examples/build_corpus.rs`.
- Added `docs/ALPHA7_HASH_FIX.md`.
- Retained the alpha.6 PSNR infinity fix.

## 0.1.0-alpha.5

Workspace-hardening and operational smoke-test release.

Added:

- `svcp doctor` to print correct workspace verification commands.
- `svcp self-test` to run deterministic fixture encode/validate/decode/similarity checks.
- `svcp pipeline` to generate packets, manifest, benchmark CSV, similarity matrix, and JSONL summaries in one command.
- `scripts/svcp-smoke.sh` for a repeatable local smoke workflow.
- `docs/LOCAL_VERIFICATION.md` and `docs/ALPHA5_WORKSPACE_HARDENING.md`.

Clarified:

- Use `cargo test -p symthaea-visual-compression-probe` for crate selection.
- Avoid `cargo test symthaea-visual-compression-probe`, which is a test-name filter and can compile unrelated workspace targets.

## 0.1.0-alpha.4

- Added typed `EncodingParams` for integration-grade callers.
- Added `VisualMemoryPacket::encode_with_params()` with configurable topology levels.
- Added `CognitiveScanSummary` and `visual_summary()` for scan triage.
- Added topology complexity, HDC activation ratio, reconstruction PSNR, and cognitive memory class summaries.
- Added `RankedPacket`, `rank_packets()`, and manifest row/header helpers for downstream corpus tools.
- Added CLI `summary`, `batch-encode`, and `matrix` commands.
- Added docs for alpha.4 integration and regression workflow.
- Kept the crate dependency-light and claim-disciplined: still not a production image codec.

## 0.1.0-alpha.3

- Added packet validation through `VisualMemoryPacket::validate()` and `svcp validate`.
- Added stable non-cryptographic `image_hash64` and `VisualMemoryPacket::stable_hash64()` helpers for regression fixtures and corpus indexing.
- Added `edge_energy()` as a simple visual-structure proxy for comparing scan changes.
- Added `svcp diff` for packet-to-packet similarity and metric deltas.
- Added `svcp sweep` for block/keep parameter sweeps with CSV/JSON outputs.
- Added `svcp index` for TSV/JSON packet corpus indexes.
- Added `svcp corpus-benchmark` for repeatable corpus reports.
- Expanded command help and claim-discipline workflow around baselines and repeatable reports.

## 0.1.0-alpha.2

Improves the prototype from a sparse-reconstruction demo into a visual-memory experiment.

Added:

- `PacketMetrics`
- `PacketSimilarity`
- `BenchmarkReport`
- `VisualMemoryPacket::metrics()`
- `topology_similarity()`
- `packet_similarity()`
- `benchmark_image()`
- CLI `metrics`
- CLI `fingerprint`
- CLI `benchmark`
- CLI `query`
- JSON output mode for automation
- format documentation
- experiment plan

Still intentionally missing:

- production binary packet format
- external image formats beyond PGM
- real HDC monorepo type integration
- FEP residual codec
- persistent Laplacian summaries
- corpus-level benchmarks

## 0.1.0-alpha.1

Fixed monorepo-ready packaging under `crates/domains/symthaea-visual-compression-probe/`.

## 0.1.0-alpha.0

Initial proof-of-concept package.
