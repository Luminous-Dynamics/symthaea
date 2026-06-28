# symthaea-visual-compression-probe

## Alpha.7 packet-hash roundtrip note

Alpha.7 fixes the fixture workflow failure where `VisualMemoryPacket::stable_hash64()` changed after a valid `.svmp` text roundtrip. Packet hashes now hash the canonical persisted packet text, not raw in-memory `f32` coefficient bits.

```bash
cargo test -p symthaea-visual-compression-probe
```


## Alpha.6 test-fix note

Alpha.6 fixes the crate test that incorrectly rejected infinite PSNR. Infinite PSNR is the expected result when reconstruction is exact and MSE is zero. The test now rejects `NaN`, accepts `+inf`, and checks for high finite PSNR otherwise.

```bash
cargo test -p symthaea-visual-compression-probe
```


## Alpha.5 workspace-hardening quick start

The real Symthaea workspace log showed `summary`, `batch-encode`, and `matrix` compiling and running successfully. Alpha.5 adds commands to make verification less ambiguous:

```bash
cargo run -p symthaea-visual-compression-probe --bin svcp -- doctor
```

```bash
cargo run -p symthaea-visual-compression-probe --bin svcp -- self-test \
  crates/domains/symthaea-visual-compression-probe/fixtures \
  --json
```

```bash
cargo run -p symthaea-visual-compression-probe --bin svcp -- pipeline \
  crates/domains/symthaea-visual-compression-probe/fixtures \
  /tmp/svcp-alpha5-pipeline \
  --json
```

Use this for crate tests:

```bash
cargo test -p symthaea-visual-compression-probe
```

Avoid `cargo test symthaea-visual-compression-probe`; that filters test names and can compile unrelated workspace test targets.

A dependency-light Rust prototype for **cognitive visual compression**.

This crate is not trying to beat JPEG, WebP, AV1, or image-specific production codecs. Its purpose is to test a Symthaea-native idea:

> Commercial codecs compress pixels so humans can look later.  
> Symthaea should compress structure so minds and machines can reason now.

The crate stores three kinds of visual memory:

1. **Spectral residual layer** — blockwise DCT-style sparse coefficients for approximate reconstruction.
2. **HDC signature layer** — deterministic binary hypervectors for query/retrieval without decoding pixels.
3. **Topology fingerprint layer** — thresholded Betti-style summaries for durable shape and anomaly detection.

The first supported image format is grayscale PGM (`P2` or `P5`) so the prototype can build with **zero external dependencies**.

## What alpha.4 adds

- typed `EncodingParams` for integration callers
- `visual_summary()` API and `svcp summary` for scan triage
- `svcp batch-encode` for folder-to-packet corpus creation
- `svcp matrix` for pairwise similarity matrices
- manifest helpers for reproducible packet indexes
- topology complexity and cognitive memory class summaries
- stronger docs for Symthaea/Symtropy integration

## What alpha.3 added

- `validate` command for strict packet consistency checks
- `diff` command for comparing two cognitive visual packets
- `sweep` command for parameter scans across block/keep settings
- `index` command for building TSV/JSON corpus indexes
- `corpus-benchmark` command for repeatable fixture/corpus reports
- stable non-cryptographic hashes for packets and images
- edge-energy proxy for visual structure/change detection
- stronger regression tests and fixture workflow
- clearer failure modes for claim-disciplined experiments

Alpha.2 additions remain: metrics, fingerprint, benchmark, query, JSON reports, HDC signatures, topology fingerprints, and prototype packet metrics.

## Commands

```bash
cargo run -p symthaea-visual-compression-probe --bin svcp -- inspect fixtures/tiny_pump_scan.pgm
cargo run -p symthaea-visual-compression-probe --bin svcp -- encode fixtures/tiny_pump_scan.pgm /tmp/pump.svmp --block 8 --keep 10
cargo run -p symthaea-visual-compression-probe --bin svcp -- decode /tmp/pump.svmp /tmp/reconstructed.pgm
cargo run -p symthaea-visual-compression-probe --bin svcp -- compare fixtures/tiny_pump_scan.pgm /tmp/reconstructed.pgm
cargo run -p symthaea-visual-compression-probe --bin svcp -- metrics /tmp/pump.svmp --json
cargo run -p symthaea-visual-compression-probe --bin svcp -- fingerprint /tmp/pump.svmp
cargo run -p symthaea-visual-compression-probe --bin svcp -- benchmark fixtures/tiny_pump_scan.pgm --block 8 --keep 10 --json
cargo run -p symthaea-visual-compression-probe --bin svcp -- query /tmp/pump.svmp /tmp/packet-corpus --top 5
cargo run -p symthaea-visual-compression-probe --bin svcp -- validate /tmp/pump.svmp --json
cargo run -p symthaea-visual-compression-probe --bin svcp -- sweep fixtures/tiny_pump_scan.pgm --blocks 4,8 --keeps 2,4,8
cargo run -p symthaea-visual-compression-probe --bin svcp -- index /tmp/packet-corpus /tmp/packet-index.tsv
cargo run -p symthaea-visual-compression-probe --bin svcp -- corpus-benchmark fixtures --block 8 --keep 10
cargo run -p symthaea-visual-compression-probe --bin svcp -- summary fixtures/tiny_pump_scan.pgm --json
cargo run -p symthaea-visual-compression-probe --bin svcp -- batch-encode fixtures /tmp/svcp-corpus --manifest /tmp/svcp-manifest.tsv
cargo run -p symthaea-visual-compression-probe --bin svcp -- matrix /tmp/svcp-corpus /tmp/svcp-similarity.csv
```

## Format

The `.svmp` file is a plain-text prototype format containing:

- dimensions
- block size
- kept coefficient count
- sparse block coefficients
- HDC block signatures
- topology threshold summaries

It is designed for debugging and review, not compact production storage yet.

See [`docs/FORMAT.md`](docs/FORMAT.md).

## Claim boundary

This crate does **not** claim state-of-the-art image compression.

It is a research probe for:

- query-without-decode visual memory
- infrastructure scan comparison
- false-green visual evidence retention
- topological anomaly fingerprints
- Symthaea/Symtropy Field Deck scan packets

The primary success metric is not “smallest pretty image.” The primary success metric is:

> Can Symthaea answer useful questions from the compressed representation without reconstructing pixels?

## Suggested next step

Wire this crate into the Old Waterworks vertical slice as a Field Deck scan artifact:

- capture pump scan as grayscale diagnostic image
- encode into `.svmp`
- retain HDC/topology packet in Chronicle evidence
- compare before/after repair scans without reconstructing pixels
- query prior packets to identify similar failures

## Monorepo placement

This archive is packaged under `crates/domains/symthaea-visual-compression-probe/` so it can be extracted from the Symthaea repository root. The crate exposes the `svcp` binary at `src/bin/svcp.rs` and fixture examples under `examples/`.

See [`docs/MONOREPO_INSTALL.md`](docs/MONOREPO_INSTALL.md) for exact commands.
