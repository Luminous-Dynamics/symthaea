# Alpha.3 Experiment Harness

Alpha.3 turns the crate from a codec sketch into a repeatable experiment harness.

## Core Principle

Do not ask whether this beats JPEG/WebP/AV1. Ask whether the packet lets Symthaea reason about visual structure without reconstructing pixels.

## New Commands

### Validate

```bash
svcp validate /tmp/pump.svmp --json
```

Checks dimensions, block counts, duplicate coordinates, coefficient bounds, finite values, and monotonic topology thresholds.

### Diff

```bash
svcp diff before.svmp after.svmp --json
```

Compares two packets using HDC + topology similarity and reports metric deltas.

### Sweep

```bash
svcp sweep fixtures/tiny_pump_scan.pgm --blocks 4,8,16 --keeps 2,4,8,12
```

Runs a block/keep parameter grid and prints CSV by default. Use this before making any compression claim.

### Index

```bash
svcp index /tmp/packet-corpus /tmp/packet-index.tsv
svcp index /tmp/packet-corpus - --json
```

Builds a corpus index with stable packet hashes and packet metrics.

### Corpus Benchmark

```bash
svcp corpus-benchmark fixtures --block 8 --keep 10
```

Benchmarks all `.pgm` files in a folder using the same parameters.

## Minimum Honest Report

For any experiment, report:

- input corpus and fixture hashes
- block size
- kept coefficients
- packet density
- text-to-raw ratio
- MSE / PSNR for reconstruction
- query-without-decode similarity behavior
- false positives / false negatives for retrieval
- failure examples

## Near-Term Use in Symthaea

Use alpha.3 to build a small visual-memory corpus:

1. Encode before/after repair scans.
2. Index packets.
3. Query similar prior failures.
4. Compare topology deltas after repair.
5. Store hashes and reports in Chronicle-style evidence logs.

The packet is not yet a production format. It is a measurement instrument.
