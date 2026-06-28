# Alpha.5 Workspace Hardening Notes

Alpha.5 is a small operational hardening pass prompted by real workspace output.

## What the local log proved

The alpha.4 crate compiled and the following commands ran successfully in the Symthaea workspace:

- `summary`
- `batch-encode`
- `matrix`

The later failure came from this command form:

```bash
cargo test symthaea-visual-compression-probe
```

In Cargo, that is **not** a package selection command. It is interpreted as a test-name filter, and Cargo may still compile unrelated workspace test targets. In the observed log, the failure was in `symthaea-probe-stream`, which was missing `adapter`, `backends`, and `recorder` modules.

Use this instead:

```bash
cargo test -p symthaea-visual-compression-probe
```

For a faster smoke path that does not rely on workspace-wide tests, use:

```bash
cargo run -p symthaea-visual-compression-probe --bin svcp -- self-test \
  crates/domains/symthaea-visual-compression-probe/fixtures \
  --json
```

## New commands

### `doctor`

Prints the correct workspace commands and explains the common `cargo test` pitfall.

```bash
cargo run -p symthaea-visual-compression-probe --bin svcp -- doctor
```

### `self-test`

Runs a deterministic fixture smoke test through encode, validate, decode, similarity, and PSNR checks.

```bash
cargo run -p symthaea-visual-compression-probe --bin svcp -- self-test \
  crates/domains/symthaea-visual-compression-probe/fixtures
```

### `pipeline`

Runs the practical corpus workflow in one command:

- encodes all `.pgm` images
- writes `.svmp` packets
- writes `manifest.tsv`
- writes `benchmark.csv`
- writes `similarity.csv`
- writes `summaries.jsonl`

```bash
cargo run -p symthaea-visual-compression-probe --bin svcp -- pipeline \
  crates/domains/symthaea-visual-compression-probe/fixtures \
  /tmp/svcp-alpha5-pipeline
```

## Claim boundary

This is still not a production codec and not a claim to beat JPEG/WebP/AV1. It is a visual-memory experiment for checking whether compact structural packets are useful for retrieval, anomaly detection, and reasoning without pixel reconstruction.
