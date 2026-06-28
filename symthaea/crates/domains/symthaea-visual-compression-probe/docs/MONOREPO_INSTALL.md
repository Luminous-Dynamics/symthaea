# Monorepo Install Notes

This archive is packaged to be extracted from the Symthaea repository root.

## Extract

```bash
cd /srv/luminous-dynamics/symthaea
unzip /path/to/symthaea-visual-compression-probe-v0.1.0-alpha.2.zip
```

It will create:

```text
crates/domains/symthaea-visual-compression-probe/
```

## Register in workspace

If your root workspace uses explicit members, add:

```toml
"crates/domains/symthaea-visual-compression-probe",
```

If it already globs `crates/domains/*`, no member edit is needed.

## Verify

```bash
cargo test -p symthaea-visual-compression-probe
cargo run -p symthaea-visual-compression-probe --bin svcp -- --help
cargo run -p symthaea-visual-compression-probe --bin svcp -- benchmark crates/domains/symthaea-visual-compression-probe/fixtures/tiny_pump_scan.pgm --json
```

## Useful first commands

```bash
cargo run -p symthaea-visual-compression-probe --bin svcp -- \
  encode crates/domains/symthaea-visual-compression-probe/fixtures/tiny_pump_scan.pgm \
  /tmp/pump.svmp --block 8 --keep 10

cargo run -p symthaea-visual-compression-probe --bin svcp -- metrics /tmp/pump.svmp --json

cargo run -p symthaea-visual-compression-probe --bin svcp -- fingerprint /tmp/pump.svmp
```

## Intended lane

Keep this crate in the experimental perception / visual-memory lane until it has corpus benchmarks and monorepo integration tests.

It should not become a production image codec. It should become a queryable visual evidence packet for Symthaea, Symtropy Field Deck scans, and Chronicle repair artifacts.
