# Local Verification

## Fast crate-only test

```bash
cargo test -p symthaea-visual-compression-probe
```

## Smoke workflow

```bash
bash crates/domains/symthaea-visual-compression-probe/scripts/svcp-smoke.sh
```

## Manual smoke commands

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
  /tmp/svcp-pipeline \
  --json
```

## Cargo pitfall

Avoid:

```bash
cargo test symthaea-visual-compression-probe
```

That form filters test names; it does not select the package. In a large workspace it can compile unrelated crates and fail on unrelated workspace issues.
