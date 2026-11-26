# Symthaea RRVR Shadow Observer v0.1

A read-only, deterministic first slice of the RRVR-Symthaea integration.

## What it does

- Validates a versioned manipulation-episode snapshot.
- Preserves separate event, acquisition and ingest semantics.
- Enforces an immutable ingest-sequence decision cut and matching evidence-ledger frontier.
- Binds the canonical snapshot content to the digest in an external admission receipt.
- Abstains on stale, unknown or disputed observations rather than filtering them.
- Uses the existing 21-channel `symthaea-manipulator` state and 16,384D HDC encoder to summarize adjacent-state representation similarity.
- Emits a constant-velocity joint-angle forecast as an **uncalibrated baseline**, not a probability or safety claim.

## What it does not do

- It does not verify issuer signatures or evidence-ledger authority. It validates receipt shape, declared scope and snapshot digest consistency only.
- It does not compare timestamps across different clock domains.
- HDC similarity is not an anomaly probability.
- Constant-velocity extrapolation is not a qualified robot dynamics model.
- The observer has no actuator, dispatch, retry, permit, endpoint-fence or evidence-ledger mutation interface.
- It does not establish a physical effect, physical observation, simulator qualification, RoboRecover reproduction or robot safety.

## Run

From the Symthaea workspace root:

```bash
cargo fmt --check -p symthaea-rrvr-observer
cargo test -p symthaea-rrvr-observer
cargo clippy -p symthaea-rrvr-observer --all-targets -- -D warnings
```

The first qualification tier is software/process-model only. Simulator, HIL, physical and host-benchmark evidence require separate adapters and evidence.
