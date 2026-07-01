# Fixtures and Replay Plans

Alpha.9 introduces named fixtures and replay plans.

Fixtures are stable local input configurations. They are not scientific golden results.

Replay plans are command lists and caveats for reproducing a local run.

## Fixture names

- `smoke-binding`: tiny wiring check, not benchmark evidence.
- `demo-binding`: small local demonstration.
- `pilot-binding`: pilot-sized run that should precede a larger replicated matrix.

## CLI usage

List fixtures:

`cargo run --bin symthaea-quantum-comp -- fixtures`

Print a replay plan:

`cargo run --bin symthaea-quantum-comp -- replay smoke`

Supported replay scopes match the named presets:

- `smoke`
- `local-research`
- `pilot-matrix`
