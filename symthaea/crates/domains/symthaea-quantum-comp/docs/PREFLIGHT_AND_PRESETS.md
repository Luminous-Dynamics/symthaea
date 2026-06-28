# Preflight and Presets

Alpha.7 adds preflight checks and named presets to reduce accidental misuse.

## Preflight checks

Preflight emits `Info`, `Warning`, or `Error` findings. Errors should block execution. Warnings should be copied into reports.

Checks include:

- zero dimensions
- low dimensions
- zero trials
- low trial counts
- invalid noise ranges
- invalid topology thresholds
- low sweep step counts
- low replicate counts
- empty matrix axes

Preflight does not prove a result is scientifically meaningful. It only catches common local mistakes.

## Presets

Preset names are stable lowercase labels:

- `smoke`
- `local-research`
- `pilot-matrix`

Use presets for examples, CI, tutorials, and notebook wrappers. Use explicit configs for serious experiments.
