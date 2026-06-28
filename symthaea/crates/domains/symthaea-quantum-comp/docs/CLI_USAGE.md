# CLI Usage

Alpha.9 includes a minimal dependency-free CLI.

## Commands

Run a binding probe:

`cargo run --bin symthaea-quantum-comp -- binding smoke`

Run a noise sweep:

`cargo run --bin symthaea-quantum-comp -- noise smoke`

Run an experiment matrix:

`cargo run --bin symthaea-quantum-comp -- matrix smoke`

List presets:

`cargo run --bin symthaea-quantum-comp -- presets`

List schema labels:

`cargo run --bin symthaea-quantum-comp -- schemas`

List fixtures:

`cargo run --bin symthaea-quantum-comp -- fixtures`

Print a replay plan:

`cargo run --bin symthaea-quantum-comp -- replay smoke`

Run a local release gate:

`cargo run --bin symthaea-quantum-comp -- gate smoke-binding`

## Presets and replay scopes

- `smoke`
- `local-research`
- `pilot-matrix`

## Fixtures

- `smoke-binding`
- `demo-binding`
- `pilot-binding`

## Claim boundary

The CLI runs local research probes only. It does not claim quantum consciousness, quantum advantage, physical backend execution, physical entanglement, or Mycelix attestation.
