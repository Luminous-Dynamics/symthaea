# Local Release Gates

Alpha.9 adds a local release-gate report shape.

A release gate combines:

- preflight findings,
- claim-audit findings,
- fixture intent,
- replay-plan presence,
- caveats.

It does not imply peer review, external hardware validation, quantum advantage, or Mycelix attestation.

## CLI usage

Run a gate over the smoke fixture:

`cargo run --bin symthaea-quantum-comp -- gate smoke-binding`

## Status values

- `Pass`: local checks found no blocking issue.
- `Warn`: local checks found caveats that must be preserved.
- `Block`: local checks found a blocking problem.
