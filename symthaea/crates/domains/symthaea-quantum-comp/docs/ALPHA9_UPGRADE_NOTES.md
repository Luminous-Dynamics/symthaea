# Alpha.9 Upgrade Notes

Alpha.9 is a release-readiness and documentation-boundary upgrade.

It does not add a new quantum probe. Instead, it makes the crate easier to inspect, package, and integrate responsibly.

## Added

- `stability` module for alpha surface annotations.
- `api_inventory` module for dependency-free inventory reports.
- `release_manifest` module for blocked claims and recommended verification commands.
- CLI commands: `inventory` and `manifest`.
- Examples: `api_inventory` and `release_manifest`.
- Tests: `tests/alpha9_tests.rs`.
- Docs: API inventory and alpha.9 release manifest notes.

## Changed

- Crate version moved to `0.1.0-alpha.9`.
- Schema labels now end in `alpha9`.
- Local verification script exercises the new examples and CLI commands.

## Claim posture

Alpha.9 remains conservative. It still does not claim quantum consciousness, quantum advantage, external backend execution, or Mycelix attestation.
