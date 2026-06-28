# Alpha.7 Upgrade Notes

Alpha.7 is a practical hardening pass. It does not add stronger scientific claims. It adds local preflight checks, named run presets, versioned schema labels, a small CLI, and research bundle packaging.

## Added

- `preflight` module for local configuration warnings and blocking errors.
- `presets` module with stable `smoke`, `local-research`, and `pilot-matrix` profiles.
- `bundle` module for local research bundle packaging.
- `schema` module with stable alpha.7 report labels.
- Minimal `symthaea-quantum-comp` binary for binding, noise, and matrix runs.
- Examples:
  - `preflight_presets`
  - `research_bundle`
- Tests:
  - `tests/alpha7_tests.rs`

## Claim boundary

Alpha.7 still does not claim quantum consciousness, quantum advantage, physical backend execution, cryptographic receipt generation, or Mycelix source-chain publication.

## Why this matters

The crate is becoming easier to use as a research artifact instead of merely a set of interesting probes. Preflight checks help catch weak local runs before they become misleading reports. Bundles help collect manifest, result, audit, and receipt text into one lab-note artifact.
