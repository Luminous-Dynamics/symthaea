# Patch 0015: security run supply chain license and advisory gates

**Series:** 27

## Objective

Qualify the release dependency and redistribution boundary.

## Intended changes

- Record exact Rust, Nix, external verifier, and packaging dependencies.
- Run license compatibility and known-advisory checks under a documented policy.
- Include source and license inventories in release evidence.

## Required tests

- Unknown or forbidden licenses block release.
- Unresolved advisories are visible and policy evaluated.
- Dependency locks and source identities are reproducible.

## Non-claims

- Does not claim absence of undisclosed vulnerabilities.
- Does not substitute policy for legal advice.
