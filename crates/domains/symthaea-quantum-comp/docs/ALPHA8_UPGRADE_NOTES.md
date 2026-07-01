# Alpha.8 Upgrade Notes

Alpha.8 is a reliability and integration-boundary release.

It adds named fixtures, replay plans, local release gates, and explicit integration declarations. The goal is to make research artifacts easier to reproduce and harder to over-interpret.

## Added

- `fixtures`: stable named local fixtures for smoke, demo, and pilot runs.
- `replay`: operator-facing replay plans for smoke/local/pilot scopes.
- `release_gate`: local gate summaries that combine preflight, audit, fixtures, and replay metadata.
- `interop`: explicit Symthaea, Mycelix, local lab, and external backend boundary declarations.
- CLI commands: `presets`, `schemas`, `fixtures`, `replay`, and `gate`.
- Examples: `fixture_catalog`, `replay_plan`, `release_gate`, and `interop_boundary`.

## Claim posture preserved

Alpha.8 still makes no quantum consciousness claim, no quantum advantage claim, no hardware backend execution claim, no physical entanglement claim, and no Mycelix attestation claim.

## Why this matters

Alpha.7 made the crate easier to run. Alpha.8 makes it easier to reproduce and package responsibly.
