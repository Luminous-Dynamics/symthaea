# Symthaea Aircraft Loading Auditor

A dependency-free Rust research prototype for deterministic mass, moment, station-capacity, and declared center-of-gravity (CG) envelope checks.

## Scope

This is the first building block for a broader **loading-master** workflow: it audits one declared loading state against one exact aircraft configuration revision. Version 0.1 accepts only an explicit Takeoff phase; Ramp, Taxi, and Landing states are rejected until phase-specific limits/calculations exist. It is deliberately not an aircraft dispatch, load-release, or flight-safety system.

- Inputs use SI-style aviation units: kilograms, metres, and kilogram-metres.
- It rejects profile/configuration mismatches, malformed SHA-256 syntax, duplicate IDs, unknown stations, non-finite values, and arithmetic overflow.
- It reports take-off mass and station-cap violations as blocking findings.
- It reports missing or ambiguous CG-envelope coverage and missing securement references as not assessable.
- It sorts station summaries and findings for deterministic reports.
- The profile-source authenticity field is always false. References and digest syntax are not proof that source documents are genuine, approved, current, or applicable.

## Explicit limitations

No fuel-burn phase model, taxi/landing weight audit, moving passengers, ULD/package geometry, floor pressure, cargo restraint-strength calculation, dangerous-goods segregation, aircraft-specific trim/performance prediction, dynamic stability, maintenance status, dispatch authorization, or live ATC interface is implemented. Station capacity and CG limits are only as trustworthy as the supplied profile.

The synthetic unit fixtures are not aircraft specifications. Never use these results to load, dispatch, or fly a real aircraft.

## Local checks

```sh
cargo test -p symthaea-aircraft-loading
cargo clippy -p symthaea-aircraft-loading --all-targets -- -D warnings
cargo fmt --check -p symthaea-aircraft-loading
```

Hosted exact-head checks must pass before merge. A queued, skipped, or absent job is not a pass.
