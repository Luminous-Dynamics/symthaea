# SOMA ↔ Mycelix Position v2 adapter

This branch introduces a feature-gated adapter to Mycelix Position's versioned claim contract.

## Ownership

- Mycelix Position owns domain-neutral observation, estimate and prediction types, reference-system/time/provenance references and validation semantics.
- SOMA owns mobile/provider input mapping and receipt of raw provider values.
- This adapter is not another estimator, clock conversion service, calibration authority, source-independence authority, navigation-health engine or actuator authority.

## Missing source timestamps

When a legacy mobile provider supplies no source acquisition time, the adapter emits:
- source observation time: raw reading absent, unit absent, explicitly unqualified;
- optional receipt time: separately preserved, including an unqualified local device tick when that is all the provider can expose.

It never substitutes receipt tick for source observation time and never inserts zero as a placeholder.

Unknown calibration/configuration/provenance and unresolved frames remain Unknown or Unqualified. An empty failure-domain list means dependency evidence is not available; it does not mean independence.

## Exact real-dependency adapter test

The dedicated workflow pins Mycelix commit dc28de1165eeb04b27d23011b6f9681fa740d66d, the exact API head whose current mirrored Position workflow passed 94 tests in each workspace.

CI copies the actual Mycelix pure positioning package into a temporary path, renames it to avoid colliding with Symthaea's offline stub, and creates a tiny standalone Cargo harness. The harness imports the exact committed SOMA adapter file with `#[path]`, builds against the real Mycelix source, then runs rustfmt check, Clippy and all-target tests.

This intentionally does not resolve Symthaea's full root workspace: that workspace currently references a missing `crates/core/symthaea/Cargo.toml` member. The isolated harness proves only that the adapter source compiles and tests against the real Mycelix API; it is not a full `symthaea-soma` package or application build.

## Claim ceiling

The adapter tests establish source mapping and package/API compatibility only. They do not validate the mobile sensor drivers, provider timestamp correctness, CRS transformations, calibration, source independence, protection levels, accuracy, physical navigation or field safety.
