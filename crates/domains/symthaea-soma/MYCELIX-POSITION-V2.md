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

## Exact real-dependency integration test

The dedicated workflow pins Mycelix commit 9722ecb7350307e4d01337254272d2808e1a7095, the v2 API head that passed the Position package workflow in both mirrored workspaces.

The CI script:
1. checks out the actual Mycelix source;
2. copies its pure positioning package to a temporary CI path;
3. renames the package to avoid colliding with Symthaea's offline stub;
4. injects that temporary package as an optional dependency in the SOMA manifest;
5. enables the SOMA positioning feature and runs format, Clippy and all-target tests;
6. restores the committed SOMA manifest and lockfile on exit.

The committed default Cargo manifest has no external path dependency and keeps the feature disabled by default. This ensures the integration test does not accidentally compile against Symthaea's local stub.

## Claim ceiling

The adapter tests establish source mapping and package compatibility only. They do not validate the mobile sensor drivers, provider timestamp correctness, CRS transformations, calibration, source independence, protection levels, accuracy, physical navigation or field safety.
