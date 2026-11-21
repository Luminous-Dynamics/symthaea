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

The production SOMA manifest now pins Mycelix commit 85d09458ba05d5da68b2702b1e13ee6692330f9f through the uniquely named Git package `mycelix-positioning-v2`. Its library name remains `positioning`, so the adapter retains its existing import paths. The facade manifest points directly to the canonical `mycelix-position/lib/positioning/src/lib.rs`; it does not duplicate or fork the implementation. This avoids ambiguous package discovery caused by mirrored `positioning` manifests in the Mycelix repository.

The dedicated workflow extracts the immutable revision from the production SOMA manifest and builds a temporary adapter harness against that exact Git dependency. It runs feature-disabled `cargo check`, then feature-enabled Clippy and all-target tests. Cargo fetches the package from the same public Git URL and immutable revision used by production.

This intentionally does not resolve Symthaea's full root workspace: that workspace currently references a missing `crates/core/symthaea/Cargo.toml` member. The harness now verifies the production dependency's package identity, Git source, revision and Rust library import, plus the exact adapter code. It is still not a full `symthaea-soma` package/application build or Android/iOS provider integration.

## Claim ceiling

The adapter tests establish source mapping and package/API compatibility only. They do not validate the mobile sensor drivers, provider timestamp correctness, CRS transformations, calibration, source independence, protection levels, accuracy, physical navigation or field safety.
