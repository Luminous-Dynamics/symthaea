# subterra-inspection-evidence-gate

A Rust consumer boundary that decides whether an already-scored inspection option is eligible to enter the Subterra campaign optimizer.

## Why

A candidate's EVSI number is not sufficient provenance. The option must be bound to its precise asset/decision, asset revision, coordinate frame, region, sensor model, likelihood model, loss table, EVSI assessment, source dataset, observation receipt, calibration receipt, rights disposition and validity interval.

## Run

```sh
cargo run --manifest-path Cargo.toml --example synthetic_admission --offline
cargo fmt --manifest-path Cargo.toml -- --check
cargo test --manifest-path Cargo.toml --all-targets --offline
cargo clippy --manifest-path Cargo.toml --all-targets --offline -- -D warnings
```

## Admission rules

The gate rejects a candidate if any required digest is missing; the decision, asset or revision differs from the frozen planning context; frame, region, sensor/model, likelihood, loss-table or EVSI assessment bindings mismatch; observation evidence is future-dated beyond declared clock skew or older than the maximum age; calibration, evidence, rights grant, resource assessment or risk assessment is outside its validity interval; rights do not explicitly cover the requested purpose; the region scope is partial/unknown; the external verifier did not mark the receipt verified; or any required receipt digest is missing.

The time comparison uses one frozen evaluation timestamp from the campaign context. It does not call the system clock during evaluation, so repeated evaluations on identical inputs are deterministic.

## Important security boundary

This crate does **not** verify cryptographic signatures, certify that a rights grant is legally valid, prove source inventory completeness, calibrate sensors, establish POD/PFA, validate likelihood quality, or authenticate the upstream EVSI calculation. It consumes a disposition and opaque verifier receipt emitted by separately governed Mycelix/Sol-Atlas services. A caller must not construct these fields from untrusted candidate input and treat them as trusted attestations.

Admission means only "eligible for downstream planning under these declared bindings." It does not mean safe, defect-free, cleared, authorized for intervention or physically verified.

## Integration

The gate returns an admitted candidate wrapper containing both the original campaign option and the complete evidence manifest only after all checks pass.

The `build_evidence_bound_campaign` constructor takes the complete candidate set and frozen per-option contexts. It builds the campaign input only if **every** supplied option is admitted; one rejected option aborts construction rather than silently filtering the set. The returned `EvidenceBoundCampaign` retains all candidate manifests. Its `optimize` method returns the campaign report, all admitted evidence manifests, and the exact evidence manifests corresponding to selected options.

The same frozen contexts, admission result and campaign evidence bundle should be retained for later independent audit. Direct construction of the lower-level campaign crate remains possible; production integration should use this evidence-bound builder as its admission boundary.

The synthetic example uses fake digests and deliberately does not claim that a real external verifier or rights service ran.
