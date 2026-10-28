# Aviation Decision Readiness Gate (ADRG-001)

**Status:** composition API and unit-test definitions committed; Rust execution pending.  
**Implementation:** `crates/domains/symthaea-operations-research/src/aviation_decision_readiness.rs`  
**Related protocols:** [split integrity](./AVIATION_CALIBRATION_AND_SPLIT_INTEGRITY.md), [calibrated uncertainty](./AVIATION_DISTRIBUTIONALLY_ROBUST_BENCHMARK.md), [online freshness](./AVIATION_ONLINE_CALIBRATION_FRESHNESS.md).

## Research hypothesis

The aviation benchmark currently has separate specialist checks for feature timing, split integrity, plan verification, robust expected cost, network harms and calibration freshness. A candidate should not be promoted for human consideration just because one layer succeeds while another reports rejected, indeterminate or stale evidence.

ADRG-001 composes these declared outputs into a single fail-closed result. It deliberately stops at `EligibleForHumanReview`; it never grants dispatch, maintenance-release, airworthiness or other operational authority.

## Inputs and rules

The composition takes:
- `FeatureAuditReport`: must be `Accept`.
- `SplitIntegrityReport`: must be `Accept`.
- `VerificationReport`: must be `Verified`.
- Online drift monitor report: must be `InControl`; a drift alarm and an out-of-support alarm are separate blockers.
- Distributionally robust cost assessment: must be eligible.
- Finite-scenario network/harm assessment: must be eligible.
- Candidate calibration receipt ID and active calibration receipt ID: both must be non-empty and equal.

Every blocker is returned in deterministic order. Rejected and Indeterminate states remain distinct; neither becomes acceptance by default. Calibration-receipt equality is only a string-binding check at this layer: a future run-manifest layer must bind the receipt cryptographically to frozen input/config/code digests.

## Decision-state interpretation

- **Blocked:** at least one gate is rejected, indeterminate, stale, drifted, out-of-support or risk-ineligible. Do not promote the candidate.
- **EligibleForHumanReview:** all supplied gate reports satisfy their acceptance requirements and the candidate calibration receipt matches the active receipt. This means only that the provided artifacts passed this composition function; it is not proof that reports are authentic or that the operation is safe.

The gate does not independently verify the source data or recompute each specialist verdict. Each report must originate from its corresponding trusted verifier in the eventual integration pipeline.

## Required tests and benchmark targets

- 100% blocked on each isolated fault: feature/schema rejection, split rejection or indeterminacy, verifier rejection, missing/stale calibration receipt, drift alarm, support shift, distributional-risk ineligibility, or network-risk ineligibility.
- When several gates fail, include every blocker in stable order.
- 100% Indeterminate propagation: missing evidence cannot be promoted.
- No state that implies autonomous operational authorization.
- End-to-end fixtures should inject one fault at each layer and combinations of faults, and verify the manifest shows the exact blocker set.
- Once wired to a real run manifest, use immutable source/config digests and verifier receipts, not caller-provided labels alone.

## Relation to the state of the art

This is an integration/evidence architecture, not a new flight optimizer. Robust optimization, conformal calibration, sequential drift testing and operational safety assurance each have established approaches. The research claim to test is whether composing them into a reproducible, fail-closed decision path reduces false promotion of stale or invalid candidates while keeping useful intervention rates acceptable.

## Test status and claim ceiling

Ten unit tests are defined for the composition function. They have not been compiled or run in this environment due to missing `cargo` and `rustc`. Passing the synthetic composition tests does not establish runtime integration, data authenticity, airline cost savings, safety certification or regulatory compliance.

Follow-up: [AVIATION-ENTERPRISE-012 #8798](https://github.com/Luminous-Dynamics/symthaea/issues/8804).
