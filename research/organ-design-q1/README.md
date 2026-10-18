# Organ Design Q1 — Synthetic Candidate-Ranking Contract

**Status:** isolated research prototype, not biological validation or an organ-design tool.  
**Boundary:** synthetic objective vectors and evidence admission only. No wet-lab protocol, medical recommendation, transplant, safety, efficacy, or clinical claim is supported.

## Purpose

Q1 is the Symthaea-side counterpart to Symtropy's analytic Q0 transport work. It establishes a small candidate-ranking contract before an optimizer is built. Its purpose is to stop a future search loop from turning one opaque "organ realism" scalar into a false universal winner.

The contract distinguishes:
- **hard constraints** (must pass before a candidate can be admitted);
- **objective vectors** (declared units and minimize/maximize direction);
- **Pareto fronts** (incomparable candidates remain incomparable);
- **assumptions and evidence status** (missing, stale, contradictory, or unverified inputs fail closed);
- **training and held-out paths** (selection only on training candidates; held-out mode is evaluation-only);
- **evidence tier and claim ceiling** (this Q1 contract is fixed to `synthetic_simulation`).

Internal confidence is metadata only. It is neither an objective nor a tie-breaker and cannot promote the evidence tier. A candidate that asks to claim a tier above the profile ceiling is rejected.

## Isolated execution

The crate has its own Cargo workspace and no runtime dependencies. JSON parsing and test-only SHA-256 helpers are development-only dependencies for validating the hostile-case catalog and synthetic design identity tokens.

```sh
cargo fmt --manifest-path research/organ-design-q1/Cargo.toml -- --check
cargo test --manifest-path research/organ-design-q1/Cargo.toml
```

The public Rust API is in `src/lib.rs`. `hostile-cases.v0.json` names eighteen adversarial cases; the Rust test module checks the fixture catalog and separately executes matching typed cases. The canonical identity function creates deterministic, length-prefixed UTF-8 material from the profile, exact candidate fields, admitted evidence digests, and outcome decisions. A runner can SHA-256 the material to produce a digest; the crate intentionally avoids a runtime hashing dependency.

## Input contract

Every profile pins:
- a non-empty profile ID and purpose;
- objective IDs, units and direction;
- exact hard-constraint IDs and required assumption IDs;
- evidence source IDs plus 64-character lowercase SHA-256 digests;
- a baseline and at least the declared ablation set;
- data split, evaluation mode, and evidence tier.

Every candidate must provide a SHA-256 digest for its exact design/topology artifact, exactly one finite metric with the declared unit per objective, every declared hard-constraint result, every assumption resolution, and every required evidence reference with a matching digest and `current` status. The test suite hashes explicit synthetic identity tokens; production callers must hash the actual canonical design artifact bytes. Any constraint failure, missing item, contradictory evidence, stale source, profile/split mismatch or claim-ceiling promotion makes the candidate ineligible.

For baseline/ablation comparisons, if the baseline or any required ablation is absent or rejected, the receipt is `BLOCKED` even though other candidate rejection records are preserved. A `READY` receipt is a software contract result, not a physical result.

## Pareto semantics

Objective vectors are compared component-by-component after unit identity is checked. Candidate A dominates B only when A is no worse for every objective and strictly better for at least one. The implementation does not choose implicit weights and does not compare uncalibrated confidence scores. Equal or trade-off vectors are kept on the same Pareto front, with IDs used only for deterministic ordering within the front.

Held-out evaluation deliberately emits `EvaluatedOnly` decisions without Pareto fronts and sets `selection_permitted = false`. A caller must not route those cases back into candidate selection.

## Evidence ceiling

The result tier is hard-coded by profile validation to `synthetic_simulation`; changing it to physical-bench, tissue-specific, or translational evidence is rejected by this contract. Even a future passing test suite will prove only the behavior of the typed synthetic ranking semantics. It will not establish:
- pressure solving or physical flow accuracy;
- vessel compliance, pulsatile flow, barrier permeability or oxygen delivery;
- tissue viability, disease-model validity, whole-organ function or transplant readiness.

## Exact-head verifier receipt

Use the receipt-producing entrypoint (from the repository root):

```sh
bash research/organ-design-q1/verify-q1.sh \
  target/organ-design-q1-receipt.json \
  "$(git rev-parse HEAD)"
```

The receipt binds the exact source commit, isolated manifest, ranking implementation source, and hostile-case catalog hashes, plus formatting/test command outcomes and the synthetic-only claim ceiling. The dedicated `Organ Design Q1` workflow checks out the exact pull-request head, pins Rust 1.96.0, runs the verifier, and uploads the receipt even when verification fails. A previous-head result or a queued/cancelled workflow is not a pass.

## Current validation status

This editing environment has not run Cargo/Rust. The source and fixtures must be checked out and executed on the exact branch SHA. A queued/cancelled/skipped workflow is not a passing receipt. The Q1 tranche should stay separate from Symtropy's Q0 gate and must not be described as qualified until an exact-head run completes.
