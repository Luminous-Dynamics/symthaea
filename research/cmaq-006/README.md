# CMAQ-006: Evidence-Gated World-Model Use

**Status:** executable policy-gate smoke harness, v0.1  
**Tracking issue:** [#9029](https://github.com/Luminous-Dynamics/symthaea/issues/9029)  
**Evidence status:** Rust source, known-answer TSV fixture and focused workflow added; exact-head Rust run is not yet confirmed  
**Claim ceiling:** evidence-routing policy prototype only; no production safety or SOTA claim

## Research question

When should an intelligent system trust a learned world model for the next decision—and when should it fall back to a simpler policy?

Three neighboring benchmark tracks answer different questions:

- CMAQ-003 tests detection and response to regime shifts: [PR #8984](https://github.com/Luminous-Dynamics/symthaea/pull/8984).
- CMAQ-004 measures world-model net decision lift after model-query costs and reports paired uncertainty intervals: [PR #9024](https://github.com/Luminous-Dynamics/symthaea/pull/9024).
- CMAQ-005 scores held-out causal-effect and safety predictions: [PR #9017](https://github.com/Luminous-Dynamics/symthaea/pull/9017).

CMAQ-006 is the **model-use gate** connecting those evidence types. It should use a model only when current-epoch evidence is tied to the exact model artifact, held-out predictive adequacy passes, the lower 95% bound on net decision lift exceeds a reviewed minimum margin, and the independent action authorization permits the proposed action.

## Decision rule

The minimum net-lift margin is code-pinned to **0.02 utility units** in v0.1, so the case input cannot silently lower the bar.

1. If the independent direct-action fallback is not authorized, emit **BLOCK**.
2. If a shift alarm is active, or the model epoch differs from the current environment epoch, emit **DIRECT_FALLBACK**.
3. If the proposed model action is not independently authorized, emit **DIRECT_FALLBACK**.
4. If the model digest is invalid, an evidence digest is invalid, or predictive/utility evidence was produced for a different model digest, emit **DIRECT_FALLBACK**.
5. If predictive adequacy is FAIL or UNKNOWN, emit **DIRECT_FALLBACK**.
6. If the supplied lower 95% bound on cost-adjusted net decision lift is less than or equal to 0.02, emit **DIRECT_FALLBACK**.
7. Only when all gates pass, emit **USE_MODEL**.

The fallback action must independently pass the action envelope; a model-use gate is not itself a safety validator.

## CLI

This is a dependency-free Rust crate pinned to Rust 1.96.0.

Run from the repository root:

    cargo test --locked --manifest-path research/cmaq-006/Cargo.toml
    cargo run --quiet --locked --manifest-path research/cmaq-006/Cargo.toml -- --input research/cmaq-006/fixtures/cases.tsv

Input is a strict TSV case ledger. It includes the model digest, digests attached to the predictive-adequacy and utility evidence, model/current epochs, shift signal, adequacy status, net-lift lower bound, and independent action authorizations.

The CLI emits a deterministic TSV ledger with schema version, case identifier, all three digests, both epochs, shift status, adequacy status, chosen decision, the code-pinned margin, authorizations and reason. Duplicate case IDs, malformed rows, unknown adequacy labels, non-finite numbers, invalid booleans and header drift fail closed.

## What this v0.1 gate verifies—and what it does not

It verifies **consistency and policy routing**: digest shape/equality, epoch freshness, explicit shift status, adequacy status, lower-bound decision lift, independent action authorization, strict input parsing and deterministic reasons.

It does **not** cryptographically verify that a digest corresponds to the real artifact, verify signatures, recompute CMAQ-004 confidence intervals, independently reproduce CMAQ-005 metrics, or validate the trustworthiness of the evidence ledger itself. The TSV must come from an evaluator/wrapper that verifies artifact digests and report provenance; candidate-authored claims must not be accepted as authoritative gate inputs. Digest equality alone is not proof of authenticity.

The fixed fixtures are public known-answer tests, not hidden qualification. The current crate does not execute a live model, call a physical system, or integrate into Symthaea's planner. BLOCK and DIRECT_FALLBACK are routing outputs, not certifications that any action is physically safe.

## Known-answer controls

The fixture exercises:
- eligible model -> USE_MODEL;
- shift alarm or stale epoch -> DIRECT_FALLBACK;
- unknown adequacy or insufficient lift -> DIRECT_FALLBACK;
- predictive/utility digest mismatch or malformed digest -> DIRECT_FALLBACK;
- unauthorized model action with authorized fallback -> DIRECT_FALLBACK;
- no authorized fallback -> BLOCK.

Unit tests also reject duplicate case IDs, malformed row width, non-finite lift values and unrecognized booleans.

## Qualification target (prospective, not an achieved result)

Before connecting this gate to the real system, require:
- zero USE_MODEL outcomes for all stale, mismatched, unknown-adequacy and below-margin negative controls;
- zero executed actions when the independent safety layer authorizes neither candidate;
- complete ledger-to-artifact provenance and mutation tests that prove evidence tampering is rejected;
- positive paired net decision lift on held-out mechanisms after inference costs, with an episode/family-level 95% interval excluding zero;
- published direct-fallback rate, task coverage, blocked actions, failures and costs—so “always fall back” cannot masquerade as success.

The intended paradigm shift is **evidence-conditioned capability**: model-based action is an earned privilege, revoked when its support becomes stale or its measured benefit is insufficient.
