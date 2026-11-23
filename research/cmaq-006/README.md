# CMAQ-006: Evidence-Gated World-Model Use

**Status:** executable policy-gate reference, v0.2  
**Tracking issue:** [#9029](https://github.com/Luminous-Dynamics/symthaea/issues/9029)  
**Parent smoke gate:** [PR #9030](https://github.com/Luminous-Dynamics/symthaea/pull/9030)  
**Claim ceiling:** evidence-routing policy prototype only; no production safety or SOTA claim

## Research question

When should an intelligent system trust a learned world model for the next decision—and when should it fall back to a simpler policy?

CMAQ-006 composes shift/freshness evidence, held-out adequacy, decision-value lift and independent action authorization. Version 0.2 removes the single aggregate `adequacy_status=PASS` input. Model use now requires ten separately typed evidence dimensions, each of which must be `PASS`.

## Decision rule

The minimum net-lift margin remains code-pinned to **0.02 utility units**. Untrusted episode input cannot lower it.

1. If the independent direct-action fallback is not authorized, emit **BLOCK**.
2. If a shift alarm is active or the model epoch differs from the current environment epoch, emit **DIRECT_FALLBACK**.
3. If the proposed model action is not independently authorized, emit **DIRECT_FALLBACK**.
4. Invalid or mismatched model/evidence digests emit **DIRECT_FALLBACK**.
5. Every adequacy dimension below must be `PASS`. A `FAIL` or `UNKNOWN` produces a dimension-specific fallback reason.
6. The lower 95% bound on cost-adjusted net decision lift must strictly exceed 0.02.
7. Only when all conditions pass does the gate emit **USE_MODEL**.

The ten dimensions are independent and conjunctive; they are not averaged into a confidence score:

| Input field | Evidence obligation | Existing owner / source |
|---|---|---|
| `physics_adherence_status` | Physical coherence and physics-law evidence | MiraBench-inspired diagnostics; Symtropy physical/reference evidence |
| `action_fidelity_status` | Predicted outcome follows the commanded action and task intent | MiraBench action-following diagnostics |
| `constraint_compliance_status` | Predicted behavior preserves object identity, spatial/trajectory constraints, obstacles, and task-specific restrictions | RoboTrustBench constraint-sensitive scenarios |
| `counterfactual_grounding_status` | Predictions remain grounded when instructions conflict with scene facts or physical feasibility; no invented objects/attributes | RoboTrustBench counterfactual scenarios |
| `failure_preservation_status` | Failure-inducing actions remain failures rather than being overwritten with success | MiraBench optimism-bias / failure-preservation diagnostics |
| `horizon_support_status` | Requested target distance is within the supported `P*(K)` range for this model/family/shift regime | #10257 / #10276 |
| `latent_rollout_fidelity_status` | Horizon-wise latent rollout consistency for the exact encoder/model/action history | #10257 LRC crosswalk |
| `state_sufficiency_status` | Held-out probes recover task-relevant state; collapsed/incomplete latents fail | #10257 LRC/state-sufficiency crosswalk |
| `selected_action_calibration_status` | Uncertainty/support remains qualified on planner-selected actions, not only behavior/random actions | #9127 and #10256 |
| `trajectory_provenance_status` | Required action-coupled trajectory and stochastic-draw lineage is independently qualified | #13375/#13377; Symtropy #3690/#3743 |

### Why failure preservation is its own gate

MiraBench reports a 906-video human annotation corpus with 16,704 structured decisions and finds that nominal task completion can improve while failure preservation deteriorates. In one paper-specific DreamDojo-2B post-training ablation, nominal completion rose from 14% to 92% while failure preservation fell from 48.7% to 23.1%. These are MiraBench findings, not Symthaea results. The corpus is a diagnostic source; this crate does not reproduce MiraBench's human/VLM scoring pipeline.

The current v0.2 profile deliberately accepts only `PASS`, `FAIL`, and `UNKNOWN`. `NOT_APPLICABLE` is not accepted as an episode-level escape hatch. A genuinely different task/model class requires a separately reviewed, versioned adequacy profile with its own predeclared applicability obligations.



### Constraint and counterfactual grounding are separate obligations

RoboTrustBench (EMNLP 2026 Findings; arXiv:2606.01600) uses 1,207 expert-validated image–instruction samples drawn from real DROID episodes: 533 normal, 345 constraint-sensitive, 228 counterfactual, and 101 adversarial. Its constraint-sensitive cases include distractors, occlusion, obstacles, underspecified references and trajectory constraints. Its counterfactual cases include object/attribute contradictions, geometric impossibility, infeasible interaction, object absence and wrong location.

These are not equivalent to nominal action-following or failure preservation. A prediction can respond to an action while still violating a trajectory/placement constraint; it can preserve one failure while hallucinating a missing object or changing its attributes to satisfy an instruction. Therefore v0.2 requires separate constraint-compliance and counterfactual-grounding evidence.

Project page: https://huiqiongli.github.io/RoboTrustBench/
Dataset: https://huggingface.co/datasets/Huiqiong0124/RoboTrustBench_Dataset
Paper: https://arxiv.org/abs/2606.01600

The published dataset card exposes the 1,207 image–instruction records and companion metadata. The complete model-output evaluation materials, human-evaluation instructions and rubric must still be verified and pinned before claiming an official benchmark reproduction. The dataset viewer does not by itself prove that every scorer, prompt and baseline asset is published.

**Unsafe-instruction suppression remains independent.** The model cannot self-certify a harmful instruction as safe. The model-action and direct-action authorization inputs must be produced by a separately controlled action-authority/RTA layer. A passing physics, constraint, counterfactual or other model-evidence status never grants physical authority.

## CLI

This is a dependency-free Rust crate pinned to Rust 1.96.0.

Run from the repository root:

    cargo test --locked --manifest-path research/cmaq-006/Cargo.toml
    cargo run --quiet --locked --manifest-path research/cmaq-006/Cargo.toml -- --input research/cmaq-006/fixtures/cases.tsv

Input is a strict 20-column TSV containing model/evidence digests, epochs, shift status, ten evidence statuses, net-lift lower bound and independent action authorizations. The CLI emits a deterministic 24-column decision ledger.

Unknown labels, missing fields, malformed rows, duplicate case IDs, non-finite numbers, invalid booleans, invalid digests and header drift fail closed.

## What this v0.2 gate verifies—and what it does not

It verifies **policy routing and internal consistency**: digest shape/equality, epoch freshness, explicit shift status, eight separate adequacy statuses, the lower-bound decision-lift threshold, independent action authorization, strict parsing and deterministic reasons.

It does **not** cryptographically verify that a digest corresponds to the real artifact, verify signatures, recompute upstream confidence intervals, re-run physical/reference evaluators, prove that a caller's status accurately summarizes its evidence, or authenticate the evidence ledger. The TSV must come from a trusted wrapper that independently verifies every report and artifact against its frozen profile and manifest. Candidate-authored claims must not be accepted as authoritative gate inputs. Digest equality alone is not proof of authenticity.

In particular, a `PASS` value from the CLI input is a routing datum—not a certificate. #3743 now has a successful PR merge-ref qualification for finite source-to-address re-projection and Philox receipt conformance, but its draft/review status and external anchoring boundary remain. Downstream consumers must independently retrieve and validate its referenced artefacts.

The fixed fixtures are public known-answer controls, not hidden qualification. The crate does not execute a live model, call a physical system, or integrate into Symthaea's planner. BLOCK and DIRECT_FALLBACK are routing outputs, not certifications that any action is physically safe.

## Known-answer controls

The 19-row fixture covers:
- eligible model -> USE_MODEL;
- shift alarm and stale epoch -> DIRECT_FALLBACK;
- each of the ten dimensions independently set to FAIL or UNKNOWN -> DIRECT_FALLBACK;
- insufficient net-lift bound -> DIRECT_FALLBACK;
- predictive/utility digest mismatch and malformed model digest -> DIRECT_FALLBACK;
- unauthorized model action with authorized fallback -> DIRECT_FALLBACK;
- no authorized fallback -> BLOCK.

The Rust tests also check per-dimension fail/unknown routing, parser/header integrity, duplicate IDs, malformed statuses, non-finite values, authorization priority and deterministic output.

## Qualification target (prospective, not an achieved result)

Before connecting this gate to the real system, require:
- zero USE_MODEL outcomes whenever any required dimension is FAIL or UNKNOWN;
- zero action execution when the independent authority authorizes neither candidate;
- complete ledger-to-artifact provenance and mutation tests rejecting tampered evidence;
- positive paired held-out net decision lift after inference, observation, latency and fallback costs, with independent-cluster uncertainty;
- published fallback rate, model-use coverage, blocked actions, failures and costs—so "always fall back" cannot masquerade as success.

The paradigm is **evidence-conditioned capability**: predictive control is an earned, revocable privilege. The gate demands evidence about physical coherence, command following, failure preservation, planning range, latent dynamics, state sufficiency, selected-action uncertainty and trajectory lineage—not a single headline confidence score.
