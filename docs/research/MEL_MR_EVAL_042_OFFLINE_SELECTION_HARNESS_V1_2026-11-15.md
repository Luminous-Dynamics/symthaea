# MEL-MR-EVAL-042 — Offline selection-exploitability harness v1

**Date:** 2026-11-15  
**Status:** Implementation authored; exact-head Rust qualification pending  
**Parent:** [Issue #12190](https://github.com/Luminous-Dynamics/symthaea/issues/12190)  
**Specification:** [PR #12191](https://github.com/Luminous-Dynamics/symthaea/pull/12191)

## Why this implementation exists

A musical evaluator must be qualified under the distribution induced by the selector that consumes it. Pairwise evaluator accuracy on randomly sampled candidates is not sufficient evidence for best-of-N, iterative repair, or multi-backend portfolios.

This first implementation makes the selection trajectory reproducible and makes the proxy/ground-truth boundary explicit. It is intentionally an offline contract harness, not an audio-generation system and not a product-quality claim.

## Implemented in this slice

- Standalone Rust 2024 harness under \`tools/mel-music-selection-exploitability-harness\`; it is isolated from the large workspace dependency graph.
- Versioned trajectory request and manifest, with frozen task, source lineage, instruction, evaluator-suite, seed, candidate count, budgets, adaptive-round ladder, backend-portfolio size and selector policy.
- Candidate observations contain only optimizer-visible proxy vectors, uncertainty, backend/renderer identities, content digests and explicit hard-gate verdicts.
- Synthetic ground truth is represented by a separate type and only accepted by the evaluator function. The selector API cannot receive hidden truth.
- Candidate counts are bounded at 128. Supported adaptive-round declarations are 1/2/4/8 and backend portfolios 1/2/4. Each materialized candidate must identify a declared round; observed distinct backends may not exceed the declared portfolio.
- Every assigned slot remains in the denominator, including failed-before-render, render-failed, timed-out, malformed and unrendered slots.
- Actual render attempts are counted against the frozen render budget. Duplicate output digests are reported rather than silently deduplicated.
- Policies: deterministic random; best-of-N similarity, loudness and prompt-alignment; mean ensemble; uncertainty-penalized ensemble; and hard-gate-first Pareto selection.
- Hard-gate-first does not convert rights, integrity, preservation, edit execution, source lineage or missing evidence into weighted preferences. All six declared gates must be Verified before the candidate enters its Pareto pool; Rejected and Unknown are not admissible.
- Stable tie-breaking, deterministic seed handling, full-pool SHA-256 commitment and replay verification of the policy outcome.
- Separate hidden-truth evaluation reports the selected human-outcome vector, shortcut family and hard-gate false acceptance. A gate result marked Verified against hidden invalid truth is explicitly counted as a false accept.
- JSON Schema and a frozen two-candidate no-op-similarity manifest.
- An exact-head workflow pinned to Rust 1.96.0, with contract tests, manifest validation, the N ladder, strict Clippy and reproducibility artifacts.

## Synthetic challenge

The frozen fixture contains two materialized candidates for one task. Candidate 1 has a higher similarity score but the requested edit is absent, so the independent edit-execution gate rejects it. Candidate 0 is a lower-scoring but valid edit.

The validator requires the vulnerable best-of-N similarity arm to select the high-proxy candidate. The ladder example expands the same known-answer pattern through N = 1, 2, 4, 8, 16, 32, 64 and 128, and runs all implemented selector policies. The example is synthetic and deterministic; it contains no real audio, listening data, commercial backend result or human adjudication.

## Validation contract

The workflow is designed to:

1. Check out and verify the exact pull-request head.
2. Validate the schema and fixture as JSON.
3. Run formatting, standalone Cargo tests and the fixture validator on Rust 1.96.0.
4. Execute the N-versus-policy ladder and validate its JSON output.
5. Run Clippy with warnings denied.
6. Upload exact source hashes, Cargo.lock, the candidate-selection report and the exact source revision.

**Current evidence status:** the code and workflow are authored on this branch. No local Cargo/Rust toolchain is available in the current execution environment. Until the workflow reports success for the exact head, compilation, tests, formatting and Clippy remain pending—not passed.

## Deliberate limitations

This slice does not yet:
- run actual adaptive repair rounds; it records and bounds round identity in the frozen candidate manifest;
- generate real candidate sets or invoke a vendor model, renderer, plugin or live audio device;
- implement a dynamic greedy-repair loop or compare live generation-call efficiency;
- establish statistical confidence intervals, human preference, musical plausibility or preservation on real audio;
- prove rights, authorship, cultural validity or commercial dataset permission;
- establish superiority over Suno, Moises, Lyria or any other commercial system.

The initial fixture models a no-op similarity shortcut. Over-edit, loudness, renderer/style, prompt/caption, repetition/novelty, collateral masking, duplication, rights/provenance and metric-disagreement families remain planned confirmatory controls. A synthetic fixture pass is only a known-answer test of the harness.

## Qualification targets retained from the research contract

These remain prospective targets, not achieved results:

- Zero false accepts on frozen deterministic hard-gate, rights, integrity, no-op and receipt-substitution controls at every tested N.
- Upper 95% confidence bound of no more than 1 percentage point additional hard-gate false acceptance versus N=1 at matched task and accepted-candidate coverage.
- At least 90% detection of preregistered shortcut trajectories with at most 5% false exploit alarms on benign controls, subject to a pilot power analysis.
- No regression in protected preservation or hard obligations when increasing selection pressure.
- Fewer renders and lower p95 cost to verified acceptance than full regeneration, greedy single-metric repair and exhaustive selection at matched verified acceptance.

These targets require a preregistered pilot, paired and cluster-aware uncertainty intervals, held-out tasks and separate human adjudication. They cannot be established by the synthetic harness alone.

## Next implementation slice

1. Add the remaining adversarial generators and their hidden known-answer truth.
2. Execute multi-round candidate generation/search trajectories rather than merely validating round declarations.
3. Add selected-set calibration and risk/coverage analysis with a sealed evaluator process.
4. Bind candidate bytes and backend execution to EVAL-035 and EVAL-041; route admissibility through EVAL-040 rather than the selector.
5. Add a consented, licensed, blinded human cohort and held-out renderer families before making any real-audio claims.
