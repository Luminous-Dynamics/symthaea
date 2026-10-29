# CMAQ-001 Factor-Composition Generator

**Status:** research generator prototype; not a qualified benchmark result  
**Version:** 0.1.0  
**Implementation:** dependency-free Rust  
**Purpose:** create candidate-visible intervention history and qualification contexts while keeping counterfactual labels in a separate evaluator file

## Why this tranche exists

The parent CMAQ-001 scorer can measure constrained decision value, regret, calibration, provenance, authorization, and postcondition gates. It does not create a causal task distribution. This generator supplies that missing interface with a deliberately testable factor-composition challenge.

A task composes a left factor and a right factor. Every factor appears in many observed training compositions, while one composition per left factor is withheld from support data. The reward law is additive in factor-specific effects plus a shared interaction term. This lets a factorized world model estimate reusable effects and compose them on an unseen pair. A pair-key lookup or nearest-episode policy has no exact outcome record for the withheld composition.

This is a narrow synthetic test of **compositional transfer**, not evidence of general causal understanding. The generator does not implement an agent, train a world model, or claim to beat a published benchmark.

## Inputs and outputs

Run from this directory:

    cargo test --locked
    cargo run --locked -- --seed 4242 --factors 8 --radius 2 --out /tmp/cmaq-dataset

The output directory must not already contain any of the generated filenames. This prevents accidentally overwriting a qualification bundle. Options:

- --seed N: deterministic 64-bit seed. If omitted, CMAQ_SEED is read; otherwise the public development default is 7.
- --factors N: number of reusable factors, from 4 through 24; default 8.
- --radius N: feature values span the integer interval -N..=N, with N from 1 through 10; default 2.
- --out DIR: output directory; required.

The dataset contains:

| File | Candidate-visible? | Purpose |
|---|---:|---|
| support-observations.tsv | Yes | One observed intervention per row: factors, context, chosen action, reward and observed safety/outcome. Unchosen counterfactuals are absent. |
| qualification-cases.tsv | Yes | Held-out factor compositions and context features, allowed actions, the evidence ID required by the scorer, and whether explicit approval is required. |
| policy-evidence.tsv | Yes | The stable evidence record referenced by the qualification cases. |
| answer-key.tsv | **Evaluator only** | Counterfactual reward/safety/outcome for every action. Never pass this file to the candidate. |
| manifest.json | Evaluator only by default | Schema, split rule, seed commitment, row counts, and SHA-256 digests of the exact generated files. |

The qualification cases have the same case_id values as the answer key and can be joined by that ID. The answer key columns are compatible with the parent CMAQ-001 trace scorer. The visible cases deliberately do not contain mechanism IDs, reward labels, safety labels, or action outcomes.

## Split semantics

For factor IDs left and right, the pair is held out when:

    (left + right) % factor_count == 0

Every factor still appears in multiple support compositions. For each supported pair, all feature-grid contexts are emitted and exactly one action is intervened upon per row. The qualification set enumerates all feature-grid contexts for each withheld pair. The feature grid is synthetic and bounded; it is not a claim about any real environment.

The reward law for action a is:

    bias[a] + left_bias[left,a] + right_bias[right,a]
      + x * left_slope[left,a] + y * right_slope[right,a]
      + x * y * shared_interaction[a]

Coefficients are generated from a specified seeded SplitMix64 stream. The safety rule combines hidden left/right factor risk and action-specific feature risk. A successful outcome is a safe action with positive reward. The exact coefficients and all counterfactual labels are only represented through the evaluator answer key.

## Reproducibility and split hygiene

- Generation is deterministic for the same source version and inputs; the manifest contains file hashes and a seed commitment, but not the seed itself.
- The public seed and fixtures are for development only. For a genuinely blinded qualification run, the evaluator must draw a uniform 64-bit seed from a cryptographically secure random generator outside candidate control, keep the seed and answer key private, publish neither them nor the manifest before the run, and bind the exact candidate, generator source, data digests, and scoring code in a separate evaluator receipt.
- A SHA-256 digest is a commitment to bytes, not proof that the evaluator was fair or that the generated distribution is scientifically adequate.
- The seed commitment is an integrity aid, not strong secrecy: the seed space is 64-bit and an adversary with enough compute could exhaust it. Do not give candidates the manifest or answer-key digest; a later qualification version should support a larger secret seed and a signed evaluator receipt.
- The factor-composition split is public by design. It tests compositional transfer, not secrecy of the split rule.
- This v0.1 generator does not yet generate independent held-out causal-law families, distribution-shift schedules, noisy outcomes, or multi-step transition dynamics. Those should be separate preregistered lanes, not silently mixed into the first score.

## Suggested baselines

Run these on the exact same support data and qualification cases:

1. constant-prior policy;
2. nearest-support episode retrieval;
3. factor-pair lookup with an explicit abstention fallback;
4. independent linear model per action without factorization;
5. factorized causal/world model;
6. factorized model plus search/planning;
7. oracle policy, reported only as a non-deployable ceiling.

Use paired evaluation cases and identical action/context/runtime budgets. Report constrained regret, safe-action rate, optimal action rate, Brier score, per-case-family results, p50/p95 latency, retained memory, and seed-to-seed uncertainty. Do not compare this synthetic score directly to AMA-Bench, MemoryArena, LongMemEval, or leaderboard scores from a different protocol.

## Tests and qualification boundary

The unit tests cover deterministic generation, seed sensitivity, the official SHA-256 test vector, the withheld-composition split, factor/action support coverage, counterfactual-label separation, and scorer-compatible answer-key columns. These tests qualify the data-generation interface only. They do not qualify any cognitive architecture, and they do not establish state of the art.
