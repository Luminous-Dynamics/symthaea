# Open-World Physics Evaluator v0.3 — Paired-Seed Selection Diagnostics

A small Rust reference harness for the Symthaea OPEN-WORLD-PHYSICS-001 research track.

## What this implements

- Three analytically specified, dimensionless response-law families: inverse power, screened inverse power, and saturating response.
- A query interface that exposes observations only and refuses non-finite or out-of-discovery-domain inputs.
- Two transparent experiment-selection baselines with a matched budget: fixed-grid and seeded-random.
- A bounded grid-search model-selection baseline with a Gaussian-likelihood/BIC-like score using declared observation noise, plus an explicit indeterminate disposition when the best two structural families are too close (ΔBIC-like score < 2).
- Separate training-fit and structural extrapolation scores over a fixed held-out domain.
- A controlled candidate-pool sweep with 1, 3, 10 or 30 proposals per structural family, using deterministic nested pools shared across matched cases. The three original calibration worlds are accompanied by six off-grid transfer worlds whose true parameters fall between the candidate grid values. This gives nine source-visible diagnostic worlds. Each world/policy/pool-size case is repeated across five deterministic paired seed replications, producing 360 per-case candidate-pool measurements across two query policies and four pool sizes. It separates training-score selection from an evaluator-only held-out oracle and reports selection regret, separate training/held-out errors, and a clearly labelled held-out-minus-training NRMSE delta. Every row includes a zero-based replication index so policies and candidate budgets can be paired under the same seed schedule. Because the training and extrapolation domains have different target scales, that delta is descriptive and is not a distribution-matched generalization-gap estimate.
- Deterministic machine-readable JSON reports with separate SHA-256 digests for the three-world calibration corpus and nine-world candidate-selection corpus, plus evaluator source, dependency lock and report receipt. Verification checks both corpus digests independently of the report's self-computed receipt. The receipt also binds compiler/Cargo identities, build-time Git commit and source-tree state.
- Known-answer, domain-rejection, query-repeatability, noisy reference-family recovery under both policies, deterministic-report, model-selection ambiguity semantics, and receipt-tampering tests.

## Run

From the Symthaea repository root:

```sh
cargo test --locked --manifest-path research/open-world-physics-001-evaluator/Cargo.toml
cargo run --locked --manifest-path research/open-world-physics-001-evaluator/Cargo.toml --release
```

The CLI emits a JSON report to stdout. The report receipt is checked before output. Missing toolchain/lock metadata and dirty source state are explicit report fields, not silently treated as clean.

## Important limitations

**This is an open, source-visible reference suite, not a secret blind leaderboard.** The API keeps the truth out of the agent-facing query response, but source access means the reference laws are not genuinely hidden from a determined agent. The report sets `blind_evaluation=false` and identifies its claim ceiling explicitly.

The grid-search fit and candidate-pool sweep are bounded reference baselines, not Ramanujan, EUREKA, or a claim of novel law discovery. The ΔBIC-like threshold is only a model-selection heuristic, not a posterior probability, formal proof, or physical-truth oracle. The held-out oracle is strictly evaluator-side: it measures the best generalization available in the exact candidate pool and is never used to choose the submitted candidate. Selection regret diagnoses the gap between that ceiling and the training-selected candidate; it must not be reported as an agent-accessible score. The harness does not yet connect to the canonical ScientificModel / PhysicsIR handoff, PR #8034 experiment selector, Symtropy Physics Experiment Atlas, or an EUREKA qualification service. Those integrations must use the existing contracts rather than introduce parallel ones.

The three analytic laws are simple synthetic fixtures; no historical apparatus is reproduced, no measured dataset is used, and no claim is made about real-world physical novelty. The nested `Cargo.lock` is committed and the focused workflow uses `--locked`; the authoring environment lacks a Rust toolchain, so it cannot supply local compile/test evidence. The v0 scores quantify behaviour on this declared grammar—not whether the evaluator exceeds the state of the art.

## Next gate

1. Execute formatting, unit tests and the release CLI on this exact commit.
2. Independently recompute the report and add adversarial malformed/counterexample cases.
3. Add a candidate submission interface that can be evaluated after the candidate is frozen, without leaking held-out truth into the agent loop.
4. Add a real blind execution boundary before publishing any blind-recovery score.
5. Integrate the already-existing model handoff and experiment-selection components only after exact-head qualification.
6. Use the selection curve to distinguish candidate-generation coverage from candidate-ranking quality. Then add a separate blind or evaluator-owned track with more varied hidden mechanisms; this fixed-grammar sweep is only a diagnostic. Off-grid parameter transfer makes the grid-search ceiling less artificial but is still not structural novelty, real-data validation, or a blind evaluation.
