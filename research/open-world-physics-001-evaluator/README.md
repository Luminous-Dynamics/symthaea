# Open-World Physics Evaluator v0.1

A small Rust reference harness for the Symthaea OPEN-WORLD-PHYSICS-001 research track.

## What this implements

- Three analytically specified, dimensionless response-law families: inverse power, screened inverse power, and saturating response.
- A query interface that exposes observations only and refuses non-finite or out-of-discovery-domain inputs.
- Two transparent experiment-selection baselines with a matched budget: fixed-grid and seeded-random.
- A bounded grid-search model-selection baseline with a Gaussian-likelihood/BIC-like score using declared observation noise, plus an explicit indeterminate disposition when the best two structural families are too close (ΔBIC-like score < 2).
- Separate training-fit and structural extrapolation scores over a fixed held-out domain.
- Deterministic machine-readable JSON reports with a SHA-256 corpus fingerprint and a receipt digest that detects report-byte changes.
- Known-answer, domain-rejection, query-repeatability, noisy reference-family recovery under both policies, deterministic-report, model-selection ambiguity semantics, and receipt-tampering tests.

## Run

From the Symthaea repository root:

```sh
cargo test --manifest-path research/open-world-physics-001-evaluator/Cargo.toml
cargo run --manifest-path research/open-world-physics-001-evaluator/Cargo.toml --release
```

The CLI emits a JSON report to stdout. The report receipt is checked before output.

## Important limitations

**This is an open, source-visible reference suite, not a secret blind leaderboard.** The API keeps the truth out of the agent-facing query response, but source access means the reference laws are not genuinely hidden from a determined agent. The report sets `blind_evaluation=false` and identifies its claim ceiling explicitly.

The grid-search fit is a bounded reference baseline, not Ramanujan, EUREKA, or a claim of novel law discovery. The ΔBIC-like threshold is only a model-selection heuristic, not a posterior probability, formal proof, or physical-truth oracle. The harness does not yet connect to the canonical ScientificModel / PhysicsIR handoff, PR #8034 experiment selector, Symtropy Physics Experiment Atlas, or an EUREKA qualification service. Those integrations must use the existing contracts rather than introduce parallel ones.

The three analytic laws are simple synthetic fixtures; no historical apparatus is reproduced, no measured dataset is used, and no claim is made about real-world physical novelty. The v0 scores quantify whether the baseline fitting pipeline recovers these declared families and generalizes into the held-out input range—not whether it exceeds the state of the art.

## Next gate

1. Execute formatting, unit tests and the release CLI on this exact commit.
2. Independently recompute the report and add adversarial malformed/counterexample cases.
3. Add a candidate submission interface that can be evaluated after the candidate is frozen, without leaking held-out truth into the agent loop.
4. Add a real blind execution boundary before publishing any blind-recovery score.
5. Integrate the already-existing model handoff and experiment-selection components only after exact-head qualification.
