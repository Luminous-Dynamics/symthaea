# ICCB evaluator prototype

This is the first isolated Rust reference kernel for the Intergenerational Capability-Compounding Benchmark (ICCB) proposed in [Symthaea PR #7974](https://github.com/Luminous-Dynamics/symthaea/pull/7974).

## Run locally

From the repository root:

~~~sh
cargo test --manifest-path research/iccb/Cargo.toml
cargo run --manifest-path research/iccb/Cargo.toml
~~~

The prototype intentionally has no third-party dependencies. Its manifest defines a nested workspace boundary so it can be tested separately from Symthaea's larger workspace while its data contracts stabilize.

## What is implemented

- Candidate-facing synthetic structs for evidence claims, decision options, mandate constraints and scenarios.
- Five named, stable-ID public development fixtures: productive expansion, asset-price-only boom, hidden maintenance burden, critical-dependency overlap, and stale evidence.
- Evidence usability checks for explicit source, observation time, validity window and qualification state.
- Hard gates for budget, liquidity, maintenance burden and delegated authority.
- Reported valuation change, expected net cash, productive-capacity delta, maintenance burden and critical dependency overlap are separate dimensions.
- An outcome vector that keeps net cash, liquidity, productive capacity, maintenance burden and dependency exposure separate.
- Unit tests for negative controls, stale/expired/missing-source evidence, repeated dependency IDs, constraints and deterministic equality.
- A demo in which the higher-cash option fails the maintenance gate while a lower-cash, higher-capacity option passes.

## Important limits

This is a **prototype contract and rule kernel**, not an investment model. It has no portfolio optimizer, probabilistic forecast, currency/period normalization, live-source adapter, real-world asset model, cryptographic receipt, benchmark corpus loader, hidden holdout evaluator, or integration with Symthaea, Sol-Atlas, Symtropy, Mycelix or Luminous Platform.

The EvidenceState::Qualified label is supplied by the fixture; it does not independently validate a source's truth. The current day fields are synthetic integer day indices. The dependency counter reports overlap with a supplied list of known critical dependencies; it does not compute cascades or infer dependency criticality. expected_net_cash_minor is an input value, not a modeled or validated forecast.

The demo should not be interpreted as proving superior decisions or a real investment result. A clean local test run and independent review are still required before treating the code as validated. The current authoring environment did not have rustc or cargo installed, so no local compile/test result is claimed in this change.

## Market validation

The buyer-side falsification plan is in [BUYER_WORKFLOW_VALIDATION.md](BUYER_WORKFLOW_VALIDATION.md). It recommends testing one asset-heavy operating-company capital-planning workflow rather than positioning ICCB as a replacement for incumbent family-office reporting or accounting products. No customer interviews or paid-pilot commitments are claimed.

## Next steps

1. Run cargo test and cargo fmt against this exact head; fix any compiler, formatting or contract issues.
2. Implement a simple spreadsheet-equivalent baseline over the public fixtures.
3. Move larger fixture definitions into versioned data files and separate candidate-visible observations from a separately controlled holdout evaluator.
4. Introduce deterministic run receipts, manifest/configuration hashes and replay.
5. Add adversarial fixtures and compare per-dimension outcomes against simple baselines, then extend only where results justify it.
