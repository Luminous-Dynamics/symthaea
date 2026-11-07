# Research-taste decision reference

A standalone offline Rust crate with two narrow responsibilities:

1. Validate the checked-in synthetic research-taste decision fixture pack against its canonical 16-case contract.
2. Compute finite-state expected value of sample information (EVSI) for an explicitly declared prior, action set, utility matrix, experiment-outcome likelihood model and cost.

The fixture validator does not score natural-language agent responses. The EVSI calculator does not estimate priors, likelihoods or utilities for the researcher. Those are inputs that must be independently justified and preregistered.

## Commands

cargo fmt --manifest-path tools/sym-rsi-research-taste-reference/Cargo.toml -- --check
cargo clippy --manifest-path tools/sym-rsi-research-taste-reference/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path tools/sym-rsi-research-taste-reference/Cargo.toml
cargo run --manifest-path tools/sym-rsi-research-taste-reference/Cargo.toml -- fixtures docs/research/fixtures/SYM_RSI_RESEARCH_TASTE_DECISION_FIXTURES_V0_2026-11-07.json
cargo run --manifest-path tools/sym-rsi-research-taste-reference/Cargo.toml -- evsi docs/research/fixtures/SYM_RSI_RESEARCH_TASTE_EVSI_EXAMPLE_V0_2026-11-07.json

## EVSI contract

The finite decision problem declares mutually exclusive states and a prior; available actions and their utilities under each state; mutually exclusive experiment outcomes and their likelihood under each state; and full experiment cost in the same units as utility.

The reference computes the best expected utility without the experiment, expected best utility after observing it, EVSI and net EVSI after cost.

Information gain and EVSI differ. An observation may be informative while never changing the optimal decision, producing zero EVSI. The calculator validates probability mass, dimensions, unique labels and finite utilities, but it does not validate whether states, priors, likelihoods, utilities or cost boundaries are scientifically defensible.

The crate is research tooling, not an evaluator authority, receipt service, scientific oracle or promotion service. It is intentionally outside the root production Cargo workspace.
