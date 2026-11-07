# Research-taste decision reference

A standalone offline Rust crate with two narrow responsibilities:

1. Validate the checked-in synthetic research-taste decision fixture pack against its canonical 16-case contract.
2. Validate a separate nine-case synthetic pivotal-correction pack for project-context sensitivity, irrelevant-context invariance, evidence-based correction rejection, abstention, value-aware stopping and analysis-integrity failures.
3. Compute finite-state expected value of sample information (EVSI), keeping direct experiment cost, foregone-alternative opportunity cost and outcome-conditioned downside risk as separate inputs and outputs.

The fixture validator does not score natural-language agent responses. The EVSI calculator does not estimate priors, likelihoods or utilities for the researcher. Those are inputs that must be independently justified and preregistered.

## Commands

cargo fmt --manifest-path tools/sym-rsi-research-taste-reference/Cargo.toml -- --check
cargo clippy --manifest-path tools/sym-rsi-research-taste-reference/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path tools/sym-rsi-research-taste-reference/Cargo.toml
cargo run --manifest-path tools/sym-rsi-research-taste-reference/Cargo.toml -- fixtures docs/research/fixtures/SYM_RSI_RESEARCH_TASTE_DECISION_FIXTURES_V0_2026-11-07.json
cargo run --manifest-path tools/sym-rsi-research-taste-reference/Cargo.toml -- pivotal docs/research/fixtures/SYM_RSI_RESEARCH_TASTE_PIVOTAL_CORRECTION_FIXTURES_V0_2026-11-07.json
cargo run --manifest-path tools/sym-rsi-research-taste-reference/Cargo.toml -- evsi docs/research/fixtures/SYM_RSI_RESEARCH_TASTE_EVSI_EXAMPLE_V0_2026-11-07.json

## EVSI contract

The finite decision problem declares mutually exclusive states and a prior; available actions and their utilities under each state; mutually exclusive experiment outcomes and their likelihood under each state; direct experiment cost; foregone-alternative opportunity cost; outcome-conditioned downside losses; and a CVaR confidence level and maximum admissible downside threshold. Cost and loss values are expressed in the same preregistered utility units.

The reference reports:

- Gross EVSI: expected improvement in the best downstream action after observing the experiment, before costs.
- Net EVSI: gross EVSI minus direct experiment cost and opportunity cost.
- Expected downside loss and upper-tail conditional value at risk (CVaR).
- A separate risk-limit result and decision-eligibility flag. Positive net EVSI does not override a failed risk gate.

For discrete outcomes, CVaR averages the worst 1 - confidence fraction of the predictive downside-loss distribution. At confidence 0.95, the final 5% tail may split the probability mass of an outcome. The risk threshold is a constraint, not an arbitrary penalty subtracted from expected value. The declared loss model must include adverse consequences relevant to the experiment and cannot represent unmodeled hazards.

Information gain and EVSI differ. An observation may be informative while never changing the optimal decision, producing zero EVSI. The calculator validates probability mass, dimensions, unique labels, finite utilities/costs and risk inputs, but it does not validate whether states, priors, likelihoods, utilities, costs or downside models are scientifically defensible.

The crate is research tooling, not an evaluator authority, receipt service, scientific oracle or promotion service. It is intentionally outside the root production Cargo workspace.

The EVSI input's structural contract is also published at `docs/research/fixtures/SYM_RSI_RESEARCH_TASTE_EVSI_SCHEMA_V0_2026-11-07.json`. The Rust semantic validator additionally checks cross-array dimensions and probability mass. The JSON Schema and semantic checks are separate; neither authenticates the source estimates or proves the model complete. The crate now has 25 authored Rust tests, including six tests for the pivotal-correction pack; all remain unexecuted in the current environment.


## Pivotal correction contract

The `pivotal` CLI validates nine synthetic policy cases across three context behaviors: six require a decision change when decision-relevant evidence changes, one requires invariance to irrelevant context, and two require principled abstention when evidence is non-identifying or qualified reviewers conflict. Cases also cover unsupported corrections, sunk-cost anchoring, negative net EVSI, downside-risk veto and analysis-integrity failure.

These are deliberately **not** SciΠvot's 248 real project-correction cases and are not expert gold. They test fixture semantics and the intended context-sensitivity/invariance contract. Any real correction dataset requires access/licence review, independent expert adjudication and selection-bias analysis. The named human correction remains a candidate, not an automatic oracle.
