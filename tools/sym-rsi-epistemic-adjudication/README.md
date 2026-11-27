# Epistemic adjudication contract harness

This standalone Rust crate compiles the repository's production
`src/web_research/types.rs` and `src/web_research/meta_learning.rs` by path.
It does not copy their logic or replace the full Symthaea crate.

## Contract

- `VerificationOutcome.was_correct` is `Some(true)` or `Some(false)` only when an independent correctness outcome is available.
- `None` means the claim has been assessed by the system's verifier but has not been independently adjudicated.
- Unadjudicated records remain countable, but do not update source credibility, domain expertise, empirical accuracy, readiness or Meta-Phi.
- `LearnerStats` reports total, adjudicated and unadjudicated record counts separately.
- History is bounded to 1,000 records. Pending records are evicted before adjudicated records so an unbounded stream of pending claims cannot erase measured calibration history.
- The former serialized boolean representation remains compatible: a legacy boolean deserializes as `Some(bool)`; a missing correctness field deserializes as `None`.

The high-level web-research bridge now records each newly verified claim as unadjudicated. It no longer maps `confidence > 0.7` to a ground-truth correctness label.

## Run

```sh
cargo test --locked --manifest-path tools/sym-rsi-epistemic-adjudication/Cargo.toml
cargo clippy --locked --manifest-path tools/sym-rsi-epistemic-adjudication/Cargo.toml --all-targets -- -D warnings
```

This harness compiles the repository's production type and meta-learning source modules and runs **17 tests** (five type tests, nine learner tests, and three serialization-compatibility tests). It does not compile the entire web-research runtime or prove that external adjudication references are authentic. The caller remains responsible for ensuring any supplied `Some(bool)` is backed by an independent outcome rather than the verifier's own confidence.

## Interpretation boundary

A high epistemic confidence score is still a prediction about a claim, not an observed correctness outcome. The web-research bridge records its claims as pending/unadjudicated, preserving the predicted confidence separately. Only an explicit independent adjudication may call `record_outcome` with `Some(true)` or `Some(false)`.

The current verifier remains a source-evidence heuristic. This change prevents its confidence estimate from training the calibration learner as ground truth; it does not turn multi-source agreement into independent scientific validation.
