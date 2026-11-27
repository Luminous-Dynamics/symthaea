# Epistemic adjudication contract harness

This standalone Rust crate compiles the repository's production
`src/web_research/types.rs` and `src/web_research/meta_learning.rs` by path.
It does not copy their logic or replace the full Symthaea crate.

## Contract

- `VerificationOutcome.was_correct` is `Some(true)` or `Some(false)` only when an independent correctness outcome is available.
- `None` means the claim has been assessed by the system's verifier but has not been independently adjudicated.
- Unadjudicated records remain countable, but do not update source credibility, domain expertise, empirical accuracy, readiness or Meta-Phi.
- `LearnerStats` reports total, adjudicated and unadjudicated record counts separately. It also reports adjudication-only Brier score and ten-bin expected calibration error (ECE); both remain `None` until valid independent labels exist, and pending claims cannot alter them.
- History is bounded to 1,000 records. Pending records are evicted before adjudicated records so an unbounded stream of pending claims cannot erase measured calibration history.
- The former serialized boolean representation remains compatible: a legacy boolean deserializes as `Some(bool)`; a missing correctness field deserializes as `None`.

The high-level web-research bridge now records each newly verified claim as unadjudicated. It no longer maps `confidence > 0.7` to a ground-truth correctness label.

## Run

```sh
cargo test --locked --manifest-path tools/sym-rsi-epistemic-adjudication/Cargo.toml
cargo clippy --locked --manifest-path tools/sym-rsi-epistemic-adjudication/Cargo.toml --all-targets -- -D warnings
```

This harness compiles the repository's production type and meta-learning source modules and now runs **20 tests** (five type tests, twelve learner/calibration tests, and three serialization-compatibility tests). It does not compile the entire web-research runtime or prove that external adjudication references are authentic. The caller remains responsible for ensuring any supplied `Some(bool)` is backed by an independent outcome rather than the verifier's own confidence.

## Interpretation boundary

A high epistemic confidence score is still a prediction about a claim, not an observed correctness outcome. The web-research bridge records its claims as pending/unadjudicated, preserving the predicted confidence separately. Only an explicit independent adjudication may call `record_outcome` with `Some(true)` or `Some(false)`.

The current verifier remains a source-evidence heuristic. This change prevents its confidence estimate from training the calibration learner as ground truth; it does not turn multi-source agreement into independent scientific validation.

## Calibration metrics

Brier score is the mean squared error between predicted confidence and the independent binary correctness outcome. ECE uses ten fixed confidence bins and weights each bin's absolute confidence/accuracy gap by its sample proportion. Both metrics use only adjudicated observations with finite probabilities in [0,1]; they are not computed from pending claims or the verifier's own confidence labels. The valid calibration sample count is reported alongside the scores, and both scores remain absent when that count is zero. A missing score is reported as `None`, not as an artificial zero or a neutral calibration claim.


## Selective prediction and abstention

The production module `src/web_research/selective_prediction.rs` implements the separately versioned `symthaea-risk-coverage-v1` evaluator. The standalone harness imports this exact source file; the metric logic is not copied into the tests.

- Only independently adjudicated binary truth values with finite confidence in [0, 1] enter risk, AURC, operating points or confidence intervals. Pending and invalid records remain visible as separate counts.
- Risk is the fraction of wrong accepted predictions. Confidence is confidence in the predicted label, not a correctness label.
- Equal binary64 confidence values are admitted as one group. The requested coverage is rounded upward to the next complete tie group; no record is arbitrarily selected within a tie.
- AURC is the conventional item-wise empirical area, with equal-confidence groups handled by the expected item-wise risk over all within-tie permutations. It therefore reduces to standard AURC for unique confidence values while remaining deterministic and input-order invariant.
- Fixed 50%, 80% and 90% targets are accompanied by achieved coverage, selected sample count and a two-sided 95% Wilson interval when the selected sample meets the configured minimum. Always-answer, random-abstention expectation and inclusive confidence-threshold baselines are also reported.
- Reports stratify by domain, source family, delayed/censored status and adjudicator. Unknown values remain an explicit group.
- Decision costs are absent unless the evaluation supplies explicit false-positive, false-negative and abstention costs. Costs are normalized per original eligible case.
- The default minimum eligible sample size is 20. Tests may lower it to exercise small-cohort edge cases; doing so is not an evaluation recommendation.

Run the same locked harness commands above. The metric evaluates predictions against supplied labels; it does not authenticate adjudicator independence, establish truth, choose a production abstention policy or prove that the live researcher uses the metric.
