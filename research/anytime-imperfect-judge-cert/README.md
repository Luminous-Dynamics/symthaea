# Anytime Imperfect-Judge Certification Kernel

First std-only mathematical slice for issue #9450.

Run:

    cargo fmt --manifest-path research/anytime-imperfect-judge-cert/Cargo.toml -- --check
    cargo test --locked --manifest-path research/anytime-imperfect-judge-cert/Cargo.toml

The crate implements summable alpha spending, one-sided Hoeffding bounds, the four-bound measurement-channel envelope, the TPR-lower / FPR-upper discrimination gate, a consistency gate for test-positive upper versus FPR lower bounds, checked count aggregation, immutable validated bound objects, and a strict affirmative certification threshold with an inconclusive fallback.

The deterministic sequential-null Monte Carlo test is an implementation smoke test, not a coverage proof or empirical certification.

Scope limitations:
- Requires predeclared iid Bernoulli streams; adaptive stream choice is not the same as adaptive item choice within a stream.
- Calibration abstentions must already be counted as positive/fail actions in TP/FP; test abstentions must be included in test_positive.
- No JSON receipts, authenticated sampling sequence, external references, GLMM, finite-bank sampling without replacement, propensity weighting, or provider adapter.
