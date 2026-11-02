# Conformal Policy Control Research Slice

Standalone Rust research crate for exact finite-support policy clipping and risk diagnostics, tracked by issue #10042.

## Implemented

- Exact rational arithmetic with checked overflow.
- Likelihood-ratio clipped policy on finite support: pi_beta is proportional to min(pi_candidate, beta times pi_safe).
- Exact normalization and total variation from the safe reference.
- Exact expected evaluator loss, reward and discrete upper-tail CVaR.
- Exact finite-support supremum importance weight for a declared calibration mixture.
- CPC-inspired weighted empirical diagnostic, augmented with the declared worst-case bounded test loss.
- Full beta-grid enumeration without assuming risk monotonicity, including a fixture whose evaluator risk rises and then falls as beta increases.
- A JSON run receipt bound to the fixture SHA-256 and exact source commit, with schema in run-receipt-v0.1.schema.json.
- Synthetic fixture with non-monotonic loss over actions and a rare high-risk action.

## Commands

    GIT_COMMIT_SHA=$(git rev-parse HEAD) cargo run --manifest-path research/conformal-policy-control/Cargo.toml --bin cpc-finite-support
    GIT_COMMIT_SHA=$(git rev-parse HEAD) cargo test --manifest-path research/conformal-policy-control/Cargo.toml

Generate Cargo.lock on first dependency resolution, commit it, then use --locked. The CLI refuses to emit a receipt unless GIT_COMMIT_SHA is an exact 40-character lowercase commit SHA; this is metadata, not a signature or proof the worktree was clean.

## Scientific boundary

This is not a full CPC implementation, not an action-conditional conformal method, and not a production safety gate. The weighted empirical diagnostic alone does not establish a formal beta-selection guarantee. Finite support removes sampling error in policy normalization and the supremum test-weight calculation, but calibration outcomes remain random.

The fixture's evaluator risks are synthetic truth for benchmark diagnostics. Never substitute a model's own predicted risks as ground truth. CPC and AC-CP theory may be claimed only after paper-matched reproduction and a theorem-to-code review. Dynamic, delayed, revised, nonstationary feedback is out of scope until separately qualified.
