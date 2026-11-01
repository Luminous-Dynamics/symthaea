# Online Repeated-Selection Feedback Ledger V1

**Status:** typed replay and descriptive metric-accounting baseline for issue [#9720](https://github.com/Luminous-Dynamics/symthaea/issues/9720). It is not CAP, an online calibration algorithm, a selection-corrected estimator, or a coverage guarantee.

## Objective

Measure how predictive statements behave after repeated adaptive operation selection, while preserving the timeline and provenance of delayed and partial feedback. Each statement binds the episode and decision sequence, candidate pool, frozen selection policy, selected-population cohort, pre-action state, frontier, selected operation, declared resource dimension, forecast interval, and issue tick.

## Feedback state machine

- **Exact verified cost:** final feedback contains the full six-dimensional measured remaining resource vector, the execution action digest, a meter receipt digest, and an independent completion-verifier digest. The statement's declared resource dimension is projected from that vector.
- **Coverage bit only:** final feedback reveals whether the interval covered the selected label, with the execution action digest and independent evaluator and meter receipts, but does not reveal the exact cost. This models partial feedback without pretending the exact label is available.
- **Right-censored:** final feedback contains a six-dimensional observed lower-bound vector, execution action digest, and meter receipt. It is counted as a definite miss only when the declared dimension's lower bound is strictly above the interval's upper endpoint. Otherwise coverage is unknown and reported separately.
- **Competing terminal:** permission revocation, user abandonment, task abort, or external-input unavailability is recorded separately with the execution action digest and is not converted into a proof-completion cost or ordinary miss.
- **Action outcome unknown:** an interim receipt identifies the ambiguous action. At most one later final receipt may reconcile it, and that final receipt must carry the exact same action digest. A different digest is rejected. Unknown is never counted as covered.
- **Pending:** no feedback has yet arrived for a selected statement at the chosen replay tick.

Receipt sequence is monotone across the full stream with checked overflow; selected decision sequence is strictly increasing within an episode; feedback cannot predate its statement or move backwards in logical receipt time; a final outcome is single-assignment. Every final receipt is action-bound, not merely statement-bound. Late feedback is preserved and replay-prefix reports exclude it until its received tick.

## Metric semantics

At replay tick `t`, `observed_fcr` is:

`coverage_misses / resolved_coverage_statements`

Resolved statements include exact cost labels, coverage-bit labels, and right-censored labels that already prove an interval miss. They exclude pending feedback, unresolved action outcomes, ambiguous right-censoring, and competing terminal events. The report returns `None` when the denominator is zero.

This is intentionally named observed FCR. If feedback resolution is selective or censoring depends on latent cost, the complete-feedback subset can be biased. The report is not a valid population estimate unless the analysis separately establishes the required feedback/censoring assumptions or applies a valid correction. It must not be reported as CAP's selection-conditional or online FCR guarantee.

Each statement declares exactly one resource dimension. Re-run identical episodes for tokens, tool calls, CPU milliseconds, GPU milliseconds, wall milliseconds, and micro-USD. V1 does not claim simultaneous six-dimensional coverage.

## Validation tests

The module tests exact verified-label scoring, delayed replay prefixes, coverage-bit-only feedback, definite versus ambiguous censoring, competing terminal separation, unknown-outcome reconciliation, monotone stream ordering, duplicate final prevention, missing provenance, and dimension projection.

The deterministic CLI matrix has five rows: complete/partial resolved feedback; delayed prefix; unknown then reconciled; definite censored miss; and competing terminal. The matrix verifies state-machine accounting only. It does not measure a predictor or calibrator.

## Algorithm baselines and comparison protocol

Compare at least selection-blind split conformal, the frozen selected-policy split-conformal baseline from #9698, a faithful CAP implementation for a selection rule covered by its published construction, online conformal/ACI, and an adversarial semi-bandit baseline where feedback is partial. Replay the same candidate pools, selections, episode seeds, labels, and policy digests in chronological order.

CAP's JMLR 2025 result gives exact selection-conditional finite-sample coverage under the assumptions and selection-rule constructions in that paper, and discusses long-run FCR under its dynamic extensions. The ICLR 2026 adversarial semi-bandit work connects long-run coverage to regret under partial feedback. Neither result automatically transfers to proof-cost censoring or simultaneous resource vectors.

Report selection rate, resolved-feedback fraction, per-tick and rolling-window observed FCR, interval width and proper scores, abstention, delay/backlog, censoring and competing-risk counts, coverage after policy/feature shift, and verified proof cost/completion. Add run-clustered uncertainty intervals. Keep task families and chronological holdouts disjoint.

## Theory and production gates

Before claiming coverage control, specify the filtration, selection mechanism, feedback process, exchangeability or adversarial assumptions, censoring mechanism, target event, and repeated-statement unit. Reproduce CAP for a supported scalar selection rule before extending it. A six-resource statement needs a separate simultaneous event and theorem. Production use additionally needs cryptographic frontier receipts, signed state/evaluator/meter evidence, durable event storage, and an independent enforcement/verification authority.