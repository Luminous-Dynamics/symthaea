# Online Repeated-Selection Feedback Ledger V1

**Status:** typed replay ledger plus a narrow fixed-initial-holdout, decision-driven CAP baseline for issue [#9720](https://github.com/Luminous-Dynamics/symthaea/issues/9720). The replay ledger's observed FCR is descriptive accounting; the separate CAP interval routine implements only the fixed-holdout decision-driven construction described below.

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

## Fixed-holdout, decision-driven CAP baseline

`decision_driven_cap.rs` implements the fixed initial-holdout construction for a single scalar resource dimension. The selection rule is explicit: select the current example when its frozen score is no greater than `base_threshold + increment * prior_selected_count`. Its state uses only the prior binary selection count. After selection, the exact same current threshold is applied to every example in the independent, fixed initial holdout; only those picked residuals enter the split-conformal quantile. Later selected observations are never appended to this fixed holdout.

The routine requires separate declared training and holdout partition digests, frozen predictor/score/policy/cohort identities, exact metered labels, and independent completion-verifier receipts. If the selected calibration subset is too small for the finite-sample conformal rank, it returns an unbounded interval rather than extrapolating. The three `CAP_DECISION_ROW` matrix rows cover a selected fixed-holdout interval, a later decision with an updated threshold, and insufficient support.

The relevant CAP paper proves selection-conditional finite-sample coverage and real-time FCR control for its fixed-holdout decision-driven construction under i.i.d. holdout/test examples and a selection rule independent of the initial holdout; its full-holdout and symmetric-threshold variants require their own adaptive-pick rules/conditions. Our code implements the narrow fixed-holdout threshold family only. Partition digests are declarations, not cryptographic evidence of independence. This implementation does not handle arbitrary decision-driven policies, data-dependent model retraining, online drift, censored labels, partial feedback, competing risks, or simultaneous six-resource coverage.

Before treating this as a production guarantee, the independence assumptions must be established by the data pipeline, not merely asserted in a config. Reproduce both the CAP positive cases and known-invalid selection/calibration constructions as negative controls. The 2025 analysis by Sale and Ramdas is a required cautionary reference: it identifies incorrect exchangeability arguments in some online selective-conformal strategies and provides corrected calibration-selection procedures.

## Online feedback ledger distinction

The online replay ledger in `online_selection.rs` remains a state-machine and descriptive-metric substrate. Its `observed_fcr` over resolved labels is not the CAP estimator or guarantee. CAP calibration currently consumes a declared fixed holdout of exact verified scalar costs; it does not ingest `RightCensored`, `CoverageBitOnly`, or competing-terminal observations. Those require a separately justified censoring/partial-feedback calibration method.

## Algorithm baselines and comparison protocol

Compare at least selection-blind split conformal, the frozen selected-policy split-conformal baseline from #9698, a faithful CAP implementation for a selection rule covered by its published construction, online conformal/ACI, and an adversarial semi-bandit baseline where feedback is partial. Replay the same candidate pools, selections, episode seeds, labels, and policy digests in chronological order.

CAP's JMLR 2025 result provides finite-sample selection-conditional coverage and real-time FCR control for specific decision-driven and symmetric-threshold constructions; compare our fixed-holdout implementation against the decision-driven theorem, not against every CAP variant. Sale and Ramdas (TMLR 2025) show that seemingly plausible online selection/calibration procedures can violate the required exchangeability, so include their corrected methods and failure cases. Online conformal under adversarial semi-bandit feedback is a separate comparator for partial labels. None of these results automatically transfers to proof-cost censoring or simultaneous resource vectors.

Report selection rate, resolved-feedback fraction, per-tick and rolling-window observed FCR, interval width and proper scores, abstention, delay/backlog, censoring and competing-risk counts, coverage after policy/feature shift, and verified proof cost/completion. Add run-clustered uncertainty intervals. Keep task families and chronological holdouts disjoint.

## Theory and production gates

Before claiming coverage control, specify the filtration, selection mechanism, feedback process, exchangeability or adversarial assumptions, censoring mechanism, target event, and repeated-statement unit. Reproduce CAP for a supported scalar selection rule before extending it. A six-resource statement needs a separate simultaneous event and theorem. Production use additionally needs cryptographic frontier receipts, signed state/evaluator/meter evidence, durable event storage, and an independent enforcement/verification authority.
## Primary statistical references

- Bao, Huo, Ren and Zou, CAP (JMLR 26, 2025): https://jmlr.org/papers/v26/24-0452.html
- Sale and Ramdas, Online Selective Conformal Prediction: Errors and Solutions (TMLR 2025): https://arxiv.org/abs/2503.16809
- Online conformal prediction with adversarial semi-bandit feedback (ICLR 2026): https://proceedings.iclr.cc/paper_files/paper/2026/hash/4eb5daabc45b45a9a312aa2c8fca8a74-Abstract-Conference.html
