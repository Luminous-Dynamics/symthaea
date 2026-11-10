# Finite-sample selective-risk certificates for RoboDojo and embodied-policy gating

**Status:** dependency-free Rust contract implementation; synthetic unit tests only; hosted exact-head validation required. This is not a robot safety certificate.

## 1. Research motivation

The existing RoboDojo calibration module reports Brier score, log loss, expected calibration error (ECE), high-confidence failure rate and a risk–coverage curve. Those are useful diagnostics. However, a good overall calibration score does not by itself establish a finite-sample upper bound on failure risk among the *selected subset* of predictions that a runtime gate allows to proceed.

A useful distinction is:
- **Probability calibration:** do predicted probabilities agree with observed frequencies on average?
- **Selective risk control:** after applying one frozen acceptance rule, what upper bound can we place on the actual failure rate among predictions that pass?
- **Physical safety:** did an independently measured physical system remain within its hazard constraints? Statistical task-risk bounds do not establish this.

Recent research makes this a timely direction:
- *Uncertainty-Calibrated Safety Gating for Vision-Language-Action Manipulation Under Domain Shift* (Sensors, 2026) evaluates a model-agnostic gate that routes among policy execution, re-observation and a fallback planner. The paper reports two Isaac Sim household tasks and explicitly examines the reliability/intervention trade-off. Its scores are external results, not a Luminous baseline: https://www.mdpi.com/1424-8220/26/10/3140
- *Learning Robot Safety From Sparse Human Feedback Using Conformal Prediction* (IEEE RAS news, 23 June 2026) describes finite-sample conformal warning regions learned from human trajectory labels and reports a quadrotor study with 30 flights across six navigation tasks: https://www.ieee-ras.org/news/learning-robot-safety-from-sparse-human-feedback-using-conformal-prediction/
- *Learnable Conformal Prediction for Safe and Efficient Robotics under Perception and Planning Uncertainties* (ICRA 2026) explores context-adaptive prediction regions while retaining conformal coverage statements: https://ieeexplore.ieee.org/abstract/document/11697249/
- *Fixed-Sequence Confidence-Interval Calibration for Finite-Sample Selective Risk Control* (2 October 2026) develops one-sided Hoeffding/Clopper–Pearson risk bounds over pre-specified acceptance thresholds. Its experiments are LLM selective answering, not robotics; the statistical framing is relevant, but the benchmark result must not be transferred: https://www.mdpi.com/2571-905X/9/5/109

The Luminous implementation deliberately starts with one pre-registered selective rule rather than a learned threshold selector. That keeps the first claim narrow and auditable.

## 2. Contract

Each held-out episode contributes:
- a stable sample/episode ID;
- task ID;
- exact policy-artifact SHA-256;
- exact environment/shift-condition SHA-256;
- exact evaluator SHA-256;
- predicted probability of full task success;
- predicted probability of an unsafe event;
- observed official success outcome, when available;
- independently adjudicated unsafe-event outcome, when available;
- valid-run status.

The calibration dataset/manifest SHA-256 is recorded separately. The crate checks digest shape and exact identity equality. It does **not** hash the files, authenticate their contents, establish held-out status, or prove that the score was emitted before the outcome; those remain acquisition/manifest responsibilities.

## 3. Frozen selective rule

Before looking at calibration outcomes, preregister:
- `p_success >= p_success_min`;
- `p_unsafe <= p_unsafe_max`;
- maximum accepted task-failure rate;
- maximum accepted unsafe-event rate;
- family-wise confidence error `delta`;
- minimum accepted sample count.

Only episodes that satisfy both score thresholds enter the accepted cohort. Other episodes remain counted as rejected and are not mixed into accepted-risk denominators.

This is a single fixed threshold pair. Trying many thresholds and reporting whichever passes would introduce selection/multiplicity; that requires a separate fixed-sequence or multiplicity-controlled procedure and is intentionally not implemented here.

## 4. Exact one-sided bounds

Let (n) be the number of accepted episodes, (k_f) the accepted task failures, and (k_u) the accepted unsafe events.

For either event count (k), the one-sided Clopper–Pearson upper confidence bound (U(k,n,\alpha)) is the smallest (p) satisfying:

[
\Pr[X \le k \mid X \sim \mathrm{Binomial}(n,p)] = \alpha.
]

For zero observed events the bound is:

[
U(0,n,\alpha)=1-\alpha^{1/n}.
]

The failure and unsafe-event claims each use (\alpha=\delta/2). By the union bound, the two bounds hold simultaneously with confidence at least (1-\delta), under the sampling assumptions below.

The implementation solves the binomial-tail equation with a stable log-probability recurrence and bisection. It has synthetic regression tests for the closed form, nonzero-event quantiles, monotonicity in sample count and monotonicity in event count.

### Sample-size consequences

At `delta = 0.05`, each one-sided bound uses `alpha = 0.025`. With zero observed events:

| Accepted calibration episodes | Upper event-rate bound |
|---:|---:|
| 10 | 30.85% |
| 50 | 7.11% |
| 72 | 4.99% |
| 80 | 4.51% |
| 100 | 3.62% |
| 200 | 1.83% |

Therefore:
- A 5% risk limit needs at least 72 accepted zero-event episodes for this simultaneous two-rate claim.
- A 1% limit needs at least 368 accepted zero-event episodes.
- Fifty seeds per task/condition cannot certify a 5% limit, even if no failure is observed.
- The accepted cohort may be smaller than the planned cohort because of the score filter; size the data collection for accepted (n), not only total attempts.

These are mathematical sample-size consequences, not claims about any benchmark's observed performance.

## 5. Missingness and validity

The implementation fails closed:
- Invalid/interrupted runs count as task failures, not as removed rows or zero-cost successes.
- A valid run with no official task outcome leaves task-risk unmeasured.
- A missing unsafe-event label leaves hazard-risk unmeasured.
- If either outcome is missing in the accepted cohort, the corresponding bound is withheld and the certificate status is `IncompleteOutcomeEvidence`.
- Duplicate sample IDs, mixed task/policy/environment/evaluator contexts, malformed digests and invalid probabilities reject.
- If accepted sample count is below the minimum, status is `InsufficientAcceptedSamples`.
- The certificate's status and numerical bounds are private after creation; runtime callers can inspect them but cannot mutate a qualified certificate into a different claim.

## 6. Runtime use and limits

A runtime score check requires:
- exact task, policy, environment-condition and evaluator identity match;
- predicted scores within the frozen acceptance region;
- a qualified certificate whose two upper bounds satisfy their declared limits.

A certificate for one condition does not transfer to a shifted camera, lighting condition, embodiment, sensor calibration, action executor, policy checkpoint or evaluator. Use separate frozen contexts and calibration cohorts; do not reuse an ID certificate as an OOD guarantee.

The module assumes independent/exchangeable calibration and deployment episodes within the exact context. It cannot verify that assumption. Temporal correlation between repeated episodes, adaptive data collection, leakage from benchmark seeds, threshold selection after seeing outcomes, or post-hoc removal of failed trials invalidate the intended interpretation. A cross-context programme must preregister a multiplicity plan or allocate the overall error budget across certificates.

This is only a statistical selective-risk gate. It does not model rare catastrophic hazards outside the labelled unsafe-event definition, verify instrument truth, bound low-level joint/contact risk, prove controller authority, replace collision/workspace/interlock checks, or issue/authorize motor commands.

## 7. Relation to active sensing

Compose this certificate with the corrected active-sensing decision rule as a separate final evidence gate:
1. The sensing primitive decides whether another observation is mandatory, optional and worth its cost, or whether to stop.
2. The observation is acquired and its provenance/currentness checked.
3. The policy recomputes the task-success and unsafe-event forecasts.
4. The exact-context selective-risk certificate checks whether the recomputed score lies inside the qualified acceptance region.
5. Existing Symtropy runtime authority and physical safeguards independently admit or reject any controller request.

A sensing recommendation is not an execution authorization. A risk certificate is not a controller capability. Unknown, missing or mismatched evidence must not be converted into permission.

## 8. Benchmark experiment

Use the same policy, action executor, task/simulator, task seeds and score computation. Split by episode/scene before tuning; never calibrate on evaluation outcomes.

Compare at matched operating points:
- no gate (base policy);
- raw point-probability threshold;
- Brier/ECE-calibrated threshold;
- this fixed selective-risk certificate;
- conformal or fixed-sequence selective-risk baseline where reproducible.

Report for every task × ID/OOD/shift stratum:
- accepted coverage and abstention/intervention rate;
- accepted full-task failure rate and exact upper confidence bound;
- accepted unsafe-event rate and exact upper confidence bound;
- official success over all attempted episodes (not just accepted);
- unresolved/missing-label count;
- clean competence and shift-specific transition matrix;
- sensing/action energy, latency, replan/fallback counts and total episode resources;
- independent scenario-family clustered intervals for performance deltas.

The primary frontier is **coverage versus certified selective risk**, subject to full-attempt completion, cost and operational-failure constraints. A system cannot claim a win by lowering risk through refusing nearly every task, hiding unresolved labels, or moving more episodes into unreported human fallback.

## 9. Implementation location and claim ceiling

Implementation: `experiments/robodojo-contract/src/selective_risk.rs`, exported by the nested Rust crate.

The unit tests are synthetic contract/algorithm tests. They do not demonstrate a calibrated Symthaea model, native benchmark performance, risk reduction, HIL or physical safety. The exact-head hosted Rust workflow must pass before this code slice is considered build-qualified.
