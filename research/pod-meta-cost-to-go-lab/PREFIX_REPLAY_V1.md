# Progressive Forecast Prefix Replay V1

**Status:** deterministic evaluator and budget-bound feasibility slice for [#9752](https://github.com/Luminous-Dynamics/symthaea/issues/9752). It is not yet a full BAGEN reproduction or simultaneous-vector calibration result.

## Data contract

The evaluator reads the existing append-only ForecastJournal. Each ForecastReceipt is a pre-execution forecast at one run/decision sequence and carries task family, operation family, the point q50 estimate, marginal q90/q95 upper estimates, mandatory verification reserve, support count, shift status, and immutable pre-state/frontier/model/calibration digests.

A separately appended ForecastOutcomeReceipt provides measured resource work after that forecast, terminal cause, evaluator digest, independent meter digest, and an optional independent proof verifier digest. The evaluator does not rewrite or infer outcomes.

## Prefix replay semantics

One forecast receipt is one frozen prefix. The evaluator emits all rows sorted by run ID, decision sequence, and forecast ID. Each row retains the exact forecast and any observed terminal outcome.

## Label handling

- Sufficient proof: require the verifier digest; compute q50 absolute error, signed bias, marginal q90/q95 upper coverage for all six resource dimensions, and empirical all-six-dimension q90 coverage.
- Right-censored: preserve the censor reason. An observed lower bound strictly above a forecast q90/q95 upper estimate is a definite exceedance diagnostic; otherwise the future cost remains unknown. Censored outcomes never enter exact point-error or exact-label coverage denominators.
- Competing terminal: preserve as a separate event and do not call it proof completion or an ordinary censored proof-cost label.
- Pending: no outcome receipt exists at replay time; retain as pending and exclude from exact-label denominators.

Task-family reports expose counts and all six per-dimension metrics. The aggregate joint q90 coverage is purely empirical; marginal q90 guarantees do not imply joint coverage.

## Budget-bound feasibility

An optional `PrefixBudgetReceipt` binds a remaining six-resource hard-budget vector to the exact `(run_id, decision_sequence, pre_state_digest)` for the same forecast. Duplicate, unknown, missing-digest, or pre-state-mismatched receipts are rejected.

A prediction is declared feasible only when the forecast is `InDomain`, support is nonzero, the completion-probability threshold passes, and q90 remaining-cost plus mandatory verification reserve fits all six budget dimensions. A suspected or unsupported shift abstains rather than issuing a binary feasibility claim. This is a conservative engineering heuristic using marginal q90 values; it is not a joint six-resource coverage guarantee.

Actual feasibility is known when an independently verified sufficient proof completed and measured remaining work fit or exceeded the bound budget, or when a right-censored lower bound already exceeds at least one budget dimension (which proves infeasibility). A censoring lower bound that still fits the budget does not establish completion feasibility. Competing terminals and pending observations remain unknown. Abstentions and unknown outcomes do not enter false-feasible or false-infeasible denominators.

The evaluator reports confusion counts, abstentions, resolved false-feasible rate, and resolved false-infeasible rate. The deterministic fixture includes true feasible, true infeasible, false feasible, false infeasible, competing/pending unknown, and out-of-domain abstention cases. These are implementation checks, not population performance estimates.

## Current metrics

- q50 mean absolute error and signed bias per resource dimension, only on independently verified sufficient-proof labels.
- marginal q90 and q95 empirical upper coverage per dimension, on those same exact labels.
- empirical all-six-dimensions q90 upper-envelope coverage, descriptive only.
- counts of q90/q95 definite exceedances from right-censored lower bounds.
- completed, censored, competing-terminal, pending, and per-task-family counts.

## Qualification fixture

The deterministic CLI fixture emits four PREFIX_REPLAY_ROW records: terminal counts; exact-label token q50/q90/q95 metrics; censored token q90/q95 definite exceedances; and task-family partition size/order. Unit tests cover exact-label scoring, censoring separation, competing-terminal separation, pending forecasts, family partitioning, and deterministic row order.

## Known exclusions

V1 does not yet include:

- paired evaluation of the frontier model, historical baseline, last-observed-cost, conformal and CAP prediction outputs on a large real task suite;
- a baseline comparison against static estimates, last-observed-cost, selection-blind conformal, CAP, or BAGEN;
- forecast-query overhead in all-in resource usage;
- model/selection-policy replay or the generation of fresh estimates from prefixes; it evaluates already recorded forecasts;
- chronological/task-family split construction or confidence intervals;
- an estimator for informative censoring or a joint six-resource conformal theorem.

## Next increments

1. Expand the budget-sidecar replay to paired task-family and chronological holdouts, add cluster-aware confidence intervals, and include forecast-query overhead in all-in cost.
2. Add a frozen rollout/replay fixture that generates multiple forecasts from one trajectory and verifies no future label leaks into earlier prefixes.
3. Add paired baseline predictions on identical frozen prefixes (static, last-observed-cost, linear frontier model, split-vector conformal, and the supported scalar CAP protocol).
4. Record forecast-query overhead and include it in all-in cost.
5. Report confidence intervals and worst-family calibration; preserve censoring and competing risks.

## Remaining-budget feasibility

`evaluate_with_budgets` accepts a `PrefixBudgetReceipt` keyed by exact `(run_id, decision_sequence)` and bound to the forecast's `pre_state_digest`. It rejects duplicate, unknown, zero-identity, and state-mismatched receipts. A prediction is feasible only when the forecast is in-domain, has nonzero support, exceeds the caller's completion-probability threshold, and its q90 resource vector plus mandatory-verification reserve fits all six remaining budget dimensions. Shifted or unsupported forecasts abstain.

Actual feasibility is known for a verified sufficient-proof outcome when measured remaining work fits the budget, or for a right-censored episode when its observed lower bound already exceeds at least one resource limit. Otherwise the actual outcome remains unknown. Competing terminals and pending feedback are excluded from confusion-matrix rates. False-feasible and false-infeasible rates use only resolved predictions of the corresponding class; abstentions are separately counted.

## Frozen historical baseline

`historical_baseline.rs` fits a task-family conditional empirical median and nearest-rank q90/q95 from exact independently verified training completions. Censored, competing, and pending training episodes do not contribute exact cost labels. Unseen task families fall back to a global empirical profile. Evaluation rejects any run ID that overlaps the training cohort and emits q50 absolute error/signed bias plus marginal q90/q95 empirical coverage on the same verified evaluation labels as the frontier-conditioned predictor.

This baseline is a deliberately simple comparator, not a conformal method. Its q90/q95 are empirical order statistics and provide no distribution-free coverage guarantee. The deterministic fixture includes a censored training run (excluded), a family-specific prediction, an unseen-family global fallback, and a disjoint evaluation set.
## Claim limits

This slice validates journal replay and deterministic feasibility accounting. It does not demonstrate better forecast accuracy, lower false-feasible rates, improved task success, or superiority over BAGEN/industry baselines. Those claims require the paired prefix benchmark, disjoint task-family and chronological holdouts, and externally reviewable raw receipts.