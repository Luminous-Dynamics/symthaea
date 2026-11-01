# Progressive Forecast Prefix Replay V1

**Status:** first deterministic evaluator slice for [#9752](https://github.com/Luminous-Dynamics/symthaea/issues/9752). It is not yet a full BAGEN reproduction, a feasibility classifier, or a simultaneous-vector calibration result.

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

- a remaining hard-budget receipt per prefix, so false-feasible/false-infeasible classification is not computed;
- a baseline comparison against static estimates, last-observed-cost, selection-blind conformal, CAP, or BAGEN;
- forecast-query overhead in all-in resource usage;
- model/selection-policy replay or the generation of fresh estimates from prefixes; it evaluates already recorded forecasts;
- chronological/task-family split construction or confidence intervals;
- an estimator for informative censoring or a joint six-resource conformal theorem.

## Next increments

1. Add a prefix-level remaining-budget receipt bound to the exact pre-state digest, then report false-feasible and false-infeasible outcomes without relabelling unresolved terminals.
2. Add a frozen rollout/replay fixture that generates multiple forecasts from one trajectory and verifies no future label leaks into earlier prefixes.
3. Add paired baseline predictions on identical frozen prefixes (static, last-observed-cost, linear frontier model, split-vector conformal, and the supported scalar CAP protocol).
4. Record forecast-query overhead and include it in all-in cost.
5. Report confidence intervals and worst-family calibration; preserve censoring and competing risks.

## Claim limits

This first slice validates journal replay and metric accounting only. It does not demonstrate better forecast accuracy, feasible-budget detection, improved task success, or superiority over BAGEN/industry baselines. Those claims require the paired prefix benchmark, disjoint task-family and chronological holdouts, and externally reviewable raw receipts.