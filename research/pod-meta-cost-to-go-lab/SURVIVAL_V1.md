# Time-to-Sufficient-Proof Survival Baseline V1

**Status:** non-parametric research baseline. Not a frontier-conditioned remaining-cost predictor and not a claim of calibrated future coverage.

## Estimand

The episode-level estimand is work units from the frozen task start until all mandatory proof obligations are independently verified. A completion event is recorded only when the proof evaluator accepts sufficient verified completion.

Each episode ends in exactly one of three classes:

- **Event:** `SufficientProof`.
- **Right-censoring:** budget exhaustion, timeout, infrastructure interruption, or an unknown action outcome. The observation contributes to risk sets through its last observed work endpoint, but it is not labeled a completion failure.
- **Competing terminal event:** permission revocation, explicit abort, or policy abandonment. This is recorded separately from right-censoring because the terminal condition can prevent proof completion in the current episode.

The reason for each censor and competing event remains separately counted. In particular, a budget-stopped episode is not interchangeable with an infrastructure timeout or an ambiguous external-action result.

## Estimators

The baseline emits two distinct quantities.

### Cause-specific product-limit curve

For the sufficient-proof cause, estimate the cause-specific hazard at endpoint (t) as:

[
hat h_{proof}(t) = d_{proof}(t) / n(t)
]

and the cause-specific net survival as:

[
hat S_{net}(t) = prod_{u le t} (1 - hat h_{proof}(u))
]

Competing terminal events are treated as censored for this **net** curve. A median or 90th-percentile net-work estimate is returned only when the observed curve crosses the corresponding survival threshold. Otherwise the quantile is unknown, not extrapolated.

This quantity is not the real-world probability of completion in the presence of competing terminal events.

### Aalen–Johansen completion cumulative incidence

Maintain all-cause event-free survival and update the completion cumulative incidence by:

[
hat F_{proof}(t) =
hat F_{proof}(t^-) +
hat S_{all}(t^-) \, d_{proof}(t) / n(t)
]

where competing terminal events reduce all-cause event-free survival. Right-censored observations do not reduce the survival estimate at their censor time.

The resulting cumulative incidence keeps proof completion distinct from permission loss, abort, and abandonment.

## Risk-set and tie convention

An episode with an endpoint at work (t) belongs to the risk set immediately before the endpoint at (t). Completions, competing events, and censors at the same integer work value therefore share the pre-endpoint denominator. All terminal observations at (t) are removed from the next risk set.

Work is integer-valued in this simulator. The method does not claim sub-unit ordering within a tied interval.

## Assumptions and limits

- Censoring must be independent of latent completion time conditional on the declared cohort for the non-parametric estimates to have the intended interpretation.
- Budget exhaustion is often policy-dependent and can be informative. The baseline does not correct informative censoring.
- Permission revocation / abandonment are competing events; the estimator does not estimate the counterfactual completion time had they not occurred.
- The task-family filter prevents accidental pooling across families; it does not learn transfer to an unseen family.
- The estimator is unconditional from task start. It does not yet condition on the live proof-frontier graph, unresolved obligation structure, model state, or remaining resources.
- It is a survival baseline only, not an IPCW estimator, Cox model, discrete neural hazard model, conformal predictor, or HARP reproduction.
- Synthetic work units are not tokens, FLOPs, dollars, joules, or wall-clock time.

## Qualification cases

The deterministic runner includes:
1. a cohort with budget right-censoring between proof events;
2. a cohort with a permission-revocation competing terminal event and an unknown-action censor.

Unit tests assert risk-set handling, censor/competing cause separation, family filtering, quantile non-identifiability, and invalid-cohort rejection.

The next research step is to add a separate fit/calibration/test protocol with censoring-aware prediction of **remaining** work conditional on a frozen proof frontier, then evaluate coverage and decision utility on held-out task families.
