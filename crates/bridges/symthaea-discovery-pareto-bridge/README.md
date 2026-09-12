# symthaea-discovery-pareto-bridge

Strict adapter from `symthaea-discovery::Evaluation` records to the shared domain-neutral `symthaea-pareto` kernel.

## Cohort rule

One Pareto comparison must use one exact objective schema. Objective order in an `Evaluation` does not matter, but metric, unit, direction, target, and tolerance do.

A mixed cohort such as EV objectives versus long-duration-grid objectives fails closed instead of producing a meaningless frontier.

## Evidence rule

An evaluation receives a Pareto rank only when:

1. its discovery record validates;
2. hard-constraint feasibility is `Feasible`;
3. every declared objective has a matching prediction in the declared unit;
4. one unique highest-fidelity prediction exists for every objective.

`Infeasible`, constraint-`Unknown`, missing-objective, empty-objective, and ambiguous-highest-fidelity evaluations remain explicitly unranked. Unit mismatch is treated as an integrity error rather than as missing evidence.

## Objective mapping

- `Minimize` → kernel `Minimize` with the selected prediction value;
- `Maximize` → kernel `Maximize` with the selected prediction value;
- `Target { value, tolerance }` → kernel `Minimize` of `max(abs(prediction - value) - tolerance, 0)`.

The target transform means all predictions already inside the declared tolerance band are equally optimal on that objective. The bridge does not manufacture a preference for the exact midpoint that the discovery contract did not declare.

## Mutation boundary

`rank_evaluations` is pure with respect to the supplied evaluations and returns a report.

`rank_and_assign` first computes a complete valid report; only after success does it clear/rewrite `Evaluation::pareto_rank`. Thus a schema/integrity error cannot leave a partially updated cohort.

## Authority boundary

Pareto rank is comparative decision-support metadata. It is not evidence fidelity, scientific confidence, feasibility, promotion authority, experimental validation, or deployment approval.
