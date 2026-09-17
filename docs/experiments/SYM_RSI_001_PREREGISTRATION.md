# SYM-RSI-001 — Historical Replay Improves Exploration

**Status:** preregistered design only; no result is claimed by this document.

## Question

Does replay-grounded policy improvement reduce search cost and/or improve held-out solution quality, and does adding counterfactual dreaming provide value beyond exact replay alone?

## Frozen conditions

A. **Fixed exploration** — incumbent exploration policy; no replay adaptation and no dream-derived priors.

B. **Existing dream feedback** — incumbent exploration policy plus Symthaea counterfactual dream feedback, with generated worlds kept non-empirical until separately validated.

C. **Exact replay policy improvement** — candidate exploration policies are selected only from exact historical replay worlds. The incumbent remains in every candidate set.

D. **Exact replay + counterfactual dreaming** — condition C plus dream-generated candidate hypotheses/action priors. Dream outputs do not count as empirical evidence and cannot directly increase confidence.

## Primary contrasts

1. **C vs A** — tests the Dream-RSI-like replay mechanism independently of imagination.
2. **D vs C** — tests whether counterfactual imagination adds value beyond replay-grounded policy improvement.

These contrasts are primary. B vs A and D vs B are secondary/descriptive unless separately powered.

## Domains

Use at least three executable domains with materially different structure. Each domain must expose deterministic or seed-controlled scoring and a held-out task split. The exact domains and seeds must be frozen before executing the first measured run.

No domain may be added or removed after looking at primary-outcome results without starting a new evidence lineage.

## Metrics

Primary:
- held-out best solution quality
- total environment/evaluator calls
- total compute cost or normalized wall-clock cost

Secondary:
- out-of-distribution performance
- calibration / Brier score where probabilistic predictions exist
- regression rate against incumbent
- policy instability / churn
- replay-pool coverage
- unsupported-action rate
- safety/constraint violations

## Evidence rules

- Exact replay may only traverse recorded transitions in `ExperienceTree`.
- Missing replay edges must fail closed as `UnsupportedAction`.
- Generated counterfactuals, model predictions, interpolations, extrapolations, and adversarial worlds are non-empirical unless independently validated.
- The incumbent policy must be present in every replay-selection candidate set.
- Replay monotonicity is claimed only for the frozen replay objective over the frozen replay pool; it is not a claim of universal capability improvement.
- Any environment/toolchain change after evidence collection begins starts a new evidence lineage.

## Holdouts

Every domain must have:
1. training/replay worlds available to the policy selector,
2. held-out replay worlds not used during policy selection,
3. temporally later or independently sampled execution tasks,
4. an OOD slice with a preregistered construction rule.

## Promotion gate

A candidate policy may enter shadow execution only if:
- it does not underperform the incumbent on the frozen replay objective,
- it does not materially degrade held-out replay performance beyond the preregistered tolerance,
- it has zero new authority-boundary or safety-constraint violations,
- all evidence/provenance fields are complete.

A candidate may enter limited real execution only after passing shadow execution and a fresh controlled evaluation.

## Claim boundary

A positive C vs A result supports: **historical exact replay can improve exploration policy under the tested domains and protocol**.

A positive D vs C result supports: **under the tested domains and protocol, grounded counterfactual dreaming adds value beyond exact replay alone**.

Neither result by itself establishes recursive general intelligence, open-ended self-improvement, consciousness, or monotonic improvement outside the tested distributions.
