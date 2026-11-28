# MEL-HEAR-014 — EVSI-driven active auditory analysis v0.1

**Status:** standalone decision-policy implementation. Likelihood models are supplied and externally calibrated; this crate does not infer them from audio.

## Paradigm

Choose the next audio analysis based on the expected reduction in task decision risk, rather than always running every transform, using the highest-entropy uncertainty measure, or selecting the most expensive model.

For prior `p(h)`, decision set `d` and loss `L(d,h)`, current Bayes risk is:

`R(p) = min_d Σ_h p(h) L(d,h)`

For candidate measurement `a` with outcome likelihoods `P(y|h,a)`:

`EVSI(a) = R(p) - Σ_y P(y|a) R(p(h|y,a))`

A declared cost conversion gives net value:

`U(a) = EVSI(a) - λ cost(a)`

Only an admissible, in-domain measurement within the resource budget and with strictly positive net value is selected. Otherwise the policy returns an explicit no-action reason.

## Why this is different

Entropy reduction is not the objective. A measurement can substantially reduce uncertainty while failing to change the decision that matters. EVSI uses the actual task loss matrix, so the selected analysis is the one expected to improve the decision, not merely sharpen a posterior distribution.

The current policy is deliberately one-step and myopic. Multi-step planning must account for observation-contingent future action sets and costs, and is not claimed here.

## Implemented contract

- Validated prior, hypothesis identities, decision losses, measurement actions and outcome likelihoods.
- Per-hypothesis outcome probabilities must form a complete distribution.
- Likelihood-model digests must be bound and action costs finite/positive.
- Exact Bayes-risk and expected posterior-risk calculations.
- EVSI, expected posterior risk, net value and EVSI-per-cost emitted separately.
- Deterministic tie-breaking independent of caller action order.
- Explicit stop reasons: out-of-domain, no candidates, no admissible actions, budget exhausted and no positive net value.
- Zero-probability outcomes are retained as impossible and never turned into fabricated posteriors.
- A posterior update helper accepts only an actually observed named outcome.

## Known-answer example

A two-hypothesis decision starts with prior `[0.5, 0.5]` and 0/1 classification loss. A calibrated 90%-accurate context-window action reduces expected Bayes risk from 0.5 to 0.1 (EVSI 0.4) at cost 0.1. A perfect stereo probe reduces risk to zero (EVSI 0.5) but costs 0.8. With `λ = 0.2`, net values are 0.38 and 0.34, so the cheaper context window is selected. These are synthetic known-answer probabilities, not measured auditory performance.

## Benchmark programme

Compare EVSI/cost against fixed schedules, random action selection and entropy-reduction-only policies under equal resource budgets. Measure final decision risk, coverage, abstention, analysis count, latency/energy/review cost, calibration of action likelihoods, and regret against exhaustive oracles on small action spaces.

Prospective target: at least 20% lower decision risk at equal budget, or at least 20% lower resource use at equal decision risk, against preregistered fixed and entropy-only baselines on held-out synthetic and real-audio strata. These are targets, not results.

## Claim boundary

The policy is only as reliable as the prior, loss matrix and calibrated observation likelihoods. It cannot make a poorly calibrated likelihood model trustworthy merely by computing EVSI. Real integration needs held-out calibration, explicit model versioning and independent action-outcome evaluation.
