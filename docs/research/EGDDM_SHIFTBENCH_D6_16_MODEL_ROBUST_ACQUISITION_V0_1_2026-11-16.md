# EGDDM ShiftBench D6.16 — Robust evidence acquisition under model ambiguity v0.1

**Date:** 2026-11-16  
**Status:** evaluator-only synthetic known-answer prototype; no agent-performance or real sensor-calibration claim  
**Related:** [Issue #12515](https://github.com/Luminous-Dynamics/symthaea/issues/12515); D6.15 [#12475](https://github.com/Luminous-Dynamics/symthaea/issues/12475); finite model-family posterior envelopes [#10995](https://github.com/Luminous-Dynamics/symthaea/issues/10995); decision-aware robust investigation [#10109](https://github.com/Luminous-Dynamics/symthaea/issues/10109); dependence provenance [#10410](https://github.com/Luminous-Dynamics/symthaea/issues/10410).

## Research question and paradigm

Does an acquisition policy remain safe and useful when several plausible models disagree, measurements are noisy, and apparently separate checks share a common cause?

D6.15 begins with one supplied point distribution and deterministic measurement outcomes. D6.16 retains a **family of models** and evaluates the worst-case posterior risk across them. It does not average an unweighted model family. Each model updates independently using history-conditioned likelihoods, allowing a second measurement to depend on the first result and to encode common-cause dependence.

The model-family engine belongs to #10995; decision regret/utility semantics belong to #10109; measured cost distributions remain D6.7/#12352. This evaluator owns only the matched acquisition-policy benchmark.

## v0.1 contract

Two files are required:
- candidate view: actions, candidate measurement names/availability, cost estimates, and observed measurement trace;
- protected evaluator oracle: separate authoritative action/cutoff/budget, candidate-measurement cost samples, finite models, state priors/safety labels and outcome likelihoods.

A model contains states with rational basis-point prior masses and safe-action sets. Each likelihood row is keyed by the complete prior measurement/outcome history and the next measurement. For each state, probabilities across that row's possible outcomes must sum to 10,000 basis points. Shared-source checks can therefore have conditional likelihood 100% after an earlier result rather than being multiplied as independent evidence.

After each acquired observation, the evaluator updates each model separately, retains zero-support models as explicitly unsupported, never discards a model to improve a score, and takes the maximum risk across the unweighted model family. An irreversible action requires every model to support the evidence and every model to have zero unsafe posterior mass.

Only depth-2 traces are accepted in v0.1. Integer posterior weights are reduced by their greatest common divisor after each update to preserve exact ratios and bound growth. This is not a general POMDP solver or unrestricted robust-policy optimizer.

## Synthetic fixture

Two plausible models share three semantic states (safe, hazard, critical) but have different priors and likelihoods:
- **model-average-trap:** nominal model risk is 20%; stress model risk is 40%; the bounded-probe allowance is 35%. The unweighted average would be 30% and would act, while the correct worst-case envelope rejects it.
- **primary-only-release:** one sensor leaves unsafe posterior mass in both models.
- **shared-dashboard-release:** a dashboard derived from the sensor is conditionally uninformative after the sensor pass; the robust risk and normalized model posteriors do not improve.
- **independent-reference-release:** a discriminating reference outcome removes unsafe mass in both models.
- **noisy-bounded-probe:** noisy evidence leaves nonzero posterior risk but supports a reversible probe within the declared 35% envelope.
- **partial-model-support:** one model assigns zero mass to the observed outcome; the model is retained as unsupported and the action remains blocked.
- **out-of-model-observation:** no model supports the result; all model support is lost and action is blocked.
- **always-abstain-positive-control:** sufficient evidence exists, so blanket abstention is scored as unnecessary.
- **future-reference:** late evidence cannot change the earlier decision.

Priors, likelihoods, cost samples and traces are synthetic known-answer inputs, not real measurements, calibrated sensor characteristics, production safety estimates or agent trial results.

## Baselines and metrics

Compare D6.15 point-prior behavior, full revalidation, cheapest-first, information-gain-first, point EVSI, robust envelope from #10995, minimax regret from #10109, common-cause-unaware vs dependency-aware evidence handling, one-step greedy vs exact depth-2 oracle, always-act, always-abstain and expert/operator decisions.

Report maximum model-conditioned unsafe risk, worst-case regret, model support coverage, unsupported conditioning, shared-source false corroboration, cost/reviewer burden, latency and justified/unnecessary abstention as separate axes. Never average models or collapse metrics without a separately justified weighting protocol. Critical safety and authority failures remain hard gates.

## Limits and next gates

v0.1 uses evaluator-supplied finite models and synthetic likelihoods. The public split oracle is for unit tests, not a secret challenge set. It does not authenticate physical measurements, prove conditional independence, calibrate priors, discover omitted model families or estimate real-world harm.

Next: integrate exact posterior envelopes from #10995; compose decision regret from #10109; add independent source-dependency manifests from #10410; extend to held-out noisy/common-cause families; build an exact depth-2 policy-tree oracle; obtain blinded adjudication and real measured costs. Keep this work draft until exact-head Rust formatting, compilation, tests, Clippy and CI run.
