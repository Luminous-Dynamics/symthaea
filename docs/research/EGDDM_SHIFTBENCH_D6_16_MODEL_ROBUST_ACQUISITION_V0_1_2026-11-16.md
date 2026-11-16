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

## External benchmark landscape (checked 2026-11-16)

- **Active Causal Discovery Benchmark (ACDB, 16 September 2026)** evaluates causal-graph recovery under budgeted hard interventions and separates skeleton recovery, DAG recovery and intervention efficiency. On its dense v0 ladder, the reported PC + greedy active-orientation baseline reaches 42.7% directed F1 / SHD 4.79; Claude Sonnet 4.6 raw-active reaches 31.7% / 7.25, and GPT-5.4 raw-active 22.9% / 9.27. The authors identify the current numbers as a benchmark audit/calibration stage and note structure-blind baselines and density shortcuts. D6.16 is complementary: it scores safe evidence acquisition over model ambiguity and measurement dependence, not raw graph recovery. Reuse ACDB task families where appropriate, match tools/budgets, and report its native metrics separately. [ACDB](https://arxiv.org/abs/2609.31675).
- **Strategic simulation evaluation (March 2026)** argues that decision benchmarks need uncertainty, complexity, irreversible multiperiod moves and delayed/noisy feedback. This is directly relevant to consequence-aware acquisition policies, but it is a simulation benchmark rather than a model-robust evidence-acquisition benchmark. [Allen & McDonald](https://doi.org/10.1287/stsc.2025.0444).
- **Trustworthy active-learning review (2026)** recommends moving beyond uncertainty-only selection toward multi-criteria acquisition, empirical supervision cost, multi-seed reporting and robustness under realistic shifts. D6.16 should therefore compare robust-risk and cost-aware policies against competitive information-gain/diversity baselines, not random selection alone. [Beyond uncertainty in modern active learning](https://doi.org/10.3389/frai.2026.1844765).
- The **September 2026 agent-evaluation survey** maps 259 primary studies and separately audits 17 prominent benchmarks. Its purposive 17-benchmark matrix reports 0/17 with a complete standardized run-cost record and none with documented joint control of contamination, nondeterminism and execution cost. It explicitly says the mapped corpus is a reproducible sample, not a census of the whole field. D6.16's supplied cost samples are still synthetic, so this is an evaluation-infrastructure gap to address later—not a gap this prototype has already solved. [Nageshwaran et al.](https://link.springer.com/article/10.1007/s10462-026-11678-4).

These numbers are **not directly comparable** to D6.16 fixture calculations: datasets, metrics, model families, budgets and tasks differ. The appropriate path to a SOTA comparison is a matched adapter using the same task families, action budgets, evidence and risk constraints, plus separate native benchmark reports. Do not claim a higher directed-F1 or benchmark win from these synthetic controls.

## Implementation status — 2026-11-16

- Standalone Cargo binary: `tools/development-map-validator/src/bin/model-robust-acquisition.rs`.
- Candidate and protected-oracle schemas are separate; the CLI takes separate candidate-view and evaluator-oracle paths.
- Public synthetic development fixture: two models, three latent states per model, six measurement options and nine candidate traces; an expected report is generated by a separate JavaScript reference calculation.
- **14 Rust unit tests are authored** for the known-answer report, prior-averaging trap, correlated dashboard, independent reference, noisy bounded action, partial/out-of-model support, positive abstention control, future evidence, policy/cutoff/budget tampering, likelihood normalization and unknown cost.
- The candidate fixture contains no model states, likelihoods or oracle fields. This static artifact check does not prove runtime filesystem isolation; public fixtures remain known-answer development controls.
- Candidate/oracle/report fixture JSON and custom structural checks against the committed schema documents report zero errors. This is not a full Draft 2020-12 schema-validator run.
- An independently authored JavaScript evaluator reproduces the complete expected report exactly.
- Rust source delimiters, include paths and test count were statically inspected. **Rust tests have not been compiled or run** because this environment has no Cargo, rustc, rustfmt or Nix. No exact-head CI evidence, blinded trials, real sensor/cost calibration or independent adjudication exists.

## Limits and next gates

v0.1 uses evaluator-supplied finite models and synthetic likelihoods. The public split oracle is for unit tests, not a secret challenge set. It does not authenticate physical measurements, prove conditional independence, calibrate priors, discover omitted model families or estimate real-world harm.

Next: integrate exact posterior envelopes from #10995; compose decision regret from #10109; add independent source-dependency manifests from #10410; extend to held-out noisy/common-cause families; build an exact depth-2 policy-tree oracle; obtain blinded adjudication and real measured costs. Keep this work draft until exact-head Rust formatting, compilation, tests, Clippy and CI run.
