# SDCB scientific measurement intelligence roadmap — 2026 Q4

**Purpose:** consolidate the exact-reference, measurement-assurance, identifiability, continuous common-cause, and real-data evaluation work into one executable research programme.

**Claim boundary:** the current portfolio consists of bounded research references and evaluation plans. It is not yet a validated physical measurement platform or an autonomous laboratory system.

## 1. Research thesis

A scientific agent must distinguish:

1. the physical quantity or mechanism being investigated;
2. the measurement system's shared and local uncertainty;
3. the available experiments' ability to distinguish rival explanations;
4. the downstream decision value of collecting more evidence;
5. whether a particular evidence receipt is authentic, context-valid, and safe to apply.

The intended architecture is:

**scientific hypotheses + measurement graph → candidate experiment design → budget/identifiability gate → authorized execution → independent receipt verification → measurement-assurance gate → atomic belief transition → independent replay.**

The central rule is that agreement between channels is not independent confirmation unless the relevant shared causes and provenance relationships have been accounted for.

## 2. Public research stack

| Contribution | Research capability | What it does not yet establish |
|---|---|---|
| #13762 | Exact finite instrument-aware sequential experiment design | Physical calibration, production selector |
| #13793 | Exact finite multi-device Markov drift and replication reference | Shared common-cause transitions or OU dynamics |
| #13802 | Real-sensor calibration/drift plan and measurement-assurance benchmark matrix | Raw-data benchmark execution |
| #13804 | Exact finite two-stage shared-cause experiment-selection oracle | Continuous factors or compiled Rust result |
| #13805 | Evidence-gated joint science/common/device posterior transition | Cryptographic authenticity or dynamic common-state propagation |
| #13831 | Paired-world observational-equivalence and intervention-separation oracle | General causal discovery or sequential intervention trees |
| #13848 | Continuous linear-Gaussian shared-factor measurement subset oracle | Full prior/residual covariance, dynamic drift, or physical validation |

The PRs are research drafts. A mergeable GitHub state is not test evidence. Authored tests, independent analytical checks, local execution, hosted CI, and physical validation must remain distinct evidence levels.

## 3. Paradigm shifts

### 3.1 Measurement graph before sensor fusion

Use an explicit dependency graph for:
- physical measurands and spatial/temporal field dependence;
- environment and acquisition-system factors;
- calibration and reference-chain ancestry;
- batch, operator, preparation, firmware, power and clock factors;
- device-local bias and drift;
- residual measurement noise.

This graph prevents two different structures from being collapsed into one correlation coefficient. Correlation in the measurand is not the same as correlation in sensor error.

### 3.2 Identifiability before confidence

Before raising confidence in a scientific claim, maintain a set of plausible competing measurement/science worlds. Ask whether their predicted observation distributions differ under the current candidate experiment set. If not, confidence must not be manufactured by repetition or channel count.

### 3.3 Target decision value before generic information gain

Choose the quantity of interest and terminal action first. Report joint information gain as a diagnostic, but optimize target-parameter uncertainty or expected downstream utility when that is the scientific objective.

### 3.4 Measurement assurance as a state, not a timestamp

Calibration validity should depend on evidence about the specific instrument, standard, environment, use, stability and uncertainty budget. Keep contractual or safety capture-window rules as hard constraints, but do not mistake an arbitrary TTL for a measurement-assurance argument.

### 3.5 Evidence receipts as part of scientific state

A posterior transition must be bound to the exact task, model, hypothesis set, device/reference graph, calibration context, evidence versions, observation digest, verifier and cost. Invalid evidence must leave the entire scientific and measurement state unchanged.

## 4. Next engineering increments

### M0 — compile and exact-head verification

- Obtain a Rust 2024 toolchain in a reproducible Nix/CI environment.
- Run rustfmt, strict Clippy, and every test on the exact PR head.
- Recompute JSON/schema/known-answer fixtures with a second implementation.
- Record full SHAs and distinguish local from hosted execution.
- Add adversarial negative controls before describing a reference as ready for merge.

### M1 — full-covariance continuous Gaussian model

Extend #13848 from independent Gaussian priors to a validated symmetric positive-definite prior covariance and, separately, an explicit residual covariance matrix.

Represent a finite temporal OU/AR(1) process as correlated time-indexed latent states. Do not claim OU dynamics when the code only has independent per-time states.

Required controls:
- correlated prior states;
- shared versus local factor attribution;
- residual-correlation misspecification;
- ill-conditioning and near-collinear science/factor loadings;
- covariance symmetry, positive-definiteness and numerical stability;
- matched exact small-matrix oracle.

### M2 — sequential and robust design

Extend #13831 and #13804 into exact two-stage contingent experiment trees:
- second action selected from first outcome;
- multiple paired causal worlds;
- calibration refresh/reset actions;
- common-cause state transition between stages;
- hard pathwise and expected budgets kept separate;
- abstention and partial-identifiability outcomes;
- robust maximin over a frozen set of plausible covariance/noise scenarios.

Keep finite exact policy enumeration as an evaluator; do not imply that a scalable policy is exact.

### M3 — dynamic EGBT measurement state

Propagate the global science posterior, shared-cause states, device-local states, calibration/reference ancestry, and evidence versions through time. Include stale-context handling, atomic update/reject semantics, verifier revocation, and independent replay.

Keep three claims independent:
1. receipt authenticity;
2. measurement validity and uncertainty;
3. scientific/model validity and identifiability.

### M4 — real-data tournament

- Reproduce Inha's published S1–S4 calibration baselines and splits.
- Hold the calibration predictor fixed while comparing selection policies.
- Hold the selection policy fixed while comparing calibration/uncertainty models.
- Add a separately labelled synthetic common-cause intervention track with known ground truth.
- Use UCI chronological batches only for their supported drift/channel tasks until dataset licence wording is clarified.
- Do not infer physical common-cause mechanisms from unlabelled residual correlation.

### M5 — HDC/CfC policy integration

Use Symthaea's HDC/CfC systems to propose candidate experiments, approximate belief states, and generalize across candidate sets. Train against the exact finite oracles and test on held-out causal mechanisms, reference ancestries, device cohorts and time periods.

The learned policy must not override:
- exact finite negative-control results;
- hard resource/safety budgets;
- evidence acceptance rules;
- abstention when the supplied model cannot discriminate the hypotheses.

## 5. Benchmark programme

Run three competitions, with a factorial decomposition:

| Competition | Main question | Representative comparators |
|---|---|---|
| Prediction/calibration | How accurately and cheaply does the calibration model predict reference measurements? | Inha baselines: iTransformer, TimeXer, DLinear, XLinear, TimeMixer, SenDaL, TESLA, authors' model |
| Experiment selection | Which measurements best reduce target scientific uncertainty or regret at matched cost? | Random/fixed, EIG, D-optimal, A-optimal, c-optimal, EVSI, hierarchical calibration, robust maximin, SDCB |
| Evidence assurance | Are measurements correctly bound, independently verified, and replayable? | Stale/wrong-chain/duplicate/revoked controls, invalid-receipt mutation tests, replay |

Mandatory factorial experiments:
- fixed calibration model, change only selection policy;
- fixed selection policy, change only uncertainty/calibration model;
- cross both to measure interaction.

## 6. Competitive gates

Proposed research targets, not achievements or industry standards:

- 100% agreement with exact finite oracles on frozen known-answer cases.
- Zero hard-budget violations.
- Zero invalid, stale, wrong-chain, or duplicate receipt acceptance in frozen controls.
- Zero false independent-confirmation claims in shared-cause negative controls.
- Non-inferiority on genuine independent-sensor controls.
- At least 10% lower target scientific posterior variance per equal cost than matched i.i.d./hypothesis-only EIG on held-out shared-cause tasks, with paired confidence intervals excluding zero.
- Approximate-policy regret no greater than 2% against exact small-instance oracles on the declared evaluation set.
- Nominal 90% and 95% interval coverage assessed by task family, not just pooled.
- Reproduced external calibration results before claiming an empirical SOTA comparison.
- At least one independently executed, licensed real-data result before claiming a real-world benefit.

Do not aggregate all measures into one score. Publish Pareto curves for scientific quality, reference cost, latency, memory, energy and assurance failures.

## 7. What comparison with the field must acknowledge

- NIST does not prescribe one universal recalibration interval; it points to application accuracy requirements, instrument stability, environmental effects, measurement-assurance programmes and control-chart evidence.
- Metrological traceability requires an unbroken calibration chain with uncertainty contributions. A device label or a green status flag does not create traceability.
- Published sensor-network studies already model space-time-correlated sensor errors and show that i.i.d. assumptions can change uncertainty calculations.
- Recent optimal-design work already demonstrates that correlated observation noise can change optimal sampling schedules.
- The Inha 2026 benchmark already evaluates reference-efficient calibration, unseen-location transfer, long-term drift, event-conditioned shifts, and edge deployment.

Therefore, do not claim that SDCB invents correlated sensor models, calibration networks, optimal experiment design, or intervention-based identifiability. The testable differentiation is their integration with target-decision-aware selection, paired-world abstention, exact evaluator oracles, reference ancestry, and evidence-bound posterior transitions.

## 8. External references and dataset caveats

- NIST, [Recommended Calibration Interval](https://www.nist.gov/calibrations/recommended-calibration-interval).
- NIST Technical Note 2156, [Metrological Traceability](https://www.nist.gov/metrology/metrological-traceability).
- Kok et al. (2025), [Modelling and determining correlations in sensor networks](https://doi.org/10.1016/j.measen.2024.101793).
- Qi & Baker (2026), [Optimal experimental design with observation noise](https://doi.org/10.1016/j.mbs.2025.109571).
- Yun et al. (2026), [Low-cost indoor sensor calibration dataset and benchmark](https://arxiv.org/abs/2610.11236).

The Inha study uses five locations in one building and fixed-interval event metadata; its S4 results vary by event. It is not five independent laboratories and does not identify the physical source of common residuals.

The UCI Gas Sensor Array Drift pages contain conflicting licence descriptions: narrative text says research-only/no commercial use, while a structured field says CC BY 4.0. Treat as unresolved pending written clarification; no commercial-use or redistribution claim is authorized by this roadmap.

## 9. Operational claim vocabulary

Every public report must label each result as one of:
- **Specified:** mathematical or protocol definition only.
- **Analytically checked:** independent calculation of a known-answer fixture.
- **Structurally checked:** fetched committed artifacts and schema/shape validation.
- **Locally executed:** exact-head code was run locally with reproducible toolchain evidence.
- **Hosted CI passed:** required workflows passed on the exact head.
- **Externally reproduced:** independent implementation or evaluator reproduced the result.
- **Physically validated:** declared real measurement procedure was executed with traceable evidence.

Never promote a result to a stronger class because a branch is mergeable, a workflow is skipped, or an approximate policy agrees on one fixture.


## Full-covariance and adaptive-planning follow-on

The next stacked increment adds:
- \`scripts/research/sdcb_full_covariance_gaussian.rs\`: exact subset selection with full latent prior covariance and full candidate residual covariance; selected actions use the corresponding principal residual-covariance submatrix.
- \`scripts/research/sdcb_two_stage_adaptive_oracle.rs\`: exact categorical two-stage planning where the second action may depend on the first outcome and every path must fit the hard budget.
- \`docs/research/SDCB_FULL_COVARIANCE_GAUSSIAN_V1.md\` and \`docs/research/SDCB_ADAPTIVE_TWO_STAGE_ORACLE_V1.md\`: mathematical references and limits.
- \`docs/research/SDCB_COVARIANCE_ADAPTIVE_TOURNAMENT_V1.md\`: full covariance, adaptive-policy, model-misspecification and mobile short-collocation tournament protocol.

A key mathematical point: for a fixed linear-Gaussian model with known covariance, posterior covariance does not depend on observed values. Therefore, a variance-only design over a fixed action subset is not outcome-adaptive. The separate categorical planner is used for non-Gaussian paired-world discrimination, outcome-dependent actions and terminal abstention.

A new real-world track is short mobile-sensor collocation with a slower reference instrument. It must account for unequal averaging windows and process covariance; paired rows do not necessarily represent the same measurand average. See Kok & Tabandeh (2026), DOI 10.1002/qre.70319.

The same evidence rules apply: exact arithmetic and schema parsing are not Rust execution; skipped hosted workflows are not passes; synthetic common-cause perturbations do not establish a physical failure mechanism.


## Finite-scenario covariance robustness

The full-covariance Gaussian reference now includes an exact finite-scenario maximin selection path. One shared action subset is evaluated under every declared prior/residual covariance scenario, and the objective minimizes the worst-case scientific posterior variance. This is not a continuous distributionally robust guarantee; the scenario envelope must be justified and its provenance retained.

The frozen two-scenario control places the residual correlation on science-A/reference in one scenario and science-B/reference in the other. The same selected subset is used for both. The scenario-specific posterior variances and binding worst-case result are retained in the report.


## Dynamic temporal covariance follow-on — 27 November 2026

This increment bridges existing discrete Markov drift references (#13793 and #13804) with continuous full-covariance design (#13881/#13882). It is not a second binary drift oracle.

- `docs/research/SDCB_DYNAMIC_COVARIANCE_BELIEF_V1.md`: continuous-time state-space formulation for physical-process, shared, reference-chain and device-local states; interval-integrated covariance; Bayesian covariance-model uncertainty separated from robust support sets; dynamic adaptive-policy and evidence boundaries.
- `scripts/research/sdcb_ou_window_covariance.rs`: standalone scalar stationary OU covariance oracle for arbitrary rectangular averaging windows, including overlap, duplicate-window, disjoint-window and shared white-noise controls.
- `docs/research/SDCB_TEMPORAL_MOBILE_REFERENCE_TOURNAMENT_V1.md`: preregistered task families and matched temporal, calibration, experiment-design, mobile-reference and edge comparators.
- Versioned example, expected-result fixture and JSON Schema accompany the OU primitive.

The exact OU primitive is a bounded foundation, not a completed vector state-space filter or adaptive laboratory planner. Eight Rust tests are authored but not compiled/executed here; no raw dataset, physical measurement, or hosted workflow pass is claimed. Follow-on increments are vector OU propagation, interval-integrated mobile-reference observations, covariance-model posterior updates, and an elapsed-time-aware adaptive policy.


## Vector continuous-time state-space follow-on — 27 November 2026

The DC1 follow-on now adds `scripts/research/sdcb_vector_state_space.rs`, a bounded linear-SDE discretization reference using the Van Loan block exponential and diagonal [13/13] Padé scaling/squaring. It produces both elapsed-time transition F and full process covariance Q, and propagates Gaussian means/covariances. The state dimension is capped at eight. Frozen scalar-OU, coupled two-factor and non-normal fixtures are included; coupled/non-normal covariance values are independently checked by elementwise quadrature. This is a numerical reference, not a production filter. Ten Rust tests are authored but are not compiled or executed in this environment. Follow-on work remains interval-integrated observations, covariance-model uncertainty and dynamic adaptive planning.


## Covariance-model uncertainty follow-on — 27 November 2026

The DC3 research reference adds `scripts/research/sdcb_covariance_model_mixture.rs`: prequential Gaussian predictive-density updates for finite model weights, within-model scientific conditioning, and mixture variance with explicit between-model contribution. The caller's robust support set remains independent of Bayesian weights, including the ability to retain a zero-prior-weight scenario for worst-case evaluation. It is paired with `docs/research/SDCB_COVARIANCE_MODEL_MIXTURE_V1.md` and `docs/research/SDCB_COVARIANCE_MODEL_UNCERTAINTY_TOURNAMENT_V1.md`, plus schema/example/expected fixtures.

The reference does not estimate covariance parameters or detect omitted models itself. The tournament adds those as separate evaluation dimensions, along with weight sensitivity, model-set adequacy, and generalized Bayesian robust-design comparisons. Ten Rust tests are authored, not compiled/executed here; fixtures are finite Gaussian calculations, not physical measurements.


## Interval-integrated observation follow-on — 27 November 2026

The DC2 increment adds `scripts/research/sdcb_interval_observation.rs`, which reuses the vector state-space oracle to construct exact joint moments for multiple rectangular interval-average observations with arbitrary start/end times and loading vectors. A piecewise augmented state activates/finalizes one accumulator per observation at its declared boundaries, preserving observation/observation process covariance and endpoint-state/observation process cross-covariance. The update conditions the final state using both sources of covariance and keeps acquisition/electronics covariance separate. Overlapping, nested, duplicate and disjoint windows are supported within a bounded augmented dimension of eight.

The OU fixture cross-checks the #13903 known-answer matrix and endpoint posterior under nested reference/mobile windows. Eight Rust tests are authored but not compiled/executed here. This is a bounded linear-Gaussian rectangular-window reference, not yet arbitrary integration kernels or nonlinear/time-varying observation dynamics.


## Open-world covariance-model adequacy — stacked after interval observations

The covariance-model mixture path also includes a separate open-world diagnostic in `scripts/research/sdcb_model_set_adequacy.rs`. It prevents normalized Bayesian weights from being mistaken for proof that any registered model is adequate. A caller supplies a best-model prequential log-density threshold calibrated on independent chronological validation data; when all positive-prior candidates fall below it, the report is flagged inadequate and downstream scientific decisions are not eligible for automatic update. Robust support membership remains separate. The reference does not infer unknown physical causes or calibrate a universal threshold. Four Rust tests are authored, not compiled/executed here.


## Block-conformal open-world adequacy follow-on — 27 November 2026

The fixed-threshold adequacy diagnostic now has an exact finite-rank calibration companion in `scripts/research/sdcb_block_conformal_adequacy.rs`. It computes a conservative upper-tail conformal p-value from block-maximum predictive nonconformity scores, requires a frozen model-set/scoring-protocol digest, reports finite-sample p-value resolution, and rejects reuse of calibration epoch/unit IDs across adaptive stages. The guarantee is only for exchangeable calibration/test units; serially dependent time-series blocks are not automatically exchangeable. Eight Rust tests are authored, not compiled/executed here. This is not a time-series validity guarantee or physical calibration result.
