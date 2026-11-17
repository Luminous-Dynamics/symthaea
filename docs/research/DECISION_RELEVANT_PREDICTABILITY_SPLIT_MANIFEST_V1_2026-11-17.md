# Decision-Transfer Benchmark — Split Manifest Contract V1

**Status:** Research protocol proposal, 2026-11-17. This document specifies reproducible split identity and independent sampling units; it contains no benchmark result.

**Parent:** [#12608 — held-out dynamical transfer and intervention decision benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/12608)

**Composes with:** #12502 (reference generation), #12505 (numerical/Lyapunov qualification), #12506 (model baselines), #12551 (forecast evaluator), #12567 (information-channel integrity), #12407 (observation events), #9817 (active sensing).

## 1. Why freeze a split contract first?

A chronological split inside one generated trajectory measures continuation. It does not estimate transfer to a new initial condition, parameter cell, forcing regime, equation family, observation process, or intervention mechanism. Adjacent samples are dependent and must not be counted as independent trials.

The first evaluator (#12551) correctly labels its current result as a within-trajectory diagnostic. V1 therefore defines the *identity of an experiment and its holdout groups* before adding more models or reporting aggregate scores.

The manifest is evaluator-owned. Candidate processes receive only the authorized observation/task stream and a public task-contract digest. They never receive hidden family labels, future suffixes, evaluator seeds, split membership for confirmatory holdouts, or truth records.

## 2. Independent units and split axes

Every run declares exactly one primary independent sampling unit for inference:

- `trajectory`: independently initialized and generated trajectories within a fixed system/parameter cell;
- `parameter_cell`: a preregistered parameter region, with trajectories nested within it;
- `equation_family`: an entire governing-equation family, with parameter cells and trajectories nested within it;
- `intervention_mechanism`: a controlled intervention family, with scenarios nested within it.

Nested observations are not independent replicates. The analysis must cluster at the declared unit or a higher grouping level. Timesteps, forecast leads, sensor channels, repeated rollouts from the same initial state, and multiple seeds sharing a single fitted system are not independent units by default.

A task may define several split axes, but it must nominate one primary confirmatory axis and list all nesting relationships. The report must publish per-group results so an aggregate cannot hide a failed family.

## 3. Split classes

Each scenario is assigned to exactly one split role:

- `train`: available for fitting and feature/model selection;
- `validation`: available for declared hyperparameter and policy selection;
- `confirmatory_test`: hidden until the protocol, code, model-selection rule and thresholds are frozen;
- `negative_control`: known-answer leakage, nuisance, solver, or null-effect controls; never pooled into positive-control skill;
- `positive_control`: known-answer detectable effects under declared assumptions; reported separately from naturally difficult cases.

The manifest declares the holdout relation, not just a random row assignment:

| Track | Required separation |
|---|---|
| Continuation | chronological within-trajectory; diagnostic only |
| Initial-condition transfer | no initial-condition family overlap |
| Parameter transfer | no parameter-cell or held-out-region overlap |
| Forcing/coupling transfer | held-out forcing or coupling configuration |
| Equation transfer | complete equation-family holdout |
| Observation shift | held-out noise, delay, missingness or cadence mechanism |
| Intervention transfer | held-out intervention mechanism or target family |

No row-wise random split is accepted as evidence of dynamical transfer. If a task intentionally allows a shared axis (for example, same equation family but new parameter region), that is a distinct named track, not an exception silently mixed into a broader claim.

### 3.1 Nested sampling hierarchy

`sampling.nesting` is an ordered outer-to-inner path, such as equation family → parameter cell → trajectory. Every scenario must declare one opaque group identity for every axis in that path, including contextual axes that are not held out. Each child group identity must map to exactly one parent group identity across the manifest. A parameter-cell group cannot silently appear under two equation families, and a trajectory group cannot migrate between parameter cells.

Only axes explicitly listed in `split_axes` impose train/validation/confirmatory group disjointness. Contextual axes may be shared across roles, but their hierarchy must remain coherent. For example, parameter-transfer evaluation can share one equation-family parent while holding out parameter-cell IDs; equation-family transfer holds out the outer family groups as well. This allows separate, interpretable tracks without confusing one kind of generalization with another.


## 4. Manifest fields

A canonical manifest is UTF-8 JSON with deterministic key ordering for digesting. The schema version is explicit. The following logical fields are required; implementation may use JSON, CBOR or another canonical encoding only after defining byte-level canonicalization and cross-language test vectors.

```json
{
  "schema": "luminous.decision-transfer.split-manifest.v1",
  "protocol_revision": "sha256:<protocol-digest>",
  "generator": {
    "repository": "Luminous-Dynamics/symthaea",
    "commit": "<full-commit-sha>",
    "generator_id": "<versioned-generator-id>",
    "solver_id": "<solver-and-version>",
    "solver_configuration_digest": "sha256:<digest>"
  },
  "task": {
    "task_id": "<immutable-task-id>",
    "task_contract_digest": "sha256:<digest>",
    "loss_contract_digest": "sha256:<digest>",
    "observation_operator_digest": "sha256:<digest>"
  },
  "sampling": {
    "primary_independent_unit": "parameter_cell",
    "nesting": ["equation_family", "parameter_cell", "trajectory"],
    "minimum_independent_units_per_role": 1
  },
  "split_axes": [
    {
      "axis": "equation_family",
      "policy": "group_disjoint",
      "train_groups_digest": "sha256:<digest>",
      "validation_groups_digest": "sha256:<digest>",
      "confirmatory_test_groups_digest": "sha256:<digest>"
    },
    {
      "axis": "parameter_cell",
      "policy": "held_out_region",
      "region_definition_digest": "sha256:<digest>"
    }
  ],
  "scenarios": [
    {
      "scenario_id": "<opaque-random-id>",
      "split_role": "train",
      "group_ids": ["<opaque-group-id>"],
      "seed_commitment": "sha256:<commitment>",
      "reference_data_digest": "sha256:<digest>",
      "solver_qualification_digest": "sha256:<digest>"
    }
  ],
  "randomization": {
    "seed_commitment_algorithm": "sha256",
    "permutation_suite_digest": "sha256:<digest>"
  },
  "freeze": {
    "protocol_frozen_at": "<UTC timestamp>",
    "confirmatory_manifest_digest": "sha256:<digest>",
    "analysis_plan_digest": "sha256:<digest>"
  }
}
```

This is an illustrative shape, not a JSON Schema or implemented parser. The actual implementation must reject unknown schema versions, duplicate scenario IDs, duplicate group IDs where prohibited, missing required digests, malformed hashes, role overlap, and invalid nesting.

### 4.1 Opaque identities

Scenario and group identifiers are generated independently of the hidden mechanism, parameter value, outcome, array position, file path, and evaluator seed. Candidate-visible IDs are newly randomized per run and must not be a stable lookup key across confirmatory episodes.

The mapping from opaque IDs to evaluator truth remains evaluator-only. A public manifest may expose schema, task contract, and permitted metadata, but not hidden group membership or seed material for confirmatory data.

### 4.2 Seed commitments

A commitment is not a random seed and must not be mistaken for secrecy by itself. The evaluator stores the seed and a cryptographic commitment before the run; reveal only after the confirmatory run is sealed when reproducibility policy permits. The commitment binds the chosen seed but does not prove that the generator or solver is correct.

Use a cryptographic digest such as SHA-256 for integrity receipts. Non-cryptographic hashes (including FNV-1a) may be used for in-process table lookup only and must never be described as authentication, provenance or tamper evidence.

### 4.3 Solver qualification

Every reference trajectory is tied to equation/version, parameter values, initial conditions, integration method, tolerances, step size, burn-in, output cadence, and numerical qualification. For continuous-time systems, include convergence evidence under tighter tolerances and/or smaller steps. For delay equations and stochastic systems, declare their respective reference semantics.

A solver failure, unresolved trajectory, or out-of-domain parameter cell is retained as a status; it is not silently dropped or relabelled as a negative result.

## 5. Confirmatory freeze and model-selection firewall

Before confirmatory evaluation, freeze and digest:

1. task, observation and loss contracts;
2. generator and solver versions/configuration;
3. split manifest and hidden group assignment;
4. baseline implementations and eligible hyperparameter spaces;
5. tuning budget, stopping rule and resource limits;
6. model checkpoint or reproducible training recipe;
7. metric definitions, aggregation and missing/failure policy;
8. statistical analysis plan and multiplicity family;
9. candidate/evaluator protocol and leakage attack suite.

Model selection may use training and validation groups only. If a candidate is changed after confirmatory outcomes are viewed, the next evaluation is exploratory unless a fresh untouched confirmatory group is allocated.

A test report must include the exact manifest digest, source revision, model/checkpoint digest, task-contract digest, analysis-plan digest and evaluator revision. A result without those identities is not a reproducible benchmark result.

## 6. Statistical reporting contract

For every metric and model, report:

- independent-unit count and nested scenario/trajectory count;
- per-unit scores and per-family distributions;
- paired difference against each eligible baseline;
- uncertainty interval and exact resampling/inference unit;
- simultaneous confidence procedure and declared family of comparisons;
- missing, failed, timed-out and abstained units;
- coverage, accepted-case risk, abstention rate and worst-group risk;
- full resource vector with units, hardware and measurement boundary.

Do not compute confidence intervals by treating forecast leads or timesteps as independent. For paired scenarios, resample or otherwise infer at the paired independent-unit level. The procedure must preserve pairing and clustering. If the number of independent groups is too small for the declared inferential method, report descriptive results and mark the confirmatory claim inconclusive.

For multi-horizon or multi-regime claims, control the preregistered family-wise error or use another justified simultaneous procedure. A pointwise interval is not a simultaneous guarantee.

## 7. Leakage and negative controls

The frozen suite must include:

- ID-to-world memorization and stable-ID replay;
- order, index, filename, serialization and deterministic-seed leakage;
- future suffix, hidden solver state, evaluator label and loss access;
- train/test group overlap and parameter-cell leakage;
- duplicate, stale, delayed and out-of-order observations;
- observation timestamps or metadata that encode hidden class without task justification;
- a sham intervention with zero physical effect;
- an observation-only intervention that changes sensor output but not the physical system;
- a positive control whose intervention effect is known and identifiable.

A candidate must be invariant to nuisance renaming/permutation when the authorized information is unchanged. Prefix-equivalent histories must produce identical candidate input transcripts irrespective of hidden future suffixes. These are harness/interface properties; they do not prove a model is non-leaking outside the tested boundary.

## 8. Baseline fairness and industry comparability

Do not force every model into every track. Maintain two related but separate lanes:

**Known-equation dynamics lane:** persistence, mean, AR, ESN, NG-RC, faithfully reproduced NGRC-MDS, suitable state-space/nonlinear filters, and matched HDC/CfC/LTC variants. Models receive identical observations, permitted covariates, training groups, tuning budget and compute/resource ceilings.

**Real-world forecasting lane:** official-compatible public datasets and splits for Chronos-2, TimesFM-3 and other eligible time-series foundation models, alongside classical/statistical and task-specific baselines. Record model version, weight availability, licence, covariate support, context limits, inference hardware, preprocessing and data-contamination checks. Report probabilistic scores (for example CRPS/quantile loss) as well as point metrics where supported.

The two lanes must not be collapsed into one leaderboard: known-equation transfer tests dynamical and intervention generalization, while general forecasting benchmarks test performance on their own task distributions. A foundation model's strong GIFT-Eval/TIME result does not establish causal intervention competence; a strong synthetic-chaos score does not establish broad industrial forecasting skill.

## 9. Proposed decision-quality acceptance gates

These are targets, not achieved results:

1. Zero overlap across every declared held-out group axis.
2. Zero accepted leakage attacks in the frozen negative-control suite.
3. Every confirmatory score traceable to immutable source, data, task, split and analysis digests.
4. At least one preregistered held-out family where decision regret improves by at least 10% over the strongest eligible baseline, with a simultaneous paired interval supporting improvement.
5. Critical-event risk is non-inferior under a preregistered margin; no aggregate gain can conceal a material worst-family regression.
6. Selective prediction improves accepted-case risk at a declared coverage, with abstention cost and coverage-risk curve reported.
7. Any resource-efficiency claim demonstrates equivalent decision quality and complete resource-vector accounting, not a single latency or energy number.
8. Causal language is limited to controlled, identifiable interventions; observational prediction remains predictive.

If a target is underpowered, numerically unresolved or not yet independently reproducible, the correct result is **inconclusive**, not a relaxed threshold.

## 10. Implementation sequence

- **R0:** review and freeze this manifest/split contract; obtain independent review of the primary unit and nesting rules.
- **R1:** add a strict Rust manifest parser/validator with deterministic canonicalization and known-answer digest tests.
- **R2:** add grouped split construction and overlap/leakage rejection tests.
- **R3:** connect multi-trajectory generator and solver qualification without duplicating #12502/#12505.
- **R4:** add held-out parameter and equation-family evaluation.
- **R5:** add controlled intervention and counterfactual scoring.
- **R6:** add clustered/simultaneous inference and reproducible receipts.
- **R7:** run matched baseline tournament and Symthaea architecture ablations.

No runner is required to review the protocol. Code acceptance still requires exact-head focused tests and all relevant independent checks.

## 11. External benchmark context (retrieved 2026-11-17)

- The TIME benchmark argues that reused legacy datasets, weak data integrity and task mismatch limit conclusions from older forecasting benchmarks; it proposes 50 fresh datasets and 98 task formulations with pattern-level analysis. Its public dataset card lists CC BY-NC 4.0, so the research/evaluation license must not be mistaken for production/commercial rights. [TIME paper](https://proceedings.mlr.press/v306/qiao26g.html) · [TIME dataset card](https://huggingface.co/datasets/Real-TSF/TIME).
- IND-Eval emphasizes industrial source-balanced and high-resolution multivariate tasks; its reported TimesFM-3 lead is benchmark-specific, and its simulator case study shows that domain specialization can trade off against broad benchmark performance. [IND-Eval](https://arxiv.org/abs/2610.16652).
- Google Research announced TimesFM-3 on 2026-08-31 and reports top average rank among pretrained models on GIFT-Eval, FEV-Bench and TIME. That is the publisher's benchmark result, not proof of universal forecasting superiority. TimesFM-3 self-hosted pretrained weights are under a separate non-commercial/non-production license; commercial/production deployment is permitted through authorized Google Cloud services, subject to those service terms. [TimesFM-3 announcement](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) · [TimesFM repository and license notice](https://github.com/google-research/timesfm).
- Mistral announced Timestral on 2026-10-15 and claims its 209M model ranks first among pretrained zero-shot models on GIFT-Eval, TIME and FEV-Bench. The Timestral paper reports GIFT-Eval MASE/CRPS 0.658/0.446, TIME MASE/CRPS 0.626/0.522, and FEV skill scores 50.39 (SQL) / 39.49 (MASE). These are Mistral's evaluation results, not independent reproductions. [Timestral paper](https://arxiv.org/abs/2610.37585) · [Mistral announcement](https://mistral.ai/news/timestral/).
- Dated third-party snapshots accessed 2026-11-17 do **not** support one universal winner across all public aggregations: (a) TIME leaderboard by CRPS lists Timestral-209M 0.5223, t0-gamma 0.5340 and Simons-297M 0.5341; (b) TSFM.ai's FEV-Bench MASE skill score view lists TimeRSI-1.5 41.44, ci-proto 40.80, TimeRSI 40.58 and Timestral-209M 39.49; (c) TSFM.ai's GIFT-Eval average MASE-rank view lists Taichu-TimeSeries-Agent-v2 first (18.02) and Timestral-209M eleventh (32.52); and (d) The Aggregate's FEV average-win-rate snapshot from 2026-11-02 lists timeh2o-v1 first (89.6) and Timestral-209M second (87.66). These rankings use different metrics, model inclusion rules and refresh dates; they are discovery snapshots, not interchangeable official results. [TIME CRPS snapshot](https://paperswithcode.co/benchmark/time-benchmark) · [FEV MASE-skill snapshot](https://tsfm.ai/benchmarks/fev-bench) · [GIFT-Eval rank snapshot](https://tsfm.ai/benchmarks/gift-eval) · [FEV win-rate snapshot](https://theaggregate.ai/benchmark?slug=fev-bench).
- Chronos-2 remains a relevant zero-shot multivariate/covariate-informed probabilistic baseline, but its paper-reported strengths must be stated against the benchmark and release it actually evaluated. [Chronos-2 technical report](https://arxiv.org/abs/2610.41956).
- Chronos-2 remains a relevant zero-shot multivariate/covariate-informed probabilistic baseline, but its paper-reported strengths must be stated against the benchmark and release it actually evaluated. [Chronos-2 technical report](https://arxiv.org/abs/2610.41956).

**Benchmark policy:** maintain separate scoreboards for (1) real-world general forecasting (TIME, GIFT-Eval, FEV-Bench and industrial telemetry), (2) known-equation dynamical inference, and (3) held-out intervention/decision transfer. Do not infer causal intervention competence from a forecasting leaderboard, or broad industrial forecasting quality from a synthetic-chaos result. Every comparator row must record model revision, access mode, license, covariates, context window, data split, hardware, resource boundary, and whether the result is publisher-reported, aggregator-reported, independently reproduced, or internally reproduced.

All external model figures are publisher- or aggregator-reported until independently reproduced under this protocol. Benchmark rankings change; the date, model revision, licence and exact task must accompany every comparison.

## 12. Claim boundary

This document is a protocol proposal, not an implementation or a result. It does not establish forecasting superiority, causal discovery, general intelligence, physical safety, or any real-world risk guarantee.
