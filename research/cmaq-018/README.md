# CMAQ-018 — Matched-History Counterfactual Branch Fidelity

**Status:** deterministic synthetic v0.1 harness authored; exact-head Rust qualification is pending. The candidate predictor boundary is now expressed as the VIS-005 adapter trait and a content-addressed interface contract.

## Research question

Given one exact observed history and a fixed action interface, can a world model predict what changes under `do(a_i)` versus `do(a_j)`? Does it rank a common action set correctly, and does it know when an intervention is unsupported?

The primary estimand is the difference between outcomes from the same history:

`Delta_Y(h; a_i, a_j) = Y(do(a_i) | h) - Y(do(a_j) | h)`

This is not ordinary observational prediction. A model can predict the average future well while getting the causal effect of changing an action wrong.

## Relationship to adjacent work

- CMAQ-004 (#9010) measures whether model access improves realized decisions over matched direct policies.
- CMAQ-017 (#9407) measures planner-induced distribution shift and multi-step model exploitability.
- CMAQ-011 (#9094) bridges to pinned external closed-loop environments.
- **WorldOracle** measures action-conditioned rollouts against paired real-world executions from the same initial observation and action sequence. It is the intended external fixed-action rollout bridge; it should not be confused with an exact real-world counterfactual of an unexecuted alternative action. Paper: https://arxiv.org/abs/2610.18826. Project: https://worldoracle-benchmark.github.io/
- **MiraBench** complements this track with action-following fidelity and optimism-bias diagnostics across human-annotated judgments. Paper: https://arxiv.org/abs/2605.29360
- **Rollback** uses rewind-and-branch counterfactual action comparisons for policy improvement. It is a downstream algorithmic comparator, not itself a guarantee that the imagined branch ordering is correct. Paper: https://arxiv.org/abs/2610.36068. Project: https://rollback-wm.github.io/

WorldOracle's current official repository and Hugging Face card describe a public validation release. The current validation manifest lists 67 episodes across AgiBot (20), YAM (15), Human Hand (20), and Policy (12), totaling about 60.6 GB. Data are CC BY-NC-SA 4.0; use only within its non-commercial/share-alike terms. The 268-episode test-input dataset is currently labelled private and contains no ground truth; test ground truth remains withheld. Do not access or mirror test inputs: https://github.com/WorldOracle-benchmark/WorldOracle and https://huggingface.co/datasets/WorldOracle/worldoracle-validation

## v0.1 protocol

A dependency-free Rust executable defines seven controlled intervention families:

1. `null_effect`: every intervention has the same true utility.
2. `monotone`: action effect increases over the tested force range.
3. `non_monotone`: the best force is interior, not maximal.
4. `state_dependent_reversal`: the same force intervention changes sign across observed histories.
5. `action_interaction`: combined force/grip effects are not additively separable.
6. `termination_seam`: one combined action crosses a terminal/failure boundary.
7. `hidden_regime`: the same observed history can conceal two different action-effect regimes.

Each episode uses one observed history, seven canonical interventions, and 21 pairwise action comparisons. Every true branch shares the exact history ID and common disturbance draw. This is a paired counterfactual simulation protocol, not an observational estimate.

There are 64 deterministic episode seeds per scenario. The branch set contains both in-support actions and deliberately unsupported magnitudes/combinations.

## Information boundary

The candidate-facing prediction interface accepts only:

- the observed physical history `(x, v)`;
- the canonical action identities and parameters.

The branch/history identifier is evaluator-only metadata and is not placed in the candidate-facing observation.

It does not receive the hidden regime bit, evaluator disturbance, scenario label, or realized branch outcomes. The oracle is implemented as a separate evaluator-only branch. The branch observations retain an identical history ID and a unique action ID; the evaluator checks this invariant.

The common disturbance is shared across alternatives. This common-random-number design isolates action effects by cancelling episode-level noise. It does not model all forms of branch-dependent stochasticity; those belong in a later randomized paired-repeat track.

## Content-addressed configuration

The fixed candidate, generator, metric, observation, action, reward, wrapper and dataset contracts are now stored as separate JSON artifacts beside this README:

- `candidate-config.v0.1.json`
- `generator-config.v0.1.json`
- `metric-bundle.v0.1.json`
- `observation-schema.v0.1.json`
- `action-schema.v0.1.json`
- `reward-definition.v0.1.json`
- `wrapper-stack.v0.1.json`
- `data-manifest.v0.1.json`

The frozen-manifest example in `research/cmaq-common` binds the SHA-256 digest of each artifact, the exact benchmark source commit and the 64-episode seed schedule. There are 64 episode histories **per scenario/model condition** (448 scenario-history cases across seven scenarios, with the same seed panel paired across the five predictor arms); pairs within one history remain clustered.

## VIS-005 candidate predictor boundary

The candidate-facing API is defined by CandidateBranchPredictor and CandidateEffectPredictor in src/main.rs, with a machine-readable contract in PREDICTOR_INTERFACE.json and an integration mapping in VIS-005_ADAPTER.md. Inputs are observed history, canonical hypothetical action and a requested forecast-kind tag. Outputs explicitly echo Predicted or Counterfactual provenance, or abstain. The interface exposes no evaluator keys and no observation-memory mutation API. The CMAQ-018 branch bank always requests Counterfactual forecasts because it evaluates hypothetical alternatives from the same observed history.

## Model arms

- `point_outcome`: approximate absolute branch outcome predictor with a shared history-dependent bias and a misspecified linear action response.
- `explicit_effect_head`: predicts pairwise action effects directly, without emitting absolute branch forecasts.
- `shuffled_action_labels`: deliberately associates predictions with the wrong canonical action.
- `support_aware_point`: emits predictions only inside a declared training-support region.
- `oracle`: evaluator-only branch outcomes.

The explicit effect head is not presumed superior. The comparison tests whether common-mode errors cancel when subtracting two absolute predictions, and whether a separately parameterized effect estimate introduces different errors. The predictors are synthetic fixed functions, not trained Symthaea models.

## Metrics

Reported independently for each scenario/model pair:

- Pairwise effect-sign accuracy on non-tied true effects, with a 95% percentile-bootstrap interval and the number of independent episode histories contributing.
- Pairwise effect-magnitude MAE, with a 95% percentile-bootstrap interval and per-metric episode denominator.
- Kendall tau-b action-ranking agreement, including explicit tie handling, per-metric episode denominator and a 95% percentile-bootstrap interval.
- Top-1 regret against the best action in the same frozen branch bank, with per-metric episode denominator and a 95% percentile-bootstrap interval.
- Terminal false-safe rate with a 95% percentile-bootstrap interval where actual terminal branches have predictions.
- Null-effect false-positive rate: predicted nonzero effects when all true effects are zero, with per-metric episode denominator and a 95% percentile-bootstrap interval.
- Pair-coverage and branch-prediction coverage, so abstention is not hidden.
- Absolute branch-utility MAE and next-state MAE where the model emits branch predictions; each reports a separate defined-episode denominator (zero for pairwise-effect-only models).
- Branch prediction evaluations, direct pair-effect evaluations, and oracle branch evaluations, reported separately.

Intervals resample the 64 independent episode histories as clusters (2,000 deterministic percentile-bootstrap replicates). The 21 action pairs inside one episode are never treated as independent samples. Each metric's per-episode value is computed first, then episodes are resampled. Each metric reports its own defined-episode count. Undefined metrics have denominator zero and JSON `null`; percentile bounds are also null when fewer than two episode histories define the metric.

No single composite score is emitted.

## Reproduction

From this directory:

```sh
cargo test --locked
cargo run --release --locked > /tmp/cmaq-018-report.json
```

The focused GitHub Actions workflow runs formatting, tests, Clippy, JSON Schema validation, report invariants, and creates a digest-bound run receipt.

## Next steps

1. Qualify the exact synthetic report and known-answer fixtures.
2. Add confidence intervals and support-shift sweeps after a sample-size pilot.
3. Add action-composition and held-out intervention magnitudes.
4. Join CMAQ-017 exploitability results to identify whether branch-ranking failures are discovered by adaptive planning.
5. Build a pinned WorldOracle adapter only after public data, release policy, action interface and license are verified.
6. Evaluate Rollback-style branch-and-compare policy improvement only after the evaluator and qualification split are frozen.

## Claim ceiling

A synthetic pass demonstrates only the branch identity, known-answer intervention and metric contracts implemented by this generator. It does not establish real-world counterfactual correctness, physical safety, learned Symthaea world-model reliability, or SOTA. No physical robot intervention or actuator-control authority is enabled.
