# CMAQ-020 Failure Fidelity Scorecard v0.1

**Status:** evaluator prototype with synthetic known-answer tests.  
**Claim ceiling:** scorecard arithmetic, custody validation and synthetic fixtures only. No WorldFailBench, MiraBench, ScratchWorld, LIBERO, RoboTwin or physical robot result is produced by this crate.

## Research premise

A world model used to evaluate or train a policy must preserve not only whether a task succeeds, but how an attempted action fails and where the failure boundary lies.

A binary success/failure label is insufficient. Two models can have the same overall success rate while one invents successful grasps where the reference has missed contact, or places failures at the wrong stage. That gives a policy the wrong learning signal and a deployer the wrong diagnosis.

The evaluator therefore treats the following as separate observables:

1. outcome fidelity: success versus failure;
2. failure reason fidelity: the observable error category;
3. failure stage fidelity: the registered execution stage and ordinal position;
4. failure-boundary fidelity: whether a controlled scene/action perturbation crosses the success/failure boundary, and whether the candidate gets both sides right;
5. false-success and false-failure rates, separately.

These are inspired by WorldFailBench's reason/stage/boundary tasks and MiraBench's action-following and optimism-bias probes. They are not a reimplementation of either benchmark's original scorer.

## Input record

Each evaluator-owned record binds:

- unique record ID;
- frozen domain, environment/mechanism family and independent cluster IDs;
- initial-state digest;
- realized-action digest;
- frozen candidate checkpoint digest;
- evaluator reference outcome;
- candidate-predicted outcome;
- optional boundary pair identity, lower/upper side and preregistered perturbation value.

A failure signature contains a stable reason ID, stage ID and zero-based stage index. Stage IDs and reason IDs are vocabulary entries fixed by the task contract before evaluation. Candidate predictions are stored separately from reference labels at the runner boundary; the candidate must never receive evaluator truth.

A boundary pair contains exactly two records: lower and upper perturbation of the same initial state and same realized action in the same environment family, domain and independent cluster. The candidate checkpoint must be identical. The lower perturbation value must be strictly less than the upper value.

Digest strings are format-checked by this library only. The runner must recompute them from exact source bytes, verify the action/observation identity and attach signed custody receipts. These checks are not proof of source authenticity or execution truth.

## Metrics

Let the evaluator reference be y and candidate prediction be y-hat.

- Reference failure count: number of cases where y is a failure.
- False success: reference is a failure but candidate predicts success.
- Failure capture recall: candidate predicts failure on reference-failure cases / reference-failure cases.
- False-success rate: false successes / reference-failure cases.
- Reason accuracy: exact reason match / reference-failure cases. A predicted success on a reference failure scores as incorrect.
- Stage accuracy: exact stage ID and stage index match / reference-failure cases. A predicted success on a reference failure scores as incorrect.
- Reference success count and false failure: candidate predicts failure where the reference succeeded.
- False-failure rate: false failures / reference-success cases.
- Boundary flip accuracy: fraction of boundary pairs where candidate correctly predicts whether crossing the preregistered perturbation interval changes success to failure or vice versa.
- Boundary pair outcome accuracy: fraction where the candidate predicts the exact reference outcome on both lower and upper cases.

The two boundary metrics are intentionally distinct. A candidate may correctly predict that a boundary exists but place it on the wrong side; flip accuracy can be one while pair-outcome accuracy is zero. That is not an adequate substitute for exact boundary localization.

No aggregate reliability scalar is emitted. Each denominator and cluster summary remains visible. Undefined rates remain absent when the corresponding reference denominator is zero.

## Statistical and evidence rules

- The independent environment-generation cluster is the inferential unit, not the number of episodes.
- Repeated traces from the same cluster remain grouped in the output.
- One frozen candidate checkpoint is used per evaluation.
- Boundary pairs share exact source-state, action, environment family, domain, independent cluster and checkpoint identity.
- Every failed, timed-out, invalid, partial, rolled-back or abstained execution must retain its original denominator and disposition in the upstream evidence store; the v0.1 failure signature scorer currently scores recorded Success/Failure outcomes and should be extended with explicit Partial/Abstain dispositions before external admission.
- Never treat a candidate's own generated trace as ground-truth reference evidence.
- No model prediction changes a real-world safety authorization or execution decision.

# Benchmark alignment

## WorldFailBench

WorldFailBench asks whether world models reproduce the observable reason a robot policy fails, the execution stage at which it fails, and the conditions under which a controlled scene change turns success into failure. It evaluates manipulation in LIBERO and RoboTwin and includes real-world G1 manipulation. Its paper reports substantial failure-fidelity gaps across seven tested world models.

CMAQ-020 uses these dimensions as a scorecard shape, while requiring a separately pinned adapter, action interface, task split, reference execution and cluster manifest before any WorldFailBench comparison. We do not copy or pool its published scores.

Primary reference: https://arxiv.org/abs/2610.33565

## MiraBench

MiraBench separates physics adherence, action-following fidelity and optimism bias under failure-inducing actions. CMAQ-020 adds false-success rate and exact failure reason/stage diagnostics to the broader state-response contract. A model that predicts a failure for the wrong physical reason should not receive full credit merely for getting the binary outcome right.

Primary reference: https://arxiv.org/abs/2605.29360

## AMD failure-fidelity study on RoboTwin

AMD's 20 October 2026 study, "Learning to Fail", reports that BWM on RoboTwin 2.0 adjust_bottle rendered 82.5% of simulator failures as successes. Fine-tuning on the policy's natural failure rollouts reduced fake success to 26.3% on its reported held-out episodes, while a success-only control did not improve. The same paper reports that action-fork outcomes from the same initial state remained unreliable after that repair.

This is a single task/model/policy setup, not a general benchmark ranking. Its most important CMAQ-020 lesson is that failure examples help, but do not guarantee action-conditioned counterfactual fidelity. We must test both failures and near-boundary action forks.

Reference: https://www.amd.com/en/developer/resources/technical-articles/2026/rl-inside-world-models.html

## Recovery benchmark

LIBERO-RECOVER moves beyond task completion into retry, action adaptation, object-state recovery and environmental recovery. It is a complementary downstream capability test, not a failure-prediction test: a world model may faithfully predict a failure without knowing how to recover, and a policy may recover despite imperfect predicted details. Keep the two claims separate.

Reference: https://arxiv.org/abs/2609.05178

## Required controls

1. No-op/copy baseline: do not award credit for unchanged fields or unchanged success predictions.
2. Missed-contact negative control: contact expected but not realized; model must not hallucinate lift/transport success.
3. Partial execution: command partially executes; separate assigned from realized action and preserve the resulting state.
4. Same-start action fork: same initial scene and policy, different action variants lead to distinct outcomes; test both branches.
5. Boundary sweep: preregister lower and upper scene/action perturbations and sample across the transition region.
6. Failure-data leakage control: keep reference failure labels evaluator-only; training data and evaluation labels must be disjoint as declared.
7. Closed-loop policy comparison: same policy with and without the world model, plus a direct-policy baseline. This is a separate evaluation lane.

## Promotion gates (targets, not results)

- Zero false-successes on deterministic known-answer failure fixtures.
- Exact reason and stage recovery on deterministic fixtures.
- 100% rejection of malformed/incomplete boundary pairs and mixed checkpoint/source identities.
- Publish false-success and false-failure rates separately.
- On external tests, use held-out initial-state/environment clusters; report 90% and 95% cluster-level intervals plus weak-cell worst cases.
- Boundary flip detection alone is insufficient: report exact two-sided boundary outcome accuracy.
- Failure-data training and success-data controls must have equalized training steps, model architecture, input window and resource budget.
- No claim of “failure understanding” based solely on generated-video similarity or global success-rate correlation.

No cross-benchmark pooling: WorldFailBench, MiraBench, AMD's RoboTwin task study, LIBERO-RECOVER and CMAQ synthetic failure fixtures use different task populations, models, interfaces and metrics.
