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

A failure signature carries an optional reason ID, optional stage ID and optional zero-based stage index. A missing reason represents WorldFailBench's Unjudgable / indeterminate category; a missing stage ID and index together represent an unknown decisive stage. These observations remain in the reference-failure denominator. Known reason/stage IDs are vocabulary entries fixed by the task contract before evaluation. Candidate predictions are stored separately from reference labels at the runner boundary; the candidate must never receive evaluator truth.

A boundary pair contains exactly two records: lower and upper perturbation of the same initial state and same realized action in the same environment family, domain and independent cluster. The candidate checkpoint must be identical. The lower perturbation value must be strictly less than the upper value.

Digest strings are format-checked by this library only. The runner must recompute them from exact source bytes, verify the action/observation identity and attach signed custody receipts. These checks are not proof of source authenticity or execution truth.

## Metrics

Let the evaluator reference be y and candidate prediction be y-hat.

- Reference failure count: number of cases where y is a failure.
- False success: reference is a failure but candidate predicts success.
- Failure capture recall: candidate predicts failure on reference-failure cases / reference-failure cases.
- False-success rate: false successes / reference-failure cases.
- Reason accuracy: exact known/unknown reason match / reference-failure cases. Unjudgable is not dropped or coerced into a known class. A candidate inventing a reason for an indeterminate reference, or abstaining on a known reason, is scored as a mismatch; predicted success is also incorrect.
- Stage accuracy: exact known/unknown stage ID and index match / reference-failure cases. Unknown reference stages remain in the denominator; a candidate guessing a specific stage where the reference is unknown (or abstaining where it is known) is a mismatch.
- Reference success count and false failure: candidate predicts failure where the reference succeeded.
- False-failure rate: false failures / reference-success cases.
- Boundary flip accuracy: fraction of boundary pairs where candidate correctly predicts whether crossing the preregistered perturbation interval changes success to failure or vice versa.
- Boundary pair outcome accuracy: fraction where the candidate predicts the exact reference outcome on both lower and upper cases.

The two boundary metrics are intentionally distinct. A candidate may correctly predict that a boundary exists but place it on the wrong side; flip accuracy can be one while pair-outcome accuracy is zero. That is not an adequate substitute for exact boundary localization.

No aggregate reliability scalar is emitted. Each denominator and cluster summary remains visible. Undefined rates remain absent when the corresponding reference denominator is zero.


## Replicated success/failure boundary rates

The binary lower/upper boundary pair in the unit fixtures is not sufficient for the released WorldFailBench boundary subset, which contains multiple rollouts on each side. The evaluator now has a separate `BoundaryRatePair` API for replicated side rates.

Each pair binds:
- pair/domain/environment-family/independent-cluster IDs;
- perturbation dimension and the two side perturbation values;
- common initial-state, realized-action and candidate-checkpoint digests;
- reference success counts and trial counts for both the success-labelled and failure-labelled scene variants;
- candidate-predicted success counts and trial counts for both variants.

The counts preserve repeated rollouts; they do not turn them into independent environment samples. The independent unit remains the source cluster/pairing structure declared by the manifest.

For each side, the evaluator calculates success rate in basis points. It reports separately:
- success-side rate MAE;
- failure-side rate MAE;
- error in the success-side minus failure-side rate gap;
- whether the reference boundary exceeds the contract's frozen minimum gap;
- whether the candidate predicts a boundary of at least that size;
- boundary-presence agreement and gap-direction agreement;
- per-cluster pair counts and boundary-detection summaries.

The minimum detectable gap is preregistered in `FailureFidelityContract.minimum_boundary_gap_basis_points`. The current synthetic fixture uses 2,000 basis points (20 percentage points) as an example only; an external benchmark adapter must pin the threshold and side semantics to the benchmark protocol before evaluating a candidate. This is a scorecard option, not a replacement for WorldFailBench's published boundary metric.

The scorecard does **not** assume the success side corresponds to a smaller numeric perturbation. The dataset's position, orientation and size dimensions can have different directions; labels in the evaluator manifest determine which scene is the success side and which is the failure side.

The current API still does not reproduce the original WorldFailBench boundary scoring code. Its site describes that evaluation code as not yet released. The adapter must preserve dataset side metadata, policy variants, task/pair identities, all 10 rollout outcomes per side and the benchmark's factor-specific reporting. Do not compare a one-pair mean rate score directly against the paper's published aggregate.

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

WorldFailBench asks whether world models reproduce the observable reason a robot policy fails, the execution stage at which it fails, and the conditions under which a controlled scene change turns success into failure. Its paper scope covers seven world models, 32 tasks and 6,900 reference rollouts across LIBERO, RoboTwin and physical G1 manipulation.

The documented public simulator release lists 5,000 RoboTwin and 1,500 LIBERO episodes (6,500). A separate boundary set lists 120 pairs and 2,400 rollouts (10 rollouts per side). The additional 400 G1 reference rollouts in paper scope are not part of those documented public simulator counts. The dataset's failure taxonomy includes Task Planning Failure, Action Execution Failure and Indeterminate/Unjudgable; stage annotations can be unknown.

The linked Hugging Face dataset card and DAXIAORobotics/WorldFailBench dataset repository declare Apache-2.0. The separate tangxiaoyang18/WorldFailBench public project-site README says the evaluation code is not yet released. The open dataset documentation is therefore not the original executable scorer. Before dataset admission, pin the exact archive revision, compute SHA-256, inventory all files, check the license for all referenced assets and record policy/checkpoint provenance.

The website's published RoboTwin score table gives its Ref–Ref diagnostic TFE 0.061, SFE 0.094, TPC 0.814 and SPC 0.903. Across the seven world-model rows, DW05 has the lowest listed TFE (0.159) and highest listed TPC (0.402); Ctrl has the highest listed world-model SPC (0.095). TFE/SFE are lower-is-better and TPC/SPC higher-is-better. These metrics are not combined into one score, and the public results site does not publish a single pooled boundary score. This is a benchmark-reported snapshot, not a Symthaea comparison.

Primary references:
- Paper: https://arxiv.org/abs/2610.33565
- Dataset card: https://huggingface.co/datasets/ACERobotics/WorldFailBench
- Dataset docs and Apache-2.0 license: https://github.com/DAXIAORobotics/WorldFailBench
- Public results / code status: https://github.com/tangxiaoyang18/WorldFailBench
- RoboTwin results table: https://tangxiaoyang18.github.io/WorldFailBench/

## MiraBench

MiraBench separates physics adherence, action-following fidelity and optimism bias under failure-inducing actions. CMAQ-020 adds false-success rate and exact failure reason/stage diagnostics to the broader state-response contract. A model that predicts a failure for the wrong physical reason should not receive full credit merely for getting the binary outcome right.

Primary reference: https://arxiv.org/abs/2605.29360

## AMD failure-fidelity study on RoboTwin

AMD's 20 October 2026 study, "Learning to Fail", reports that BWM on RoboTwin 2.0 adjust_bottle rendered 82.5% of simulator failures as successes. Fine-tuning on the policy's natural failure rollouts reduced fake success to 26.3% on its reported held-out episodes, while a success-only control did not improve. The same paper reports that action-fork outcomes from the same initial state remained unreliable after that repair.

This is a single task/model/policy setup, not a general benchmark ranking. Its most important CMAQ-020 lesson is that failure examples help, but do not guarantee action-conditioned counterfactual fidelity. We must test both failures and near-boundary action forks.

Reference: https://www.amd.com/en/developer/resources/technical-articles/2026/rl-inside-world-models.html

## LIBERO-RECOVER

The current project page advertises 2,178 recovery scenarios, 130 subtasks across four suites and 3,184 human demonstrations, and marks the submission under review / paper forthcoming. The GitHub README for the data release instead advertises 4,803 annotated failure scenes (including 465 constructed L4 scenes) and 1,446 raw rollouts. These figures may describe different snapshots/subsets/denominators; they are not reconciled by one immutable manifest in the checked sources and must not be collapsed into one benchmark size.

Its four levels are action retry, action adaptation, object-state recovery and environmental recovery. This is a downstream recovery-policy capability test, not a failure-prediction test.

References:
- Paper/preprint: https://arxiv.org/abs/2609.05178
- Current project page: https://liulin815.github.io/LIBERO-Recovery/
- Dataset and code README: https://github.com/liulin815/LIBERO-Recovery

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
