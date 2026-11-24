# Individual Counterfactual Effect Retention Evaluator v0.1

**Status:** Evaluator-side finite reference; implementation proposal pending exact-head Rust qualification.  
**Parent:** Symthaea #13255 — REPRESENTATION-HORIZON-001.  
**Related:** #13239 (single-epoch decision-preserving compression), #13258 (temporal representation preservation), #13445 (imagination-budget decision value), #13375/#13377 (causal trajectory and stochastic evidence pairing), #9437 (candidate/evaluator isolation).

## 1. Research question

Does an action-conditioned world model preserve the history-dependent treatment effect of the same individual through three distinct stages?

1. **Environment / paired ground truth:** the realized outcome contrast from applying two interventions to the same restored individual under a shared factual history, branch-point state and exogenous forcing commitment.
2. **Frozen-latent readout:** the contrast recoverable from the same history-inferred latent state before the imagined transition.
3. **Imagined rollout:** the contrast expressed by the model's action-conditioned output for the two interventions.

This isolates a different failure from decision-sufficient compression. A representation may preserve action-relevant partitions and its latent may retain individual-specific information, while the transition/decoder path loses that information. Conversely, a population-average effect can be correct while opposite individual effects cancel.

This is motivated by Okita, *Where Individual-Effect Information Survives: A Diagnostic Ladder for World Models* (2026-11-04), DOI 10.3390/make8110363: https://www.mdpi.com/2504-4990/8/11/363 . The paper's empirical world-model studies use small, low-dimensional systems. This evaluator is a proposed Symthaea diagnostic, not a replication or a claim that the same failure has been established in large robotics models.

## 2. Estimands

For individual (i), baseline action (a_0), alternative action (a_1), and a frozen task utility scale:

[
Delta_i^{mathrm{real}} = U_i(a_1) - U_i(a_0)
]

[
widehat{Delta}_i^{mathrm{latent}} =
widehat U_i^{mathrm{latent}}(a_1) -
widehat U_i^{mathrm{latent}}(a_0)
]

[
widehat{Delta}_i^{mathrm{rollout}} =
widehat U_i^{mathrm{rollout}}(a_1) -
widehat U_i^{mathrm{rollout}}(a_0)
]

The evaluator reports the following independently:

- Population effect sums and mean-effect bias.
- Latent-readout absolute error and sign agreement per individual.
- Rollout absolute error and sign agreement per individual.
- The rollout-minus-latent absolute-error sum.
- Per-individual realized/predicted contrasts, retained for cluster-level analysis.

The critical distinction is:

[
operatorname{error}!left(mathbb{E}_i[widehat{Delta}_i],
                              mathbb{E}_i[Delta_i]ight)

e
mathbb{E}_i!left[|widehat{Delta}_i-Delta_i|ight]
]

A zero population-mean error does not imply low individual error. F8 is the deterministic mean-cancellation example.

The Rust reference uses fixed integer utility units, exact signed effect differences and unsigned absolute-error numerators. Reported sums have the individual count as denominator; the reference does not round through floating-point means.

## 3. Pairing contract

Every pair must bind:

- Hidden individual identity digest.
- Factual-history digest.
- Restored branch-point state digest.
- Shared exogenous-forcing manifest digest.
- Intervention-pair and action IDs.
- Task, loss-scale, evaluator-profile and evaluation-split digests.
- Model configuration, latent representation, latent state, latent-readout and rollout profile digests.
- Paired seed.
- Existing causal/stochastic pairing receipt and the exact verifier-profile digest.

Both branch records must match the frozen factual history, branch-point state, exogenous forcing, model configuration, latent representation, latent state and paired seed. Branch run IDs must be unique. The baseline and alternative action IDs must be distinct and exact.

The pairing receipt digest is only a reference. This module does not authenticate receipt bytes, verify source-to-actuator event correspondence, or prove that the external verifier executed. A trusted evaluator wrapper must independently retrieve and validate the source artifacts and #13377 / Symtropy verifier receipt before it supplies outcomes to this module.

The two branches share the same restored individual/history and exogenous-forcing commitment. Endogenous state and event consequences may differ when caused by the intervention. Do not replay an identical post-intervention future-state trajectory after the branches diverge.

## 4. Missingness and denominator rules

A non-applied branch returns a typed `BranchNotApplied` error with terminal time and reason. It is never assigned zero utility or zero treatment effect. The caller must retain the failed/late/abstained branch and the task-contract no-output penalty in the overall benchmark denominator.

Missing receipt references, model/latent mismatch, branch-point mismatch, exogenous-forcing mismatch, duplicate individual IDs, duplicate branch run IDs and mixed task/loss/evaluator profiles reject the cohort.

The module evaluates one intervention pair per individual within an evaluation cohort. If multiple action pairs or repeated episodes are required, keep the individual/episode/cluster assignment explicit upstream and do not count correlated pairs as independent samples.

## 5. Environment eligibility

Before interpreting an individual-effect result, the environment/task/intervention formulation must be audited for:

1. Action efficacy: the intervention changes the outcome.
2. Branchability: the same individual can be restored and both interventions executed under the declared exogenous conditions.
3. Individual dependence: treatment effects vary meaningfully across individuals.
4. Measurability: the effect is distinguishable from the estimator's error floor.
5. Convergence: the effect estimate stabilizes as independent individuals are added.

The five checks are separate. A simulator family is not automatically eligible; each task/action/utility formulation needs its own audit. The source paper's (A geq 3phi) convention is not a universal threshold. Any adopted threshold must be preregistered with sensitivity analysis.

This evaluator does not compute the abducible-share decomposition (A, A_{lat}, A_{pred}), estimate a structural causal model, prove identifiability, or convert observational records into individual counterfactual ground truth. It consumes actual paired branch outcomes from a qualified branchable environment.

## 6. HDC/CfC-LTC experiment

This module supports, but does not implement, the following training/evaluation comparison:

- Reconstruction-only.
- Reconstruction plus action-conditioned transition prediction.
- Reconstruction plus individual-contrast supervision on training branch pairs.
- HDC-only latent representation/readout.
- CfC/LTC-only temporal dynamics.
- HDC+CfC/LTC composition.
- Strongest reproducible value-aligned latent baseline.
- PCA, random projection, fixed quantization and identity controls.

For each frozen checkpoint, evaluate the same held-out individuals and intervention pairs through both the latent readout and the deployed imagined rollout. A post-hoc probe using evaluator labels is diagnostic only; it does not establish that the deployed world model preserves or uses the information.

Keep training, calibration/development and confirmatory individual/branch families separate. Preserve per-individual comparisons for the existing cluster-level inference path. Do not claim a memory/energy benefit without measurement on the actual candidate representation path.

## 7. Implementation boundary

The evaluator module is `research/temporal-evaluator/src/individual_effect_retention.rs`. It is registered in the existing standalone evaluator binary and reuses:

- #13239 for decision-partition/regret qualification.
- #13258 for temporal representation-preservation profiles.
- #13375/#13377 and the Symtropy verifier for paired branch/event evidence.
- #13445 for model-budget and downstream decision-value evaluation.
- #9437 for candidate/evaluator process isolation.

No second simulator, causal oracle, regret implementation, calibration engine, bootstrap/statistics engine, branch scheduler or action authority is added.

The initial source adds 15 deterministic Rust test functions for F8 mean cancellation, separate latent/rollout error, paired identity/provenance mismatch, duplicate IDs, non-applied branches, incompatible task/loss scales and extreme signed utility differences. The Rust toolchain is not available in the authoring environment, so these tests have not been compiled or executed here. A dedicated Rust 1.96 GitHub Actions workflow is added for exact-head qualification.

## 8. Claim ceiling

A passing Rust workflow would qualify only the finite arithmetic and input/provenance contracts implemented here. It would not establish:

- empirical abducible-share values or general counterfactual identification;
- calibration or predictive superiority;
- improved HDC/CfC-LTC representations;
- benchmark/SOTA gains;
- physical safety, actuator authorization or deployment readiness.

Those claims require separate source-bound experiments, held-out individuals/interventions, valid environment formulations, independent evaluator custody, complete failure denominators and reproducible closed-loop outcomes.


## 9. Exact counterfactual restoration tiers and first pilot

A digest match is a commitment to an identity, not proof that an individual was restored to the same physical or simulated state. The experiment must declare which restoration tier produced the paired outcomes.

### Tier S — deterministic simulator snapshot/restore

This is the first individual-effect evaluator pilot.

- Restore the same serialized simulator state and hidden individual parameters before each intervention.
- Freeze the same dynamics/build, timebase, intervention definitions and exogenous disturbance schedule.
- Use the #13375/#13377 action-coupled branch contract: shared initial state and exogenous event identity, independent arm-local future state/observation trajectories after actions diverge.
- Keep simulator state, hidden individual parameters and exogenous forcing outside the candidate process.
- Run the independent replay and stochastic correspondence verifier before admitting the paired outcomes.

A simulator snapshot digest and branch receipts support exact finite simulator counterfactuals only under the simulator's declared semantics. They do not establish that the simulator is physically faithful.

### Tier H — hardware-in-the-loop with measured restoration

A physical or HIL claim needs an additional restoration-measurement protocol: fixture/setup ID, instrument/calibration identity, pre-intervention state vector and tolerance, post-reset state vector, restoration residual, environmental conditioning, component wear, and an independently reviewed restoration receipt. Any variable outside tolerance invalidates the pair or demotes it to a non-exact matched restart. A source-state digest alone cannot qualify physical restoration.

### Tier R — randomized matched physical units

Where the same physical individual cannot be restored and receive both interventions, randomized matched-unit trials can estimate population-average or subgroup treatment effects under a declared design. They do **not** reveal both potential outcomes for the exact same physical individual. Do not report them as exact individual counterfactual ground truth or a third-rung reproduction.

Observational logs alone are weaker still: they do not provide the missing potential outcome without additional identification assumptions and a qualified estimator.

### First low-cost pilot: dm_control cartpole swing-up

Prioritize the exact `dm_control/cartpole/swingup` formulation audited in Okita (2026-11-04), not a generic claim about every DMC task. The paper reports it passes its C1-C5 formulation/measurement checks, with abducible share (A=0.396) and estimation floor (phi=0.093). Its source-specific (A ge 3phi) criterion is met (0.396 vs 0.279). These values are the paper's result, not Symthaea's.

Pilot sequence:

1. Recover/pin the paper's exact environment version, latent/individual parameterization, intervention pair, outcome variable, seed schedule, estimator and sample-size/convergence procedure. Do not approximate its task and call that a reproduction.
2. Reproduce the C1-C5 audit and estimation-floor sensitivity before training Symthaea. Include threshold sensitivity (2x/3x/4x) and held-out individuals.
3. Generate exact simulator snapshot/restore pairs. Share the exogenous event/tick schedule by stable identity, but let post-action endogenous trajectories diverge.
4. Feed the same factual history to HDC-only, CfC/LTC-only, HDC+CfC/LTC, reconstruction-first, individual-contrast and simple-latent baselines.
5. Compare the frozen-latent effect readout with the actual imagined rollout on held-out individuals and held-out intervention pairs. Keep mean-effect error, individual-effect MAE/sign, action-ranking, decision regret and resource cost separate.
6. Only after the low-dimensional mechanism reproduces should we choose a visual/dynamic manipulation task and rerun C1-C5 for its exact task/intervention/utility. The paper's robosuite push formulation fails individual dependence; its stack formulation does not clear the full measurability/convergence checks. A robot benchmark name does not by itself create individual-counterfactual ground truth.

### Decision relevance is a separate axis

PR #13468's F10/F11 fixtures demonstrate that the same individual-effect fidelity error can be decision-sufficient at a declared epsilon and decision-critical at a tighter epsilon. Reuse #13239/#13152 as the only common epsilon-good action / regret authority. This module reports counterfactual fidelity; it does not compute decision sufficiency or replace the independent hard-constraint gate.

Links:
- [Decision-Equivalent Imagination v0.1 and F1-F11](https://github.com/Luminous-Dynamics/symthaea/pull/13468)
- [Causal trajectory pairing #13375/#13377](https://github.com/Luminous-Dynamics/symthaea/pull/13377)
- [Okita paper](https://www.mdpi.com/2504-4990/8/11/363)

### Application-time outcomes are not pairing identities

Each individual comparison now retains the baseline and alternative branch's actual `applied_at_us` plus the signed difference (alternative minus baseline). These timestamps are not required to match: delay introduced by compute, observation, queueing or execution is part of the treatment. The evaluator does not shift either branch onto an artificial common timestamp. The existing #13377 receipt must bind each realized utility to its own exact application timestamp, action digest and branch trajectory. The new Rust reference preserves those times for downstream paired/cluster analysis; it does not independently validate the trajectory tick or recompute the outcome.
