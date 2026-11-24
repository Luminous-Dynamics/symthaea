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
