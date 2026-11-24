# Symthaea Temporal Evaluator

This is a separate, evaluator-side Rust package for paired retain/evict effects and exhaustive small-subset upper bounds. It is not a candidate runtime dependency.

The current source provides:

- seeded, reproducible retain-first / evict-first order assignment using evaluator-held randomness (SplitMix64 for reproducibility only; not cryptographic secrecy);

- paired intervention checks that require identical episode/task/future-stream/model/initial-state/candidate-feature digests and identical byte/compute/refresh/observation budgets;
- componentwise wrong-belief, abstention and missing-output exposure differences;
- group interventions to expose non-additive memory interactions;
- exact subset enumeration up to 16 candidate items;
- separate best-known subsets for each exposure component and a Pareto frontier, with no weighted scalar objective;
- hard checks for duplicate IDs, unordered selections, incorrect bytes, budget overruns, incomplete subset enumeration and exposure totals exceeding episode duration.

The exact subset result is an evaluator-only clairvoyant upper bound, not a deployable candidate policy. It must never be exposed to the candidate or used to train against confirmatory outcomes.

Package separation is an architectural boundary only. It does not provide OS-level confidentiality. Before confirmatory use, complete #9437: separate OS identities and protected oracle storage, candidate sandboxing, evaluator-owned signing keys, signed receipts, replay protection and adversarial filesystem/descriptor tests.

No Cargo build, format, Clippy or test execution is claimed until a Rust toolchain is available.

## Individual counterfactual effect retention

The evaluator-side module `src/individual_effect_retention.rs` compares, for each restored individual and a fixed pair of interventions:

- the independently measured realized utility contrast;
- the contrast predicted from the same frozen history-conditioned latent readout;
- the contrast predicted by the model's action-conditioned imagined rollout.

It reports per-individual MAE/sign matches and exact cohort sums for realized, latent and rollout effects. Keeping per-individual rows prevents a correct population mean from hiding errors that cancel between individuals. Integer utility units and exact numerators are retained; no new loss scale or floating-point aggregation is introduced.

### Pairing and failure contract

Both sibling branches must bind to the same individual, factual history, restored branch-point state, exogenous-forcing manifest, latent state, model configuration and paired seed. Their action IDs must be the declared baseline/alternative pair. The task, intervention pair, loss scale, evaluator profile/split and model/readout/rollout profiles are frozen across the cohort.

The module requires nonempty digests referencing the existing #13377 / Symtropy pairing verifier artifacts, but it does **not** authenticate receipt bytes or independently execute the verifier. The evaluator wrapper must retrieve and validate the actual receipt and source artifacts before admission. This module is a consumer of paired evidence, not a second causal-pairing implementation.

A branch that was not applied is returned as a typed error. It is not imputed as zero effect or removed silently; the caller must retain the non-application and its task-contract penalty in the overall experiment denominator. No confidence intervals or statistical inference are computed. Preserve the per-individual comparison rows for the existing independent-cluster analysis path.

The module does not estimate the paper's abducible share, prove structural causal identification, calibrate outcomes, authorize actions, or establish physical safety. It is valid only for independently qualified branchable formulations that satisfy the relevant action-efficacy, individual-dependence, measurability and convergence conditions. Candidate/evaluator process isolation under #9437 remains mandatory.

For external interpretation, distinguish deterministic simulator snapshot/restore (Tier S), hardware-in-the-loop with measured restoration (Tier H), and randomized matched physical units (Tier R). Only Tier S/H with qualified branch identity can support an individual counterfactual claim; Tier R estimates population/subgroup effects, not both potential outcomes for the same physical unit. The first low-cost pilot is the paper-audited `dm_control/cartpole/swingup` task, after exact upstream parameterization/intervention reproduction and C1-C5 re-audit. Its published A=0.396 and phi=0.093 are prior-paper values, not Symthaea measurements.

The individual comparison retains both actual branch application timestamps and their signed alternative-minus-baseline delay. Application-time differences are treatment outcomes (e.g. compute-induced delay), not pairing errors. #13377 remains responsible for binding each realized outcome to its exact trajectory tick, action digest and application timestamp; this evaluator only preserves those values for downstream analysis.

## DMC individual-effect pilot boundary

The versioned DMC pilot contract is in [DMC_INDIVIDUAL_EFFECT_PILOT_V0_1.md](../../docs/research/DMC_INDIVIDUAL_EFFECT_PILOT_V0_1.md), with its machine-readable source/parameter manifest at [dmc_individual_effect_pilot_manifest_v0.1.json](../../docs/research/fixtures/dmc_individual_effect_pilot_manifest_v0.1.json).

Important distinction: the paper's `cartpole.swingup` result is an environment-eligibility / observation-masking diagnostic (150 individuals, six interventions, 15 pairs, 12 history blocks, A=0.396, phi=0.093). The paper's data table reports no trained world-model checkpoint for this DMC formulation. Symthaea model training on this task is therefore a new extension, not a reproduction of a paper DMC latent-versus-rollout result. The exact physical-parameter perturbations, action sequences, masks, seeds and simulator driver still require recovery before calling the DMC eligibility track a reproduction.

No DMC benchmark execution or Symthaea model-side result is claimed by the pilot contract.

