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



## Randomized-policy decision regret

The evaluator-side module `src/randomized_decision_regret.rs` extends decision-observability evaluation to stochastic policies. The deterministic oracle remains the canonical validator and deterministic comparator; this module accepts one frozen action distribution for an otherwise identical candidate-visible information set.

- Action probabilities are integer PPM and must sum exactly to 1,000,000.
- Expected loss and per-world expected regret retain exact integer numerators; displayed PPM values round upward.
- The certificate reports worst-case expected regret across hidden worlds, equal-world mean expected regret, and maximum regret among every action with nonzero support as separate values.
- The deterministic minimax-regret score is only a comparator. It is not the optimal randomized-policy floor.
- The module does not sample actions, calibrate a probability model, optimize a mixed policy, authenticate candidate execution, or establish safety.

Fifteen unit tests cover a balanced mixture that halves worst-case expected regret relative to the deterministic comparator, support-risk separation, deterministic degeneration, exact probability mass, canonical action ordering, inadmissible actions, candidate-view identity, duplicate worlds, digest requirements and conservative rounding, empty/bounded inputs, action-set consistency, loss-range validation and asymmetric mixtures.


## Individual counterfactual decision transfer

The evaluator-side module `src/individual_decision_transfer.rs` measures whether a multi-action utility ranking from the history-conditioned latent readout or the imagined rollout selects an action with lower independently realized regret for the same individual.

- Each scorecard declares the complete expected action set before evaluation and binds every action row to one individual, factual history, full branch-point snapshot, shared exogenous-forcing manifest, task/loss/split/model profiles and the same latent state.
- Every row must have a unique action/run identity, a verified upstream pairing/trajectory reference, and an applied outcome. Missing action rows or non-applied branches reject; no outcome is silently removed or imputed as zero.
- Utilities use one frozen integer scale (higher is better). Ties are resolved by lexicographically smallest action ID for selection; all exact realized top-1 actions and the complete epsilon-good set remain reported.
- The certificate separates latent and rollout choices, realized regret, top-1/epsilon-good membership, the paired rollout-minus-latent regret difference, and pairwise ranking concordance/discordance/ties. Ranking ties are retained so Kendall tau-b can be reconstructed.
- Output is canonicalized by action ID and echoes the complete frozen decision context plus per-action pairing/trajectory digest references. It contains per-individual rows only; no confidence interval or cross-individual inference is computed. #8578 remains the independent-cluster inference owner.

This implements a known decision-transfer metric family already discussed in Okita (2026), rather than claiming the metric itself as novel. The Symthaea-specific increment is its bounded, typed and provenance-linked Rust evaluator. Digest references are not receipt authentication, and package separation is not a security boundary: validate the underlying artifacts and retain #9437 isolation before confirmatory use.

Research contract: [INDIVIDUAL_DECISION_TRANSFER_V0_1.md](../../docs/research/INDIVIDUAL_DECISION_TRANSFER_V0_1.md). Semantic fixture: [individual_decision_transfer_v0.1.json](../../docs/research/fixtures/individual_decision_transfer_v0.1.json). Exact-head Rust 1.96 formatting, tests and strict Clippy are required before qualification.
