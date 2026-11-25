# Individual Decision Transfer Evaluator v0.1

**Status:** bounded Rust evaluator and semantic fixture added; exact-head Rust 1.96 qualification pending.  
**Issue:** [#13623](https://github.com/Luminous-Dynamics/symthaea/issues/13623).  
**Parent branch:** `research/randomized-decision-regret-v0-1-20261125`.  
**Module:** `research/temporal-evaluator/src/individual_decision_transfer.rs`.

## 1. Research question

For one individual and one frozen decision point, does the action selected from a history-conditioned latent utility readout or from a decoded/imagined rollout agree with the independently realized best action, and what realized regret does each choice incur?

This is the decision-side bridge between the finite effect-fidelity arithmetic in #13520 and the supplied stochastic-policy evaluation in #13592. It is a per-individual action-set scorecard, not a new causal estimator, generic regret oracle, policy optimizer, simulator, or statistical-inference implementation.

## 2. Prior-art and claim boundary

The metric family is prior art. Okita (2026), Section 9.2, already reports per-individual intervention regret, top-1 action choice, within-individual rank correlation, and comparisons against a constant policy. Section 9.1 also describes paired contrast supervision and checkpoint selection. This work does not claim those metrics as novel.

- [Okita (2026), article and supplementary materials](https://doi.org/10.3390/make8110363)
- [Section 9.2: Calibration slope and the decision](https://www.mdpi.com/2504-4990/8/11/363#sec9dot2)

The contribution here is an auditable, bounded Rust implementation that binds a complete per-individual action set to existing branch/run, snapshot and pairing evidence; separately evaluates frozen-latent and rollout action choices; retains ranking ties; and leaves inference to #8578. It should reproduce published metric semantics where data and definitions permit before introducing extensions.

## 3. Input contract

One `IndividualDecisionTransferInput` contains:

- a single decision-point context with nonzero digests for the individual, factual history, full branch-point snapshot, common exogenous realization, task, loss scale, evaluation split, evaluator, model/checkpoint, latent representation/state, readout, rollout and pairing-verifier profiles;
- a declared complete expected action-ID set, fixed before candidate predictions or evaluator outcomes are inspected;
- a nonnegative integer epsilon in the same utility units;
- one executed branch row for every expected action.

Each branch row carries a unique action ID and run ID, the same factual-history / full-snapshot / exogenous / model / representation / latent digests, nonzero pairing-receipt and trajectory digest references, an application disposition and two predicted utilities.

All realized and predicted values use one frozen integer utility scale, with **higher values better**. The action set must be identical across the three evaluators. Every branch must have been applied in v0.1. A missing or non-applied action is a typed refusal; it is never dropped or imputed as zero. A future version may accept upstream task-contract penalties only if their semantics and independent verification are explicitly specified.

Digest fields are identity references, not proof of artifact authenticity. The evaluator launcher must retrieve and verify the underlying branch, trajectory and pairing receipt bytes before it supplies these records. #9437 process/OS isolation remains mandatory.

## 4. Output semantics

The evaluator sorts action rows by action ID and returns deterministic output. The certificate echoes the complete frozen decision context, and each output action row retains its pairing-receipt and trajectory digests so the evidence references survive scoring. On exact predicted-utility ties, the lexicographically smallest action ID is selected. It reports:

1. realized best utility and the complete realized epsilon-good action set;
2. latent-readout and imagined-rollout selected action IDs;
3. realized utility and regret of each selected action;
4. whether each selection is an exact realized top-1 action and whether it belongs to the realized epsilon-good set;
5. paired `rollout_regret - latent_regret` (positive means rollout choice is worse);
6. action-level realized/latent/rollout utilities and regret values;
7. pairwise ranking components for each predictor: concordant pairs, discordant pairs, realized-only ties, predicted-only ties and ties on both axes.

The ranking components are sufficient to reconstruct Kendall's tau-b without silently treating tied pairs as concordant or discordant. They are not pooled across individuals. Per-individual top-1, epsilon-good hit and regret are the units passed downstream; #8578 remains responsible for independent-cluster inference.

The scorecard deliberately does not compute confidence intervals, rank correlations across individuals, calibration slopes, posterior coverage, effect-retention ratios, or causal identified sets. Keep all of those quantities separate.

## 5. Required negative controls

The unit suite covers:

- latent readout selects the realized best action while rollout selects a worse action;
- rollout selects the realized best action while latent readout selects a worse action;
- shuffled rows and expected-action order yield identical canonical certificates;
- epsilon-good sets include all actions within the declared tolerance;
- exact argmax ties use the stable lexical rule while all tied realized winners remain top-1;
- predicted and realized ranking ties are retained in distinct counters;
- a common additive utility shift does not change action choices or regret;
- changed exogenous-realization and full branch-point-snapshot identities reject;
- missing/extra action rows, duplicate action IDs and duplicate run IDs reject;
- non-applied actions reject instead of receiving fabricated utility;
- missing receipt references, negative epsilon and zero decision-point ID reject;
- extreme signed utilities use wider checked arithmetic without wraparound;
- the certificate echoes the exact context and preserves action-specific receipt/trajectory references.

## 6. Research hypothesis and falsification

**Hypothesis:** individual-specific information that is present in a branch-point latent representation may be attenuated, distorted or occasionally improved by the decoded rollout. The meaningful consequence is whether the action ranking and realized per-individual regret change—not whether a reconstruction or effect-magnitude metric improves in isolation.

Falsify the hypothesis where rollout rankings match or outperform latent rankings across held-out individuals, especially under conditions where the latent estimator is above its floor. Also include null-effect and all-actions-epsilon-equivalent strata. A model that predicts the population-average action but fails individual-specific ranking must not pass as individualized decision competence.

## 7. Benchmark design

Use whole-individual / physical-parameter-draw splits and separate intervention-pair holdouts. Sibling branches, posterior draws and action rows belonging to one individual remain in the same split and cluster.

Required model arms:

- constant/population policy and direct reactive policy without a learned world model;
- history-only and branch-point-only predictors;
- matched GRU/RSSM;
- HDC-only and CfC/LTC-only;
- HDC+CfC/LTC;
- shared dynamics with persistent individual-effect memory;
- latent-readout and rollout-readout decisions from the same frozen checkpoint;
- evaluator-only realized-best action as an upper bound, never candidate-visible.

Report three layers independently:

**Representation:** individual-effect MAE and error against zero-effect baseline; sign agreement; rank correlation; calibration slope, intercept and bin error; latent-to-rollout effect retention above the declared estimator floor.

**Decision:** per-individual realized regret; top-1 and epsilon-good hit rates; Kendall tau-b / tie counts; improved and worsened fractions versus the constant policy; concentration of the most common action; the paired latent-versus-rollout regret delta.

**Systems:** all-in training and inference compute; simulator interactions; p50/p95/p99 decision latency; peak memory; measured energy with host, transfer, preprocessing and verification boundaries declared. Report quality-versus-cost Pareto curves.

The constant/population baseline must be frozen on training individuals, then evaluated on held-out individuals. Do not select its best common action using the confirmatory cohort. The paper's same-evaluation-cohort constant baseline is a historical metric definition; the preregistered holdout is a stricter deployment-oriented extension and must be labelled as such.

## 8. Comparison strategy

- **EAWM:** direct DMC model-based-RL comparison only under matched task, observation, update, interaction and compute protocols. It reports gains on DMC 500K and DMC-GB2 500K, but those scores are not individual-effect or decision-transfer scores. [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/fc09b26b85ab3abb2832bd555a2e4215-Abstract-Conference.html).
- **World Model Evaluation Lab / CPG:** reuse fixed-planner oracle-dynamics versus learned-dynamics attribution. Change only the dynamics callable; hold planner, initial states, scoring and budgets fixed. [CPG protocol](https://denis-hamon.github.io/world-model-eval-lab/07_cpg.html).
- **MRS.Q:** compare search with optimistic versus minimum-over-ensemble value aggregation. Search can degrade decisions even with accurate dynamics; value overestimation is a separate failure mode. [ICML 2026 paper](https://proceedings.mlr.press/v306/chang26a.html).
- **WorldArena 2.0:** later visual/tactile, online-RL and real-robot validation on the benchmark's own task interfaces. Do not pool these outcomes with DMC individual-effect metrics. [Challenge definition](https://iros2026challenge.world-arena.ai/).
- **Field-level gap:** a 2026 survey catalogued 160 benchmarks and identified only 11 explicit VLA-vs-world-model comparisons and four prediction-to-executed-action bridges in its corpus. These counts are survey-specific, but they motivate reporting a matched direct-policy baseline and actual executed decision outcomes. [Jena et al. (2026)](https://arxiv.org/abs/2609.29669).

No cross-benchmark scalar leaderboard is proposed. Report matched head-to-heads and their exact task, data, action, planner, compute and hardware conditions.

## 9. Decision-world-model evidence ladder

A June 2026 decision-making-centric position paper recommends reporting an L0–L7 profile rather than a single pooled world-model score. It explicitly warns that the levels span partly independent axes (observational/interventional, open-/closed-loop, fixed-policy/optimizer-induced), and that the ladder is not monotone in practice. [Yu et al. (2026)](https://arxiv.org/abs/2606.15032).

This scorecard is a **bounded L4 / per-decision outcome slice**, not a complete L6 policy evaluation and not L7 policy optimization:

- **L4 — action controllability / interventional fidelity:** the upstream branch protocol must prove matched histories, full branch-point restoration and shared exogenous correspondence. The scorecard requires those identities and compares action-specific predicted utilities with executed outcomes.
- **L5 — reward/value/outcome fidelity:** this scorecard computes realized utility and regret for a supplied finite action set, but it does not yet establish full-horizon reward/value calibration or success-probability calibration.
- **L6 — fixed-policy evaluation/ranking:** requires closed-loop execution of behavior, anchor and shifted policies in the trusted environment and model, with full-horizon return/ranking/occupancy agreement. #9094 owns the external comparison protocol; #9289 owns the causal advantage-of-prediction controls. The current per-decision ranking metric is only one component.
- **L7 — policy optimization/planning utility:** requires optimizing inside the model under a fixed budget, executing the resulting policies in the trusted environment, and reporting policy lift, optimization regret, safe-improvement probability and optimizer-induced exploitability. #9094 owns the matched external benchmark bridge; #9396 owns adaptive-search/model-error exploitability (XGap). Do not infer L7 from a good fixed-policy score.

### Why the L7 exploitability experiment is essential

A fixed policy set can miss narrow optimism pockets that an adaptive planner discovers. #9396 already specifies matched-global-MSE structured error families and search ladders over candidate count, search rounds and rollout horizon. Keep the random-action prediction error, optimizer-selected prediction error and realized selected-policy regret separate. Use #13652's per-individual scorecard as an action/effect diagnostic feeding the chain; do not duplicate the XGap owner.

## 10. Prospective gates (targets, not results)

- positive held-out paired regret improvement with a 95% independent-cluster interval excluding zero;
- stretch target of at least 10% relative regret reduction versus the strongest matched-cost baseline;
- no material regression in hard constraints, non-application rate or failure severity;
- separate effect-fidelity and decision-quality results, with transfer claims restricted to strata above the estimator floor;
- all-in compute/latency/memory/energy reported, with no efficiency claim from unmeasured estimates.

These are preregistration targets. They are not current benchmark results.

## 11. Execution and claim ceiling

This PR adds only a finite evaluator primitive, tests and a semantic fixture. It does not authenticate receipts, establish causal identification, train HDC/CfC, reproduce the DMC paper, demonstrate model advantage, provide physical safety evidence, or establish SOTA.