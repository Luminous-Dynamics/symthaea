# Randomized Decision Regret v0.1

**Status:** evaluator-side finite contract and unit tests added; exact-head Rust qualification and integration with a trained candidate are pending.  
**Parent:** #13568, stacked after DMC restore/prefix qualification.  
**Owner:** new module research/temporal-evaluator/src/randomized_decision_regret.rs.  
**Reuses:** decision_observability.rs as the sole decision-loss/action-set validator and deterministic minimax comparator.

## 1. Why this increment exists

The existing decision-observability evaluator deliberately qualifies deterministic policies: when hidden worlds have the same candidate-visible history, recurrent state, model identity and frozen policy seed, the candidate must emit the same action in every world. That is the correct contract for a deterministic policy, but it does not cover a stochastic policy that emits one probability distribution and then samples an action.

This matters for model-based RL, stochastic MPC, distributional planners, and many learned action policies. Comparing a stochastic candidate only by its sampled action discards the distribution that generated the action. Comparing only by its expected loss can also hide an action with very poor consequences that still has positive probability.

The new module evaluates the policy distribution directly and reports three different quantities:

1. **Worst-case expected regret:** the largest expected per-world regret over all evaluator-supplied indistinguishable worlds.
2. **Mean expected regret:** the equally weighted mean over those worlds.
3. **Worst supported-action regret:** the largest regret of any action to which the policy assigns nonzero probability in any supplied world.

The third quantity is deliberately not called a tail-risk guarantee. A positive-probability action may be arbitrarily rare; a genuine CVaR or chance constraint requires a calibrated draw model and a separately qualified sampling/uncertainty contract.

## 2. Finite mathematical contract

Let W be the evaluator's finite set of hidden worlds, A the common admissible action set, L(w,a) in [0, 1,000,000] the evaluator-only normalized loss, and p(a|v) the candidate's action distribution for a candidate-visible information state v.

For each world:

    r(w,p) = sum_a p(a|v) * (L(w,a) - min_b L(w,b))

The primary robust statistic is:

    R_wc(p) = max_w r(w,p)

The mean statistic averages r(w,p) uniformly over the supplied worlds. The support statistic is:

    R_support(p) = max_(w,a:p(a|v)>0) (L(w,a) - min_b L(w,b))

The evaluator receives action probabilities as integer parts-per-million. They must be strictly positive for listed rows, canonically sorted by unique action ID, and sum to exactly 1,000,000. Zero-mass actions are omitted. The reported result is exact for the encoded PPM distribution; it is not an exact evaluation of an arbitrary higher-precision source distribution. Any quantization from a float/softmax policy must be defined and reported by the policy adapter, not silently assumed here. The weighted numerators are retained exactly; PPM outputs round upward, so rounding cannot make the encoded candidate distribution appear better. The policy-artifact digest is an identity reference only, not a hash of the action-probability rows; a surrounding run receipt should bind the emitted rows verbatim to the run.

The deterministic minimax-regret action/floor is also emitted for context. It is a comparator only. A randomized policy can achieve a smaller worst-case expected regret than every deterministic action, so the deterministic floor is not a lower bound for the randomized problem.

## 3. Implementation boundary

The Rust module:

- records one policy-artifact identity alongside the exact action-distribution rows for an equivalence class (the policy digest is not a hash of those rows);
- requires all hidden worlds to have identical candidate-visible observation history, recurrent-state identity, model/configuration identity and policy seed through the existing canonical oracle;
- checks that every positive-probability action is admissible in every world;
- reuses the existing decision-observability implementation for action-set and view validation and for the deterministic comparator;
- retains exact expected-loss and expected-regret numerators per world;
- reports expected regret separately from supported-action regret;
- rejects incomplete probability mass, zero-mass rows, noncanonical ordering, unknown actions, duplicate worlds, empty commitments and arithmetic overflow.

It does not implement a randomized minimax optimizer, estimate the probability distribution from samples, sample candidate actions, fit a calibrated tail model, authorize actions, or authenticate execution receipts. A policy digest is an identity reference, not proof that the candidate executed the declared distribution. Confirmatory use still requires #9437 isolation and evaluator-owned receipts.

## 4. Falsifying fixture

The semantic fixture uses two observationally indistinguishable worlds:

| Hidden world | Left loss | Right loss |
|---|---:|---:|
| W1 | 0 | 1,000,000 |
| W2 | 1,000,000 | 0 |

Either deterministic action has worst-case regret 1,000,000 PPM. A 50:50 mixture has expected regret 500,000 PPM in each world, while its worst supported-action regret remains 1,000,000 PPM.

This fixture demonstrates the intended distinction. It is not a trained-model result, an empirical benchmark score, or evidence that randomization is generally beneficial. In asymmetric or more complex games, the best mixture need not be uniform.

## 5. Proposed benchmark: Decision Advantage Protocol

The central research question should shift from "does the world model predict plausible futures?" to "does its counterfactual information improve decisions under a fixed resource budget?"

For each frozen decision point and hidden-world equivalence class, compare policies on identical worlds, admissible actions, observation histories, deadlines, and cost envelopes:

- direct reactive policy with no learned world model;
- deterministic policy using the same candidate representation;
- stochastic policy without a learned world model;
- matched GRU/RSSM or other recurrent world-model policy;
- HDC-only, CfC/LTC-only, and HDC+CfC/LTC policies;
- model-based search with both optimistic and pessimistic / ensemble-minimum aggregation;
- evaluator-only clairvoyant action oracle as an upper bound, never available to candidate training or execution.

Report model fidelity and decision value separately. Required outcome families:

- individual-effect MAE versus the zero-effect baseline, sign accuracy, pair-centred correlation and effect calibration;
- worst-case expected regret, mean expected regret, support regret, and candidate regret relative to the evaluator's admissible-action oracle;
- policy ranking correlation and regret as rollout depth grows;
- closed-loop task success, recovery, constraint violations, and failure severity;
- p50/p95/p99 latency, peak memory, inference FLOPs or measured compute, simulator interaction count, and measured energy where hardware permits.

A world model only earns a decision advantage when it improves held-out regret or task outcomes against a matched no-model policy at equal or lower cost. Lower pixel error, better reconstruction, or a larger latent dimension is not sufficient.

### Split discipline

Hold out entire individuals/physical parameter draws and separately hold out intervention pairs. Keep all sibling counterfactual rows for one individual in one split and one cluster during inference. Confirmatory outcomes must not be used for checkpoint selection, prompt tuning, planner tuning or threshold selection. Freeze action sets, deadlines, support manifests, seeds, and utility scales before evaluation.

### Prospective targets

These are target gates, not current results:

- positive paired held-out decision-regret improvement with a 95% individual-cluster interval excluding zero;
- at least 10% relative regret reduction at matched all-in compute as a stretch target;
- no regression in the task's predeclared hard constraints or unacceptable failure classes;
- calibrated 90% and 95% predictive/decision intervals on held-out individual strata where interval semantics are valid;
- at least 90% retention of measurable individual-effect signal from latent readout to imagined rollout, only when latent signal is demonstrably above the estimator floor;
- a Pareto improvement across regret, p95 latency, memory and energy, rather than an isolated accuracy claim.

Do not average task-success percentages across benchmarks, or compare DMC's abducible share with a manipulation success rate.

## 6. Industry and research comparison

The August 2026 survey *Do World Models Make Better Robots?* reports a corpus of 160 benchmarks, of which 11 contain an explicit VLA-versus-world-model comparison and four bridge prediction to executed action. Those counts are specific to that survey's corpus, not a universal census. Its central gap is exactly the one this protocol addresses: prediction quality and closed-loop action success are usually assessed in separate tracks.

WorldArena 2.0 provides a useful external anchor because it explicitly adds visuo-tactile evaluation, world models as online RL environments, and simulated plus real-robot platforms. Its announced tracks include video quality, a world-model-as-RL-environment track, and real-world manipulation. This makes it a strong future validation target, but not a directly interchangeable score table: Symthaea must use the published task split, action API, sensors, scoring and hardware conditions before claiming a head-to-head result.

Other comparison roles remain distinct:

- World-In-World: closed-loop embodied task success and action-interface comparison.
- Rollback: checkpoint rewind and branch-outcome supervision on real robots; compare only with the same task, robot, reset and recovery protocol.
- MRS.Q: model-based search baseline; report search-induced overestimation and the effect of pessimistic ensemble aggregation, not prediction error alone.
- Causal-JEPA: visual causal representation comparison in a matching observation/action setting; do not transfer its visual benchmark numbers to low-dimensional DMC.

## 7. Paradigm shift

A model should not be rewarded for imagining every possible future. It should be rewarded for preserving precisely the distinctions that can change the decision.

A next-generation system should:

1. Build decision-relevant equivalence classes over hidden worlds, rather than compressing states solely by reconstruction similarity.
2. Maintain distributions over actions and outcomes, and expose expected regret separately from support and calibrated tail risk.
3. Ask for additional sensing, simulation or verification only when it can change the epsilon-good action set or reduce regret enough to justify its cost.
4. Use model-based imagination as a proposal mechanism, then independently verify the narrow set of decisions that matter.
5. Abstain or fall back to a certified baseline when the information set is not decision-sufficient.
6. Measure the complete cost of prediction, search, observation and verification, including latency and energy.

This is a decision-sufficient, evidence-carrying world model—not a claim that one model architecture dominates every task.

## 8. Ownership and non-claims

- #13520: individual-effect arithmetic.
- #13543/#13562: posterior-draw coupling and per-individual cohort summaries.
- #13568/#13566: DMC restore and factual-prefix qualification.
- #10372/#10377: structural support, cross-world assumptions and analytic identification fixtures.
- #13239/#13152 and existing decision-observability modules: deterministic decision sufficiency and regret.
- #9094 (CMAQ-011) owns the external decision-centric benchmark bridge; #9289 (CMAQ-013) owns causal advantage-of-prediction/sham-model controls; #9396 (CMAQ-016) owns optimizer exploitability. This PR adds a randomized-policy scoring lane for those programmes; it does not replace their environment adapters or broad benchmark ownership.
- #10693 (HLB-002) owns HDC/CfC neuron correctness and gradient qualification. Its one-step-vs-sequence-BPTT boundary must be resolved before any HDC/LTC temporal trainer result is described as sequence-level learning.
- #9437: candidate/evaluator process isolation.
- #8578: independent-cluster inference.

The new module is an evaluator primitive, not a candidate trainer, not a randomized minimax optimizer, not a standalone benchmark suite, and not a second causal-pairing verifier. No DMC paper replication, trained HDC/CfC gain, randomized-policy optimality, closed-loop improvement, physical result, SOTA win or safety authorization is claimed.


## 9. External evidence snapshot (checked 2026-11-25)

These projects evaluate different capabilities and must remain distinct comparators:

- **MRS.Q (ICML 2026):** search can reduce performance even with an accurate model; taking the minimum over an ensemble of value estimates is an effective control for search overestimation in the paper's benchmark domains. Compare search optimism and realized return at matched model-query and compute budgets. Source: https://proceedings.mlr.press/v306/chang26a.html
- **World-In-World (ICLR 2026):** a closed-loop platform with Active Recognition, Active Embodied QA, Image-Goal Navigation and Robotic Manipulation. It standardizes a planner/action API and uses task success rather than video quality as its primary embodied utility signal. The authors also report that controllability, action-observation post-training and inference-time compute matter for task success. Source: https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b4263be85820683d78675cc18d2efc7-Abstract-Conference.html
- **MiraBench (May 2026):** action-conditioned reliability is decomposed into physics adherence, action-following fidelity and optimism-bias detection; the paper reports more than 16,000 human judgments across 12 model configurations. This directly motivates failure-inducing actions and explicit false-success prediction metrics in our protocol. Source: https://arxiv.org/abs/2605.29360
- **Causal-JEPA (ICML 2026):** object-level latent masking improves counterfactual visual question answering and lets its planner use a much smaller latent input than patch-based world models in its tested settings. It is a relevant visual-representation comparator, not evidence that low-dimensional HDC/LTC or arbitrary counterfactual policies will win. Source: https://proceedings.mlr.press/v306/nam26c.html
- **WorldArena 2.0 (IROS 2026 challenge):** its challenge separates video quality, world-model-as-online-RL-environment, and real-world manipulation tracks; the broader benchmark direction includes visuo-tactile evidence and simulated/real robot platforms. We should target its native task API and scorecards rather than translate its visual score into DMC regret. Source: https://iros2026challenge.world-arena.ai/
- **World-model benchmark survey:** a separate 2018–2026 survey snapshot lists 130 benchmarks with 108 cross-category entries. Source: https://world-model-benchmarks.github.io/
- **Prediction-to-action survey (30 August 2026):** a different survey's 160-benchmark corpus counts 11 explicit VLA-versus-world-model comparisons and four prediction-to-executed-action bridges. Source: https://arxiv.org/abs/2609.29669

The two survey counts (130 and 160) use different corpora and coding schemes. They are not additive and should not be presented as a census contradiction. The common conclusion is that prediction, action selection, and closed-loop outcomes are still measured in fragmented ways.

### Resulting benchmark additions

Add these explicit tests to the Decision Advantage Protocol:

1. **Action-following falsification:** hold the initial observation and action history fixed; propose actions that should produce visibly different outcomes; score whether the model predicts the correct direction and magnitude of the change.
2. **Optimism traps:** include actions that are physically impossible, contact-unstable, out-of-support or known to fail. Report false-success rate, failure severity calibration, and whether search preferentially selects those errors.
3. **Decision-rank stability:** correlate predicted action ranking with independently executed candidate outcomes at fixed action-bank and model-query budgets. Report top-1 regret, top-k regret and rank correlation separately.
4. **Inference-time scaling:** measure whether extra imagination/search buys real held-out return, or only increases model-query count and optimism. Include direct policy and compute-matched controls.
5. **Modality ablation:** on embodied tracks, compare vision-only and visuo-tactile models under the same action tape and sensor budget. Keep low-dimensional DMC separate.
6. **Counterfactual observability:** vary which latent state features are available (e.g., position only, position + velocity, history probes) and test whether this changes individual-effect information and decision regret, not just reconstruction.

The strongest Symthaea claim would not be "our model is more realistic." It would be: *under a frozen, matched budget, it retains more decision-relevant individual-effect information, chooses actions with lower held-out regret, detects when imagined futures are unreliable, and falls back before model exploitation causes a failure.* This is a falsifiable research hypothesis, not a current result.
