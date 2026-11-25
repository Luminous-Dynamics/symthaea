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
- World Model Evaluation Lab v0.18 / CPG: fixed-planner oracle-dynamics versus learned-dynamics task-success gap with interval gating, paired initial-state comparisons and task-distribution checks. Reuse its protocol; keep this as the planner/dynamics layer, distinct from individual-effect retention.
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

## 7A. Existing randomized-mixture work: do not duplicate the optimizer

The repository already has a randomized minimax policy-frontier programme in #8646 (with the benchmark work in #8493). Its recorded horizon-two experiment found only a negligible average randomized-mixture gain on the tested grid (0.000064 bits). Its column-generation path was approximately 2.56× slower than exhaustive-frontier plus LP over the reported 36-case aggregate. Epsilon-frontier pruning reduced the largest horizon-two kernel frontier from 450 to 131 at epsilon 0.01, with no observed regret on that tested grid; the issue explicitly does not claim a general regret theorem.

This materially changes the implementation boundary:

- #13592 measures any supplied stochastic action distribution over candidate-observationally-equivalent worlds. It is not a randomized policy optimizer.
- #8646 remains the owner of randomized minimax mixture optimization, frontier pruning and the approximation-error proof obligation.
- Future integration should feed #8646's selected mixture into this evaluator and compare its exact expected-regret certificate against deterministic, direct/no-model, and learned stochastic policies.
- A uniform mixture's improvement in the two-world semantic fixture is an arithmetic sanity check only. It is not evidence that stochastic mixtures improve real tasks or that the minimax solver is useful on the target benchmark.

The next research question is not "can we randomize?" but "does the particular uncertainty set and individual-effect task distribution contain a real robust decision improvement after the full compute/runtime cost of mixture construction?" The benchmark must report the solver's certified/actual regret gap, policy support, query/compute cost and comparison with exhaustive-frontier plus LP. Do not create a second solver in this PR.

## 8. Ownership and non-claims

- #13520: individual-effect arithmetic.
- #13543/#13562: posterior-draw coupling and per-individual cohort summaries.
- #13568/#13566: DMC restore and factual-prefix qualification.
- #10372/#10377: structural support, cross-world assumptions and analytic identification fixtures.
- #13239/#13152 and existing decision-observability modules: deterministic decision sufficiency and regret.
- #9094 (CMAQ-011) owns the external decision-centric benchmark bridge; #9289 (CMAQ-013) owns causal advantage-of-prediction/sham-model controls; #9396 (CMAQ-016) owns optimizer exploitability. This PR adds a randomized-policy scoring lane for those programmes; it does not replace their environment adapters or broad benchmark ownership.
- #10693 (HLB-002) owns HDC/CfC neuron correctness and gradient qualification. Its one-step-vs-sequence-BPTT boundary must be resolved before any HDC/LTC temporal trainer result is described as sequence-level learning.
- #8646/#8493: randomized minimax policy-frontier/mixture optimization and epsilon-pruning claims. This PR only scores supplied mixed policies; it does not duplicate that optimizer.
- #11105: robust discounted state-action-rectangular MDP LPs; a distinct model/uncertainty semantics.
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


## 10. 2026 comparator expansion: event-aware, video-predictive and world-action models

The comparator matrix must include the strongest model class for each track rather than treating a single "world model" score as universal.


### Decision-quality baseline: Counterfactual Planning Gap (CPG)

The public **World Model Evaluation Lab v0.18** is a particularly relevant external reference for the planner × dynamics part of this protocol. It defines the Counterfactual Planning Gap by running the same planner against oracle dynamics and learned dynamics, then comparing success rates. It uses Agresti–Caffo confidence intervals, paired-bootstrap intervals when both arms share initial states, and a verdict gate that can return inconclusive. Its task-distribution rerun is instructive: a fixed-initial-state Acrobot result that suggested a model bottleneck changed to a planner bottleneck when initial states were sampled from the actual task distribution and the oracle planner solved only about 3% of starts. Reacher retained a model bottleneck, while a high-capacity Cartpole/CEM cell showed the learned model outperforming oracle dynamics. The heterogeneity is a feature, not a defect.

Reference implementation and results: https://github.com/Denis-hamon/world-model-eval-lab and https://denis-hamon.github.io/world-model-eval-lab/index.html

We should **reuse CPG rather than reimplement it**. CPG diagnoses the effect of substituting learned dynamics for oracle dynamics under one fixed planner; it does not measure individual-effect identification, same-individual cross-world consistency, or whether the model retains a subject-specific treatment effect. In Symthaea, report the CPG-compatible planner/dynamics arm separately from:

- environment abducible-share certificate `(A, phi; Sigma)`;
- latent and rollout individual-effect retention;
- within-individual intervention ranking/regret and calibration;
- randomized-policy expected regret and positive-support regret;
- all-in model/planner cost.

This creates a layered failure diagnosis: planner ceiling, learned-dynamics contribution, individual-information retention, decision quality, and runtime economics. It avoids blaming the world model when the planner cannot solve the task even under oracle dynamics.
### Vector-control / DMC track

**Event-Aware World Model (EAWM, ICLR 2026)** is a direct comparator for low-dimensional control. It learns event boundaries from observations and predicts meaningful event transitions; its paper reports 10–45% gains over strong MBRL baselines across Atari 100K, Craftax 1M, DeepMind Control 500K and DMC-GB2 500K. This is important prior art: event segmentation as such is not a new Symthaea contribution. The Symthaea question should be whether HDC/CfC plus the paired-individual-effect objective preserves *which individual responds to which intervention*, improves held-out decision regret, and does so at a better complete resource cost than EAWM and matched RSSM baselines. Reproduce the exact DMC task subset, training interactions, seeds, and evaluation protocol before comparing numbers.

Source: https://proceedings.iclr.cc/paper_files/paper/2026/hash/fc09b26b85ab3abb2832bd555a2e4215-Abstract-Conference.html

### Visual / embodied track

**V-JEPA 2 / V-JEPA 2-AC (Meta, June 2025)** is a major representation-and-planning baseline. Its paper reports pretraining on more than one million hours of internet video and post-training V-JEPA 2-AC with less than 62 hours of unlabeled DROID robot video; the authors demonstrate zero-shot Franka pick-and-place planning in two labs. The released results include per-task manipulation outcomes and comparisons with Octo and Cosmos. This is not a matched low-dimensional DMC comparator. It belongs in a visual/robot track with the same image-goal interface, action space, robot task, reset protocol, and physical success criteria.

Sources:
- https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/
- https://github.com/facebookresearch/vjepa2

**Cosmos-Predict 2.5 (NVIDIA)** provides a current large-scale generative/action-conditioned video baseline. The technical report describes 2B and 14B models, 200 million pretraining video clips, domain-specific robot and driving variants, and an action-conditioned robot model. On the reported action-conditioned robotic video comparison, Cosmos-Predict2.5-2B reports PSNR 24.95, SSIM 0.85, latent L2 0.28 and FVD 146 versus Cosmos-Predict1-7B at 21.14, 0.82, 0.32 and 190. These are video-prediction metrics, not closed-loop task-success numbers; they must not be presented as action or causal-effect superiority.

Source: https://research.nvidia.com/labs/dir/cosmos-predict2.5/

**OSCAR (June 2026)** fine-tunes Cosmos-Predict2.5-2B with a skeleton-conditioned action interface to improve action following and cross-embodiment coverage; its paper also evaluates generated-world policy scores against real RoboArena evaluations. It is a relevant baseline for action-conditioned virtual policy evaluation and cross-embodiment transfer, not an apples-to-apples DMC score.

Source: https://arxiv.org/abs/2606.04463

**WorldArena 2.0** remains the integration target for the visuo-tactile, online-RL-environment, and real-world manipulation lanes. Its 2026 challenge exposes separate tracks for video quality, world model as RL environment and real-world WAM manipulation; use the official task/data/API and report the correct track-specific score.

Sources:
- https://world-arena.ai/
- https://iros2026challenge.world-arena.ai/

### Required benchmark consequences

1. Add an **event-abstraction ablation**: no event segmentation, fixed-window segments, ground-truth event boundaries for evaluator-only ceiling, learned event boundaries, and EAWM-style generic event segmentation. Compare individual-effect retention and decision regret, not just transition-prediction loss.
2. Add **action-following and optimism** arms from MiraBench: physically impossible actions, action-sign reversal, contact/failure cases, and out-of-support actions; score physics adherence, action-conditioned state change, false-success rate, and planner-selected optimism.
3. Keep three independent scoreboards: (a) vector-control/DMC and model-based RL; (b) visual/video prediction and action-conditioned reliability; (c) closed-loop and real-robot task utility. Never rank them by a pooled scalar.
4. A head-to-head claim requires matched interfaces, actions, data access, model-query/interaction budgets, training compute, inference latency and task/reset protocol. Otherwise report "related external result; not directly comparable" and use it to motivate a local reproduction.
5. The intended Symthaea edge is a falsifiable systems claim: higher individual-effect information retention and lower held-out action regret under matched resource constraints, with calibrated abstention/fallback when the model is unreliable—not visual realism or parameter count.

No benchmark numbers from these external systems are treated as reproduced Symthaea results.
