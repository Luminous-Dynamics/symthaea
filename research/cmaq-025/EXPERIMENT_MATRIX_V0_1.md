# CMAQ-025 — Experiment Matrix v0.1

**Status:** proposed qualification design. No experiment in this document is a measured result.  
**Purpose:** compare representation quality, identifiability, active evidence acquisition, and downstream decision utility under one frozen contract.

## 1. Research thesis: decision-sufficient state, with an explicit identifiability envelope

The programme should no longer optimize a world model solely for low next-state prediction error. It should seek the smallest representation that preserves all permitted future intervention/query/horizon responses needed by a declared decision contract.

For a candidate representation \(z=\phi(x)\), define a contract-indexed response map:

\[
\mathcal R_{\mathcal C}(x)
=\{R(x, a, q, h): (a,q,h)\in\mathcal C\}.
\]

Two states may share one abstract state only when their registered response maps are equivalent within the contract tolerance and all hard invariants remain satisfied. The equivalence is contract-relative; it is not a claim that the states are physically identical or equivalent for every possible future question.

The new object is an **identifiability envelope** attached to every prediction:

- which observable context and mechanism scope the prediction is supported on;
- which intervention, query and horizon were qualified;
- the independent-cluster support behind the prediction;
- which assumptions/invariance warrants are active;
- which permitted probes could reduce ambiguity;
- what output mode is justified: point, response set, or UNKNOWN.

This separates three conditions that ordinary confidence scores often collapse:

1. **Known response:** the candidate-visible view is supported and the contract response is consistent across admissible held-out clusters.
2. **Observable shift:** the inputs changed in a way that can be tested against support, invariants and known response families.
3. **Observational alias:** identical candidate-visible inputs produce different evaluator-only responses. No candidate can distinguish those worlds without additional information. The correct result is a response set, a qualified invariance warrant, or UNKNOWN—not a more confident point estimate.

## 2. Factorial test matrix

All arms must use the same frozen intervention/query/horizon contract, independent clusters, evaluation episodes, probe budget and action policy unless a factor explicitly changes.

| Dimension | Required cells | What it diagnoses |
|---|---|---|
| Family transfer | Known family; unseen ID/same map; observable covariate shift; new context region | Retrieval versus extrapolation |
| Mechanism identifiability | Identifiable; observational alias; probe-resolvable alias; probe-unresolved alias | When a scalar response can be known |
| Evidence quality | Full; sparse support; conflicting support; stale evidence; integrity failure | Support gating and fail-closed behavior |
| Execution outcome | Success; non-execution; failure; partial; rollback; delayed effect; outcome unknown | Whether assigned intent is confused with realized transition |
| Temporal regime | One-step; multi-step; delayed effect; irreversible boundary | Horizon-conditioned response fidelity |
| Optimizer pressure | Direct policy; small/medium/large search; adversarial optimistic-error search | Search-amplified model exploitation |
| Resource budget | Fixed memory; fixed latency; fixed probe cost; fixed rollout count | Capability per unit cost |

The hidden mechanism label, reference response and held-out family identity remain evaluator-only. They are never predictor inputs. Training/evaluation splits are made by independent mechanism/environment cluster, not random transitions.

## 3. Matched representation baselines

| Baseline | Purpose | Must not be misrepresented as |
|---|---|---|
| Identity / full-state | Upper-bound memorization and over-splitting control | A deployable compressed model |
| No-op / identity transition | Copying shortcut and false-merge control | A learned response model |
| Global per-action HDC mean delta | Current NixWard additive-delta baseline | Context-aware causal reasoning |
| Contextual HDC delta | Current candidate preserving observable context distinctions | A CfC/LTC model |
| Context + action-parameter HDC | Test whether action-parameter structure adds response sufficiency | Proven causal mechanism discovery |
| Symbolic / SCM | Structured intervention baseline where variables and equations are available | A universal oracle |
| Recurrent transition model | Temporal baseline; use CfC/LTC only when actual training, checkpointing and inference are integrated | A model that exists merely because a library module exists |
| Large video/world foundation model | External frontier comparison on common tasks and interfaces | A directly matched memory/latency competitor without accounting for its full stack |

Do not pool families or tune on held-out evaluator outcomes. Every model receives identical training records and allowed candidate-visible fields. A parameter-count comparison alone is insufficient: report total serialized model state, support memory, encoder cost, inference latency, energy, probe use and rollout/search compute.

## 4. CMAQ-025 identifiability cases

The scorer consumes two separate channels: candidate predictions and evaluator-only annotations. The candidate output is materialized first; the scorer joins the channels afterwards.

| Case | Candidate view | Evaluator response | Expected interpretation |
|---|---|---|---|
| A: identifiable | Supported view | One response across clusters | Point forecast may be scored |
| B: benign ID shift | Same view, new family/cluster IDs | Same response | Do not over-abstain merely because IDs are new |
| C: observable shift | New candidate-visible features | Response changes consistently with observable features | Test generalization or abstention against support |
| D: hidden alias | Identical complete candidate view | Conflicting responses across families | Point forecast has an irreducible error floor |
| E: probe resolves | Probe result changes candidate view | Responses become conditionally consistent | Charge probe cost; permit a new conditional prediction |
| F: probe fails | Probe result does not change informative content | Responses remain conflicting | Do not claim the alias was resolved |
| G: support conflict | Same view has contradictory training response evidence | Conflicting support | Return UNKNOWN or a justified set, not an arbitrary tie-break |

A finite response set is complete only relative to evaluator-observed outcomes in this finite benchmark. It is not proof that the set contains every possible outcome in an open world.

## 5. Metrics and scorecard

Report metrics separately; never hide a weak cell behind one weighted average.

**Representation and equivalence**
- pair-level false merges and false splits;
- contract response equivalence by intervention × query × horizon;
- changed-response exact accuracy;
- response regret and invariant violations.

**Identifiability and selective prediction**
- point coverage and selective risk;
- abstention rate on empirically unidentifiable views;
- unnecessary abstention on identifiable views;
- response-set miss rate, width and complete observed-set coverage;
- irreducible scalar point-error floor for aliased groups;
- view-consistency, custody and leakage violations (hard failure, not an averaged metric).

**Active evidence acquisition**
- probe cost and useful-response rate;
- reduction in false merges and decision regret per unit probe cost;
- false splits introduced by refinement;
- value of information against random, uncertainty-only, and evaluator-hinted acquisition;
- oracle-blind acquisition reported separately from evaluator-hinted CEGAR.

**Planner and execution**
- selected-action regret, not merely average one-step model error;
- overestimation rate and worst-case consequence of optimized predictions;
- recovery success, recovery depth, irreversible action count and time-to-safe-state;
- task success, constraint violations, recovery after deviations, and fallback use.

**All-in resource cost**
- resident bytes and serialized bytes for model plus support/evidence index;
- train/update/inference latency and energy;
- probe and rollout compute;
- tail latency and cost per successful, invariant-preserving task.

Use independent mechanism/environment clusters as the inference unit. Report per-family values, weakest family/cell, and cluster-bootstrap confidence intervals. Repeated transitions from one environment do not create independent evidence.

## 6. Frontier benchmark and industry comparison

The external benchmarks are complementary test beds, not one interchangeable leaderboard.

| External programme | What it tests | CMAQ/Symthaea contribution to test alongside it |
|---|---|---|
| [World-in-World (ICLR 2026)](https://world-in-world.github.io/) | Closed-loop utility across four environments; task success and controllability rather than video quality alone | Whether the model knows when its response forecast is identifiable, and whether that improves replanning |
| [WorldArena 2.0](https://world-arena.ai/) | Visuo-tactile world models, interactive RL environments, simulation and real-robot platforms | Contract-level response fidelity, evidence scope, resource cost and failure/UNKNOWN behavior |
| [ContactWorld](https://arxiv.org/abs/2606.13877) | 12 contact-rich tasks; structured spatial and temporally continuous visuo-tactile representations | Test whether HDC composition preserves contact geometry and transition-relevant tactile structure; do not assume current HDC does so |
| [LIBERO-Recover](https://arxiv.org/abs/2609.05178) and [RoboRecover](https://arxiv.org/abs/2609.28952) | Recovery from real execution deviations, action failures and intermediate off-nominal states | Typed realized-action episodes, partial/rollback outcomes, outcome-unknown reconciliation and safe fallback |
| [MRS.Q / The Surprising Difficulty of Search](https://proceedings.mlr.press/v306/chang26a.html) | Search can harm performance through overestimation even when the model is accurate | Test selected-action optimism and error exploitation as search budget increases; compare robust ensembles and explicit UNKNOWN |

Industry comparison should include publicly accessible versions of:
- NVIDIA Cosmos 3 / Cosmos-Predict 2.5, where multimodal world generation, reasoning, action prediction and interactive simulation are relevant;
- Meta V-JEPA 2 / 2-AC, where latent prediction and image-goal robot manipulation are relevant;
- Google DeepMind Genie 4, where interactive generated-world simulation is relevant;
- strong direct VLA/WAM policies and native simulators as controls.

Use a shared environment, same action API, same sensor availability, same task distribution and matched inference/rollout budgets. Some external models are large generative simulators or policy backbones; they are not like-for-like with a small symbolic/HDC transition module. Compare full end-to-end capability per resource, and separately compare the incremental value of adding Symthaea's evidence/identifiability layer to the same underlying policy.

## 7. Proposed qualification gates (targets, not results)

| Stage | Proposed gate |
|---|---|
| Finite contract witnesses | Zero false merges on all registered deterministic equivalence witnesses; zero scalar forecasts accepted as universally correct for deliberately aliased groups |
| Selective behavior | 100% of deliberately unidentifiable deterministic witnesses return complete observed response sets or UNKNOWN; no blanket abstention on the benign-ID-shift controls |
| Mechanism transfer | Positive held-out-cluster benefit over pooled HDC on pre-registered shifted-context cells, without hiding a negative worst-family cell |
| Active probes | Lower decision regret than random and non-separating probes at the same total probe cost; report the full cost/benefit curve |
| Planner robustness | Increasing search must not increase hard-invariant violations; selected-action regret must not become worse than the matched direct-policy baseline |
| Compression | Retain the earlier stretch target: at least 25% lower all-in representation cost with no more than 1 percentage point of task-success loss |
| Physical recovery | Demonstrate statistically supported recovery improvement on held-out deviations before claiming an embodied benefit; keep sim and physical results separate |

The first gates are deterministic software qualification, not evidence of physical safety. Statistical gates require independent clusters, pre-registered splits, negative controls, and exact-head source/checkpoint receipts.

## 8. Highest-value next experiments

1. **CMAQ-025 → NixWard adapter:** export predictions from the actual contextual HDC delta path into the candidate channel; join evaluator annotations only in the scorer.
2. **CMAQ-023 × CMAQ-025:** test whether oracle-blind probes reduce decision regret per cost, not just whether they split state partitions.
3. **CMAQ-016 × CMAQ-025:** increase search budget over controlled optimistic errors; measure whether selection amplifies errors and whether robust response sets/UNKNOWN prevent harmful choices.
4. **CMAQ-024 successor:** move beyond unseen IDs to shifted observable context, partial support, and explicitly unidentifiable hidden-mechanism cells.
5. **Embodied handoff:** only after the software response contract is stable, run matched closed-loop tests in World-in-World / WorldArena-compatible tasks, then contact-rich and failure-recovery suites.

## 9. Claim discipline

The following distinctions are mandatory in reports:

- implemented vs structurally represented vs source-tested vs exact-head CI-qualified vs physical-world validated;
- family-ID transfer vs covariate shift vs mechanism shift vs unidentifiable aliasing;
- candidate-visible evidence vs evaluator-only labels/outcomes;
- observed finite response-set coverage vs a guarantee over all possible outcomes;
- open-loop prediction quality vs closed-loop decision utility;
- simulator result vs physical robot result;
- accuracy under a fixed policy vs error exploitation under policy optimization/search.

No SOTA or safety claim follows from a deterministic toy fixture. A credible claim must name the test environment, checkpoint, source manifest, independent clusters, action budget, uncertainty/abstention protocol and reproducible evidence receipt.
