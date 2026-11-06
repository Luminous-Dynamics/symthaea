# Query-discriminating connectome intervention benchmark V1

**Tracking:** #10670 — SCI-CONNECTOMICS-092
**Composes:** #10603, #10656, #10661, #9522, #9587, #9549, #9575, #9578, #9934, #9255
**Status:** protocol + synthetic contract fixture; no biological intervention or performance result is claimed.

## 1. Objective

Select feasible perturbations that distinguish currently admissible reconstruction worlds where their scientific query decisions differ. Identify when no available intervention can make the target decision identifiable under the declared nuisance/observation model envelope.

This is not a new general intervention scorecard. #9587 owns broad intervention prediction and active-experiment evaluation; #10603 owns exact finite reconstruction-world expansion and adaptive observation policies; #9204 owns design-based sampling; #9248 owns all-list-missed sensitivity; #9255 owns scientific certificates.

## 2. Test-directed reconstruction

The benchmark reverses the conventional workflow. Instead of treating reconstruction as complete before experimental design:

1. Freeze the query, loss, source revision, base worlds, candidate alternatives and all admissibility rules.
2. Freeze the feasible intervention universe, costs, measurement/indicator kernel, state/context scope and nuisance-model family.
3. Predict an observable response distribution for every intervention × world × nuisance-model combination.
4. Partition worlds by query decision and determine which interventions separate those decision classes.
5. Select an intervention by the preregistered primary objective: robust reduction in expected decision loss/regret per total intervention cost.
6. Compare entropy/EIG, expert and graph heuristics under the same information and budget constraints.
7. After an observed response, update world compatibility using calibrated likelihoods and recompute query robustness.
8. Stop only when the remaining compatible worlds agree on the query or return an explicit non-identifiability / no-feasible-intervention disposition.

Maximizing entropy reduction is not the primary objective. An intervention can have high information gain while distinguishing only worlds that agree on the scientific decision. Such an intervention may be scientifically unhelpful for the frozen question.

## 3. Objects and claim boundaries

Keep these objects separate:
- direct structural edges, with source-bound EM/synapse/gap-junction evidence;
- multi-hop structural paths in a versioned connectome;
- latent neural-state dynamics;
- the sensor/indicator/observation kernel;
- measured finite-horizon responses to interventions;
- mechanism attribution: direct, polysynaptic, extrasynaptic/latent, shared-input, state-mediated or unresolved;
- scientific query decision and loss for each reconstruction world.

Synapse counts are not conductances. A measured response does not prove a direct synapse. A structural path does not guarantee transmission in every state. An anatomical non-edge is not a functional zero outside the declared coverage and detection limits. Extrasynaptic signalling is not an EM edge.

## 4. Primary decision objective

Let the frozen world set be W, query q(w), prior sensitivity set Π, feasible interventions A, costs c(a), nuisance models m in M, and predictive observation distributions P(y | w,a,m).

The primary score is the worst-case expected posterior query loss per total intervention cost over the preregistered prior and nuisance envelope. The objective must state its exact loss table and whether the prior is fixed, sensitivity-bounded or minimax. For this V1 contract fixture, the prior is fixed and the loss is 0–1 query-decision error; nuisance robustness is a maximum over the two declared nuisance models.

A selected intervention must be feasible and must bind its prediction/separation evidence to exact world, query, observation-model and nuisance digests. The evaluator—not the input author—recomputes entropy, expected posterior loss, risk reduction and objective ranking.

If no feasible intervention improves the preregistered decision objective, return NO_FEASIBLE_DISCRIMINATING_INTERVENTION. If competing query decisions remain observationally equivalent under every feasible intervention and nuisance model, return NOT_IDENTIFIABLE. These are not failed runs and must not be coerced into a positive result or a zero-effect claim.

## 5. Baselines

All baselines receive the same candidate worlds, data, feasible intervention universe, nuisance envelope, experiment budget, optimization compute and prior information.

- random feasible intervention;
- expert-ranked intervention;
- node degree / centrality / shortest-path heuristics;
- expected entropy reduction / mutual information;
- expected KL separation / Bayesian EIG;
- greedy one-step decision value of information;
- fixed non-adaptive intervention bundle;
- exact exhaustive optimum on bounded synthetic cases;
- activity-only, shuffled, and degree-/type-preserving rewired controls.

Report information gain as a secondary measure. Do not combine it with posterior query risk, predictive calibration, feasibility or acquisition cost into a single opaque scalar.

## 6. Synthetic known-answer fixture

Three admissible worlds have frozen query decisions and priors:

| World | Query decision | Prior |
|---|---:|---:|
| H-PLUS | true | 0.10 |
| H-MINUS | false | 0.10 |
| H-SAME | true | 0.80 |

The baseline Bayes decision is true, with expected 0–1 loss 0.10.

Two nuisance models are declared: nominal and indicator-saturation.

| Intervention | Nominal observations (PLUS / MINUS / SAME) | Saturation observations | Feasibility |
|---|---|---|---|
| high-eig-irrelevant | X / X / Y | X / X / Y | feasible |
| cheap-confounded | X / Y / X | X / X / X | feasible |
| orthogonal-robust | X / Y / X | X / Y / X | feasible |
| infeasible-perfect | X / Y / Z | X / Y / Z | disallowed |

Known-answer properties:
- high-eig-irrelevant yields H(0.2, 0.8) ≈ 0.721928 bits, but leaves H-PLUS and H-MINUS compatible after observation X. Its expected posterior query loss remains 0.10.
- cheap-confounded separates the query classes under the nominal model but loses that separation under indicator saturation. Its worst-case expected posterior query loss remains 0.10.
- orthogonal-robust separates H-MINUS from both query-true worlds under both nuisance models. Its worst-case expected posterior query loss is 0.
- infeasible-perfect is maximally informative and resolves the query, but must never be selected because it is explicitly disallowed.
- Entropy-greedy chooses high-eig-irrelevant; robust decision-loss selection chooses orthogonal-robust.
- Baseline expected loss is 0.10; the robust intervention reduces it to 0 under this synthetic known-truth model.

This fixture tests objective semantics, not a biological or statistical population claim.

## 7. Required adversarial cases beyond the starter fixture

1. Latent states differ, but the observation kernel maps them to the same observable distribution: return NOT_IDENTIFIABLE.
2. A nuisance model erases apparent anatomical separation: robust selection must account for it.
3. A query-irrelevant world has large prior mass and attracts entropy-greedy selection.
4. An infeasible/disallowed intervention looks optimal: it is excluded before optimization.
5. An adaptive second intervention is needed after the first outcome.
6. A non-significant result has insufficient power: preserve NOT_IDENTIFIABLE, not functional non-connection.
7. Extrasynaptic response produces an effect without a direct EM edge: attribution remains extrasynaptic/latent or unresolved.
8. Prior sensitivity changes Bayesian selection: report the range and robust solution.
9. Target/animal/session leakage occurs: reject the run.
10. The intervention policy is trained or tuned on its own lockbox: reject the run.
11. Missing/failed/inaccessible response data are not negative outcomes.
12. Multiple worlds with different query decisions remain compatible after all feasible interventions: explicit non-identifiability.

## 8. Biological benchmark strata

### C. elegans

Randi et al., Neural signal propagation atlas of Caenorhabditis elegans, Nature (1 November 2023), reports optogenetic activation and whole-brain calcium measurement for 23,433 neuron pairs across 113 animals. Its response map measures effective propagation over all paths and can reflect extrasynaptic signalling. It is an intervention-response benchmark, not direct synaptic ground truth. Pin task, response horizon, animal/target splits, preprocessing and the exact source/connectome revision.

### MICrONS

Use its co-registered structure/function data for observational and stimulus-conditioned predictions. Do not describe stimulus response as a universal optogenetic intervention set or infer same-animal proofreading benefit where the exact pairing is absent.

### Drosophila walking CPG

Pugliese et al., Connectome simulations identify a central pattern generator for fly walking, Nature, version of record 28 October 2026, is a high-value circuit case. The paper reports DNg100-driven rhythmic motor activity and a separate DNb08 pathway with optogenetic experimental support. It explicitly reports that its rate-model simulations do not reproduce tripod inter-leg coordination. That gap creates discriminative hypotheses involving local rhythm generation, proprioceptive feedback, biomechanical coupling and descending inter-leg pathways. Keep computational screening and headless-fly intervention results as separate evidence tiers. See #9934 for exact reproduction and source custody.

### Fish datasets

Fish1.5/Fire&Wire are eligible only where exact same-specimen identity, data rights, observation modality and intervention evidence support the task. Calcium imaging alone does not become a perturbation trial.

## 9. Metrics and promotion targets

Report separately:
- worst-case expected posterior query loss;
- expected posterior query loss and prior-sensitivity range;
- query-loss reduction per unit of total intervention cost;
- query-critical class separation / resolvability;
- false-resolution rate;
- expected and realized information gain;
- predictive proper scores, calibration and interval coverage;
- full-trajectory error on held-out target/animal/session;
- abstention/non-identifiability rate;
- intervention feasibility, acquisition, model, compute, reviewer and analysis costs.

Aspirational promotion targets (not current results):
- 100% exact optimum agreement on bounded synthetic cases;
- 0 false query-resolution certificates with an opposing compatible world;
- 0 infeasible/disallowed interventions in selected policies;
- 100% correct non-identifiability on known observational-equivalence controls;
- ≥20% lower decision loss per total intervention cost than the strongest frozen baseline on preregistered held-out cases, with uncertainty supporting improvement;
- nominal 90% predictive interval coverage near 90%, with sharpness and proper scoring reported separately;
- zero target/animal/session leakage.

Use paired uncertainty intervals, clustered by the independent biological unit, and report all failed/unresolved cases. These thresholds require independent prospective evidence before biological promotion.

## 10. Staged execution and claim ceiling

Stage 1: schema + synthetic contract qualification.
Stage 2: replay the C. elegans atlas and #9587 observation-model artifacts under genuine animal/target lockboxes.
Stage 3: Drosophila CPG discriminative hypotheses, contingent on exact artifact custody and reproduction in #9934.
Stage 4: prospective blinded biological interventions only with qualified collaborators, approved protocols and exact data custody.

The benchmark can establish distinguishability under a declared finite hypothesis set, intervention universe, nuisance envelope and observation model. It cannot prove that the hypotheses are biologically complete, identify synaptic causal effects from anatomy alone, or guarantee an in-silico optimal intervention will work in vivo.