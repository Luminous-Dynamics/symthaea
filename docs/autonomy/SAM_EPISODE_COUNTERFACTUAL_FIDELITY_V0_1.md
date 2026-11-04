# SAM Episode-Specific Counterfactual Fidelity v0.1

**Status:** research protocol / analytic and controlled-simulation fixtures  
**Issue:** [SAM-EPISODE-COUNTERFACTUAL-001 #10356](https://github.com/Luminous-Dynamics/symthaea/issues/10356)  
**Composes:** #10257, #10266, #10237, #10291, #10318, #10320, #10338, #10043; Symtropy #3471, #3387 and #3502  
**Claim ceiling:** episode-specific counterfactual fidelity conditional on the admitted structural mechanism, factual evidence, abduction set and scenario context.

## 1. Motivation: action-conditioned is not counterfactual-conditioned

A world model can generate a plausible future after receiving a shared history and alternative ego action, yet omit an event that the factual continuation has already revealed in the specific episode.

The distinction is between:

- **Observational prediction:** what tends to happen given observed history/action.
- **Interventional prediction:** what outcomes follow if an action is set across a target population.
- **Episode-specific counterfactual:** what would have happened in this particular already-observed episode if the ego action had been different.

The third requires episode-specific evidence from the factual continuation. A direct action-conditioned rollout is a required baseline, not automatically an R3 counterfactual.

The August 2026 paper *How Can Driving World Models Do Counterfactual Prediction?* (arXiv:2608.11601) demonstrates this failure in a controlled CARLA benchmark. Its 186 cases span 72 placements, three towns and three scenario types. Direct predictions from two tested model families generally score 0.23–0.45 on their event-recovered-fraction metric, while the paper's training-free evidence-transport pipeline reports 0.64–0.70 overall. Its surrounding actors follow open-loop scripts, so the study does not test interactive actor responses to changed ego behavior.

The September 2026 CausalDriveBench paper (arXiv:2609.32157) adds 7,285 verified causal QA pairs and 1,000 counterfactual trajectories. It reports a best QA result of 70.6%, four of 13 models below chance, and no statistical correlation between causal QA and trajectory accuracy. Its initial release is based on urban nuScenes data at 2 Hz, and its R3 trajectories are synthesized by planner/LLM and audited because human counterfactual futures are unobservable in recorded data.

These are complementary benchmarks, not interchangeable evidence.

## 2. Formal episode-specific estimand

Let a structural causal model be:

`X_j := f_j(PA_j, U_j)`

where:
- `PA_j` are the declared parent variables;
- `U_j` is the episode-specific exogenous state/noise;
- `f_j` is the frozen structural mechanism.

For factual action `a_i` and factual observations/continuation `F_i`, the abduction stage produces a set or posterior:

`U_i^* = {u : SCM(u, a_i) is compatible with F_i under the declared observation model}`

The episode-specific counterfactual under alternative action `a'` is:

`Y_i(a') = {SCM_Y(u, do(A=a')) : u in U_i^*}`

If `U_i^*` is a singleton under justified assumptions, a point counterfactual may be reported. If several exogenous states remain compatible, preserve the resulting outcome set/bounds. A single plausible sample is not an identified counterfactual.

A random seed is not an abduction result. It must be tied to the observed factual episode through its actual latent state, draw receipts or admissible posterior.

## 3. Three estimands must remain separate

### R1 — observational prediction

`P(Y | H, A=a)`

This is a prediction under observed associations. It cannot be promoted to an intervention or episode-specific counterfactual claim without additional assumptions/evidence.

### R2 — intervention effect

`P(Y | do(A=a))`

This estimates a distribution of outcomes under an intervention for a declared population. It requires a target population and justified sampling/intervention assumptions. It is not an answer about one specific previously observed episode.

### R3 — episode-specific counterfactual

`Y_i(a')` conditional on factual episode `(H_i, a_i, F_i)`

R3 requires abduction from the factual continuation, replacing the declared action, then propagating the same admitted structural mechanisms/exogenous episode state through the counterfactual branch.

Missing factual continuation or missing abduction evidence makes R3 unavailable/NotIdentified; it must not silently collapse to R1 or R2.

## 4. The central invariant: preserve exogenous facts, recompute descendants

A valid episode-specific counterfactual must neither discard all factual continuation evidence nor copy the entire factual continuation unchanged.

Each variable is classified as one of:

- `PreInterventionState`
- `EpisodeExogenous`
- `InterventionTarget`
- `CausalDescendant`
- `NonDescendant`
- `SharedCauseOrConfounder`
- `UnobservedOrUnidentified`

Rules:

1. **Pre-intervention common state:** preserve the exact shared prefix and initial episode identity.
2. **Episode exogenous state:** preserve the same abduced exogenous cause across branches where the structural model declares it exogenous to the intervention.
3. **Non-descendant:** preserve the factual episode value if the graph and context justify non-descendence.
4. **Descendant:** recompute through the same frozen structural mechanism under the new intervention.
5. **Shared cause/confounding:** preserve the common cause; do not propagate it as if it were an effect of the intervention.
6. **Interactive actors:** when the ego intervention changes an actor-visible cue, re-invoke the actor policy only after its declared delivery/reaction delay. The actor's exogenous policy noise may be coupled across worlds, but its endogenous response must be recomputed.
7. **Post-treatment observables:** do not copy or force factual values merely because the scenario seed matches.
8. **Not identified:** preserve multiple compatible outcomes or return `NotIdentified`; do not invent a unique hidden state.

This is stronger than fixed-script replay. A fixed-script actor remains a useful baseline but is not interactive.

## 5. Abduction, action and prediction pipeline

### 5.1 Abduce

Consume the immutable factual episode, factual action history, synchronized sensor/time metadata, and admitted structural mechanism. Infer the episode-specific exogenous state or posterior/set. Retain:
- source evidence and observation model;
- inferred variables and their uncertainty;
- compatible latent-state alternatives;
- non-identifiable dimensions;
- abduction algorithm/artifact digest.

For camera/sensor outputs, depth/geometric lifting and reprojection can transport supported observations into the alternative viewpoint. Every pixel/region/track must distinguish transported factual evidence from synthesized unsupported content. Geometry-based evidence transport is not a substitute for causal mechanism propagation.

### 5.2 Act

Replace only the declared intervention variable(s), such as the ego command/trajectory over a frozen action interval. Bind exact command payload, intervention time, units, route/trajectory and scope. Do not silently change the rest of the model.

### 5.3 Predict

Re-run all structural descendants with the same admitted mechanisms and episode exogenous variables. For interactive policies, regenerate actor observations from the counterfactual world at each tick and apply the actor policy after delivery/reaction delay. Record every event and state transition in the Symtropy replay chain.

### 5.4 Score

The evaluation-only counterfactual reference must remain hidden from the prediction model and planner until the model output/action is committed. Compute event, state, trajectory, geometry, causal and decision metrics independently.

## 6. Required execution arms

For each eligible episode/action pair, retain separately:

1. **Factual:** observed episode replayed under factual ego action.
2. **Direct-action baseline:** shared pre-decision history + alternative ego action, no factual continuation.
3. **Factual-evidence/AAP arm:** factual continuation-derived abduction set + alternative ego action + frozen structural mechanisms.
4. **Null / event-free control:** same action with a declared event absent, when defined.
5. **Interactive actor branch:** actor policy recomputed from counterfactual ego-visible observations.
6. **Fixed-script actor control:** same script applied regardless of ego intervention; explicitly not interactive.
7. **Oracle/reference:** evaluation-only reference execution, never a model prediction or deployable system.

Freeze the factual episode and alternative action before exposing the reference/counterfactual outcome.

## 7. Identity, pairing and causal receipts

Every episode/action result binds:
- factual episode and immutable factual trace digest;
- shared history, decision tick and factual action payload;
- alternative intervention payload and interval;
- SCM/mechanism graph and version digest;
- abduction evidence/posterior digest and method;
- exogenous variable identities/draws and coupling policy;
- actor-policy identity, observation allow-list and reaction-delay profile;
- variable classification graph and causal edges;
- scenario, environment, dynamics, camera/sensor, map and clock profiles;
- prediction model, reference, evaluator and artifact digests;
- execution receipt, output artifact and causal event-log roots;
- uncertainty, applicability, missingness and NotIdentified dispositions.

Use Symtropy #3387/#3459/#3502 for exact execution/replay. A hash-chain detects mutation relative to a retained root but does not authenticate the origin of factual data or prove model truth.

## 8. Metrics — keep them orthogonal

### Episode-event fidelity
- event/actor identity preservation;
- event presence/omission and false insertion;
- event onset/offset timing error;
- event track/trajectory residual;
- recovered fraction only when factual/counterfactual/null references make its denominator well-defined.

### Causal fidelity
- correct non-descendant invariance;
- correct descendant recomputation;
- confounder/shared-cause preservation;
- structural mechanism consistency;
- interactive actor response after delivered cues;
- future-action/evaluator leakage.

### Sensor/visual fidelity
- geometric support fraction;
- transported factual evidence fraction;
- synthesized/unsupported fraction;
- reprojection/track alignment error;
- perceptual similarity only where a valid matched ground truth exists.

### Decision fidelity
- action-ranking reversals;
- selected-action regret under #10237;
- reference-sensitive decisions under #10318;
- action-space adequacy under #10338.

### Epistemic fidelity
- counterfactual set coverage and size where a reference set is available;
- NotIdentified / abstention rate;
- compatibility of abduction posterior with factual trace;
- missing, censored, timed-out and unsupported denominators.

### Cost
- inference/execution time p50/p95/p99;
- compute/memory/energy per query;
- exact replay time and resource use.

No single metric (ADE, image embedding, QA accuracy, recovered fraction or route success) can stand in for all these dimensions.

## 9. Analytic and adversarial fixture matrix

The companion JSON fixture includes:
- event exogenous to ego intervention: event must persist across alternative ego action;
- interactive actor descendant: observed ego change triggers a later actor policy re-evaluation;
- non-descendant invariant: unrelated event remains fixed;
- null/event-free: absent factual event cannot be hallucinated as recovered evidence;
- over-copy: factual post-treatment actor state cannot be frozen when policy should change;
- missing factual continuation: R3 is NotIdentified/unavailable;
- ambiguous abduction: multiple latent episodes remain compatible and must yield a set/bound;
- wrong mechanism reuse: changed structural mechanism invalidates prior abduction/counterfactual identity;
- future leakage and clock mismatch;
- direct plausible rollout omits a factual event: must not pass R3 fidelity.

## 10. Benchmark ladder and external baselines

| Benchmark | Role | Important limitation |
|---|---|---|
| Analytic SCM fixtures | Exact R1/R2/R3 correctness and known latent-state recovery | Not physical validation |
| arXiv:2608.11601 CARLA subset | Episode-specific visual-event preservation; direct baseline vs evidence transport/AAP | 186 cases, 72 placements, scripted actors; ego action is varied while other scripted motion is replayed |
| CausalDriveBench arXiv:2609.32157 | Causal QA, active/dormant/distractor entities and R2/R3 action trajectories | nuScenes urban Boston/Singapore, 2 Hz; R3 targets are synthesized/audited rather than observed human counterfactuals |
| Symtropy #3471/#3502 | Interactive actor response after counterfactual ego cues | First policies are synthetic, not human-behavior models |
| Waymo Open Motion / Argoverse 2 | Forecast and scene-history transfer only with explicit counterfactual construction | Recorded factual future is not a human counterfactual ground truth |
| Fail2Drive / CARLA / Bench2Drive | Downstream closed-loop outcomes under separate protocols | Do not substitute route success for R3 episode fidelity |

The August paper's visual recovered-fraction values (direct ~0.23–0.45; AAP ~0.64–0.70 overall for the two tested backbones) are reproduction targets on that exact case set and metric, not universal SOTA numbers. A fair comparison must pin cases, encoder, reference/null runs, model/checkpoint, seeds and scoring code.

The CausalDriveBench paper reports 70.6% best causal-QA accuracy and no statistical correlation between its causal QA and trajectory error. Preserve that separation; a fluent explanation is not proof of action-level causal fidelity.

## 11. Falsification controls

1. A non-descendant episode event revealed only in factual continuation must persist in the alternative branch.
2. Ego action changes an actor-visible cue; actor state cannot change before cue delivery/reaction delay.
3. The same visible cue/history with changed hidden controller state must not change actor policy output.
4. A descendant variable must be recomputed through its frozen mechanism.
5. A shared confounder must remain a shared cause, not become an intervention effect.
6. An event-free null control prevents plausible but unsupported event insertion.
7. Copying all factual post-treatment states must fail when the intervention changes their causal parents.
8. Direct action-conditioned plausible output that omits a known factual event must fail R3 episode fidelity.
9. Missing factual continuation prevents R3 classification.
10. Multiple compatible exogenous states preserve a counterfactual set or NotIdentified state.
11. Changed structural mechanism/profile invalidates the abduction/counterfactual artifact.
12. Changed camera pose/time/coordinate/action payload rejects pairing.
13. Future action or evaluator-only outcomes cannot leak into model/actor inputs.
14. Same random seed without matching episode-specific exogenous identity is insufficient pairing.
15. Stale/missing/unsupported/timeout outputs never count as event preservation or success.

## 12. Acceptance and claim ceiling

A runtime implementation is accepted only after independent recomputation of analytic fixtures, exact episode/action/context pairing, explicit R1/R2/R3 labelling, correct descendant/non-descendant behavior, policy-mediated actor recomputation, NotIdentified handling, and exact fresh-process replay. Rust compile/test/format and exact-head CI must be observed; skipped or stale-head checks are not passes.

Passing establishes episode-specific counterfactual fidelity only under the named structural mechanisms, factual evidence, abduction method, exogenous coupling and scenario contexts. It does not establish general identifiability from ordinary logs, human-behavior validity, simulator physical fidelity, crash probability, controller safety, roadworthiness or SAE L4/L5 capability.

## 13. Prospective targets (not results)

- zero false R3 claims when factual continuation or abduction evidence is absent;
- zero non-descendant event omissions in admitted analytic/controlled cases;
- zero incorrect freezing of post-intervention interactive actor states;
- 100% of unsupported/occluded/synthesized regions and NotIdentified variables are retained as such;
- at least +0.15 absolute recovered-fraction improvement versus direct action-conditioned baseline on a rights-cleared, exact-protocol reproduction, with event identity/timing and geometric consistency non-inferior;
- zero future-action/evaluator leakage;
- exact matched-branch replay for all controlled synthetic fixtures.
