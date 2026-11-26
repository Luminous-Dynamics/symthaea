# ErgoStream Benchmark Preregistration v0.1

**Date:** 2026-11-25  
**Status:** proposed confirmatory protocol; no benchmark results or superiority claims  
**Programme:** Symthaea #13401 / #13402; Symtropy #7149 / #7152 / #7171 / #7193 / #7230  
**Purpose:** measure task-agnostic structural acquisition and its downstream transfer value without conflating representation quality, causal validity, policy quality or task success.

## 1. Central claim and falsifiable question

ErgoStream asks a capability that ordinary task-conditioned benchmarks do not isolate:

> Before a learner knows which downstream questions will be asked, can it acquire a compact, revisable library of reusable structure from causally arriving, unlabeled streams—and does that frozen library improve independently evaluated transfer to unfamiliar worlds under matched information and resource budgets?

This is an engineering hypothesis, not a claim of universal learning, a biological ergobrain, consciousness, or a new general theory of intelligence.

The benchmark contribution is the **pre-task acquisition → frozen library → hidden structural/dynamical probes → decision and action outcomes** protocol. Existing suites should be adapters and comparison points, not be mixed into one score.

## 2. Five distinct estimands

Report these separately. A result at one level cannot be promoted into a stronger claim.

1. **Acquisition quality:** what units, relations, motifs, transformations and alternatives were inferred from the available stream, and at what observation/compute/memory cost?
2. **Structural transfer validity:** does a frozen, scoped correspondence survive an independent target structure oracle? Are multiple maps possible, or is the map non-identifiable?
3. **Predictive / behavioural / mechanism transfer:** do frozen predictions survive hidden continuations, action-conditioned transitions and, separately, evaluator-owned intervention outcomes?
4. **Sequential research-policy quality:** on tiny finite worlds, how much terminal decision regret remains relative to an exact finite-horizon policy oracle? What was the cost and risk of the selected sequence?
5. **Downstream utility:** does the acquired library improve a decision or closed-loop task relative to cold-start, no-transfer and direct-policy controls under equal budgets and latency?

A structural isomorphism is not causal identity. A finite transition witness is not continuous-time conjugacy. A calibrated forecast is not a qualified mechanism. A lower prediction loss is not automatically a better decision.

## 3. Evaluation lanes and anti-saturation design

Maintain three physically separate task lanes:

### A. Public development lane

Public examples, known-answer fixtures, source-visible canaries and benchmark debugging. Scores here are never confirmatory. They may be used to tune code, baselines and thresholds.

### B. Immutable anchor lane

Versioned generator recipes, task instances, seeds, evaluator semantics and reference outputs. Once released, an anchor is never silently replaced. This lane measures longitudinal progress and detects regressions.

### C. Sealed refresh lane

After the learner, configuration, acquisition budget and analysis plan are frozen, an independent evaluator instantiates new seeds, parameter combinations, nuisance transforms, compositions and mechanism shifts. Task identities and hidden outcomes remain evaluator-held until the run is complete. The recipe may be public; the specific seed and task instance are not.

An adaptive development corpus may use solver trajectories to discover shortcuts and refine task recipes. It must never silently alter the immutable anchor or the sealed confirmatory cohort. This design is motivated by AutoSciBench's finding that refined generated benchmarks can expose shortcuts hidden by curated suites; recipe/judge dependence remains a limitation requiring independent review.

## 4. First-release task matrix

The first confirmatory release should include at least three independently generated paired-world families. A larger programme should progressively add the remaining rows.

| Family | Positive transformation | Hard negative / ambiguity control | Claim tested |
|---|---|---|---|
| Relabel / equivariance | entity, channel, symbol, coordinate and direction relabeling | superficial signature match with incompatible relation topology | structural correspondence |
| Matched marginals | same local and degree statistics, altered higher-order relations | K3,3 vs triangular prism; motif collisions | relation and motif transfer |
| Intervention divergence | observations appear structurally similar | hidden action effects disagree; mechanism mutation | mechanism-transfer rejection |
| Partial scope / multiscale | valid substructure or quotient at declared resolution | omitted context changes a required observable | scoped map and abstraction validity |
| Non-identifiability | multiple maps explain the public prefix | symmetry, missing modality, delayed evidence | set-valued maps / query / abstention |
| Temporal / hybrid | time scaling, asynchronous observation, event/reset mapping | stale observation, dropped event, clock drift, mode switch | continuous-time and hybrid transfer |
| Cross-modal | paired audio/visual/action event alignment ([Symtropy #7442](https://github.com/Luminous-Dynamics/symtropy/issues/7442)) | shuffled alignment, clock offsets, delayed arrival, missing/disagreeing modality | cross-modal structure acquisition |
| Embodied dynamics | source-to-target action and state mapping | target-side transition or latency changes | behavioural transfer and action utility |

### Cross-modal factorization: timing is not semantics

The first cross-modal campaign must be factorial rather than using one blended "synchronization" label. Use two independently controlled axes:

- **Clock registration:** shared calibration anchors present / absent; affine offset and rate mismatch; timestamp jitter; delayed and out-of-order arrival; missing anchors.
- **Event correspondence:** aligned; marginal-matched shuffled; periodic/repeated-pattern non-identifiability; missing or uninformative modality.

Required crossed cells include:

| Clock evidence | Event relation | Correct interpretation |
|---|---|---|
| Shared anchors; affine map identifiable | Aligned | clock and event correspondence can both be qualified independently |
| Shared anchors; same clock schedule | Marginal-matched shuffled | clock map remains valid while same-time semantic/event pairing is false |
| Shared anchors; affine map identifiable | Periodic event-phase ambiguity | clock is identifiable but event correspondence remains set-valued |
| No shared anchors | Periodic event-phase ambiguity | both clock registration and event correspondence require abstention/set-valued output |

Keep sensor measurement time, arrival/availability time and evaluator reference time separate. Delayed transport must not be absorbed into sensor-clock skew. Do not infer causality from temporal coincidence.

This design follows AV-SyncBench's independent temporal and semantic challenge construction. Its official repository reports pairwise accuracies (not offset-classification accuracy) of 0.583 overall for Synchformer/VGGSound and 0.569 for SparseSync/VGGSound-Sparse, with per-offset tiers at 50/100/200/300/500 ms. Ties count as incorrect. Reproduce the exact official transforms, model score functions, preprocessing mode and aggregation before any Symthaea comparison. The real-data reproduction adapter is tracked in Symtropy #7516. MATH-SYNC supplies a complementary clock-registration baseline: it solves a joint inverse-affine problem over known event correspondences and reports Joint Huber vs Joint OLS median macro-RMSE reductions of 73.7% nominal / 62.4% stress on a frozen 10-participant Harvard cohort, and 82.1% / 67.5% on 15-participant WESAD schedule replay; clean timing showed near parity. Crucially, it assumes event correspondences have already been identified, so it does not solve event-identity ambiguity. Our OLS/Theil–Sen/Huber estimator module is only a paired-anchor affine-regression baseline, not a reproduction of MATH-SYNC. The initial Symtropy implementation is #7442 / draft PR #7493; generic estimator baselines are #7495 / draft PR #7498.

Score clock offset/rate error, held-out anchor residual and false-identifiable rate separately from event-pair precision/recall, false-unique rate, set-valued correctness and held-out motif transfer. No aggregate synchronization score may conceal a valid clock estimate paired with a false event relation.

For v0.1, keep the first three families tiny, deterministic, replayable and independently enumerable. Do not introduce raw high-dimensional video/audio or continuous physics before clock, provenance, cost and evaluator-custody contracts are qualified.

## 5. Visibility, freeze and custody contract

**Learner-visible during acquisition:** causally available observations; modality-native data and public schema; source/frame references; observation time and arrival time; clock uncertainty; legal actions and declared costs; resource counters; candidate-local IDs; evidence-prefix identity.

**Never learner-visible before freeze:** generator family/parameters when hidden by the task; seed; latent unit/relation/mechanism labels; future frames; hidden target transitions; holdout probes; intervention outcomes; evaluator strata; calibration labels; scores; oracle policy trees.

Before any target holdout or hidden intervention outcome is released, freeze and bind:
- acquisition trace and resource ledger;
- structural-library snapshot and revision lineage;
- source and target episode IDs and evidence-prefix digests;
- selected/excluded scope, map alternatives and transformations;
- target predictions and exact claim forecasts;
- model/config/preprocessing identity and software revision.

Use the existing #11060 blind episode shell, Symtropy owner-side episode custody and the optional #7173 AgentTransferBundle sidecar. Do not create a second episode runner, truth store, sensor stack, evidence ledger or scheduler. A process boundary is not an OS sandbox; untrusted learners require a separately qualified isolation boundary.

## 6. Mandatory baseline ladder

Baselines must receive the same causal observations, action permissions, tuning opportunity and inclusive resource budget as the system being tested.

### Acquisition / representation
- random and fixed-window segmentation;
- passive gap/pattern and n-gram / frequent-motif extraction;
- predictive-error novelty, random-network-distillation novelty and surprise-only;
- learning-progress and information-gain policies;
- MDL / predictive-compression with held-out predictive-sufficiency checks;
- modality-specific encoders with explicit event association;
- frozen dense multimodal / sequence-model baseline where input contracts match;
- object/event-centric recurrent or transformer world model;
- HDC-only, CfC/LTC-only, HDC–CfC, explicit graph/hypergraph, graph-plus-symbolic/e-graph, dense and modular hybrid ablations.

### Structural transfer
- random injective maps with degree/density-matched controls;
- unique-incidence signature baseline;
- exact bounded exhaustive oracle on tiny whole-structure worlds;
- VF2/Bliss/nauty or another exact graph-isomorphism reference on compatible graphs;
- graph edit/subgraph matching where the declared map semantics permit it;
- graph embedding / GNN / graph-transformer baseline with matched inputs and training budget.

### Predictive / behavioural / mechanism transfer
- cold-start target learner;
- source-only transfer;
- target-only learning;
- source plus target adaptation;
- full-feature ERM and shift-aware representation baselines;
- model-free / direct reactive policy;
- deterministic, stochastic and model-free dynamics baselines where the environment supports them.

### Research policy
- fixed workflow and no-adaptive-branch controls;
- random legal action under matched cost;
- maximum partition / minimax / entropy / expected remaining-version-space baselines;
- structure-balanced information gain;
- Symthaea prediction-disagreement selector;
- calibrated one-step EVSI only when a justified prior/likelihood/loss model is supplied;
- exact prior-free sequential policy oracle on tiny worlds;
- frozen production Symthaea incumbent once #11171 pins a real callable.

The exact finite oracle is an evaluator diagnostic, not a feasible competitor on large worlds. Never present a one-step disagreement selector or a hypothesis-update helper as the learned researcher.

## 7. Scorecard (no composite leaderboard score)

### Structural acquisition
- unit/event boundary precision, recall and F1;
- equivalence-class recovery, false merge and false split;
- typed relation / hyperrelation precision and recall;
- motif and composition recovery on undisclosed occurrences;
- structural library size, revision churn and retained counterexamples;
- predictive log score on unseen continuations/actions;
- MDL–predictive-sufficiency Pareto frontier;
- cross-modal alignment under missingness and disagreement.

### Transfer integrity
- exact map correctness and scope coverage;
- relation-kind / transform correctness;
- node-map multiplicity and unresolved-map rate;
- false accepted isomorphism, false unification and false invariance;
- false causal/mechanism transfer (non-compensable guardrail);
- positive transfer recall and missed-transfer rate;
- set-valued correctness, forced-unique and false-unique rate;
- negative-transfer cost and cost to first independently qualified transfer.

### Decisions and behaviour
- action-conditioned prediction accuracy / log score;
- simulation vs bisimulation results separately;
- intervention-effect prediction and intervention efficiency;
- terminal loss / regret versus exact tiny-world policy oracle;
- net decision value after experiment, compute, energy and opportunity costs;
- downstream action success, task progress, failure recovery and safety constraints;
- sync vs async outcomes and end-to-end inference latency.

### Uncertainty and calibration
- Brier, log loss, fixed-bin ECE and reliability plots;
- selective risk, coverage and tie-aware AURC (always report AURC beside decision coverage);
- set coverage and average set size for admitted conformal regimes;
- false high-confidence mechanism claims by shift stratum;
- calibration drift and abstention under out-of-support inputs.

### Resource accounting
Report observations, interventions, wall time, CPU/GPU time, FLOPs if measured, memory, model calls/tokens, energy if measured, bytes stored, human review and actual experiment cost. Report acquisition and inference cost separately. Proxy costs must be labelled as proxies.

## 8. Statistical and acceptance gates

- Use at least five seeds for stochastic development comparisons; this is a floor, not a guarantee of adequate power. Precompute confirmatory sample size from the primary paired estimand, expected effect and variance.
- Pair candidate and baseline on identical world/seed, initial evidence and inclusive budgets. Preserve invalid attempts, timeouts, refusals, no-action and incomplete traces.
- For exhaustive finite authored suites, report the complete census and exact denominators; do not attach sampling p-values to a complete finite population.
- For sampled confirmatory suites, report paired effect sizes, confidence intervals, per-family denominators, multiplicity plan and predeclared exclusions.
- Begin with at least 100 independently generated pairs per core family for development; expand confirmatory negative-transfer evaluation to at least 300 independent negative cases before making a 1% false-accept upper-bound claim. Zero errors among 300 independent cases gives only an approximate 95% upper bound near 1% under an IID/binomial interpretation; dependence or shared generator blocks requires cluster-aware analysis and a larger sample.
- A false confident causal/mechanism transfer is a non-compensable failure. High average structural accuracy cannot offset it.
- Promote a performance claim only when the paired lower confidence bound beats the strongest compatible baseline on the preregistered primary metric, with no integrity, false-confidence, resource or negative-transfer regression.
- A claim of field-leading performance requires matched native-protocol external replication, not an unrelated benchmark percentage.

## 9. Proposed stage gates

**Gate 0 — contract integrity:** exact replay; hidden-label separation; freeze-before-holdout; schema and digest validation; independent known-answer oracles; exact-head formatting/tests/Clippy.

**Gate 1 — task-agnostic acquisition:** on at least three hidden structural families, frozen libraries improve held-out structural/relation/motif probes over simple passive, MDL, dense and graph baselines at matched cost. Report failure families, not only aggregate means.

**Gate 2 — qualified transfer:** positive transfer on declared transformations and rejection/abstention on matched-marginal, intervention-divergent and non-identifiable controls. Independent oracle agreement and counterexamples are required.

**Gate 3 — decision value:** transferred knowledge reduces terminal loss or improves downstream utility over cold-start, no-transfer and direct-policy controls under equal budgets. Tiny worlds are compared with the exact sequential oracle; larger worlds use native task metrics.

**Gate 4 — embodied / scientific external validation:** evaluate adapters to Autonomous Science Bench, Unsatz, RoboDyna, V-JEPA 2 / world-model tasks and scientific reproducibility suites under their native protocols. Hardware-in-the-loop or physical evidence is required for deployment claims.

## 10. Current industry comparators and correct interpretation

| Comparator | Published result / scale | Correct use for ErgoStream |
|---|---|---|
| Unsatz (2026-10-31) | 80 systems / 20 domains; nine LLMs report 21–60% solve rates | Closest structure/model-form discovery comparator. Reproduce its native tasks; ErgoStream adds pre-task acquisition and a frozen reusable library. |
| Autonomous Science Bench 0.1 | Six pilot tasks, three trials per task; current board lists GPT-6 Astra + Codex at 72.2%, Opus 5.5 + Claude Code at 66.7% | Full model/agent/tool/harness laboratory-loop comparison. Small gaps are not decisive at this pilot sample. |
| RoboDyna | 20 conceptual and 10 household tasks; 3,500 demonstrations; 5.4K evaluated episodes; all reported policies below 20% success | Async dynamic-world and latency-aware physical transfer adapter; sync/async parity is not optional. |
| V-JEPA 2 | Pretrained on over one million hours of video; reports 77.3 top-1 on Something-Something v2 and 39.7 R@5 on EPIC-KITCHENS-100; zero-shot image-goal robot pick/place after under 62 hours of DROID post-training | Strong representation / physical-planning comparator, not a matched task-agnostic structural-library score. |
| Dreamer 3 | Evaluated across 8 domains and over 150 tasks with a fixed configuration | Strong broad model-based control comparator; test whether pre-task structural acquisition adds transfer value under matched task/hardware budgets. |
| Dreamer 4 | Separate scalable-world-model work reports an offline Minecraft diamond task requiring over 20,000 action steps, learned from diverse offline video data | Strong long-horizon world-model / imagination comparator; do not conflate its Minecraft result with Dreamer 3's 150-task evaluation. |
| AgentIdeaBench | 33 LLMs, 40 subfields, 5 disciplines; active exploration improved grounding/feasibility/clarity while measured originality did not improve | Distinguish better-grounded hypotheses from novel discovery; score independently reviewed originality separately. |
| AutoSciBench | Adaptive benchmark recipes reduced average solver accuracy by 22.4 pp (computational biology) and 25.5 pp (materials) versus curated benchmarks | Motivates a sealed refresh lane; do not let adaptive task generation contaminate immutable longitudinal anchors. |
| Continual Learning Bench (CL-Bench) | Six expert-validated domains with shared latent structure; its study finds agents often overfit immediate observations and fail to reuse knowledge across instances, and dedicated memory systems do not automatically solve this | External real-world stateful-learning adapter. Compare pre-task library value and gain over prior capability; do not pool its score with ERGOstream's synthetic structural estimands. |
| Continual Learning Exam (CLE) | 120 expert-authored workflows, 600 levels, 89 roles and 12 domains; 17 harness/model configurations; best reported configuration passes 47.7% of levels | External long-horizon professional-workflow adapter for staged experience/feedback. Its rubric-based task ladder is complementary, not a structural-isomorphism measure. |
| Continual-ARC | Recurring ARC tasks return after dormant gaps with fresh instances; learner pays for feedback, and memory-/weight-based systems are compared under a common budget | Strong memory-versus-weight adaptation comparator. Include recurrence after dormancy and fresh-instance generalization; don't mistake demonstration caching for reusable structure. |
| Compositional continual world models (2026) | Robot-manipulation curricula factor reuse across action, perception and full composition; modular dynamics trade off reuse and forgetting, with no complete solution reported | Physical-AI adapter for compositional transfer after continual acquisition; use matched simulator/task budgets and add explicit structural-library treatment arms. |
| Memora / FAMA | Multi-session memory benchmark with explicit consolidation and mutation; reports frequent reuse of obsolete/invalidated memories | Lifecycle/retraction guardrail for structural entries. Adapt its obsolete-memory principle, but keep memory QA results distinct from structural transfer. |
| AV-SyncBench (2026-07-01) | 3,269 in-the-wild clips / 38,390 samples; five models; temporal offset/jitter/speed separated from semantic source/timbre perturbations under fixed timing. Official repository reports pairwise accuracy 0.583 Synchformer/VGGSound and 0.569 SparseSync/VGGSound-Sparse overall; ties incorrect. | Direct multimodal baseline. Reproduce its exact 50/100/200/300/500 ms tiers, pairwise score, model transforms and preprocessing; do not compare those accuracies directly with ERGOstream's synthetic F1 or hidden event-identity estimands. Symtropy adapter: #7516. |
| MATH-SYNC (2026-10-13) | Joint inverse-affine clock / latent-event-time system with exact rank identifiability and Huber IRLS; Joint Huber vs OLS macro-RMSE reductions: Harvard 10 participants 73.7% nominal / 62.4% stress; WESAD 15 participants 82.1% / 67.5%; near-parity on clean timing. | Clock-registration comparator only after reproducing its joint formulation, gauge, correspondences, fixed-scale Huber and cohort protocol. It assumes event correspondences are correctly identified and does not solve semantic/event identity ambiguity. Our paired-anchor OLS/Huber/Theil–Sen is a simpler baseline, not MATH-SYNC. |
| DS-CP / TCC / KMM-CP | 2026 work on domain-shift-aware or transported conformal calibration; TCC uses paired unlabeled source/target data, KMM-CP is selective under support overlap | Required calibration-transfer baselines. Do not transport calibration across invalid structural/mechanism maps or claim arbitrary-shift coverage. |

These figures are reported by their respective papers or benchmark maintainers, not reproduced by Symthaea. Model/agent/harness configurations, trial counts, data budgets, hardware and scoring protocols differ. They cannot be ranked against ErgoStream without a matched adapter and protocol.

## 11. Research hypotheses to falsify

- **H1 — pre-task utility:** a library acquired before downstream task disclosure improves later unseen probes over task-specific encoders and raw-stream baselines at the same budget.
- **H2 — compositional reuse:** multiscale motifs and relations improve unseen compositions, not merely repeated instances.
- **H3 — cross-modal complementarity:** aligned modalities resolve structures that unimodal controls cannot; shuffled alignment must remove the gain.
- **H4 — selective transfer:** explicit scope, invariance/equivariance and counterexample search reduce false unification and negative transfer relative to embedding similarity alone.
- **H5 — decision relevance:** knowledge transfer improves terminal decisions, not only predictive fit or information gain.
- **H6 — hybrid representation value:** HDC, CfC/LTC, explicit graph and symbolic/e-graph components add measurable value in ablations; any component that does not earn its cost is removed.
- **H7 — calibrated ignorance:** the learner detects non-identifiability and distributional support failure, preserves alternatives and abstains or requests a discriminating probe.

Any hypothesis may fail. A transparent negative result is a successful benchmark outcome.

## 12. Library intervention and causal-attribution controls

The primary transfer claim is not merely that a learner with a library performs better. It is that the *structural content actually acquired before task disclosure* causes measurable downstream improvement. Cold-start/no-transfer controls are necessary but do not fully rule out extra memory capacity, prompt/context, parameter count or retrieval compute as the source of a gain.

Run a preregistered, paired library-intervention tournament. Freeze the learner, acquisition trace, source observations, library serialization, target tasks and total resource budgets before task disclosure. Randomize the library treatment within matched target-world/seed blocks; use fresh process state for each arm so one treatment cannot contaminate another.

### Treatment arms

1. **Intact acquired library:** the exact pre-task frozen entries, alternatives, transformations, counterexamples and provenance produced by the acquisition policy.
2. **Empty / masked library:** same interface, number of slots and retrieval path, but no usable structural information.
3. **Capacity-matched randomized library:** preserve entry count, type frequencies, relation arity, approximate degree/frequency profiles and serialization size, but randomize the actual structural incidence.
4. **Structure-destroyed ablation:** preserve units, relation count, local marginals and relation-kind frequencies while permuting endpoints so higher-order motifs and cross-context relations are broken.
5. **Context-shuffled library:** preserve individually valid entries and their marginal frequency, but permute which source context / evidence prefix each entry is bound to.
6. **Retrieval-disabled arm:** retain the acquired library and identical model parameters, but prevent target-time retrieval. This checks that the proposed transfer path actually consumes the library.
7. **Cold-start target learner:** no source-acquired content; same target data and total downstream budget.
8. **Oracle-library ceiling:** evaluator-generated correct structure, never exposed to the candidate. This estimates headroom and verifies the target task is solvable.
9. **Post-disclosure library sentinel:** a library built with target-task information after disclosure. This is a leakage/ceiling control, not a task-agnostic treatment and never counts as a win.

Arms 1–6 must hold constant model weights, parameter count, retrieval calls, context/token budget, memory allocation, target observations, allowed actions, wall-clock/compute budget and tuning opportunity. For arms that remove or randomize content, pad to the same declared capacity and serialization length. Do not give the intact arm a larger effective context or more inference work.

### Primary estimands

- **Library content effect:** paired difference in target terminal loss / utility between the intact library and the strongest capacity- and marginal-matched content-destroying control.
- **Transfer sample efficiency:** target observations or interventions needed to reach a preregistered quality threshold, with all acquisition and adaptation costs reported separately and jointly.
- **Transfer frontier:** target quality versus source acquisition observations, target observations, compute, memory, latency and energy; compare Pareto frontiers rather than one unqualified efficiency ratio.
- **Retrieval-mediated effect:** paired difference between intact and retrieval-disabled arms, with exact retrieved entry IDs/revisions logged.
- **Negative-transfer cost:** excess target loss, action regret and high-confidence false-transfer rate when the source library is misleading or out of support.

### Integrity and analysis

- Bind every retrieved entry to the frozen source evidence prefix and exact library revision; log retrieval, rejection, counterexample and supersession events.
- The candidate never sees the intervention assignment, evaluator task family, hidden labels, randomization seed or expected arm outcome.
- Preserve a fixed primary comparison and multiplicity plan. Use paired analysis at the generator/world seed level and cluster-aware intervals when recipes share latent structure; do not treat frames or events from one world as independent trials.
- Report each family and shift separately, including failed retrieval, no retrieval, abstention, timeout and invalid action denominators.
- The randomized capacity-matched and structure-destroyed arms are mandatory before attributing a gain to structural acquisition. If the intact library does not beat the strongest compatible controls, report that reusable content has not been demonstrated even if a larger memory or model performs well.
- The oracle-library arm is evaluator-only. The post-disclosure sentinel is not a valid task-agnostic treatment and must never be included in the primary leaderboard.

### Gate 1A — attributable pre-task library value

Advance only when the intact frozen library improves a preregistered downstream estimand over the strongest capacity/marginal-matched control on sealed target worlds, with a paired lower confidence bound above zero (or above a predeclared practically meaningful margin), while preserving the false-unification, false-causal-transfer, calibration and resource guardrails. Otherwise retain the result as a negative or inconclusive finding and revise the acquisition hypothesis rather than relaxing the controls.

## 13. Continual structural-library lifecycle

This track composes with Symthaea #55 (continual-compositional evaluation), #7029 (matched HOLD/UPDATE adaptation and retention), and #7030 (action-dependent observation and evaluator-state integrity). It does **not** create a second continual-learning runner, change-policy owner, evaluator or memory authority. Use their shared evaluation spine and attach ERGOstream-specific structural-library snapshots and transfer probes.

### Why this is a separate estimand

A frozen-library transfer test asks whether acquired structure transfers once. A continual-library test asks whether the knowledge remains useful, revisable and safe as tasks and mechanisms evolve. These are not the same capability.

Use controlled curricula with hidden boundaries and no task-family labels:

1. **A → B:** acquire source structure in A, then encounter a related world B with both shared and changed factors.
2. **A → B → A:** return to an A-like regime after a dormant gap with fresh instances; measure recognition, relearning cost and whether the system has needlessly overwritten reusable A knowledge.
3. **A → B → C-composition:** combine previously acquired primitives in a new relation/action/perception composition, not merely a fresh instance of a seen task.
4. **Within-family mechanism mutation:** preserve superficial observations or selected relations while changing one hidden action-conditioned dependency; old entries should become limited, suspended or rejected for that claim scope.
5. **Recurring and unsignaled drift:** use abrupt, gradual, cyclic and stochastic-hazard changes without revealing the change point to the learner.

### Paired counterfactual update design

At each preregistered update point, fork from the exact same parent learner/library state and evidence prefix:

- **HOLD:** continue without incorporating the newly available candidate evidence.
- **UPDATE:** apply the frozen update policy to the same permitted evidence.
- **No-op update:** execute the update pathway without decision-relevant information.
- **Irrelevant-trigger update:** vary irrelevant metadata while keeping evidence unchanged.
- **Counterexample update:** provide a valid, decision-relevant contradiction that should cause scope narrowing, alternative retention, retraction or supersession.
- **Rate-matched random update:** match update frequency and resource consumption without using decision-relevant evidence.

Reuse #7029's candidate-selection/evaluation separation, paired branch ledger and update/retention estimands. Reuse #7030's frozen evaluator state and action-dependent-observation controls. Do not expose evaluator labels, future task identities, change-point locations or HOLD/UPDATE assignment.

### Structural-library lifecycle semantics

Preserve #13401's existing lifecycle (`candidate`, `replayed`, `holdout_supported`, `qualified_in_scope`, `retracted`, `superseded`). Add evaluator dispositions for the particular transfer claim—`still_applicable`, `scope_limited`, `challenge_pending`, `suspended`, `rejected`—rather than allowing the learner to self-promote or silently overwrite prior knowledge. A previously qualified entry is not automatically valid for a new target, time, sensor or intervention scope.

Every update must retain parent snapshot identity, evidence-prefix identity, changed entries, derived descendants, counterexamples and the reason for each state transition. At use time, resolve current applicability and invalidation through existing evidence/lifecycle owners; do not treat a stored vector or digest as proof that the relationship remains true.

### Metrics

- **Forward transfer:** target adaptation gain attributable to prior library content on new tasks/compositions.
- **Backward transfer and forgetting:** changes in previously qualified family performance after later learning, reported by family and horizon.
- **Negative transfer:** loss/regret increase caused by retrieved but inapplicable prior structure.
- **Stale-use rate:** rate at which superseded, retracted, out-of-scope or contradicted entries influence a candidate prediction/action.
- **Correction latency:** observations, actions, compute and wall time from a valid counterexample becoming available to correct scope/revision disposition.
- **Counterexample propagation completeness:** proportion of derived entries that are correctly narrowed or invalidated after a necessary parent is retracted.
- **Recurrence cost:** observations/interventions required to regain prior A performance after A → B → A, compared with cold-start and HOLD controls.
- **Library churn and bloat:** bytes, entry revisions, duplicate/superseded records, retrieval candidates and index/update work over the whole stream.
- **All-in transfer efficiency:** downstream quality versus acquisition, update, replay, memory, retrieval, verification, latency and energy cost.

Do not optimize forgetting to zero. Under genuine mechanism change, refusing to reuse obsolete knowledge is correct; under recurrence, relearning a previously valid structure may be wasteful. Report adaptation and retention as competing outcomes, as #7029 requires.

### Decision-sufficient structure: learn what to retain, retrieve and forget

Compose with MACE #13284 (decision-sufficiency frontier and active evidence acquisition) and #13319 (adversarial memory lifecycle, poisoning, repair and erasure). ERGOstream must not create a second evidence graph, lifecycle authority, memory manager or decision policy.

The target is not the smallest library in bytes, nor maximum reconstruction of the acquisition stream. It is the least costly *admissible structural representation* that preserves a fixed downstream decision policy's performance within a preregistered regret margin, while satisfying provenance, scope, authority, currentness and independence constraints.

For a frozen decision need and policy, compare the full admitted library with budgeted retained subsets and abstractions. Report:

- **Decision-sufficiency rate:** fraction of target decisions for which the retained structure keeps paired regret gap within the preregistered epsilon.
- **Minimum sufficient cost:** bytes, retrieval/index work, verification work, inference compute, latency and energy required to meet the decision-sufficiency constraint.
- **Absolute and tail regret:** paired mean regret, worst-family regret, violation probability and CVaR; an average must not hide high-consequence failure.
- **Compression damage:** which relation, scope, counterexample, alternative or provenance dependency was removed and which decision changed as a result.
- **Acquisition/retrieval trade-off:** whether acquiring a new observation is cheaper than retaining/retrieving a larger ambiguous library; compare to MACE's EVSI/query baselines.
- **Hard-constraint violations:** any subset that loses required lineage, currentness, scope, authority or independence is inadmissible even if task accuracy remains unchanged.

Compare full library, recency/frequency truncation, top-k similarity retrieval, MDL/predictive-compression, graph/hypergraph simplification, counterexample-preserving pruning, decision-aware selection and an evaluator-only exact-scope ceiling. Randomized and structure-destroyed subsets remain attribution controls. Keep selector tuning on development/calibration worlds; use sealed decision outcomes only for final scoring.

Adversarially test a poisoned or superseded parent, a derived motif that still appears semantically similar, a stale backup/replay path, and a retracted relation that remains in a cached index. Reuse #13319's lifecycle and counterfactual influence contract. A compact representation does not gain truth or authority from compression; derived entries must remain traceable to their source evidence and must be invalidated when a necessary parent is invalidated.

This yields a new measurable trade-off: **decision quality × structural transfer × evidence integrity per unit of retained and verified knowledge**. A library that is smaller but forgets a critical counterexample is not efficient; a library that retains everything but cannot identify stale or irrelevant relations is not robust.

### Baselines and external adapters

At matched stream, feedback, update and resource budgets, compare:

- no persistent state; passive episodic cache; nearest-neighbour retrieval; standard retrieval-augmented memory; exact task/example lookup;
- frozen acquired structural library; library-only update; model-parameter update with replay; modular/factorized model; hybrid library-plus-model update;
- HOLD, no-op, random-trigger and counterexample-trigger branches;
- cold-start target learning and evaluator-only oracle-library ceiling.

Use CL-Bench for expert-validated cross-instance latent-structure reuse; CLE for staged professional workflows and diagnostic revision; Continual-ARC for recurring tasks after dormant gaps and memory-versus-weight comparison; and the 2026 compositional continual world-model benchmark for action/perception/full-composition reuse in robot manipulation. Use Memora/FAMA as an obsolete-memory stressor. Preserve every external benchmark's native tasks, splits, evaluator and reported metrics; an adapter is an additional lane, not permission to merge scores.

### Gate 1B — continual value without harmful persistence

Advance only when the UPDATE branch beats its paired HOLD and strongest matched memory/model control on a preregistered transfer or decision estimand, while (a) stale-use and false-transfer rates remain within the predeclared guardrails, (b) qualified prior capabilities do not regress beyond their declared margins, (c) counterexamples propagate to derived entries, and (d) total resources are charged. Include recurrence and drift families separately; a high average score cannot hide either needless forgetting or harmful retention.

## 14. Incremental value of explicit structural witnesses

### Falsifiable question

Do explicit typed structural correspondences, scope declarations, transformation classes and ambiguity profiles predict target transfer validity or decision improvement **beyond** simpler representation-overlap, task-complexity and graph-domain diagnostics?

Structural witnesses must earn their extra complexity. Correctly validating a supplied finite mapping is not, by itself, evidence that a witness improves transfer prediction, target decisions or negative-transfer detection. This lane therefore distinguishes (a) correctness of the proposed mapping, (b) prediction of whether transfer will succeed, and (c) realized decision value.

### External theory and adversarial comparators

- **Feature Overlap Rate (FOR) and Effective Task Complexity (ETC):** Gao (AAAI 2026) proposes a transfer-risk framework based on target representation overlap with the source-induced subspace and entropy of latent subtasks. On six GLUE transfer pairs, the paper reports that FOR and ETC together explain over 80% of transfer-risk variance, while the tested classical divergences do not. This is a direct diagnostic baseline, not a result on ERGOstream's graph or multimodal worlds. The Symtropy evaluator-side child for this test is [#7497 — ERGO-STREAM-WITNESS-INCREMENTAL-001](https://github.com/Luminous-Dynamics/symtropy/issues/7497). Reproduce the operational definitions and report estimator uncertainty rather than copying the headline figure. https://doi.org/10.1609/aaai.v40i25.39270
- **Graphon-based domain discrepancy and positional-encoding stability:** Zhu et al. (ICML 2026) decompose graph-foundation-model output shift into finite-sample approximation and a relabeling-invariant structural domain discrepancy under their graphon / dense-graph assumptions. Use the theory only on compatible graph regimes; do not extend dense graphon assumptions silently to sparse, directed, temporal or hypergraph worlds. https://proceedings.mlr.press/v306/zhu26e.html
- **ReCoG relative contexts:** Tong et al. (2026-11-01) report that removing a large proportion of source-graph edges improved cross-domain transfer for four graph-pretraining methods; their relative-node-context method reports gains on 18 target datasets. This is an important adversarial baseline: raw source topology can be a transfer liability. Compare full-graph structural learning, edge-thinned source learning and relative-context construction under matched parameters, data and compute. The reported gains are paper results, not ERGOstream results. https://arxiv.org/abs/2611.01753

### Nested diagnostic tournament

Freeze the acquisition representation, source library and target predictions before hidden target outcomes. Fit diagnostic models only on development/calibration world families; evaluate on held-out generator families and sealed refreshes. Do not feed evaluator truth, hidden mechanism labels, oracle correspondence results or post-freeze holdout metrics into any candidate feature.

Use the following nested models to predict independently scored transfer qualification and, in a separate analysis, downstream decision utility:

1. **M0 — task/source floor:** source validation error, declared sample count, model scale/parameter count, target support size, acquisition budget and nuisance metadata available to the candidate.
2. **M1 — generic transfer diagnostics:** M0 plus FOR, ETC, MMD/Wasserstein or other predeclared feature-distribution discrepancies, and calibration/support-overlap diagnostics where assumptions apply.
3. **M2 — graph-domain diagnostics:** M1 plus graphon discrepancy / finite-sample term and positional-encoding stability diagnostics only for graph families satisfying their assumptions. For sparse, directed, temporal and hypergraph families, use separately specified compatible statistics or mark the diagnostic unavailable.
4. **M3 — explicit witness diagnostics:** M2 plus candidate-produced witness features: declared map kind and scope, mapped/excluded fraction, relation-kind and transform commitments, node-map ambiguity, set-valued alternatives, unresolved/abstain state, counterexample count and evidence-prefix age. These are candidate-visible proposals and metadata, not evaluator-side truth.
5. **M4 — intervention / counterexample diagnostics:** M3 plus candidate-proposed discriminating probes and prospective outcomes, frozen before those observations arrive. Score only on the subsequent independently observed outcomes.

Run the comparison per transfer family and claim level (structural, predictive, behavioural/dynamics, mechanism) rather than pooling them into one label. Use paired world/seed blocks and cross-fitting; preserve generator-family holdouts to prevent leakage from near-duplicate recipes.

### Required model and representation arms

- Dense / modality-native representation with no explicit structural witness.
- Feature-overlap and ETC diagnostic baseline with a simple linear/regularized predictor.
- MMD/Wasserstein baseline and, where valid, graphon-domain discrepancy.
- Full raw-graph / full relation-structure encoder.
- ReCoG-style relative-context encoder and source-edge-thinned pretraining arm.
- Explicit witness-only diagnostic model.
- Hybrid representation plus explicit witness model.
- Randomized / structure-destroyed witness control, matched for feature count, serialized bytes and compute.

All arms receive the same public acquisition stream and target-time evidence. Normalize or report model parameters, persistent state, source and target observations, target labels, tuning budget, compute, memory, latency and energy. A larger or more expensive hybrid does not win solely because it has more capacity.

### Metrics and incremental-value gate

Primary prediction metrics: held-out Brier/log loss, AUROC and AUPRC for transfer failure, calibration slope/intercept, reliability by shift family, selective risk/coverage and false high-confidence transfer. Report structural-witness acceptance correctness separately from predicting downstream transfer success.

Primary decision metrics: paired target terminal loss/regret, negative-transfer cost, decision-sufficiency rate and cost to first qualified transfer. Include per-family tail risk and CVaR where decision costs are asymmetric.

Report M3−M2 and M4−M3 incremental effects on sealed families, with paired cluster-aware confidence intervals, preregistered multiplicity correction and practical effect margins. Also report inference/verification cost added by witness construction and checking.

**Gate 2A — witness incremental value:** explicit witness diagnostics advance only if they improve a preregistered held-out estimand over M2 on unseen generator families, with a paired lower confidence bound beyond the practical margin, improved or non-inferior calibration, and no false-transfer, negative-transfer or resource regression. An improvement confined to graph families where the baseline assumptions are mismatched does not qualify.

**Gate 2B — representation choice:** if ReCoG-style edge thinning or relative contexts outperform the full raw-graph arm, preserve that result. Do not force explicit graphs into the candidate architecture. Retain a structural-witness layer only where it demonstrably contributes independent transfer prediction, diagnostic value, decision utility or auditable scope/abstention—even if its main advantage is epistemic reliability rather than average task accuracy.

**Negative-result rule:** if FOR/ETC and compatible graph-domain diagnostics explain transfer outcomes as well as M3, report that the explicit witness has not demonstrated incremental predictive value. It may still be useful as an audit contract, but that is a separate claim from better transfer generalization.

## 15. Active calibration acquisition under identifiability and cost

### Falsifiable question

Can a policy choose which known-correspondence event/stream observation to request next so that the gauge-fixed joint clock design becomes identifiable and well-conditioned with fewer observations, less waiting and lower acquisition cost than passive, random or uniformly spaced schedules?

This extends MATH-SYNC's static anchor-count / temporal-span sensitivity analysis into a sequential decision problem. The benchmark must not confuse event correspondence with clock calibration: the acquisition policy may use only already-qualified event identities or explicit calibration pulses. A semantic event whose identity is uncertain cannot be used as clock ground truth by feeding the same unverified identity into the design matrix.

Symtropy execution owner: [#7529 — ERGO-STREAM-ACTIVE-ANCHOR-001](https://github.com/Luminous-Dynamics/symtropy/issues/7529). It composes with #7493 (clock/event factorial), #7498 (paired-anchor estimators), #7520 (joint inverse-affine estimator), #7518 (MATH-SYNC reproduction) and #7516 (AV-SyncBench external adapter).

### Factorial acquisition tournament

Policies see the current public sparse design, currently requestable known-correspondence observations, expected acquisition costs and availability metadata. They do not see future timestamps, hidden clock parameters, evaluator rank, generator seed or held-out event times. Every accepted, rejected, unavailable, late, duplicate and over-budget request remains in the action-attempt denominator.

Compare:

- passive fixed schedule and no-query controls;
- first-two-available anchors, earliest-available and uniform random selection;
- uniformly spaced temporal coverage;
- rank-greedy acquisition (maximize rank increase first);
- E-optimal greedy acquisition (maximize smallest singular value once full rank can be reached);
- D-optimal greedy acquisition (maximize information-volume / log determinant);
- cost- and latency-aware E/D-optimal acquisition, retaining a hard rank-first constraint;
- an evaluator-only oracle schedule upper bound, never exposed to candidate agents.

Use a lexicographic objective: **(1)** reach full column rank; **(2)** improve scaled conditioning; **(3)** reduce held-out latent event-time / clock parameter error; **(4)** minimize query cost, wait and energy. A regularized or high-scoring but rank-deficient fit is not an identifiable solution. Distinguish numerical pivot-ratio proxies from SVD smallest singular value and condition estimates.

### Required conditions and metrics

Cross anchor count (1/2/4/6/8/12/16/24), temporal span and clustering with missing reference events, heterogeneous native rates, out-of-order arrivals, Gaussian jitter, heavy tails, gross outliers, burst dropout, reference corruption, stream reset, piecewise drift and an explicitly non-affine negative control. Keep known-correspondence clock trials separate from shuffled / ambiguous semantic event-association arms.

Measure queries to first full rank, proportion reaching full rank under budget, rank trajectory, scaled singular-value/condition diagnostics, held-out latent event-time RMSE/MAE, inverse-clock rate/offset error, prediction interval coverage, false-identifiable / false-unique / unnecessary-abstention, rejected/late attempts, total wait, query cost and energy. Downstream decision regret remains a separate secondary estimand.

### Acceptance gate

Reproduce the passive MATH-SYNC anchor-count/span sensitivity before claiming active improvement. Then compare policies on paired generator-family blocks and held-out schedules. An active policy advances only if it improves cost or held-out reconstruction over *all* non-oracle passive baselines at matched noise, latency, sample and compute budgets, with paired confidence intervals, no increase in false-identifiability, and explicit abstention when the budget cannot close the rank gap.

Add a corrupted-reference / non-affine world where more observations can be misleading. The policy must detect model invalidity or abstain; it must not overfit a bad reference with more anchors. E/D-optimal gains must be checked against an independent SVD/eigenvalue implementation; the current QR pivot ratio is not sufficient for that gate.

### Research interpretation

This is a proposed ERGOstream research frontier, not an existing result or proven SOTA. MATH-SYNC already demonstrates that anchor number and temporal spread affect rank and reconstruction; #7529 asks whether sequential acquisition can select the best information-bearing observation under real costs. The policy's contribution must be measured against strong passive schedules and an oracle upper bound, not asserted from a synthetic demo.

## 16. References (snapshot 2026-11-25)

- Continual Learning Bench (CL-Bench, 2026-06-04): https://arxiv.org/abs/2606.05661
- Continual Learning Exam (CLE, 2026-10-25): https://arxiv.org/abs/2610.33956
- Continual-ARC (2026-11-10; v3 updated 2026-11-15): https://arxiv.org/abs/2611.14505
- Compositional continual world models (2026-09-18): https://arxiv.org/abs/2609.22055
- Memora / Forgetting-Aware Memory Accuracy: https://aclanthology.org/2026.findings-acl.1337/
- Causal-Discovery-Bench (2026-10-13): https://arxiv.org/abs/2611.09934
- Unsatz: https://arxiv.org/abs/2611.03713
- Autonomous Science Bench: https://asb.open-ssi.org/about and https://asb.open-ssi.org/leaderboard
- RoboDyna: https://arxiv.org/abs/2611.26969 and https://robodyna.github.io/robodyna/
- V-JEPA 2: https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/
- Dreamer 3 (Nature 2025): https://doi.org/10.1038/s41586-025-08744-2
- Dreamer 4 (world-model preprint): https://arxiv.org/abs/2509.24527
- AgentIdeaBench: https://arxiv.org/abs/2609.07611
- AutoSciBench: https://arxiv.org/abs/2610.05140
- AV-SyncBench (2026): https://arxiv.org/abs/2607.00726
- AV-SyncBench official implementation / reference pairwise scores: https://github.com/Duck-Boss/AV-SyncBench
- ERGO-STREAM-AVSYNC-ADAPTER-001 (Symtropy #7516): https://github.com/Luminous-Dynamics/symtropy/issues/7516
- MATH-SYNC (2026): https://doi.org/10.3390/math14203720
- DS-CP (ICML 2026): https://proceedings.mlr.press/v306/lin26t.html
- Transported Conformal Calibration (ICML 2026): https://proceedings.mlr.press/v306/doula26a.html
- KMM-CP (UAI 2026): https://proceedings.mlr.press/v337/laghuvarapu26a.html
- Unified Structural Factors / Feature Overlap Rate and Effective Task Complexity (AAAI 2026): https://doi.org/10.1609/aaai.v40i25.39270
- When Do Graph Foundation Models Transfer? A Data-Centric Theory (ICML 2026): https://proceedings.mlr.press/v306/zhu26e.html
- ReCoG: Relative Node Contexts for Multi-Domain Graph Pre-training (2026-11-01): https://arxiv.org/abs/2611.01753
- ERGO-STREAM-WITNESS-INCREMENTAL-001 (Symtropy #7497): https://github.com/Luminous-Dynamics/symtropy/issues/7497
