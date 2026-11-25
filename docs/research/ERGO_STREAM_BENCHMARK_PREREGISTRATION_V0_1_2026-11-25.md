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

## 12. References (snapshot 2026-11-25)

- Unsatz: https://arxiv.org/abs/2611.03713
- Autonomous Science Bench: https://asb.open-ssi.org/about and https://asb.open-ssi.org/leaderboard
- RoboDyna: https://arxiv.org/abs/2611.26969 and https://robodyna.github.io/robodyna/
- V-JEPA 2: https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/
- Dreamer 3 (Nature 2025): https://doi.org/10.1038/s41586-025-08744-2
- Dreamer 4 (world-model preprint): https://arxiv.org/abs/2509.24527
- AgentIdeaBench: https://arxiv.org/abs/2609.07611
- AutoSciBench: https://arxiv.org/abs/2610.05140
- DS-CP (ICML 2026): https://proceedings.mlr.press/v306/lin26t.html
- Transported Conformal Calibration (ICML 2026): https://proceedings.mlr.press/v306/doula26a.html
- KMM-CP (UAI 2026): https://proceedings.mlr.press/v337/laghuvarapu26a.html
