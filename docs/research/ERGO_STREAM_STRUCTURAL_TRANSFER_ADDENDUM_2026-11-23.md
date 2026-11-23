# ErgoStream: structural-transfer research addendum

**Date:** 2026-11-23  
**Status:** research direction and benchmark protocol; not an achieved learner result  
**Tracking:** Symthaea #13401 · Symtropy #7149 · Symtropy transfer follow-up #7171

## Executive recommendation

Extend task-agnostic structural acquisition into **falsifiable structural transfer**.

The core question becomes:

> Can a learner discover a reusable structure in one unfamiliar domain, propose an explicit correspondence to a second domain, identify which relations are invariant or transform equivariantly, and reject an attractive but false analogy when a counterexample or intervention contradicts it?

This is not a new world model or a replacement for the existing discovery, graph, physics, sensor, or evidence systems. It is a composition of the current Symthaea learner work with the Symtropy blind-world evaluator.

## 1. Paradigm shifts

### 1.1 From analogy by similarity to proof-carrying correspondence

Embedding proximity is useful for proposing candidate analogies, but it is not a sufficient transfer criterion. A candidate transfer should include:

- source and target scope;
- node/entity correspondence;
- relation-kind correspondence;
- explicit transformation per relation family;
- assumptions and uncertainty;
- predicted intervention effects;
- counterexamples and known non-transfer boundaries;
- evidence lineage and qualification state.

The mapping must say what is preserved, what changes in a known way, what is absent, and what remains unidentified. A relation should not be forced into the closest target relation simply because the representations are similar.

### 1.2 From static invariance to equivariance and counterfactual commutation

For a source transformation T_s and target transformation T_t, a candidate correspondence Φ should predict whether the following relationship holds:

    Φ(T_s(x)) = T_t(Φ(x))

The allowed relation transform must be declared before hidden test outcomes are revealed. Some transformations preserve order; others reverse it, rescale it, or change the coordinate frame. Some should invalidate transfer entirely.

For causal or functional claims, test interventions rather than correlation alone. A transfer map that fits observation but fails an intervention is a counterexample, not a partial success.

### 1.3 From one abstraction level to multiscale structural acquisition

Test whether the learner can discover reusable structure at multiple scales:

- observations / raw events;
- temporal or spatial units;
- relations and interaction motifs;
- mechanisms and reusable compositions;
- higher-level functional organization.

The learner should propose abstraction/coarse-graining maps and be scored on whether relations remain valid under the change of scale. This is a research hypothesis, not an assumption that physical renormalization theory directly applies to cognitive representations.

### 1.4 From novelty seeking to value of discrimination

Acquisition should prioritize observations that separate competing explanations and improve future transfer, not simply maximize surprise.

Compare passive exposure, random acquisition, prediction error, RND-style novelty, learning progress, information gain, cost-normalized information gain, predictive compression/MDL, and a hybrid. Keep every component visible; do not allow a scalar curiosity reward to hide false structure or excessive cost.

### 1.5 From single hypotheses to explicit non-identifiability

When multiple structural maps explain all available observations, the learner should return a set of candidates, a partial mapping, or an unresolved result. It should request a discriminating observation only when its expected value exceeds its observation/intervention cost.

A forced unique answer to an observationally non-identifiable problem is a benchmark failure.

### 1.6 From positive-only memory to scoped knowledge plus negative boundaries

Structural memory should retain:
- where a relation was supported;
- which transformations it survived;
- which interventions it predicted;
- which domains or scales were tested;
- where it failed;
- whether an analogy was rejected;
- which assumptions changed after revision.

Counterexamples and non-transfer results are reusable knowledge, not discarded errors.

## 2. Architecture composition

Keep the current division of responsibility:

- **Symthaea:** proposes units, relations, structural libraries, candidate correspondences, transformation hypotheses, and discriminating experiments.
- **Symtropy:** owns hidden world generation, available observation streams, permitted interventions, replay, and independent scoring.
- **Independent qualification:** determines which claims are supported in a declared scope; Symthaea cannot self-authorize scientific qualification.

Reuse the current AV-STREAM, Vision Mind, shared-grammar, invariance, symmetry, uncertainty, Symtropy blind-episode and bounded model-proposal work. Do not add a second graph engine, world model, sensor ontology, evidence ledger, or generic query system.

Symtropy #7171 adds a first finite validator for a proposed total structure correspondence and a separate qualitative intervention-effect witness. That is an executable contract slice, not an analogy-discovery learner or sealed evaluator.

## 3. Transfer Atlas: internal benchmark design

Build paired worlds rather than independent single-world tasks. Each pair should belong to a declared transformation family and have evaluator-owned hidden ground truth.

| Pair family | Expected behavior | Negative control |
|---|---|---|
| Symbol / entity renaming | Preserve structural relations under bijective relabeling | Many-to-one relabeling or a broken incidence |
| Time dilation / unit conversion | Preserve relations after scale normalization | Nonuniform timing change that alters the mechanism |
| Coordinate / channel transform | Preserve or equivariantly transform spatial relations | Coordinate change with a hidden frame mismatch |
| Orientation / sign reversal | Reverse only relations whose transform was declared | Unannounced reversal or inconsistent relation-kind transform |
| New composition | Reuse learned components in unseen combinations | Familiar components with incompatible composition rules |
| Matched local marginals | Discover higher-order relational difference | Same marginals, different interaction mechanism |
| Superficial motif collision | Reject a visually/statistically similar but structurally false analogy | Shared short pattern with different causal role |
| Non-identifiable pair | Return a set-valued or unresolved mapping | Penalize unjustified unique identification |
| Intervention pair | Predict transformed effects under precommitted interventions | Same observational fit but different intervention outcome |
| Multiscale pair | Preserve qualified relations under abstraction/coarse-graining | A relation that disappears or changes at the coarser scale |

Start with finite deterministic generators. Every generator family needs an independent implementation or independently derived exact oracle, a pinned version, replay inputs, and explicit negative controls. Keep seeds, hidden relation labels, future observations and holdout outcomes evaluator-owned.

### Required phases

1. **Acquisition:** unlabeled causal streams, no downstream question or task reward.
2. **Consolidation:** freeze the structural library and candidate transfer mappings.
3. **Blind transfer:** apply the frozen library to hidden paired worlds.
4. **Discrimination:** permit bounded queries/interventions only under the declared protocol.
5. **Qualification:** evaluate transfer, false analogies, uncertainty, cost and counterexamples independently.

## 4. Scorecard: no single ErgoScore

Report every metric per generator family and transformation class.

### Capability
- structural correspondence precision / recall;
- relation-kind mapping accuracy;
- invariant, equivariant and non-transfer classification accuracy;
- held-out composition and cross-domain transfer;
- interventional effect prediction;
- multiscale relation retention;
- downstream decision utility.

### Epistemic reliability
- false-unification rate;
- false-invariance / false-equivariance rate;
- false causal transfer rate;
- false-unique rate on non-identifiable worlds;
- calibration and risk-coverage;
- abstention quality;
- negative-transfer cost;
- counterexample detection and correction latency.

### Efficiency
- observations and active queries to first validated transfer;
- interventions per discriminated hypothesis;
- CPU/GPU time, model calls/tokens, training and inference compute;
- peak memory, wall time and energy when measured;
- total cost per validated structure and per successful transfer.

The primary competitive frontier should be **transfer capability versus false-analogy risk versus all-in observation/compute cost**. Publish Pareto frontiers and worst-family results. A gain in average transfer cannot compensate for an unacceptable rise in false causal claims.

## 5. Current research and industry comparison

### Direct scientific discovery: Unsatz

Unsatz (2026) evaluates scientific model-form discovery without supplying the ansatz. Its benchmark contains 80 systems across 20 domains; the authors report nine LLMs solving 21%–60% of systems. The benchmark also tests transfer by providing an agent with a program discovered for a related system. The authors report that this helps only when the mechanisms are shared and the first investigation succeeds.

**Use:** reproduce the relevant task protocol and include it as a direct scientific-discovery comparison. ErgoStream's distinct contribution is a pre-task structural library and explicit false-analogy / transfer-boundary evaluation. Do not compare a score from different tasks and claim superiority.

Paper: https://arxiv.org/abs/2611.03713

### Active causal discovery: ACDB

The 2026 Active Causal Discovery Benchmark evaluates LLM agents under budgeted interventions. Its current six-level linear-Gaussian ladder reports:
- PC plus greedy active orientation: 42.7% directed F1, SHD 4.79;
- Claude Sonnet 4.6 raw active: 31.7% directed F1, SHD 7.25;
- GPT-5.4 raw active: 22.9% directed F1, SHD 9.27;
- a structure-blind random DAG baseline: 23.6% directed F1 on the dense v0 ladder, reduced to 16.9% after a density probe.

The authors explicitly frame this as a benchmark audit and calibration report, not evidence that current LLMs solve active causal discovery.

**Use:** ACDB should be a high-priority Symtropy adapter for active discrimination, intervention-budget accounting, skeleton / DAG recovery and abstention calibration.

Paper: https://arxiv.org/abs/2609.31675

### Foundation world models: V-JEPA 2 and Dreamer 4

V-JEPA 2 is a strong adjacent baseline for self-supervised physical-video representation and prediction. Meta reports pretraining on more than one million hours of internet video; its action-conditioned V-JEPA 2-AC uses less than 62 hours of robot data for zero-shot planning demonstrations on Franka arms in new environments.

Dreamer 4 demonstrates a different industry direction: an action-conditioned world model trained from offline data and used to train an agent in imagination, including a long-horizon Minecraft diamond task.

**Use:** compare their compatible representations or task outcomes where input, training data, action interface and compute can be matched. These systems are not direct tests of a pre-task structural library, and their data scales make naive score comparisons unfair.

- V-JEPA 2: https://ai.meta.com/research/vjepa/
- Dreamer 4: https://arxiv.org/abs/2509.24527

### Dynamic robotics: RoboDyna and LIBERO-Dynamic

RoboDyna (November 2026) evaluates physically grounded dynamic manipulation with synchronous and asynchronous protocols. Its authors report sub-20% success for all evaluated policies and that inference latency flips 5%–12% of individual episode outcomes. LIBERO-Dynamic similarly exposes target motion during model inference and tests the effect of timing and replanning.

**Use:** high-value downstream action tests after structural acquisition succeeds. Prediction fidelity must not substitute for task completion, safe recovery, latency and human-gap reporting.

- RoboDyna: https://arxiv.org/abs/2611.26969
- LIBERO-Dynamic: https://arxiv.org/abs/2610.24579

### Audiovisual grounding: DCASE 2026 Task 3

The completed Task 3 challenge had 29 submissions from eight teams. The leading audio-only entry achieved 0.108 macro mAP; the leading audiovisual entry achieved 0.0131 macro mAP. The task reconstructs dense semantic acoustic maps from four-channel input, optionally using video.

**Use:** a specialist AV-STREAM adapter for event localization, temporal alignment and missing/delayed modality evaluation. Do not treat the leaderboard as general structural reasoning.

Results and task definition: https://dcase.community/challenge2026/task-semantic-acoustic-imaging-for-sound-event-localization-and-detection-from-spatial-audio-and-audiovisual-scenes

### Interactive exploration: ARC-AGI-3

ARC-AGI-3 evaluates exploration and adaptation in unfamiliar interactive games without natural-language instructions. The 2026 competition has closed; the official prize-results announcement is scheduled for December 4, 2026.

**Use:** an external challenge for interactive adaptation, not the primary measure of task-agnostic acquisition or mathematical structural transfer.

https://arcprize.org/competitions/2026

### Benchmark landscape: world-model survey

The 2026 survey catalogs 130 representative world-model benchmarks and distinguishes evaluation target, protocol, metric family and data source. This is a useful warning against collapsing open-loop prediction, closed-loop action, visual quality and downstream utility into a single league table.

**Use:** select task-supported combinations from its taxonomy and report the precise protocol and metric role. Do not quote an aggregate count as evidence that any one domain is adequately benchmarked.

https://world-model-benchmarks.github.io/

## 6. Baseline ladder

Every ErgoStream paired-world experiment should include the strongest compatible alternatives:

1. Random and passive exposure.
2. Fixed segmentation / frequent-motif and simple statistical baselines.
3. Dense embedding retrieval and nearest-neighbour analogy.
4. Explicit graph matching / graph-edit baselines with the same observed information.
5. GNN or graph-transformer representation.
6. LLM-based analogy / scientific-discovery agent with pinned model revision and prompts.
7. Active causal baselines from ACDB (PC plus greedy active orientation, plus compatible alternatives).
8. Unsatz-compatible scientific model-discovery baseline on its native task contract.
9. HDC-only, CfC/LTC-only, explicit typed graph and hybrid Symthaea ablations.
10. Exact finite oracle as evaluator sanity / upper bound only; never a competitor.

V-JEPA 2, Dreamer 4, RoboDyna and LIBERO-Dynamic are additional world-model / physical-interaction baselines on compatible downstream tracks, not substitutes for the transfer-specific ladder.

Match input information, exposure, active-intervention budget, training data, compute, wall time, memory, model/API revision and retries. Report every failed and incomplete run.

## 7. Prospective targets—not achieved results

Keep the current proposed targets, but apply them to the actual paired-world transfer task:

- 100% exact agreement with independent finite-world oracles on every in-scope test;
- zero hidden-truth, generator-seed or future-observation leakage;
- no false causal promotion on observational-only or non-interventional controls;
- no false unique map when the world is evaluator-defined as non-identifiable;
- at least six structurally distinct generator families before confirmatory claims;
- stretch target: +10 percentage points in primary transfer success over the strongest matched open baseline, **or** at least 20% lower all-in observation / compute cost at a preregistered non-inferiority margin no wider than 2 percentage points.

These targets require pilot variance and power analysis. False-unification and false causal transfer remain non-compensable guardrails. A SOTA claim requires direct matched runs, pinned artifacts, complete costs and uncertainty intervals.

## 8. Execution order

**P0 — close the transfer contract**
- Land or revise the Symtropy #7173 witness validator after exact-head format/test/Clippy.
- Define explicit scope for partial mappings in a later version; do not weaken v0.1 total-map validation.
- Add paired-world generation with one invariant, one equivariant and one false-analogy pair.

**P0 — baseline before complexity**
- Implement fixed segmentation, random, passive, frequent-motif and graph-matching baselines.
- Freeze structural snapshots before the evaluator releases hidden intervention outcomes.
- Add leakage negatives and non-identifiability controls.

**P1 — connect Symthaea**
- Build a narrow learner adapter that emits candidate units, relation maps and a frozen structural-library snapshot.
- Compare HDC, CfC/LTC, graph and hybrid variants under matched budgets.
- Do not make the hybrid the default winner.

**P1 — active transfer**
- Add ACDB-compatible interventions and expected-value-of-discrimination acquisition.
- Test when a learner should ask, stop, abstain or retract.
- Record intervention receipts and all rejected / over-budget actions.

**P2 — multimodal and physical qualification**
- Adapt AV-STREAM / DCASE, then CLEVRER and PHYRE, then RoboDyna and LIBERO-Dynamic.
- Preserve separate synthetic, public benchmark, controlled physical and operational evidence tiers.

**P2 — confirmatory comparison**
- Freeze generator families, transfer pairs, model versions and baselines.
- Run family-held-out, composition-held-out comparisons with preregistered analysis.
- Publish per-family metrics, full denominators, counterexamples and resource costs.

## 9. Falsification conditions

Narrow or reject the research claim if:
- a simpler matched graph/statistical baseline achieves equivalent transfer at lower total cost;
- HDC/CfC does not improve transfer after controlling data and compute;
- transfer gains disappear under generator-family or composition holdouts;
- the learner frequently asserts false invariance on matched-marginal controls;
- intervention effects fail despite observational similarity;
- the learner cannot represent uncertainty or abstain under non-identifiability;
- a high aggregate transfer score depends on a small number of easy families.

## 10. Claim boundary

This research direction does not establish a universal learning algorithm, biological equivalence, consciousness, broad scientific discovery, causal understanding, or SOTA performance. A valid finite correspondence is a validated result about a declared finite representation only. Real-world transfer requires independent domain evidence, authentic interventions, rights-cleared data and domain-specific safety review.
