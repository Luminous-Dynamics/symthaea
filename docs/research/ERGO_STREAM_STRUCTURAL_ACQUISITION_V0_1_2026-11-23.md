# ErgoStream: Task-Agnostic Structural Acquisition

**Version:** 0.1 · **Date:** 2026-11-23 · **Status:** research specification, not an implemented learner or benchmark result
**Tracking:** [ERGO-STREAM-001 #13401](https://github.com/Luminous-Dynamics/symthaea/issues/13401)

## 1. Research motivation

Gromov's *Ergostructures, Ergologic and the Universal Learning Problem* frames learning as a dynamical process in which a learner constructs internal structure from incoming signal flows. It proposes that meaning depends on combinatorial arrangements and statistical distributions of signal units, and that some structures may only become visible when modalities are coupled. These are conjectures and conceptual prompts, not an established theory of biological learning.

This specification turns that prompt into a falsifiable engineering question:

> Can an agent acquire a reusable structural library from heterogeneous, time-ordered observations without receiving task labels, an object vocabulary, a target model family, or an extrinsic task reward during acquisition—and can that library improve hidden downstream transfer?

The benchmark still has a post-hoc evaluator. 'Task-agnostic acquisition' means the acquisition policy does not optimize against the evaluator's downstream labels/questions; it does not mean evaluation is value-free.

## 2. Non-duplication and architecture composition

This is a composition contract, not a replacement for existing subsystems.

| Existing owner | Reused responsibility | ErgoStream contribution |
|---|---|---|
| #12837 AV-STREAM | Causal live audiovisual input and task/latency evaluation | Unlabeled acquisition phase and structural-library snapshot interface |
| #3934 Vision Mind | Visual provenance, entities, geometry, persistent world state, active vision | Modality-neutral candidate unit/relation records |
| #10842 / #10890 | Native HDC–CfC dialogue and acoustic-faithful perception research | Reusable cross-modal structure acquisition tests |
| #8919 / #8671 | Shared grammar recovery and progressive reusable knowledge | Acquisition before a grammar or intermediate vocabulary is supplied |
| #11042 / #8705 / #8667 | Open-world discovery, transportability, unit/coordinate invariance | Downstream probes that assess whether acquired structures transfer |
| #11336 | Evidence-calibrated symmetry discovery | Symmetry/equivariance may be a discovered relation family, not an assumed prior |
| #8669 | Uncertainty, non-identifiability and support omission | Set-valued and unresolved structural-library entries |
| Symtropy #5990 / #6027 | Blind episode interface and bounded model proposals | Evaluator-owned stream worlds and withheld structure probes |

Do not add a second world model, sensor ontology, clock service, evidence ledger, generic graph engine, generic active-learning API, or PhysicsIR.

## 3. Learner boundary

```text
modality-native causal streams
    -> candidate signal/event units
    -> typed relation hypotheses
    -> recurring arrangements / motifs / compositions
    -> versioned structural library
    -> task-agnostic acquisition controller
    -> frozen library snapshot
    -> hidden evaluator probes and transfer tasks
```

### 3.1 Candidate units

A unit is an *uncertain proposal*, not an asserted object identity. It may be a temporal segment, visual region/track, acoustic event, recurring token pattern, state transition, or a derived multi-modal event. Preserve:

- stable local candidate ID and schema version;
- modality and source/frame references;
- measurement support interval, arrival time and clock uncertainty;
- candidate boundary/extent and alternatives;
- evidence origin (`observed`, `inferred`, `predicted`, `simulated`);
- confidence semantics and calibration profile, if known;
- parent/child and revision lineage.

Do not convert image-plane or sample-index coordinates into world units without a qualified transform. Do not infer a hidden semantic class merely because a segment is repeatable.

### 3.2 Candidate relations

Relation hypotheses are typed and scoped. Initial families:

| Family | Examples | Qualification boundary |
|---|---|---|
| Temporal | precedes, overlaps, persists, recurs, follows-after-gap | Time uncertainty and missingness remain explicit |
| Spatial | adjacent, inside, near, aligned, moves-with | Coordinate frame and metric must be identified |
| Statistical | co-occurs, predicts, correlates, shares motif | Not causal evidence |
| Transformational | maps-to, invariant-under, equivariant-under, composition | Transformation domain and tolerance are explicit |
| Functional / causal | enables, blocks, changes, causes-under-intervention | Requires interventions or stronger independent evidence |
| Compositional | part-of-pattern, role-in-motif, participates-in-rule | Must survive held-out occurrences and negative controls |

Every relation stores source observations, context, applicability scope, alternatives, uncertainty, and the reason for its current epistemic status. Co-occurrence does not promote to causality.

### 3.3 Structural library lifecycle

Library entries should have explicit lifecycle states:

- `candidate`: proposed by the learner; no independent holdout support;
- `replayed`: reproducible against its cited source observations;
- `holdout_supported`: survived a declared held-out probe in its scope;
- `qualified_in_scope`: passed the relevant independent qualification contract;
- `retracted` or `superseded`: contradicted, out of scope, or replaced.

A learner cannot self-promote an entry to `qualified_in_scope`. Qualification comes from the relevant independent evaluator/authority. Preserve earlier versions and counterexamples rather than overwriting history.

## 4. Intrinsic acquisition without reward hacking

Do not implement one scalar 'interestingness' reward and assume it captures Gromov's ergologic. Maintain a measurable vector of candidate acquisition signals:

- predictive learning progress (change in held-out predictive quality, not raw surprise);
- uncertainty reduction over competing structural hypotheses;
- recurrence across independent contexts and modalities;
- compositional reuse potential on held-out data;
- compressive gain subject to predictive sufficiency;
- discriminability of competing explanations under an available query/action;
- counterevidence and contradiction discovery;
- acquisition cost, latency, energy and memory.

Keep the components separately visible and report their Pareto frontier. The controller may select a policy-dependent trade-off, but a single score must not conceal a rise in false structure, overgeneralization or false confidence.

Critical negative controls: random acquisition, passive exposure, raw prediction-error novelty, RND-style novelty, information gain, learning-progress, and predictive-compression/MDL. A candidate intrinsic policy must beat the strongest compatible baseline under equal observations and compute; no assumption that novelty or surprise is intrinsically good.

## 5. Candidate technology roles (hypotheses only)

| Component | Candidate role | Required ablation / comparator |
|---|---|---|
| HDC | Compact binding, similarity and compositional retrieval | Dense embeddings; random hypervectors; HDC-only |
| CfC/LTC | Continuous-time memory and temporal evolution | Discrete recurrent model; transformer sequence baseline; CfC-only |
| Explicit typed graph/hypergraph | Auditable nodes, relations and higher-order motifs | Flat-vector and pairwise-graph baselines |
| E-graph / symbolic layer | Canonicalization and equivalence of learned expressions/rules | No-saturation and bounded symbolic baseline |
| Active perception | Select frames, windows, channels or interventions | Passive, fixed-rate, random and information-gain policies |

The hybrid is not the default winner. Component inclusion requires measurable improvement in transfer, quality or cost, with no unacceptable regression in uncertainty and integrity.

## 6. ErgoStream benchmark protocol

### Phase A — blind acquisition

The policy receives time-ordered, causally available modality-native frames and public sensor/action semantics. It receives no task goal, generating grammar, target vocabulary, hidden relation ID, future data, evaluator seed, downstream question or task reward. The policy may select bounded observation/focus actions and spend a declared budget.

### Phase B — consolidation

Freeze a complete structural-library snapshot and acquisition trace. Bind them to exact learner/configuration/source identity. No downstream probe result can revise the snapshot used for the primary score.

### Phase C — independent probes

An evaluator that did not train or steer the learner tests:

1. event/unit segmentation and equivalence-class recovery;
2. typed relation and motif recovery;
3. prediction on withheld continuations and action-conditioned outcomes;
4. composition of previously learned units into unseen structures;
5. nuisance invariance and false-unification rejection;
6. cross-modal association under missing, delayed and contradictory streams;
7. causal/interventional claims only in worlds with identifiable intervention semantics;
8. calibrated uncertainty, set-valued answers and correct abstention;
9. retention and correction after new evidence in a separate continuation experiment.

### Phase D — interactive worlds

Symtropy supplies deterministic hidden worlds and permitted actions. Symthaea sees only the policy-visible stream and action outcomes. The evaluator retains truth, seeds, split labels and probe outcomes. Replay and counterexample evidence bind to exact generator and learner artifacts.

## 7. First synthetic world families

1. **Discrete motif streams:** hidden units, boundaries, temporal motifs, repetitions, gaps and nested compositions.
2. **Matched-marginal worlds:** similar local symbol/transition statistics, different higher-order relation structure.
3. **Nuisance-transformed worlds:** same structure under symbol relabeling, rate/time scaling, channel permutation and missing observations.
4. **False-unification worlds:** superficially similar motifs with genuinely different generating grammars or mechanisms.
5. **Cross-modal worlds:** coupled audio-like / visual-like channels, delayed or missing alignment, distractors and ambiguous association.
6. **Interactive rigid scenes:** object permanence, occlusion, contact/event order, focus actions and bounded interventions.

Start with deterministic finite generators and exact small-world oracles. Do not add high-dimensional raw media, continuous PDEs or real-robot actions until source, timing, compute and evaluator contracts are qualified.

## 8. Baselines and SOTA comparison policy

Use strongest compatible, executable comparators—not a weak strawman:

- modality specialists plus a calibrated event-association layer;
- dense multimodal encoder / sequence model where input modes match;
- object-centric or event-centric recurrent/transformer world model;
- HDC-only, CfC/LTC-only, explicit graph, e-graph and hybrid ablations;
- passive next-observation predictor;
- random, surprise-only, RND, learning-progress, information-gain and compression-based acquisition;
- exact tiny-world oracle for evaluator sanity only.

External results from ARC-AGI-3, CLEVRER, DCASE, OmniEchoBench, EgoSchema, Video-MME-v2 or other suites are not directly rankable unless the actual input modality, timing, split, action budget, hardware, model revision and scoring contract match. Use them as adapters on their intended tracks, not as one aggregated 'understanding' score.

### Prospective internal gates (not achieved results)

- First reproduce every admitted baseline and exact tiny-world control.
- On confirmatory worlds, target at least a 10 percentage-point improvement in structural transfer over the strongest matched open baseline **or** at least 20% lower total observation/compute cost at a pre-registered non-inferiority margin of at most 2 percentage points on primary structural outcomes.
- In all cases, false confident causal/structural claims must not increase; report false-unification and false-unique rates as non-compensable guardrails.
- Use at least six generator families, world-level split/uncertainty analysis and a preregistered precision/power plan; do not count frames from one world as independent worlds.

These are proposed go/no-go thresholds for a later confirmatory study. Estimate pilot variance first and amend thresholds before—not after—unblinding. They are not comparisons against a published leaderboard and not claims of current superiority.

## 9. Scorecard

Never collapse the benchmark to one number. Report per family and regime:

- unit/event boundary and equivalence-class precision/recall;
- typed relation/motif recovery and false-causal-relation rate;
- held-out predictive log score and calibrated error;
- compositional transfer, reuse precision and negative transfer;
- cross-modal association and temporal-alignment error;
- false invariance / false unification / false unique diagnosis;
- set-valued accuracy, risk-coverage and abstention;
- observation, intervention, model-call, candidate-evaluation, CPU/GPU, memory, latency, energy and human-review cost;
- structural description length versus predictive quality;
- all failures, invalid actions, incomplete traces and denominators.

## 10. Falsification conditions

Stop or narrow the hypothesis if:

- a simpler supervised or unsupervised baseline reaches equivalent transfer at lower total cost;
- HDC/CfC contributes no measurable improvement after controlling compute and input information;
- the acquisition policy improves training prediction but not unseen composition/transfer;
- the learner invents shared structure on heterogeneous controls or promotes co-occurrence to causality;
- results collapse under split-by-generator or split-by-composition evaluation;
- a system cannot identify missing evidence or abstain when structure is non-identifiable.

## 11. First implementation sequence

1. Define versioned stream-frame, candidate-unit, candidate-relation, library-entry and acquisition-event schemas.
2. Add one tiny discrete motif generator and a separately implemented exact evaluator in Symtropy.
3. Implement a minimal task-agnostic learner and one strong non-HDC baseline against the same interface.
4. Record full resource accounting and acquisition traces; add seed/label leakage negatives.
5. Only then add HDC/CfC and explicit graph/e-graph arms.
6. Freeze confirmatory families/splits and run the matched study.

## 12. Claim ceiling

This is a research design. It does not establish a universal learning algorithm, a biological ergobrain, semantic understanding, consciousness, general physical discovery, or SOTA performance. Success on synthetic stream worlds demonstrates only the measured capabilities under the declared contract. Real-world claims require independent, rights-cleared observations and domain-specific validation.

## References

- Gromov, *Ergostructures, Ergologic and the Universal Learning Problem*: https://www.ihes.fr/~gromov/wp-content/uploads/2018/08/ergologic31.pdf
- Symthaea AV-STREAM-001: https://github.com/Luminous-Dynamics/symthaea/issues/12837
- Symtropy ERGO-STREAM-WORLD-001: https://github.com/Luminous-Dynamics/symtropy/issues/7149
- ARC-AGI-3 official programme: https://arcprize.org/arc-agi/3