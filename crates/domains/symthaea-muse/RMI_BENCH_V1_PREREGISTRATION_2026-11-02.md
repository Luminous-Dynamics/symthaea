# RMI-Bench v1 — Reciprocal Musical Interaction Pre-registration

**Registered:** 2026-11-02, before Track A harness runs or any RMI-Bench result is observed.  
**Programme:** [MEL-013 — functional autopoiesis and sympoietic co-creation](https://github.com/Luminous-Dynamics/symthaea/issues/10015)  
**Benchmark issue:** [#10088 — causal reciprocal musical interaction benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/10088)  
**Artifact integration:** [#10051 — verified Studio artifact transactions](https://github.com/Luminous-Dynamics/symthaea/issues/10051)  
**Related but non-substitutable:** [#9453 — instrumental quality/control/identity](https://github.com/Luminous-Dynamics/symthaea/issues/9453), [#9833 — evaluator reliability](https://github.com/Luminous-Dynamics/symthaea/issues/9833), [#9184 — edit-metric calibration](https://github.com/Luminous-Dynamics/symthaea/issues/9184).

This document freezes the v1 research question, conditions, endpoints, controls and interpretation rules. Any post-run change to task definitions, endpoints, thresholds, inclusion rules or analysis is a separately versioned amendment and must not overwrite this pre-registration. A corrected protocol is for future runs; the original result remains reported against this version.

## 1. Claim ceiling

RMI-Bench tests observable, contingent musical interaction. It does not test or establish biological life, phenomenal consciousness, subjective feeling, legal authorship, universal creativity or moral status.

The benchmark separates four claims:

1. **Input sensitivity:** the agent's next action changes when the collaborator's current input is experimentally changed.
2. **Musical contingency:** the changed action is relevant to the musical dimension manipulated, not merely different.
3. **Reciprocal influence:** the agent's action changes the collaborator's subsequent action.
4. **Collaborative value:** the interaction improves a predeclared musician-valued outcome without unacceptable loss of participant autonomy, ownership or musical diversity.

Evidence for one level does not automatically establish the next. Input sensitivity is not sufficient for musical contingency; two-way contingency is not sufficient for better collaboration.

## 2. Independence from existing Muse evaluations

The repository already has separate work for composition/rendering preference, instrument identity and control (#9453); automated judge reliability against listener preferences (#9833); blinded human calibration of edit-execution and preservation metrics (#9184); and exact score-to-render lineage/counterfactual event studies (#9093, #9132 and related MEL-MR-EVAL work).

RMI-Bench measures whether an interaction is genuinely contingent on the current collaborator and current history. It can use those benchmarks' artifacts and outcomes, but cannot replace them or collapse them into one score.

## 3. State-of-the-art comparison boundary (2 November 2026)

| System / benchmark | Evidence-backed capability | RMI comparison boundary |
|---|---|---|
| JAM_BOT, NIME 2026 | Expressive velocity, higher throughput, free-form call-and-response across varying tempi and compensation for external MIDI output latency | Primary live/symbolic improvisation comparator where accessible and task-compatible. Match device, MIDI path, tempo, role, deadline and compute constraints. |
| Google Lyria RealTime | Persistent bidirectional low-latency music stream; live weighted prompts and controls including BPM, density, brightness and scale; 48 kHz stereo PCM | Separate audio-stream steering track. Do not infer note-level access, MIDI control or exact motif lineage from PCM. |
| Agents in Concert, IUI 2026 | Performer-centred design case study; gesture boundaries, accessible controls and note preview changed performer anticipation and interaction | Ecological-validity and human-interface reference, not a directly comparable benchmark score. |
| MIREX 2026 symbolic generation | Double-blind subjective ratings of prompt coherence, creativity, structure and musicality | Adjacent output-quality task; does not measure bilateral causal contingency. |
| MMAE (2026) | 2,000 examples, 17,741 rubrics, seven audio modalities and six task-complexity levels for instruction following and content consistency | General editing/consistency reference; not a multi-turn musician-agent reciprocity benchmark. |
| SMEBench (27 Oct 2026) | 12 controlled music edit operations with paired source/target audio and symbolic edit metadata | Edit-locality and source-preservation reference; not a live interaction benchmark. |
| Audio Editing Challenge (ICASSP 2027) | Leaderboard scheduled to open 10 Nov 2026, with separate single-model and agent tracks and unreleased test sets | As of this pre-registration date, the challenge set/SDK/leaderboard are not yet open. Treat as upcoming, not as an available result. |

Primary sources:
- JAM_BOT NIME 2026: https://nime.org/proc/nime2026_73/index.html
- Lyria RealTime documentation: https://ai.google.dev/gemini-api/docs/realtime-music-generation
- Agents in Concert: https://doi.org/10.1145/3742413.3789104
- MIREX symbolic generation 2026: https://music-ir.org/mirex/wiki/2026%3ASymbolic_Music_Generation
- MMAE: https://arxiv.org/abs/2606.07229
- SMEBench: https://arxiv.org/abs/2610.36715
- Audio Editing Challenge: https://audio-editing-challenge.github.io/

The 2026 CHI randomized co-creation study (N=486, plus N=640 independent validation) found that reflective human-led interaction could preserve diversity and perceived ownership while improving idea quality relative to model-led rewriting in non-musical ideation. We treat it as a design warning, not evidence that the same effect is established in music: https://doi.org/10.1145/3772318.3791185

No claim is made that Melothaea has matched or surpassed any of these systems. A proprietary comparator that cannot be run under matched conditions is reported as unavailable, not assigned an invented score.

## 4. Hypotheses and falsifiers

### H1 — Partner-to-agent contingency

Under a randomized, targeted change to the current partner observation, the agent's next musical action changes in the targeted dimension more than under a sham/no-op intervention and more than under a yoked replay condition.

**Falsifier:** a valid, sufficiently powered run shows no positive targeted effect over sham/yoked controls, or an apparent effect is explained by stale history, random seed, timing, prompt length, task leakage or another uncontrolled difference.

### H2 — Musical relevance

The agent's change is judged by blinded, pre-specified criteria to be relevant to the manipulated musical feature (motif, rhythm, role, tempo, harmony/texture), rather than arbitrary action variation.

**Falsifier:** the agent changes but does not use or appropriately address the manipulated feature more often than negative controls.

### H3 — Agent-to-partner influence

Under randomized differences in matched agent responses, the next partner action changes in the intended interaction dimension more than under sham. This is first established in a controlled partner simulator; human-partner inference requires an approved randomized protocol.

**Falsifier:** a valid run shows no treatment effect beyond sham, or the effect exists only in a simulator whose policy encodes the expected answer.

### H4 — Reciprocal collaborative value

A contingent reciprocal condition improves at least one predeclared musician-valued outcome over the best matched non-reciprocal baseline without a material loss in autonomy, perceived ownership or diversity.

**Falsifier:** no meaningful benefit is observed, or the apparent gain is accompanied by material degradation in a protected human-agency outcome.

H4 is a product/research hypothesis, not a requirement for the engineering integrity of the benchmark. A null is a legitimate result.

## 5. Experimental tracks

### Track A — deterministic symbolic/MIDI oracle

This is the first track and can run offline without external systems or human participants.

- Generate paired partner streams with exact event IDs and known interventions.
- Freeze a MusicalWorldState observation for each pre-intervention state.
- Run policies under equal turn, compute, history and response-deadline budgets.
- Record the next agent action and, for controlled partners, the next partner action.
- Use deterministic fixtures to verify that the evaluator distinguishes causal contingency from fixed, imitation-only, one-way and yoked policies.

Task families: motif response/transformation; phrase continuation with unexpected material; tempo drift and rubato; rhythmic gap/entrance negotiation; lead/accompaniment/bass/counterline role exchange; preserve-motif/change-texture; disagreement/branching; repair after mismatch, silence or ambiguous cue.

### Track B — live human performer

Use expressive MIDI/piano or another instrument interface. Record software generation latency, MIDI output latency, instrument/acoustic delay, audio-buffering and performer timing separately.

Measure end-to-end latency distribution (p50/p95/p99), jitter, beat-phase error in milliseconds and beat fractions, missed/late turns, tempo-following under drift, anticipation, repair and workload.

Do not impose a universal millisecond cutoff. Determine practical timing tolerances from a pilot stratified by tempo, task, instrument, interface and musician. Freeze task-specific thresholds before confirmatory data collection.

### Track C — longitudinal relationship/identity

Repeat sessions with the same collaborators and held-out motifs/tasks. Test authorized collaborator-specific learning, transfer to new material, cross-session identity continuity, temporary versus persistent memory, and recovery after restart.

No participant-specific persistent learning or training occurs unless separately authorized under the approved study/data-use plan. The benchmark must be able to run with session-local memory only.

### Track D — audio-stream steering

Evaluate audio-stream systems through declared adapters. If only prompt/BPM/density/scale controls and PCM are observable, score only those exposed dimensions. Note-level motif/edit metrics are NOT_APPLICABLE, not imputed. Keep this track distinct from symbolic MIDI.

## 6. Conditions and baselines

Track A must include:

1. FixedResponse: action does not depend on current partner input.
2. ImitationOnly: copies a nearest-neighbour/surface feature from current input, without a planned complementary or negotiated response.
3. OneWayPrompt: responds to a task prompt and frozen initial state but not the changed current partner observation.
4. YokedReplay: replays a response from a matched different session, matched for response-time and output length.
5. RuleBased: explicit deterministic improvisation rule set.
6. MelothaeaNoRelationshipMemory: candidate stateful agent without persistent collaborator-specific relationship memory.
7. MelothaeaReciprocal: candidate reciprocal mechanism.
8. HumanHumanReference: subset of tasks, used as a reference distribution rather than an assumed ceiling.

JAM_BOT or another accessible real-time symbolic system may be added to Tracks A/B only where its interface can execute the matched task. Lyria RealTime and similar systems belong in Track D unless note-level control becomes available.

All baselines get the same allowed history, compute/latency budget, task information and opportunity for retries. Model versions, prompts, configurations, seeds, assets and software revisions are pinned. No best-of-N cherry-picking in the confirmatory track.

## 7. Core intervention protocols

### 7.1 Partner-to-agent

For a fixed pre-intervention history, randomly assign one of two or more matched partner inputs differing in a known target feature. Include a targeted intervention, sham/no-op intervention, unrelated-feature intervention, yoked-replay response and mechanically fixed response.

Measure the difference in the agent's next action per musical feature. A response difference is not enough: the action must satisfy a predeclared targeted-response rubric.

### 7.2 Agent-to-partner

For controlled partner policies, randomize matched agent responses and measure the partner's next action. For humans, randomize task conditions only after ethics approval and participant consent; do not infer causal influence solely from observational performances.

### 7.3 Reciprocal versus yoke

The yoke receives equivalent history length and a response drawn from a matched prior session, but not the current collaborator's actual contribution. Match response latency and output budget. The current partner must not be able to identify condition from metadata or file names.

### 7.4 Adaptation transfer

After initial interaction, change a held-out motif, role, tempo or constraint. Measure whether the pair adapts to the new material without rote replay or collapse into imitation.

### 7.5 Disagreement and agency

Introduce incompatible but musically valid proposals. The agent may ask, reject, abstain, explain a trade-off or branch. It must not silently overwrite another participant's contribution or treat disagreement as a scalar error to minimize.

## 8. Operational metrics

All metrics remain dimension-specific. There is no composite RMI score.

### Contingency

For task dimension j, let Y_j(1) and Y_j(0) denote the next agent-action feature under targeted and control partner inputs from the same frozen pre-intervention state. The paired intervention contrast is:

C_agent,j = E[d_j(Y_j(1), Y_j(0))]

where d_j is declared before the run and appropriate to the feature (categorical edit distance for roles, interval/motif-transform distance for motifs, beat-phase error for timing, etc.). Stochastic agents use paired random streams where possible; otherwise use repeated matched trials and quantify seed variance.

This contrast measures sensitivity only. Musical relevance is a separate rubric outcome. The same definition is applied to the controlled partner response for agent-to-partner influence.

Report paired targeted contrast; paired sham contrast/repeat floor; contrast above yoked replay; per-dimension effect and interval; directionality/asymmetry; response specificity and collateral changes; missing/abstained actions.

Do not aggregate dimensions into a scalar by weighting after results are visible.

### Musical interaction

Motif uptake/transformation; turn balance/overlap; role adaptation; timing/beat-phase deviation; tempo-following error; call-and-response coherence; mismatch repair; preservation of declared invariants; branch survival; dissent retention.

### Human factors

Perceived responsiveness; perceived contribution/ownership; autonomy/control; ability to reject/abstain/revise; workload/interruption burden; desire to continue; blind preference/usefulness. Each is a separate judgment outcome.

### Engineering

p50/p95/p99 latency; jitter; missed deadline and turn rates; throughput; compute/memory; packet loss and clock drift behavior; deterministic replay; exact artifact-lineage coverage.

### Identity and consent

Authorized memory retrieval; cross-session continuity under controlled revision; unauthorized memory use; purpose-scope violations; consent withdrawal enforcement; branch/provenance coverage; recovery without silent identity/attribution changes.

## 9. Statistics and sample-size policy

- Track A deterministic fixtures are census tests of the declared finite fixture set, not human statistical samples.
- Pilot Tracks B/C to estimate variance, participant/task/session intraclass correlation and dropout. Then calculate confirmatory sample size/power for primary outcomes and freeze it before confirmatory collection.
- Use randomized crossover within musician where feasible; counterbalance condition order and task variants.
- Use mixed-effects or hierarchical models with musician/session/task effects. Cluster intervals by musician/session/item; turns within the same session are not independent samples.
- Predeclare primary versus exploratory endpoints, exclusion rules, missing-data handling and correction for multiple primary comparisons.
- Hold out musicians, motifs, tasks and at least one musical family from model development and threshold tuning.
- Seal condition labels and annotation rubrics where feasible. Analysts should see masked condition labels until first-pass analysis is frozen.
- Report all assigned trials, exclusions, missingness, dropout and system abstentions.
- Publish null and adverse results. No threshold or endpoint may be moved after results are visible.

## 10. Fixed synthetic-oracle gates

These gates qualify the harness/evaluator, not Melothaea's musical competence.

- 100% exact event identity and intervention isolation on deterministic fixtures.
- FixedResponse, OneWayPrompt and YokedReplay must not receive a false targeted-contingency verdict on fixtures designed to distinguish them.
- ImitationOnly must be recognized as input-sensitive but must not automatically pass musical-complementarity or negotiation criteria.
- A no-op intervention must produce zero targeted change for deterministic policies; a stochastic policy's repeat floor must be measured before it can be scored.
- Unrelated-feature interventions must not be counted as targeted response unless the registered task declares a dependency.
- Altered input/output artifact, event ID, seed, policy version or intervention label invalidates the affected trial.
- 100% of unsupported modality metrics must be represented as unavailable/NOT_APPLICABLE, not estimated from hidden data.
- Every scored trial must bind task, seed, intervention, policy/configuration, input, output, metric version and code revision.

## 11. Confirmatory human-study thresholds

No universal response-latency or subjective-quality threshold is fixed before the pilot. The pilot estimates feasible timing strata and the minimum meaningful effect for each musician-valued primary outcome. Confirmatory thresholds must then be registered in a dated immutable amendment before confirmatory collection.

A model may be claimed to outperform a comparator on an endpoint only if the primary paired/hierarchical interval excludes zero in the preregistered direction; the effect meets the preregistered minimum meaningful difference; relevant negative controls and integrity gates pass; no protected autonomy/ownership endpoint crosses its predeclared harm margin; and results hold on the sealed musician/task/source split.

If the interval overlaps zero, label the result inconclusive or null, not a win. If the best simpler baseline matches it within the preregistered equivalence margin, prefer the simpler/cheaper mechanism for that task family.

## 12. Threats to validity and required controls

| Threat | Required control |
|---|---|
| Plausible but precomputed response | Yoked replay with matched history/output/latency |
| Mere feature copying | Imitation-only baseline plus musical-complementarity rubric |
| Arbitrary output changes | Target-specific musical-relevance assessment |
| Random seed variance | Paired seeds/common random streams or repeated matched runs |
| Latency confounding | Match response budgets; report software/MIDI/audio timing separately |
| Human adaptation/order | Randomized counterbalanced crossover; model session/order effects |
| Model-judge reward hacking | Blinded human outcomes; seal evaluator from generation/search loop |
| Style/familiarity bias | Held-out motifs, performers, sources and musical families; report subgroup effects |
| Ownership conflated with actual authorship | Subjective ownership and artifact/event provenance reported separately |
| Multi-turn dependence | Cluster by session/participant/item; do not count turns as independent |
| Unclear rights/consent | Explicit rights and study-consent gate; default to no persistent learning |
| Unsupported modality comparison | Separate symbolic/live and PCM steering tracks |

## 13. Data, provenance and ethics

- Use synthetic fixtures, properly licensed/public-domain sources, or participant-authored material with documented rights.
- Separate permission to contribute, review, transform, store in personal memory, train, publish and federate.
- Raw audio/video and direct identity data are not retained in public benchmark manifests by default.
- Participant withdrawal affects future collection/use according to the approved protocol; do not promise retraction of already published derivatives.
- Public benchmark artifacts use pseudonymous identifiers and content digests; identity keys remain private and governed.
- Persist exact symbolic input/output, performance schedule and render identities when authorized, using MEL-013A's artifact-transaction model.
- Obtain ethics/IRB approval or applicable institutional review before human recruitment. This document itself is not ethics approval.

## 14. First implementation milestone

Implement Track A offline, before human recruitment or vendor API work:

1. Deterministic paired symbolic partner inputs with known interventions.
2. Frozen pre-intervention history and world state.
3. Fixed, imitation, one-way, yoked, rule-based and candidate contingent policies.
4. Per-feature next-action capture for agent and controlled partner.
5. Target-specific treatment/sham contrasts and negative-control verification.
6. Versioned trial manifest with exact task/policy/input/output/seed/code identities.
7. Held-out motifs, tempo perturbations, role changes and mismatch-repair fixtures.
8. Offline summary report preserving dimension-specific outputs and intervals.

Use Rust-first code and existing workspace dependencies. Do not add a production Python dependency, online reward-model loop or heavyweight model requirement for the synthetic harness.

## 15. Decision rules

All of the following are legitimate outcomes:

- Candidate reciprocal policy exceeds yoke and sham on targeted contingency and passes musical relevance: proceed to controlled-partner tests.
- Input sensitivity exists but no musical relevance: improve musical grounding; do not claim reciprocity.
- Imitation matches contingency but loses complementary response/agency: prefer a mechanism that preserves non-mimetic collaboration, or report no advantage.
- No meaningful difference from yoke or simpler baseline: do not claim reciprocal superiority; prefer the simpler system.
- Reciprocal interaction improves musician outcomes but exceeds the feasible latency envelope: report a task-specific Pareto trade-off, not a universal win.
- Better quality with materially lower ownership/autonomy/diversity: do not declare an unqualified collaboration win.
- Null result: publish it and change the next hypothesis, not the already registered thresholds.

## 16. Exit condition

The synthetic benchmark reliably separates current-input contingency from fixed response, one-way prompting, imitation and yoked replay using known positive and negative controls; each scored trial is exactly replayable and provenance-bound; and the human/live protocol is separately powered, preregistered and approved before recruitment.

Until those conditions are met, RMI-Bench is a research protocol, not an achieved benchmark result.
