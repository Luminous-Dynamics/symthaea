# Melothaea competitive frontier and research agenda v2

**Date:** 6 November 2026  
**Status:** Current external snapshot and falsifiable research agenda. Targets are goals, not results.  
**Parent:** MEL benchmark registry v1; RMI-Bench v1; MEL-013A artifact evidence; MEL-ACT live output-effect adapter.  
**Claim discipline:** No commercial victory, human-equivalent musical judgment, autonomous authorship or consciousness claim follows from synthetic tests or leaderboard position.

## Executive decision

Melothaea should not be framed as a prompt-to-song generator attempting to beat Suno on one leaderboard axis. The stronger product and research architecture is a **musical agency system with interchangeable realization backends**:

1. It interprets an evolving musical situation and maintains an explicit world state.
2. It negotiates what to contribute, what to preserve, what remains ambiguous, and who has authority to choose.
3. It plans semantic edits or continuations against explicit constraints.
4. It routes realization to a native or external renderer whose capabilities are declared and qualified.
5. It verifies the produced score/MIDI/audio, event lineage, output-effect receipts and rights/consent scope.
6. It listens to the result, detects mismatch, and recovers without silently overwriting a collaborator's choices.

The competitive separation is deliberate:

- **Production quality** is compared with Suno, Mureka, Lyria and other music generators using matched blind listening studies.
- **Editing correctness** is compared with audio-editing benchmarks using instruction execution, preservation and exact-match metrics.
- **Musical agency** is compared with improvisation, interaction and streaming-agent systems using causal intervention and control tests.
- **Human value** is assessed by musicians and listeners, separately from automatic metrics.

Never compress these into a single music-intelligence or consciousness score. A strong external renderer can improve production quality while Melothaea's agency layer remains independently evaluated.

## 1. Current commercial production frontier (6 November 2026)

Artificial Analysis maintains separate instrumental and vocal boards. Scores are blind pairwise-preference Elo estimates within their respective boards, not absolute music-quality units.

### Instrumental v1.1 snapshot

| Model | Elo | Reported 95% interval | Samples |
|---|---:|---:|---:|
| Suno v6 | 1148 | 1126–1170 | 2,208 |
| Suno v6-mini | 1114 | 1093–1135 | 2,332 |
| Mureka V9.5 | 1084 | 1063–1105 | 2,297 |
| Mureka V9 | 1083 | 1062–1104 | 2,322 |
| Lyria 3 Pro | 1080 | 1059–1101 | 2,329 |
| Lyria 3.5 | 1078 | 1057–1099 | 2,302 |

### Vocal v1.1 snapshot

| Model | Elo | Reported 95% interval | Samples |
|---|---:|---:|---:|
| Suno v6 | 1144 | 1126–1162 | 3,063 |
| Suno v6-mini | 1132 | 1114–1150 | 3,175 |
| Mureka V9.5 | 1101 | 1083–1119 | 3,141 |
| Mureka V9 | 1063 | 1045–1081 | 3,121 |
| Lyria 3.5 | 1050 | 1032–1068 | 3,203 |
| Lyria 3 Pro | 1048 | 1030–1066 | 3,191 |

These boards show that commercial systems already deliver strong one-shot generation and preference performance. Suno v6 is the current leader on both boards; the leading intervals overlap with some alternatives, so small point differences should not be represented as decisive wins. Vocal and instrumental scores are not interchangeable. Freeze model, prompt, modality, output duration, delivery format and date for every local comparison.

**Practical implication:** do not promise that Muse's native renderer beats the commercial leaders before matched blind evaluation. Design Melothaea to use the strongest permitted renderer for the task, while making semantic control, continuity, agency and verifiable execution the differentiators.

Sources:
- https://artificialanalysis.ai/music/leaderboard/instrumental
- https://artificialanalysis.ai/music/leaderboard/vocals
- https://artificialanalysis.ai/methodology/music

## 2. What the industry is already good at—and where it leaves room

| Industry capability | Current frontier | Melothaea response |
|---|---|---|
| Prompt-to-song generation | Strong end-to-end commercial systems; high blind preference | Use as renderer baseline; compete through controllable co-creation and measurable preservation |
| Fast iteration and clip editing | Commercial studio products expose section replacement, extension and arrangement workflows | Use semantic edit transactions that declare target, protected regions, alternatives, rollback and provenance |
| Audio-stream steering | Real-time audio generation/steering research and products | Keep audio-only control as its own adapter track; do not invent symbolic note access |
| Musical interaction | Live improvisers and co-creative systems demonstrate call-and-response, tempo following and performer controls | Add explicit permission, causal reciprocity, dissent retention, recovery and verified output effects |
| Editing evaluation | MMAE prioritizes exact-match execution with consistency protection | Report IFR, CR and EMR independently; preserve every attempt and failure |
| Temporal grounding | TAG-Bench exposes poor retrieval of every relevant interval and multi-occurrence counts | Treat temporal locating as set-valued retrieval, not top-1 clip selection |
| Human-centred timing | HEARTBEAT and InteractionBench stress client-observed timing and no-response accounting | Use monotonic clock-domain bridges; account every chunk and distinguish callback admission from physical playback |

## 3. Paradigm shifts that change the problem

### A. From generator-centric to agency-centric

A generator answers "what audio can be produced from this prompt?" A musical collaborator must also answer: "what is appropriate now, what is authorized, what should remain unchanged, when should I remain silent, and how can either participant reverse or redirect the contribution?"

Build a generator-neutral agency plane. Model backends become realization capabilities, not the source of truth for collaboration state.

### B. From prompt rewriting to semantic musical transactions

Every edit becomes a typed transaction:

- source artifact and exact source region;
- requested transformation;
- explicit protected invariants;
- permitted degrees of freedom;
- one or more valid result branches;
- renderer capability requirements;
- authorization and expiry;
- exact result artifact and verification receipt;
- undo/redo or branch-preservation information.

A successful edit must change the target and preserve declared invariants. Textual instruction similarity or a pleasing output is not sufficient.

### C. From plausible responsiveness to causal reciprocity

A response can look collaborative while being precomputed, replayed, copied, or insensitive to the current input. RMI-Bench therefore uses matched control, targeted, sham and unrelated interventions, plus fixed, imitation, one-way and yoked controls.

The two directions remain separate:

- Partner input changes the agent's next action.
- Agent action changes the partner's next action.

Only a joint, dimension-specific result supports a synthetic reciprocal claim. Human-partner causal claims require randomized, approved studies; live observational correlation is not enough.

### D. From raw turns to explicit uncertainty and negotiation

Musical situations frequently admit several valid continuations. A strong collaborator should preserve a viable branch set until the partner chooses, ask a question when the choice matters, and abstain when all options are vetoed or the request is outside its capabilities.

Do not reward a system for silently selecting one of several valid alternatives. Measure ambiguity preservation, explicit-choice compliance, veto compliance, and clarification cost separately.

### E. From output artifacts to verified transactions

A candidate ID, keeper key, URI, declared digest or caller-supplied "verified" boolean is not an artifact identity. Bind every event to exact content bytes and a durable transaction receipt; independently re-read and re-hash committed artifacts. Keep byte identity separate from signer trust, actor authentication, consent and musical quality.

### F. From one score to calibrated evidence

Treat failures, timeouts, abstentions, unavailable modalities, participant withdrawals and invalid outputs explicitly. A metric that cannot be measured through the declared adapter must be **NOT_APPLICABLE**, not estimated. Use cluster-aware uncertainty at the musician/session/item level rather than pretending every turn is an independent sample.

### G. From isolated sessions to sympoietic continuity without assimilation

Persistent memory should support collaborator-specific musical continuity only under explicit scope. It must not absorb disagreement, erase contributor boundaries, or turn one participant's contribution into a global preference. Preserve source-specific branches, contributor influence and consent scope across sessions.

### H. From fixed cultural taxonomies to co-governed validity

Genre and style are not universal labels with uniform meaning. Build evaluation cohorts with musicians and traditions that can define the task, instrumentation, terminology and what counts as a meaningful transformation. Report subgroup effects and uncertainty; do not treat a single world-music bucket as cultural coverage.

## 4. Benchmark portfolio and targets

Targets below are proposed stretch goals, not achieved results. Reproduce published baselines before tuning thresholds. Keep official native metrics and denominators unchanged.

| Capability | External comparator / native metric | Proposed target | Interpretation |
|---|---|---|---|
| Instrumental preference | AA-Music-Instrumental v1.1 Elo, blind pairwise | matched local study first; eventual lower 95% bound above comparator's upper 95% bound | Very ambitious; do not claim based on point estimate |
| Vocal preference | AA-Music-Vocal v1.1 Elo, blind pairwise | separate matched study; same interval criterion | Never pool with instrumental board |
| Audio editing | MMAE IFR / CR / EMR | reproduce official public baseline; stretch +10% relative EMR while retaining IFR and CR | Official Agent Track eligibility must be demonstrated |
| Controlled editing | SMEBench edit success and protected-content preservation | >=90% target operation success; >=95% protected-region preservation | Stretch goals; scorer and human calibration required |
| Temporal grounding | TAG-Bench mIoU / R@IoU>=0.7 / multi-occurrence count accuracy | >=40 mIoU; >=35% recall; >=50% count accuracy | Stretch goals against unchanged split/scorer |
| RMI partner-to-agent | targeted effect vs sham, unrelated and yoked controls | positive preregistered effect on held-out dimensions; zero false positives in deterministic negative controls | Report per dimension and uncertainty |
| RMI agent-to-partner | same reverse-direction controls | positive held-out controlled-partner effect; then randomized musician study | Simulator is not a human partner |
| Negotiation | branch preservation, choice compliance, veto compliance | zero silent commits under unresolved multi-valid choice; zero veto violations on finite oracle | Keep human value separate |
| Artifact evidence | event identity, content hashes, receipt and read-back | 100% exact event/artifact binding; zero substituted or detached artifacts accepted | Does not prove signer trust unless verifier is real |
| Live output control | queue application, epoch fence, callback admission, CPAL prediction, loopback | zero stale-epoch admissions after fence; report latency/jitter by BPM/device; physical silence only by loopback | Do not equate queue invalidation with silence |
| Human value | ownership, autonomy, workload, usefulness, desire to continue | preregistered improvement on musician-valued outcome with autonomy/ownership non-inferiority | Pilot first; musician/session clustered inference |

### Statistical policy

1. Census finite synthetic fixtures; do not call their pass rate a population estimate.
2. Pilot real musicians to estimate variance, cluster structure, task difficulty and meaningful effects.
3. Preregister primary endpoints, margins, exclusions, missingness, multiple-comparison policy and power before confirmatory collection.
4. Randomized crossover and counterbalancing where feasible; cluster intervals by musician/session/task.
5. Publish null and adverse outcomes. Prefer the simpler mechanism when it is equivalent within the preregistered margin.

## 5. Challenge and evaluation readiness (6 November 2026)

### ICASSP Audio Editing Challenge / MMAE

- **10 November 2026:** hidden challenge test set and SDK released; Single Model and Agent tracks each contain 500 hidden examples and are ranked separately.
- **25 November:** final submission and leaderboard freeze; **7 December:** reproducibility checks complete; **8 December:** final rankings.
- Public MMAE development set: 2,000 examples and 17,741 atomic rubrics.
- Native metrics: IFR, CR and EMR. Ranking is primarily EMR, with IFR then CR as tie-breakers.
- Registration closed **8 October**. Agent Track component model versions and weights had to be public before **1 November**. Do not claim official entry unless the registered team and exact eligible versions are evidenced. Public-set reproduction remains a valid research result.

Sources:
- https://audio-editing-challenge.github.io/leaderboard/
- https://audio-editing-challenge.github.io/

### HEARTBEAT 2027 — adjacent interaction prior, not a music leaderboard

Official 6 November schedule:

- **16 November 2026:** development leaderboard opens.
- **2 December:** team/resource declarations freeze.
- **16 December:** leaderboard closes.
- **23 December:** final container deadline.
- **30 December:** rankings and paper invitations.

Its released data has 132,571 samples (127,371 train / 5,200 development) across approximately 3,103 hours. Track I evaluates WAIT / RESPOND / CONTINUE / YIELD in separate offline and online settings. Online protocols use current and previously received input only, account for every chunk including no-response chunks, and prohibit backdated response timestamps. These are useful temporal-accounting principles, but speech/audio leaderboard scores are not music-generation evidence.

Source: https://heartbeat-challenge.github.io/

## 6. Industry comparison protocol

A fair generation comparison must:
- use the same prompts and modality for each model;
- freeze model version, generation settings, duration, output count and retries;
- normalize playback format and loudness consistently;
- blind and randomize presentation order;
- preregister genre, task, exclusions, sample size and analysis;
- separate one-shot and iterative workflows;
- report preference plus failure rate, controllability and output consistency;
- publish exact prompts, hashes, configuration and output lineage where licensing permits.

A fair editing comparison must:
- use the official track/split and scorer;
- preserve the native IFR/CR/EMR definitions;
- keep all failed attempts and abstentions in their denominator;
- publish separate complexity/modality strata;
- protect hidden test materials and avoid scorer leakage.

A fair agency comparison must:
- match observation history, response deadlines and compute budgets;
- use interventions, sham, unrelated and yoked controls;
- bind each action to the exact input event and policy revision;
- distinguish intention, application enqueue, callback admission, predicted playback and independent loopback observation;
- separate symbolic/MIDI, PCM steering and human live tracks.

## 7. Immediate execution sequence

1. Finish exact-head validation of the semantic MIDI parser and paired statistics additions in RMI-Bench #10721.
2. Validate the re-anchored typed benchmark registry and synthetic profile in its isolated workflow.
3. Integrate the RMI trial record with the production MEL-013A artifact verifier; fake read-back remains synthetic-only.
4. Integrate decision-to-effect receipts with the veto/output epoch reducer and clock-domain bridge; keep callback and loopback evidence distinct.
5. Reproduce public MMAE, SMEBench and TAG baselines on unchanged protocols before claiming improvement.
6. Run matched-generation comparisons with external renderers; only after that tune Muse native rendering against blind listeners.
7. Conduct a small musician pilot under appropriate ethics review, then preregister the confirmatory study.

## Explicit claim boundaries

- No single scalar music-intelligence, creativity or consciousness score.
- No SOTA claim from synthetic tests, one-shot examples, automatic judges or benchmark registry validation.
- No reciprocal collaboration claim from partner simulation alone.
- No claim of physical silence from queue invalidation or callback counters alone.
- No authorship or legal-rights determination from content hashes or provenance edges.
- No participant-specific persistent learning without a separate explicit purpose grant.

**Competitive thesis:** generators optimize audio outcomes; Melothaea should optimize the quality of the joint musical process—while producing excellent audio through the best qualified renderer and proving what was requested, authorized, changed, preserved and actually observed.
