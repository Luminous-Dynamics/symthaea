# Melothaea competitive frontier and dual-track strategy v1

**Date:** 5 November 2026  
**Status:** Research and evaluation strategy; targets are not results.  
**Parent:** [#10417 — Reproducible external SOTA and industry comparison ledger](https://github.com/Luminous-Dynamics/symthaea/issues/10417)  
**Composes:** #10307, #10390, #10393, #10402, #10409, #10088, #10190, #10133, #10249, #10421.

## Executive decision

Melothaea should compete on two distinct but connected tracks:

1. **Musical production:** create, render, arrange, edit and preserve music at a level that can be compared fairly with current production systems.
2. **Musical agency:** understand evolving musical situations, negotiate bounded contributions, preserve collaborator authority, recover from disruptions, and prove what actually occurred.

Do not collapse these tracks into one score. A system may be an excellent generator but a poor collaborator; a strong collaborator may use an external renderer whose raw generation score is higher than its own native renderer. Both facts should remain visible.

The product architecture should therefore be **generator-neutral**: Melothaea owns interpretation, forecasts, permission, semantic edit plans, interaction, recovery, provenance and evaluation. A renderer is an interchangeable, capability-qualified realization backend. Muse remains a candidate native renderer and must earn its own production-quality claims.

## 1. Competitive frontier: verified snapshot

### 1.1 Blind human preference

Artificial Analysis currently exposes separate AA-Music-Instrumental v1.1 and AA-Music-Vocal v1.1 leaderboards. Its methodology describes blind pairwise comparisons on the same prompt and modality, Bradley–Terry maximum-likelihood ratings rescaled to Elo, and 95% intervals. Audio is loudness-normalized to a -16 LUFS target (with a -1 dBTP ceiling) and delivered as 320 kbps MP3 at 44.1 kHz; the methodology also documents an October 2026 statistical update. This is a useful production-quality comparator, not a complete measure of controllability, musical structure, ownership, collaboration or rights.

**Instrumental v1.1 retrieved snapshot (5 November 2026):**

| Model | Elo | Reported 95% interval | Samples |
| --- | ---: | ---: |
| Suno v6 | 1148 | 1126–1170 | 2,208 |
| Suno v6-mini | 1114 | 1093–1135 | 2,332 |
| Mureka V9.5 | 1084 | 1063–1105 | 2,297 |
| Mureka V9 | 1083 | 1062–1104 | 2,322 |
| Lyria 3 Pro | 1080 | 1059–1101 | 2,329 |
| Lyria 3.5 | 1078 | 1057–1099 | 2,302 |

**Vocal v1.1 retrieved snapshot (5 November 2026):**

| Model | Elo | Reported 95% interval |
| --- | ---: | ---: |
| Suno v6 | 1144 | 1126–1162 | 3,037 |
| Suno v6-mini | 1132 | 1114–1150 | 3,151 |
| Mureka V9.5 | 1102 | 1084–1120 | 3,107 |
| Mureka V9 | 1063 | 1045–1081 | 3,097 |
| Lyria 3.5 | 1051 | 1033–1069 | 3,164 |
| Lyria 3 Pro | 1048 | 1030–1066 | 3,158 |

Treat these as dated snapshots, not timeless targets. Re-fetch the leaderboard and freeze its exact version/date before each comparison. The September 2 v1.0-era instrumental numbers in earlier notes must not be carried forward as the current baseline.

### 1.2 Editing and controllability

- **MMAE:** 2,000 public examples, 17,741 atomic rubrics, seven audio modalities, six task-complexity levels and multi-round editing. The paper reports current systems below 5% exact-match rate overall, with 0% on complex mixed-modality tasks. The official challenge ranks primarily by exact-match rate (EMR), then instruction-following rate (IFR), then consistency rate (CR). A plausible edit is not a successful edit if unrelated content is damaged.
- **SMEBench:** 3,600 controlled test examples covering 12 instrumentation, articulation, dynamics and pitch operations, plus a smaller real-audio multitrack complement. Matched source/target edits permit preservation, target agreement and textual compatibility to be measured separately. The authors report metric choice can change system rankings.
- **Suno Studio:** October 8 and October 29, 2026 updates include downbeat detection, clip stretching, shaped fades, pitch adjustment, clip controls and track management. These are current product capabilities. They mean basic clip editing is not a defensible differentiation by itself.

### 1.3 Temporal grounding and real-time participation

- **TAG-Bench:** 1,750 human-verified query/recording pairs over 149.5 hours; 22.1% have multiple ground-truth intervals. The leading reported model reaches 31.2 mIoU and 21.5% recall at IoU ≥ 0.7; reported one-to-many occurrence-count accuracy does not exceed 13.2%. Melothaea must enumerate all relevant occurrences and preserve parsing failures in denominators.
- **InteractionBench (2 November 2026):** evaluates timing, spoken-task accuracy, whisper/volume and long-conversation groundedness over the same seven real-time voice systems: 221 items per system, 13 scenarios and 1,519 graded runs. Timing is measured at the client-heard boundary; one server-clock correction materially changed a reported timing score. This is a speech/voice benchmark, not a music result. Its reusable lesson is to score the same input stream across timing, action correctness and grounding while keeping server, client, network and device clocks distinct. https://arxiv.org/abs/2611.02430
- **HEARTBEAT 2027:** as of the 5 November official-site snapshot, 132,571 samples (127,371 train / 5,200 development) across roughly 3,103 hours; it evaluates reciprocal audio turn-taking (WAIT/RESPOND/CONTINUE/YIELD) and streaming proactive understanding. Track I has separate offline and online settings. Online scoring covers behavior, response quality and speech onset/offset; the protocol requires only current/previously received input, per-chunk logging including no-response chunks, and no backdated response timestamps. This is a human-centered audio-agent challenge, not a music leaderboard.
- **JAM_BOT (NIME 2026):** already demonstrates relevant work on expressive velocity, higher throughput, free-form call-and-response across varying tempi and external MIDI latency compensation. These concepts are prior art; Melothaea's contribution must be their integration with permission, recovery and evidence.
- **Agents in Concert (IUI 2026):** a two-month co-design process with three jazz musicians culminated in a public performance with 97 attendees. It motivates performer-accessible controls and previews, but is not proof that preview alone improves agency in every context.

## 2. Important deadline and eligibility controls

### HEARTBEAT 2027

The dedicated challenge website snapshot retrieved on 5 November 2026 lists:

- 16 November 2026: development leaderboard opens.
- 27 November: team/resource declarations freeze.
- 11 December: development leaderboard closes.
- 18 December: final container submission.

The official site describes 132,571 samples (127,371 train / 5,200 development) across approximately 3,103 hours, with reciprocal audio turn-taking and streaming proactive-understanding tracks. Offline Track I asks for WAIT/RESPOND/CONTINUE/YIELD; Online Track I requires generated audio plus timestamped interaction logs. The online protocol requires current/past input only and logs every chunk, including no-response chunks, without backdating.

The 5 November direct read of the official dedicated site shows 16 November for opening. An earlier search snapshot showed 11 November; use the live dedicated site as operational authority and re-check before each milestone. Do not use the older ICASSP schedule as the live leaderboard calendar. Keep speech challenge results separate from music-adapted results; HEARTBEAT is an adjacent evaluation source, not a music leaderboard.

### MMAE

The official challenge site lists:
- 10 November 2026: hidden test/SDK release and leaderboard opening.
- 25 November: final submission/leaderboard freeze.
- 7 December: reproducibility checks complete.
- 8 December: final rankings and invited teams announced.

Registration was extended to 8 October. The Agent Track's specific model components and weights had to be publicly released before 1 November. Unless registration and exact-version eligibility can be demonstrated, report public-set reproduction only; do not imply official competition entry.

## 3. The paradigm shifts

### A. From generator-centric product to a generator-neutral agency layer

Generation model performance is a replaceable backend advantage. Melothaea should select among renderers based on declared capabilities and task requirements, then independently check the returned bytes and accepted controls. This means the system can compete in collaboration and editing even before Muse matches a proprietary generator's raw preference score.

Required capability manifest per backend:
- exact model/build/API version, license and permitted use;
- input/output modalities and duration/context constraints;
- supported controls and whether each is hard, best-effort or advisory;
- edit locality/preservation guarantees and required observations;
- seed, determinism, streaming and cancellation semantics;
- cost, latency, failure and retry accounting;
- output-byte digest and independent verification route.

Never label a prompt instruction as an exact control unless the backend provides evidence that it was enforced.

### B. From tracks and tokens to a multi-resolution musical world model

Keep event identity and temporal structure independent of renderer tokenization. Maintain linked views of:
- audio frame intervals and uncertainty;
- beat/tempo/meter and expressive deviation;
- notes, voices, motifs, harmony and phrase/function;
- performer/participant roles and authority;
- forecast, authorization and effect state;
- artifact lineage and protected regions.

Use explicit cross-view alignment receipts. A semantic claim must point to exact source intervals or score events; a score edit must point to its realized audio; a rendering change must not silently rewrite musical event identity.

### C. From edits to semantic transactions with preservation contracts

Treat a requested edit as a transaction:
1. identify the target region and requested musical property;
2. declare protected regions/properties;
3. produce a plan and preview before irreversible execution;
4. ask/accept/veto/branch as authorized;
5. realize the edit;
6. independently compare changed and preserved material;
7. commit, revise or roll back the artifact version.

The edit receipt should include exact source/target digests, region and event mappings, requested changes, preserved properties, renderer version and independent evaluator result. This is the practical bridge from SMEBench to trustworthy DAW workflows.

### D. From turn-taking to causal reciprocal interaction

Measure both directions of influence under controlled partner interventions:
- partner changes one relevant musical factor → agent changes its response appropriately;
- agent makes a contribution → partner's subsequent behavior changes in a measured, attributable way.

Compare fixed-response, imitation-only, one-way, yoked, wrong-feature and context-aware reciprocal controls. Use randomized/controlled interventions and participant/ensemble-clustered uncertainty. A correlation between notes is not causal influence. Reuse #10088 and #10210; do not create another generic interaction score.

### E. From happy-path latency to recovery-first reliability

Use MEL-TRG (#10390) to aggregate separate specialist receipts over the same interaction:
- timing, beat-relative error and deadlines;
- post-veto/interrupt/tempo-change/dropout recovery;
- protected-material preservation;
- actual effects at queue, callback, backend prediction and loopback boundaries;
- conditional multiparty appropriateness.

A fast wrong response is failure. A recovery that repeats or overwrites collaborator material is failure. A callback prediction is not physical observation. Keep metrics disaggregated and cohorts separate.

### F. From personalization to relational memory without assimilation

Reuse #10133. Store context-specific collaborator constraints, permissions, corrections, accepted/rejected proposals and musical conventions with attribution and revocation. Measure relationship-specific transfer on new material against session-only, generic-profile, full-history and yoked-memory baselines. Require zero cross-collaborator leakage and explicit revocation behavior. Do not make one collaborator's preferences Melothaea's universal musical identity.

### G. From dyadic turn-taking to polycentric ensemble agency

Reuse #10249. Model authority as an explicit, scoped relation, not loudness, recency, confidence or an internal Phi proxy. Test simultaneous invitations, conflicting instructions, participant-specific vetoes, role handoff, contributor attribution and unresolved lineage. Report per-participant agency and repair burden. Do not turn contributor influence into a default merit/authority weight.

### H. From generic global music to musician-governed cultural validity

Reuse #10399. Build separate cohorts under local musician governance, with explicit language/tradition/region representation, source rights and annotation authority. Do not claim an existing cross-cultural benchmark represents African traditions when its published scope does not. Preserve culturally meaningful disagreement and avoid a single flattened global preference score.

## 4. Benchmark portfolio and measurable targets

Targets below are qualification objectives, not current results. Run the published scorer and matched baseline first.

| Capability | Primary benchmark/owner | Initial target |
| --- | --- | --- |
| Raw instrumental preference | AA-Music-Instrumental v1.1 | Matched blind study; stretch: lower 95% bound above the current top model's upper bound, with version/date pinned |
| Raw vocal preference | AA-Music-Vocal v1.1 | Separate matched blind study; do not transfer instrumental result |
| Music editing | MMAE | Reproduce official IFR/CR/EMR; stretch: ≥10% overall EMR while retaining IFR and CR, with task-complexity strata |
| Localized musical editing | SMEBench | Reproduce all 12 operation families; target ≥90% requested-operation success and ≥95% protected-content preservation on deterministic controlled cohort; report real-audio complement separately |
| Temporal grounding | TAG-Bench | Reproduce FireRedAudio comparator; stretch ≥40 mIoU, ≥35% R@IoU≥0.7, ≥50% one-to-many count accuracy and ≤1% parser failure |
| Streaming turn-taking | HEARTBEAT Track I | Reproduce protocol; zero lookahead/backdating; separate offline/online and speech/music tracks |
| Proactive event grounding | HEARTBEAT Track II + #10265 | Separate precision, recall and joint F1; duplicate/missed/false responses remain explicit |
| Selective agency | #10402 | ≥90% paired act/withhold correctness on a musician-governed confirmatory cohort; show risk–coverage gain over confidence-only/local-cue baselines at matched coverage |
| Grounded authorization | #10409 / #10418 | 100% deterministic rejection of wrong-artifact, wrong-prefix, stale, yoked, expired and vetoed evidence |
| Veto/recovery | #10390 / #10382 / #10434 / #10448 | Zero post-fence old-epoch admission; 100% frame disposition accounting; zero authorization reuse after veto |
| Reciprocal influence | #10088 / #10210 | Positive intervention effect with uncertainty excluding zero on preregistered tasks; no effect on matched irrelevant interventions |
| Multiparty agency | #10249 | Zero accepted unauthorized veto bypass; per-participant agency, attribution and repair burden reported |
| Relational memory | #10133 | Zero cross-collaborator leakage; revocation prevents future retrieval/use in declared scope |
| Exact artifact/provenance | #10051 / #10421 | 100% pinned inputs, model versions, scorers, outputs, failure denominators and replay manifests |

The numeric targets on synthetic controlled cohorts must not be presented as real-musician or physical-device achievements. For human collaboration, determine non-inferiority margins and sample sizes from a blinded pilot, inter-musician agreement and participant/ensemble variance. For physical timing, report p50/p95/p99 only when the mapping is valid; distinguish decision time, queue admission, callback admission, predicted playback and loopback.

## 5. How to beat SOTA credibly

### 5.1 Beat the right baseline on the right task

Do not claim to beat Suno on generation because Melothaea has stronger causal interaction. Do not claim to beat an editing system because it has stronger preservation on unrelated symbolic fixtures. Use direct comparability labels:

- **Direct:** same benchmark version, split, protocol, modality, scorer, constraints and comparable inference budget.
- **Adapted:** musical adaptation of a non-music benchmark; report as a new track, not a native leaderboard score.
- **Diagnostic:** useful external reference, but not directly comparable.
- **Unverified:** version, rights, scorer or eligibility unresolved.

### 5.2 Use paired, adversarial, held-out evaluations

Every improvement should have:
- paired counterfactuals with one declared change;
- a negative control that should not change the action;
- a yoked/stale control;
- held-out performer, ensemble, composition, renderer and session partitions;
- explicit parse/failure/timeout/abstention denominators;
- uncertainty intervals at the correct clustering unit;
- independent annotation/oracle custody;
- clean-room replay against pinned inputs and evaluator.

### 5.3 Optimize a capability vector, not a composite score

Publish independent axes: production preference, instruction following, edit preservation, temporal grounding, timing, recovery, collaborator agency, reciprocal influence, cultural validity, provenance and operating cost. A high average must not hide a veto failure or a protected-content degradation.

### 5.4 Treat benchmark security as an engineering requirement

Pin benchmark release, split, scorer, model/API/weights, prompt, renderer, assets, toolchain, runtime and budget. Keep hidden labels out of the candidate environment. Store raw outputs and all failure dispositions. A digest binds bytes but does not prove the model saw only the permitted prefix; runtime isolation remains a separate assurance obligation.

## 6. Immediate execution plan

1. **MMAE public-set reproduction:** freeze available model versions/weights, official baseline, prompt and budget; verify rights and track eligibility; retain all 2,000 public-set attempts and exact rubrics/outputs where permitted.
2. **AA v1.1 baseline capture:** build a dated, matched prompt set for instrumental and vocal tracks; record the top models' exact snapshots; use blind randomization, level normalization and pairwise confidence intervals.
3. **SMEBench adapter:** map all 12 operations into separate target-agreement, preservation and text-compatibility metrics; record CC BY-NC-SA 4.0 and any third-party restrictions before use.
4. **HEARTBEAT protocol reproduction:** obtain the released data/baselines, separate speech and music adaptation, implement current-prefix-only trace accounting, and re-check the dedicated challenge site immediately before each milestone.
5. **Veto race reducer:** run the new dependency-free reference reducer and its exact-head workflow; then integrate its event semantics with #10382 and the live epoch/clock branch only after branch reconciliation.
6. **Waveform counterfactuals:** render #10393 fixtures through two pinned renderers, hash exact local-cue crops, and obtain independent musician labels.
7. **Human pilot design:** preregister authority, agency, interruption cost, recovery, preservation and cultural cohort measures before recruiting or collecting recordings.

## 7. Claim boundary

This programme does not establish consciousness, subjective experience, universal artistic value, legal authorship or human-equivalent musical understanding. It qualifies observable musical behavior, operational control and evidence. A leaderboard win is one narrow result. A trustworthy collaborator must also be steerable, revocable within stated boundaries, culturally accountable, recoverable and independently auditable.
