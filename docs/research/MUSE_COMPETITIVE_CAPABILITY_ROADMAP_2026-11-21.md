# Muse competitive capability roadmap — 21 November 2026

## Executive thesis

Muse should not try to win every part of music creation by training one larger audio generator. The stronger strategy is to become the **composition, collaboration, and evidence layer that can direct many renderers and generation engines**.

The product distinction:

- Foundation audio engines produce sound.
- A DAW arranges, records and processes tracks.
- Muse should understand the user's musical intent, maintain the identity and relationships of musical material, negotiate alternatives with collaborators, plan reversible changes, direct whichever renderer is best for the task, and verify what actually changed.

This is a proposed strategy, not a claim that all these capabilities are already integrated or qualified.

## 1. External landscape

Snapshot date: 21 November 2026. Leaderboards and model/API availability change; the figures below are dated observations, not permanent rankings.

| System or benchmark | Demonstrated capability / reference | Implication for Muse |
|---|---|---|
| Suno Studio 2.0 | A production environment with chat-driven audio and MIDI generation, MIDI recording/editing, timeline arrangement, automation, effects, synths, custom plugins and stem separation | We cannot define Muse as merely a chat interface around a music model or claim a better DAW without equivalent workflow coverage |
| Google Lyria RealTime | Experimental persistent bidirectional WebSocket music stream; weighted-prompt steering and real-time configuration for BPM, density, brightness, scale and other controls; 48 kHz stereo 16-bit PCM output | A serious real-time generation service to adapt to and benchmark against; its exposed controls are not equivalent to persistent note/event identity or verified score edits |
| MIT / NIME jam_bot (2026) | Expressive symbolic human–AI conversation; models velocity, supports free-form call-and-response across varying tempi, improves throughput and compensates external MIDI output latency | The closest research comparator for live symbolic improvisation; Muse must measure latency, expressive adaptation, phase error and real performer outcomes |
| YuE2 (September 2026) | Symbolic planning produces a readable melody/chord score before full-song audio realization; also advertises zero-shot covers and conversational score/arrangement editing | Score-first generation is now an industry direction, not a unique differentiator. Muse must go beyond a readable plan to stable identity, typed changes, preservation proofs, multi-agent interaction and reversible edits |
| Google MAGE (2026) | Modality-agnostic music generation and mixture-grounded editing with cross-modal temporal alignment and dynamic modality masking | Benchmark multimodal grounding and targeted source edits; do not assume a symbolic representation alone wins on audio-mixture tasks |
| SMEBench (October 2026) | 11,160 paired source/reference-edited audio examples, twelve edit operations, a 3,600-example balanced test set and a 416-pair real-audio complement | A strong external editing benchmark; report preservation, target agreement and text compatibility separately |
| MIREX 2026 symbolic generation | DAMN reported coherence 3.68, structure 3.39, creativity 3.18 and musicality 3.47; Anticipatory Music Transformer reported musicality 3.61 | A useful comparison panel, but its subjective study was small (8 prompts, 12 participants, 38 ratings per model) and selected samples from eight candidates; reproduce with a larger preregistered study |
| Artificial Analysis AA-Music v1.1 | The observed instrumental board lists Suno v6 at 1,148 ±22 Elo; the separate vocal board lists ACE Music v1 at 1,160 ±20 and Suno v6 at 1,145 ±18 | Raw audio preference is a separate competitive race. Do not compare a Muse symbolic-collaboration score to these Elo ratings |

Sources:
- Suno Studio 2.0: https://help.suno.com/en/articles/13670529
- Lyria RealTime: https://ai.google.dev/gemini-api/docs/realtime-music-generation
- jam_bot / NIME 2026: https://nime.org/proc/nime2026_73/index.html
- YuE2: https://arxiv.org/abs/2609.33757 and https://github.com/hzwangming/YuE2
- MAGE: https://arxiv.org/abs/2604.09803
- SMEBench: https://arxiv.org/abs/2610.36715
- MIREX symbolic generation: https://futuremirex.com/mirex/wiki/2026%3ASymbolic_Music_Generation_Results
- AA instrumental: https://artificialanalysis.ai/music/leaderboard/instrumental
- AA vocal: https://artificialanalysis.ai/embed/music-generation-leaderboard/leaderboard/vocals

### Important qualification

The MIREX and AA boards answer different questions. MIREX symbolic continuation uses a constrained score-generation task and human ratings. AA Music Arena uses blind pairwise preference for generated audio. Neither evaluates authorship, long-term adaptation, causal reciprocity, exact editing, or whether the system respected a musician's veto. Report the benchmark's task and sample design with every score.

## 2. Paradigm shifts

### A. From generator to renderer-independent musical director

Keep the symbolic composition and interaction state as the stable source of truth. Audio generators, MIDI renderers, DAWs and synthesis engines become replaceable realization backends.

A single intent may yield:
- an editable score and event plan;
- multiple renderer-specific realizations;
- a human-selectable branch;
- exact provenance from source material through output audio.

This lets Muse benefit from frontier audio engines without surrendering score-level control or tying the product to one provider.

### B. From genre labels to time-varying compositional grammar

Represent style as a trajectory over musical time, not a static prompt label. The structural owner, rhythm, harmony/pitch language, texture, ensemble and performance idiom can transition at different declared boundaries. A bridge must be composed and checked; an audio crossfade is not proof of a symbolic transition.

Related programme: [#12643 — time-varying style trajectories](https://github.com/Luminous-Dynamics/symthaea/issues/12643), with #7417, #9505, #9397 and #11284.

### C. From one “correct” answer to negotiated creative possibility

Music often has multiple valid continuations. Preserve a set of admissible alternatives until the collaborator chooses, explicitly asks for ranking, or authorizes exploration. Veto, abstention, clarification and branching are first-class musical actions—not failures to produce a note.

Related work: [negotiated response contract](https://github.com/Luminous-Dynamics/symthaea/blob/main/docs/research/MEL_RMI_NEGOTIATED_RESPONSE_CONTRACT_V1_2026-11-05.md) and [#11319 — co-creative agency, consent and authorship](https://github.com/Luminous-Dynamics/symthaea/issues/11319).

### D. From plausible call-and-response to causal reciprocity

A good-sounding reply may ignore the actual previous phrase, replay a cached result, or imitate a surface feature. Use paired counterfactual interventions and yoked controls to test both directions:
1. Does the partner's changed contribution alter Muse's next action?
2. Does Muse's changed action alter the partner's next contribution?
3. Is the response musically relevant, and does it preserve the collaborator's stated intent?

The first two are causal interaction questions; the third requires musical and human evidence. Never collapse them.

Related work: [#10088 — RMI-Bench](https://github.com/Luminous-Dynamics/symthaea/issues/10088), [#10159 — offline Track A runner](https://github.com/Luminous-Dynamics/symthaea/issues/10159), [#10307 — streaming musical turn-taking](https://github.com/Luminous-Dynamics/symthaea/issues/10307).

### E. From “the sound played” to grounded, clocked outcomes

Separate request, plan, scheduled event, audio callback, predicted output frame and independently observed acoustic/electrical output. Timing is a typed measurement with clock domain and uncertainty—not a generic latency number. Recovery after interruption matters as much as clean-run timing.

Related work: [#10390 — timing, recovery and grounded outcomes](https://github.com/Luminous-Dynamics/symthaea/issues/10390), CPAL clock bridge work, and [#12341 — byte-bound WAV continuity](https://github.com/Luminous-Dynamics/symthaea/issues/12341).

## 3. Competitive benchmark portfolio

Do not create one composite “Muse score”. A high musicality score must not conceal unauthorized edits, poor timing or weak collaboration. Keep these tracks independent.

| Track | External comparison | Metrics to publish | Prospective target / gate |
|---|---|---|---|
| Raw audio quality | AA-Music instrumental and vocal; matched prompts against Suno, Mureka, Lyria, ACE and other accessible systems | Blind pairwise Elo, preference by style/task, production quality, failure rate, cost and generation latency | First reach statistical non-inferiority on a declared subset; only then pursue a top-quartile or top-ranked position. No result until a matched blind study exists |
| Symbolic generation | MIREX 2026 DAMN and Anticipatory Music Transformer; fixed prompts and score lengths | Coherence, structure, creativity, musicality, constraint adherence, long-range form, voice-leading and replayable score validity | Beat the published reference means on held-out prompts with confidence intervals and a larger preregistered panel; show no cherry-picked best-of-N advantage |
| Score-native editing | SMEBench and other paired edit benchmarks; manual DAW baseline; model-based editing baseline | Target occurrence precision/recall/F1, operation and parameter exact match, boundary error, protected-event preservation, source drift, undo/replay, human acceptance | 100% preservation and selector correctness on deterministic controls; ≥95% target-occurrence F1 and ≥95% exact operation/parameter match on a musician-adjudicated held-out cohort; 0 unauthorized committed edits |
| Reciprocal interaction | RMI-Bench fixed, one-way, yoked and imitation controls; live agent adapters | Partner→Muse sensitivity; Muse→partner sensitivity; target specificity; collateral changes; causal effect under random intervention; response relevance; alternative diversity | 100% encoded positive/negative control discrimination; live Muse must outperform yoked/replay and one-way controls on preregistered causal measures before any reciprocity claim |
| Live ensemble behavior | jam_bot, Lyria RealTime, HEARTBEAT where task-compatible | p50/p95/p99 input-to-action and action-to-output latency; beat/phase error; WAIT/RESPOND/CONTINUE/YIELD F1; false entries; missed invitations; duplicate output; clock uncertainty | Zero lookahead/backdating; zero output after valid veto/stop; report device/tempo-specific deadline performance and beat-relative error. Establish numerical latency limits through instrument/performer pilots, not a universal guessed threshold |
| Recovery and resilience | Controlled interruption, tempo/meter shifts, output underruns, stream restart and stale-context controls | Time to stable phase; successful recovery; false recovery; duplicate material; repair burden; preservation of accepted commitments | Better than reset, local-cue-only and yoked-context baselines without degrading preservation or agency |
| Human agency and authorship | Manual editor; passive suggestion tool; unconstrained agent | Approval correctness, veto respected, protected material mutation, edit provenance, undo success, choice diversity, ownership and workload | 0 unauthorized committed changes; 100% of commits have approval or explicit bounded-autonomy authorization; 100% supported undo/replay |
| Musical understanding | Audio/MIDI temporal-grounding, score-observation and event-matching datasets | Event and part precision/recall, onset/offset error, beat/bar/section grounding, calibration, abstention risk/coverage | ≥95% event/operation agreement on a preregistered rights-cleared cohort, with deterministic controls exact; report unsupported/ambiguous cases separately |
| Cultural and stylistic generalization | Musician-governed cohorts and held-out traditions, ensembles, tunings and idioms | Per-cohort musical validity, instrument idiom, phrase/form convention, tuning support, disagreement and false-authenticity rate | No aggregate score may hide a failed cohort; unsupported mappings must be disclosed or abstained from |

The target values above are proposed gates, not achieved results. Sample sizes and confidence intervals must be selected from pilot variance and preregistered power analysis.

## 4. How Muse can outperform without training a frontier audio foundation model

1. **Use the best renderer for the task.** Build capability-qualified adapters for local synthesis, MIDI/DAW workflows and external audio engines. Compare output quality fairly; never pretend an unrendered symbolic score is a finished production.
2. **Make the score and relationship graph authoritative.** Stable PartId/material/relationship identity, explicit motif lineage, rational beat coordinates, typed transformations and protected regions create capabilities that are difficult to recover from a finished waveform.
3. **Plan before rendering.** Compile a natural-language request into a typed, evidence-backed edit plan. Resolve occurrence ambiguity, units, target scope and protected content before mutation; preview alternatives and preserve reversibility.
4. **Use counterfactual rehearsal.** Before accepting a phrase, arrangement or edit, evaluate several admissible branches against musical obligations, performer intent, instrumentation limits, style transitions, timing and preservation constraints. Present a Pareto set rather than hiding trade-offs in one reward.
5. **Learn collaborator-specific conventions only with authorization.** Keep per-session state distinct from persistent memory, expose what is stored, and allow reset/forget/branch. Do not infer blanket consent from participation.
6. **Treat silence, wait, clarification and abstention as skilled actions.** Do not force Muse to fill every gap or collapse ambiguity simply to maximize response rate.
7. **Qualify the physical path.** The output clock, MIDI interface, audio callback and acoustic/electrical loopback all matter. Estimate per-path latency and uncertainty, compensate per note/velocity where supported, invalidate stale queues and prove recovery.
8. **Build for pluralism.** Keep alternative harmonizations, voicings, forms and roles available. Evaluate local musical validity with practitioners of the relevant traditions rather than imposing one universal theory.

## 5. Near-term execution sequence

### Now — independent of remote runners
- Continue the RMI Track A implementation in [PR #13070](https://github.com/Luminous-Dynamics/symthaea/pull/13070); do not treat its synthetic output as a live Melothaea result.
- Connect existing score-native edit plans and their preservation verifier to the real planner, not just the schema.
- Extend the current acoustic-observer calibration work into held-out renderer and source-lineage splits.
- Add paired counterfactual examples to the style-trajectory and negotiated-response programmes.
- Freeze an external baseline inventory by version, API/model identity, date, prompt set, compute budget and output license.

### Next — first full Muse-vs-baseline experiment
Use one shared rights-cleared symbolic task set with:
- manual editor / DAW operation;
- one-way prompt baseline;
- imitation/copy baseline;
- yoked replay;
- current Muse planner;
- Muse planner plus evidence/branching/approval layer.

Render the same accepted scores with the same renderer and mix settings. Run blinded evaluation only after exact edit, preservation and provenance checks pass. Separate the human-facing preference test from the correctness test.

### Then — live performer pilot
Begin with two-person MIDI sessions, then acoustic instruments and multiparty ensembles. Randomize intervention conditions, counterbalance order, preserve every no-response chunk, and measure the complete clock chain. Obtain informed consent and collect only data the protocol explicitly authorizes.

## 6. Decision rule

Muse is competitive when it can show, on matched tasks and independent evidence, that it:
- creates outputs musicians value;
- makes requested changes more precisely than baselines;
- preserves everything outside the authorized edit scope;
- adapts causally to the actual collaborator rather than replaying or imitating;
- meets timing and recovers from disruptions;
- leaves humans in control of authorship, veto, approval and export;
- exposes uncertainty and unsupported capabilities.

A failure on any one axis should remain visible. Do not hide it behind a composite score, an attractive demo, or a model leaderboard.
