# MEL-BENCH-003 — competitive research and qualification targets

**As of:** 21 November 2026  
**Programme:** [#10417 — reproducible external SOTA and industry comparison](https://github.com/Luminous-Dynamics/symthaea/issues/10417)  
**Capability ledger:** \`benchmarks/mel/external-capability-ledger-2026-11-21.json\`  
**Matched-run contract:** \`benchmarks/mel/schemas/matched-comparison-run-v1.schema.json\`  
**Current implementation increments:** [#13088](https://github.com/Luminous-Dynamics/symthaea/pull/13088), [#13100](https://github.com/Luminous-Dynamics/symthaea/pull/13100), [#13102](https://github.com/Luminous-Dynamics/symthaea/pull/13102)

## Executive research conclusion

The market has moved beyond prompt-to-song generation. Commercial systems expose production workflows, editable MIDI, live prompt steering, and real-time musical interaction. A credible Melothaea advantage should not be framed as "more controllable generation" in general.

The stronger thesis is:

> **Melothaea is a score-authoritative, event-sourced musical collaborator whose proposals carry causal, preservation, provenance and authorization evidence. Audio is a rendering of a versioned musical state, not the only state the artist can inspect or repair.**

This shifts the comparison from a single generated-audio quality number to independently qualified dimensions: audio quality, symbolic composition, exact edit correctness, preserved musical identity, reciprocal influence, timing, recovery, artist agency, provenance and cost.

No external result in this note is a Melothaea result. All Melothaea thresholds are prospective targets.

## 1. Current industry and research frontier

### 1.1 Audio generation preference

Artificial Analysis currently separates AA-Music-Instrumental v1.1 and AA-Music-Vocal v1.1. Its methodology uses blind pairwise votes from validated private evaluators, Bradley–Terry maximum-likelihood estimation rescaled to an Elo-like score, and 95% confidence intervals. A listener must hear at least ten seconds of each track. The pipeline normalizes loudness toward -16 LUFS integrated with a -1 dBTP boost ceiling and delivers 320 kbps MP3 at 44.1 kHz; its standard generation target is approximately three minutes, with seed 42 where exposed. October 2026 methodology changes improved uncertainty accounting across the full leaderboard.

**21 November 2026 snapshot:**

| Native task | Reference system | Elo | 95% interval |
|---|---|---:|---:|
| Instrumental | Suno v6 | 1,148 | 1,126–1,170 |
| Instrumental | Suno v6-mini | 1,114 | 1,093–1,135 |
| Instrumental | Mureka V9.5 | 1,084 | 1,063–1,105 |
| Instrumental | Google Lyria 3 Pro | 1,080 | 1,059–1,101 |
| Instrumental | Google Lyria 3.5 | 1,078 | 1,057–1,099 |
| Vocal | ACE Music v1 | 1,160 | 1,140–1,180 |
| Vocal | Suno v6 | 1,145 | 1,127–1,163 |

Sources:
- [AA-Music Instrumental v1.1 leaderboard](https://artificialanalysis.ai/music/leaderboard/instrumental)
- [AA-Music Vocal v1.1 leaderboard](https://artificialanalysis.ai/embed/music-generation-leaderboard/leaderboard/vocals)
- [Artificial Analysis music methodology](https://artificialanalysis.ai/methodology/music)

Do not compare instrumental and vocal Elo. These are separate cohorts and task definitions.

### 1.2 Symbolic music generation

MIREX 2026 reports these human ratings on a 0–5 scale:

| System | Coherence | Structure | Creativity | Musicality |
|---|---:|---:|---:|---:|
| DAMN (Nested Transformer) | 3.68 ± 0.16 | 3.39 ± 0.18 | 3.18 ± 0.16 | 3.47 ± 0.14 |
| PhraseSplitGen | 2.95 ± 0.18 | 2.74 ± 0.15 | 2.89 ± 0.15 | 2.61 ± 0.14 |
| MuseCoco (BL-1) | 2.79 ± 0.17 | 2.76 ± 0.15 | 3.42 ± 0.12 | 2.68 ± 0.16 |
| Anticipatory Music Transformer (BL-2) | 3.47 ± 0.20 | 3.34 ± 0.16 | 3.34 ± 0.16 | 3.61 ± 0.17 |

Important limitations: the study used eight prompts, 12 participants, and 38 ratings per system; one sample was selected from eight generated candidates for each prompt/system. It is a useful reference point, not a sufficiently broad estimate of general musical quality.

Source: [MIREX 2026 Symbolic Music Generation Results](https://futuremirex.com/mirex/wiki/2026%3ASymbolic_Music_Generation_Results).

### 1.3 Temporal controllability — a new independent frontier

TimingBench (August 2026) evaluates whether timestamped musical events occur when requested, rather than whether a track merely resembles its text prompt. It contains 250 prompts for fixed 30-second instrumental clips: 100 single-event, 100 timed-sequence, and 50 global-tempo prompts. The prompt set defines 399 timestamped events.

Published system results:

| System | Overall instruction success (95% CI) | Timed events within ±500 ms | Tempo accuracy | FAD |
|---|---:|---:|---:|---:|
| Taan | 96.1% (93.3–99.2) | 91.3% | 92.0% | 4.8 |
| ElevenLabs Music v2 | 28.9% (25.3–32.5) | 12.8% | 87.0% | 3.5 |
| Google Lyria | 23.9% (20.3–27.5) | 8.6% | 89.7% | 2.6 |
| ElevenLabs Music v1 | 25.4% (21.9–29.2) | 8.0% | 93.0% | 3.5 |
| Suno | 22.0% (18.7–25.6) | 6.8% | 86.7% | 3.1 |

This demonstrates that timing control, tempo accuracy and audio fidelity are independent axes. Taan leads the reported timing measures, while Google Lyria has the lowest reported FAD (lower is better). The published event-timing rate uses ±500 ms; it does not represent live MIDI latency or sub-100-ms beat-phase precision.

**Evidence caveat:** TimingBench is published by Taan, one of the evaluated systems. Its public paper reports a 128-event development/calibration validation sample with 4 false credits and 6 false positives, event-family localization floors ranging from single-digit milliseconds to 218 ms median, and an unfinished separate 300-clip annotation study at the evaluation cutoff. Treat the leaderboard as a vendor-published benchmark reference, not independently replicated truth. Reproduce its prompts, event taxonomy, tolerance and detector apparatus before asserting parity or superiority.

Source: [TimingBench paper and August 2026 results](https://taan.ai/research/rushing-or-dragging/).

### 1.4 Music editing: paired target is now the critical object

SMEBench (27 October 2026) introduces controlled music edits by applying known symbolic MIDI interventions and rendering matched source and target audio. It contains 11,160 paired examples, a balanced 3,600-example test set, twelve edit operations covering instrumentation, articulation, dynamics and pitch, and a separate 416-pair real-audio complement.

Its most important result is methodological: source preservation, target agreement and text compatibility are distinct, sometimes conflicting properties. A metric can rank systems differently depending on which property it rewards. Returning the unchanged source can score well on preservation while completely failing the requested edit. The authors also identify a synthetic-to-real gap from MIDI-derived rendering.

Source: [SMEBench — Benchmarking Neural Music Editing with Synthesized MIDI Edits](https://arxiv.org/abs/2610.36715).

This is a direct warning against using only source similarity, text/audio similarity, or audio quality as a music-edit success metric.

### 1.5 Production tooling is a meaningful competitor

Suno Studio 2.0 (13 August 2026) is documented as a browser-based production environment with MIDI import/record/edit, MIDI clips used as prompts for audio generation, chat-driven generation and plugin/synth design, audio effects, a wavetable synth, musical typing, and automation.

Sources:
- [Suno Studio 2.0 official help](https://help.suno.com/en/articles/13670529)
- [Suno Studio 2.0 release notes](https://suno.com/release-notes/studio-2)

The implication: merely adding MIDI import/export, a chat interface, effects, or a timeline is not a unique product thesis. Melothaea must distinguish itself through verified event-level transformations, stable identity and replay, preservation, causal agency and cross-renderer portability.

### 1.6 Live interaction is also an established research surface

Google Lyria RealTime exposes a persistent bidirectional WebSocket stream for continuous instrumental music generation and steering. Weighted prompts can be updated during playback; large changes may be abrupt. BPM/scale changes require resending the complete configuration and resetting context, which produces a hard transition. The documentation explicitly recommends robust buffering for network jitter and variable generation latency.

Source: [Google Lyria RealTime documentation](https://ai.google.dev/gemini-api/docs/realtime-music-generation).

JAM_BOT's NIME 2026 follow-up reports expressive velocity modelling, higher sequence throughput, free-form call-and-response across varying tempi, and external MIDI output latency compensation.

Source: [JAM_BOT — Enhancing Expressive Musical Conversation](https://nime.org/proc/nime2026_73/index.html).

The implication: "real-time" and "call-and-response" are not themselves novelty claims. A serious comparison must measure actual end-to-end response time, beat-phase error, tempo-following, recovery and the musical meaning of the response.

### 1.7 Reward-model judges — CMI-RewardBench (ICML 2026)

CMI-RewardBench is not another music generator leaderboard. It evaluates reward models that judge whether generated music is musical and follows compositional multimodal instructions using text, lyrics and/or reference audio. The paper introduces 110k pseudo-labelled preference samples and a separately human-annotated CMI-Pref corpus.

The official release reports selected musicality results:

| Reward model | MusicEval SRCC | Music Arena accuracy | CMI-Pref accuracy |
|---|---:|---:|---:|
| CMI-RM | 0.8266 | 73.43% | 78.20% |
| SongEval-RM | 0.6949 | 73.88% | 72.40% |
| Gemini 3 Pro | 0.6018 | 68.85% | 65.80% |

For compositional alignment, CMI-RM reports 67.6% (text), 72.8% (text+lyrics), 76.4% (text+audio), and 79.2% (text+lyrics+audio) on CMI-Pref; Gemini 3 Pro reports 67.2%, 60.8%, 68.8%, and 64.8% respectively. Gemini still leads CMI-RM on the reported PAM alignment SRCC (0.5373 vs 0.4321). Reward-model rankings therefore depend on which dataset and property are being judged.

**Implication for Melothaea:** use specialized reward models as independent, calibrated candidate-evaluation signals, not as the authority for a score mutation. Keep musicality, instruction adherence, formal theory, preserved material and artist preference separate. Pseudo-labelled pairs are not gold human preference labels, and no reward score can authorize a commit or override an explicit artist choice.

Sources:
- [ICML 2026 / PMLR paper](https://proceedings.mlr.press/v306/ma26am.html)
- [Official CMI-RewardBench release and result tables](https://github.com/Haiwen-Xia/CMI-RewardBench)

## 2. Paradigm shifts for Melothaea

### A. From audio-first generation to score-authoritative collaboration

Treat a versioned symbolic score/event graph as the authoritative creative state. Audio is one or more renders of that state. Every event has stable identity and ancestry. A generated audio segment without a resolvable score/event representation remains an opaque external asset; it must not be presented as if it had fine-grained event editability.

### B. From prompt instructions to a typed musical change calculus

Represent each edit as a typed transformation with:
- preconditions over the current score and plan;
- explicit target event IDs and scope;
- requested musical effect;
- protected complement and invariants;
- predicted effects separated from measured outcomes;
- a reversible candidate patch;
- an independent postcondition receipt.

Examples: \`TransposeRegion\`, \`RevoiceChord\`, \`RestoreMotif\`, \`ChangeArticulation\`, \`ThinTexture\`, \`ShiftAccent\`, \`ChangeOrchestration\`, \`AlterGroove\`. Each operation declares what it cannot change.

The benchmark should test these operations as transactions, not merely judge whether the audio sounds plausible.

### C. From one-shot selection to proof-carrying edit transactions

Use the sequence:

\`Intent → Authority → Candidate generation → Independent verification → Preview → Artist approval → Atomic commit → Render → Replay receipt\`.

Candidate generation is side-effect free. A preview is not an approval. No model prediction, confidence score or benchmark pass grants commit authority. An unknown outcome requires reconciliation before another mutation. A rejected candidate leaves the source score byte-identical.

### D. From "human in the loop" to measurable agency

Agency should be measurable, not described in UI prose:
- zero unauthorized commits;
- every committed change bound to an artist approval or explicit autonomy grant;
- 100% exact undo/redo to source state;
- preview never silently trained as an artist-preference label;
- model prediction, measured outcome and artist preference kept as separate data;
- explicit decline, abstain and "do not imitate this" pathways.

### E. From conversational response to causal reciprocity

A system is not reciprocal merely because it changes output after receiving input. It must pass:
- same-history control/treatment pairs;
- independently restored mutable state per arm;
- static model/checkpoint identity binding;
- order-reversal probes;
- yoked replay controls;
- unrelated-feature negative controls;
- imitation-only controls;
- blinded musical-relevance evaluation.

The snapshot-isolated RMI Track A work addresses the synthetic state-confound slice. Production model identity remains a separate requirement.

### F. From a single generator to a plural, replaceable ensemble

Separate the score authority, cognitive planner, deterministic transformation kernels, external audio generators, real-time performer models and renderers. Use capability declarations and adapters rather than making one provider the entire architecture. This allows the artist to choose a local symbolic engine, a commercial audio generator, a human collaborator or no model at all.

## 3. Benchmark programme and prospective targets

These are proposed targets, not achieved results. Freeze the exact cohort, sample size, scorer and statistical plan before confirmatory evaluation.

| Track | Benchmark / competitor | First gate | Stretch objective |
|---|---|---|---|
| Audio preference — instrumental | AA-Music Instrumental v1.1 | Matched prompts and normalization; non-inferiority to the current leading model on a shared eligible subset | Point estimate >1,185 Elo and lower 95% confidence bound above Suno v6's current 1,170 upper bound |
| Audio preference — vocal | AA-Music Vocal v1.1 | Separate vocal cohort and same blind protocol | Point estimate >1,200 Elo and lower 95% bound above ACE Music v1's current 1,180 upper bound |
| Symbolic generation | MIREX 2026 | Reproduce DAMN and Anticipatory baselines on the same prompts/scorer | Exceed 3.68 coherence and 3.61 musicality with a larger, preregistered panel and held-out prompts |
| Instruction-following edit | SMEBench | Reproduce official scorer on 12 operation families; report target, preservation and text compatibility separately | ≥95% target-operation precision/recall on a held-out cohort; 100% exact protected-event preservation on deterministic symbolic fixtures |
| Real-audio editing | SMEBench real-audio complement + new rights-cleared cohort | Report separately from MIDI-rendered synthetic examples | Expand to ≥2,000 paired real-audio edits with adjudicated target and preservation annotations |
| Score identity / preservation | MEL Score Transaction Bench | 100% stable event ID retention outside target; 100% exact rollback | Zero silent event replacement; every change has a machine-checkable patch and receipt |
| Causal reciprocity | RMI-Bench | 100% synthetic known-answer control discrimination; zero negative-control false positives | Real-policy randomized human study beats one-way/yoked controls on preregistered musical-relevance outcomes |
| Live improvisation | JAM_BOT, Lyria RealTime | Same device/clock/MIDI path; report system, transport and output latency separately | p95 decision latency ≤100 ms for a bounded local symbolic response; p95 beat-relative onset error ≤20 ms after calibrated output compensation |
| Temporal event control | TimingBench | Reproduce its 250 prompts, 399 events and detector taxonomy; report instruction success, event timing by tolerance, tempo and fidelity separately | 100% score-time event placement in symbolic state; ≥90% rendered events within ±100 ms for event families whose independently measured detector floor supports that tolerance; no timing credit when event identity is ambiguous |
| Recovery | MEL-TRG-Music | Recover from context reset, clock drift, dropped chunks and device reconnect without state corruption | 100% deterministic state reconciliation; no duplicated, backdated or silently lost committed events |
| Agency and rights | MEL-CREATIVE-001 | 100% commits have an explicit authority receipt; 100% exact undo | Zero unauthorized commits; zero export/publication without the required permission and provenance receipt |
| Cost and efficiency | Per-track/per-edit matched budget | Report cost, latency, failure rate and quality separately | Pareto improvement: lower cost/latency without degradation in target correctness, preservation or agency |

### Statistical discipline

- Use the benchmark's native scoring method before building a new metric.
- Do not use "best of N" cherry-picked generations unless every system receives the same candidate budget and the policy is declared.
- Retain all timeouts, errors, invalid outputs, abstentions and retries in the denominator.
- Report 95% confidence intervals, task/genre strata and panel composition.
- Set sample size using a pilot variance/power calculation; do not choose a small panel because it makes a positive result easier.
- Keep synthetic MIDI-rendered and real-audio results separate.
- Do not combine the table's metrics into a single composite score.

## 4. Immediate engineering work

1. **Complete hierarchical Sonata action control.** Formal obligations remain hard constraints; only supported cognitive subintents select among independently verified candidates. Add a same-source counterfactual with different valid score selections.
2. **Add score-diff receipts.** Stable event IDs, source score digest, target set, changed-event list, protected-complement digest, obligation receipts and exact rollback.
3. **Add SMEBench-compatible edit adapters.** Start with operations that can be represented exactly in the existing score model. Keep audio-only operations explicitly unsupported by the symbolic adapter.
4. **Bind immutable production model identity.** Checkpoint/weights, adapter, configuration, tokenizer/grammar, runtime and backend identity must be separate from mutable episode state.
5. **Add live performance clock calibration.** Record capture, inference, scheduling, MIDI transport, instrument and acoustic onset separately.
6. **Build a blinded musician cohort.** Test whether responses are contextually relevant and creatively helpful, not only statistically sensitive to a changed feature.
7. **Keep the external comparison registry current.** Snapshot provider/API versions, benchmark dates, rights receipts and model identity evidence before each run.

## 5. Reward-model evaluator qualification

CMI-RewardBench shows that no one evaluator is uniformly best across musicality, text adherence and multimodal compositional alignment. Melothaea should therefore qualify its evaluators as carefully as its generators.

### Evaluation protocol

1. Freeze a rights-cleared, human-annotated holdout; keep pseudo-labelled training/augmentation data outside the evaluation denominator.
2. Have at least three independently qualified listeners judge blinded, randomized candidate pairs. Collect musicality, instruction adherence, motif/structure coherence and preference as separate labels.
3. Evaluate CMI-RM, SongEval-RM, a general multimodal judge and the deterministic music-theory/preservation verifiers on exactly the same pairs.
4. Report pairwise accuracy, tie handling, Spearman/Kendall rank correlation, calibration (Brier/log loss or reliability curve), confidence intervals and subgroup results by genre, prompt type, duration, vocal/instrumental, edit operation and participant expertise.
5. Measure evaluator disagreement against human-human disagreement. A reward model that is highly confident on examples where trained musicians disagree should be calibrated down or abstain.
6. Test adversarial pairs: unchanged-source candidates, loudness-only changes, high-fidelity but instruction-incorrect audio, exact instruction-following but aesthetically weak audio, and candidates with protected-score violations.

### Governance invariant

Reward-model outputs are **advisory measurements**, never authority:
- a high reward score cannot bypass formal theory, obligation, preservation, rights or provenance checks;
- a high reward score cannot turn a preview into an approved edit;
- reward-model predictions are not artist-preference labels;
- only an explicit artist approval or previously granted autonomy policy can authorize commit;
- disagreements between judges remain visible, not collapsed into a single uncalibrated score.

### Prospective targets

| Metric | First gate | Stretch |
|---|---|---|
| Human pairwise preference agreement | Reproduce each evaluator's native CMI-RewardBench result on its official split | ≥85% on a larger rights-cleared Melothaea holdout, reported separately per dimension |
| Rank correlation with human preference | Report SRCC and Kendall τ with 95% confidence intervals | SRCC ≥0.80 on musicality and ≥0.75 on instruction adherence |
| Calibration | Reliability curves and Brier score; no hidden thresholding | Expected calibration error ≤0.05, with abstention under high disagreement |
| Subgroup robustness | No aggregate-only report | No subgroup gap >10 percentage points without a documented scope/uncertainty explanation |
| Authority safety | Reward score can never commit or widen scope | 100% of committed edits still require the independent authorization receipt |

These are prospective qualification objectives, not current results.

## 6. Stop conditions

A task must be marked unavailable or failed—not silently downgraded—when:
- a provider cannot expose the control surface required by the claimed metric;
- model identity is unknown but the run claims immutable replay;
- the benchmark split or media rights are unresolved;
- a source/target pair is missing for an edit-correctness claim;
- a candidate changes protected content;
- a timeout, invalid output or abstention is omitted;
- a human preference is inferred from a model prediction;
- a synthetic result is described as live human collaboration.

## Sources and dated context

- Artificial Analysis, [music benchmark methodology](https://artificialanalysis.ai/methodology/music), accessed 21 November 2026.
- Artificial Analysis, [instrumental leaderboard](https://artificialanalysis.ai/music/leaderboard/instrumental) and [vocal leaderboard](https://artificialanalysis.ai/embed/music-generation-leaderboard/leaderboard/vocals), accessed 21 November 2026.
- MIREX, [2026 Symbolic Music Generation Results](https://futuremirex.com/mirex/wiki/2026%3ASymbolic_Music_Generation_Results), read-only mirror snapshot 20 November 2026.
- Cervera et al., [SMEBench](https://arxiv.org/abs/2610.36715), 27 October 2026.
- Suno, [Studio 2.0 documentation](https://help.suno.com/en/articles/13670529), 13 August 2026.
- Google, [Lyria RealTime documentation](https://ai.google.dev/gemini-api/docs/realtime-music-generation), accessed 21 November 2026.
- Blanchard et al., [JAM_BOT NIME 2026 paper](https://nime.org/proc/nime2026_73/index.html), June 2026.
