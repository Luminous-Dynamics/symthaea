# MUSE-PROCESS-REALIZATION-AUDIT-001 — Additive Process Score Verification (v0.1)

Date: 2026-11-18  
Status: source increment authored; compile/test/CI evidence pending  
Repository: `Luminous-Dynamics/symthaea`

## Research question

When a ProcessAdditive composer declares a sequence of growing and shrinking cell prefixes, does the realized symbolic score actually contain those prefixes?

The existing planner emits `ProcessPlan { trajectory_kind, prefix_lengths, cycle_starts }`. The existing grammar trace labels regions (for example, `additive-prefix-3`), but those labels are assertions made by the producer. They are not an independent test that the notes in each region match the declared process. Plan-only tests can therefore pass while a realized score is corrupted.

This tranche closes that narrow gap without changing the grammar's output format, the generic obligation ledger, the renderer, or the audio pipeline.

## Implemented contract

`verify_additive_process_score_v1(plan, score)` is a fail-closed score-side verifier. It:

1. Requires a non-empty prefix plan and a one-to-one correspondence between prefix lengths and cycle starts.
2. Requires the first cycle to start at exact rational beat zero, positive prefix lengths, strictly increasing cycle starts, and positive half-open cycle windows.
3. Independently derives the canonical prefix contour for the declared `ProcessTrajectoryKind` and maximum prefix length. It does not call the planner's contour construction.
4. Assigns every melody event to exactly one planned cycle using rational beat comparisons, not floating-point tolerance.
5. Rejects melody events outside the trajectory, events crossing a cycle boundary, an incorrect cycle event count, and a cycle whose first event does not begin at its declared start.
6. Checks that consecutive melody events are contiguous and that the final melody event ends exactly at `Score.total_beats`.
7. Requires the normalized process cell carried in ProcessPlan and checks every realized pitch and duration against that source cell's scale degree and exact duration.

The return value is a small versioned report: cycle count, observed melody event count, and maximum prefix length. The error enum reports specific rejection causes.

## Versioned expressive-envelope verification

The current generator writes a deterministic note-level envelope for each process cycle:

- velocity: `0.46 + 0.24 × prefix_length / cell_length`;
- section intensity: `0.42 + 0.44 × prefix_length / cell_length`;
- `PhraseStart` emphasis only on the first event of the first cycle; all other process melody events are `Normal`.

ProcessPlan now declares `ProcessDynamicsProfileV1::PrefixLinear`. The independent audit recomputes both f32 values using the generator's exact operation order and compares their bit patterns. It checks emphasis separately and counts all successfully checked expressive events in the serialized audit report. A legacy plan missing the profile is still readable, but returns `MissingDynamicsProfile` and is not expressively qualified.

Frozen reference values for normalized prefixes 1/4, 2/4, 3/4 and 4/4 are:

| Prefix ratio | Velocity | Section intensity |
|---|---:|---:|
| 1/4 | 0.52 | 0.53 |
| 2/4 | 0.58 | 0.64 |
| 3/4 | 0.64 | 0.75 |
| 4/4 | 0.70 | 0.86 |

The audit only verifies the current melody envelope. It does not verify the separate constant accompaniment layers (harmony/bass), acoustic loudness after rendering, perceptual crescendi, expressive timing, or whether the linear envelope is artistically appropriate. The purpose is to prove adherence to an explicit contract, not to claim that this contract is musically optimal.

## Accompaniment and identity verification

The score-side audit now covers all three roles emitted by the current ProcessAdditive generator and rejects any extra role:

- **Harmony clock:** the exact repeating scale-degree pattern 1–5–8–5 in octave 4, with eighth-note timing clipped only at the terminal score boundary; velocity 0.34, section intensity 0.62, Normal emphasis.
- **Bass anchor:** scale degree 1 in octave 2, one score-meter bar per event with a terminal clipped duration if necessary; velocity 0.43, section intensity 0.58, Normal emphasis.
- **Identity boundary:** all current melody, harmony and bass events must retain `PartId::UNASSIGNED`. The verifier rejects invented part IDs rather than reconstructing identity from role, register or ordering.
- **Role boundary:** only Melody, Harmony and Bass are currently accepted. Extra CounterMelody/other roles fail closed until a new ProcessAdditive contract is defined.

The audit report includes the exact number of verified harmony and bass events alongside melody and expressive counts. These checks establish adherence to the current score contract; they do not assert that a tonic pedal or repeating fifth-clock is harmonically sophisticated, that it is the best accompaniment, or that it renders/performs perceptually well.

## Evidence boundary

This establishes symbolic score-to-plan realization only. It does **not** establish:

- that the process is audible or perceptually recognized as additive;
- that velocity and section-intensity dynamics match an expressive profile (not checked by v1);
- that accompaniment, timbre, orchestration, or renderer output is correct;
- that the Minimalism style is culturally authentic, artistically valuable, or preferred;
- persistent part identity (current ProcessAdditive notes use `PartId::UNASSIGNED`);
- that the score is superior to commercial audio generators.

In particular, event-count growth is a structural fact, not a listener-experience result. Audio/perceptual evaluation remains a separate scorecard.

## Adversarial fixtures authored

Twenty-two new tests are present in `process_grammar.rs`:

1. Accept unmodified realizations for all four declared trajectory kinds.
2. Match hand-frozen expected contour vectors for all four process kinds.
3. Reject a plan with a false cycle count / contour.
4. Reject a duration mutation inside a cycle.
5. Reject a pitch mutation that breaks a repeated cell prefix.
6. Reject a coherent whole-cell transposition against the plan-carried process cell.
7. Deserialize a legacy plan without the optional cell field, then refuse source-cell verification with a typed error.
8. Round-trip the versioned audit report and typed failure through JSON.
9. Reject a shifted cycle boundary.
10. Reject a score with a missing melody event.
11. Exercise a deterministic 4,096-case holdout: four process kinds × 256 seeds per kind × 2, 4, 8 and 16 bars.
12. Match frozen prefix-linear velocity and section-intensity reference values.
13. Reject velocity, section-intensity and phrase-start emphasis mutations.
14. Deserialize a legacy plan without the dynamics profile, then refuse expressive verification.\n15. Reject a source cell containing a zero-duration event.
16. Reject mutations to the harmony clock pitch and bass-anchor duration.
17. Reject assigned identities on accompaniment or melody events.
18. Reject unplanned voice roles.
19. Reject source cells containing non-positive event duration.\n20. Exercise the 4,096-case generated holdout after all melody, expressive and accompaniment checks.

Existing process tests remain in place. No test has been compiled or executed in the current environment. Do not describe these tests as passing.

## Benchmark plan

Report separate measurements; do not collapse them into a single music-quality score. Include expressive-envelope correctness as a distinct symbolic score-integrity dimension.

| Dimension | Metric | Qualification target |
|---|---|---|
| Deterministic positive fixtures | Exact plan-to-score audit acceptance | 100% |
| Frozen adversarial fixtures | Invalid score rejection | 100%; zero false semantic accepts |\n| Expressive envelope | Bit-exact note-level velocity/intensity/emphasis profile adherence | 100% for the declared profile; unsupported/legacy profiles explicitly rejected |\n| Accompaniment contract | Harmony-clock and bass-anchor event correctness, role scope and unassigned identity | 100% deterministic fixtures; zero invented part identities accepted |
| Generated holdout cohort | Exact adherence across trajectory kinds, seeds, and bar counts | 100% for the pinned deterministic generator |
| Mutation robustness | Detection rate by mutation class, plus false rejection rate | 100% detection on the frozen matrix; report every class separately |
| Coverage | Accepted examples / all examples; abstentions by reason | Report explicitly; never hide abstentions in accuracy |
| Runtime | p50 / p95 / p99 verifier latency, by event count | Establish measured baseline before target setting |
| Perceptual validity | Blind listening recognition of growth/release and musical preference | Separate preregistered human study; no score-audit proxy |

The authored deterministic holdout currently uses all four process kinds × 256 seeds per kind × 2, 4, 8 and 16 input bars (4,096 examples), with engine revision pinned. This is a test definition, not an executed result; its acceptance rate remains unknown until Rust tests run. Then add mutation families: missing/extra notes, pitch substitution, duration change, wrong cycle start, wrong contour declaration, gap, overlap, cross-boundary note, and trailing/leading events. Preserve each case's plan digest, score digest, expected disposition, actual disposition and verifier version.

## Industry comparison and differentiation

Commercial music systems compete heavily on generated-audio quality, prompt adherence, iteration speed, editing experience and listener preference. This score-side verifier does not compete on those dimensions and must not be presented as doing so.

The differentiated proposition is **verifiable musical control**: a generator can make a formal promise, then an independent score-side component can reject a score that violates it. This is a complement to high-quality audio generation, not a replacement for it. A credible SOTA claim would require matched blind audio evaluations, clearly defined tasks and confidence intervals in addition to symbolic correctness.

## Paradigm shifts this increment supports

- **From plan assertions to proof-carrying scores:** a plan is not its own evidence; the realized events must independently satisfy it.
- **From global genre tags to time-indexed process:** growth, plateau, release and transition become explicit, testable temporal structures.
- **From scalar quality to a Pareto scorecard:** symbolic fidelity, editing control, identity, interaction, perceptual preference and runtime remain distinct.
- **From silent correction to fail-closed realization:** a mismatch rejects with a typed reason; the verifier does not silently retime, rewrite or reinterpret the performance.
- **From benchmark anecdotes to reproducible cohorts:** freeze inputs, seeds, engine revisions, mutation fixtures and exact outputs.

## Integration path

The next M1 integration should invoke this verifier on the Minimalism segment and retain its report beside the existing trace-integrity report. The verifier should remain narrowly scoped to ProcessAdditive until another grammar family has its own independent semantic contract. The MUSE Analyst typed-obligation work in #9797/#9835 remains the owner for general compositional obligations; this audit does not create a parallel obligation protocol.

## Source process-cell binding and remaining provenance boundary

The first draft used the maximum-prefix cycle observed in the score as the cell template. That checked internal repetition but could not detect a coherent transformation applied to every cycle. The current implementation improves this: ProcessPlan now carries an optional, serde-defaulted `process_cell` containing the exact post-filter/fallback Motif used by the generator. The verifier requires this field, checks that its length equals the maximum prefix, rejects rests in the normalized cell, and checks every realized pitch and duration against the corresponding source-cell event.

A test now requires a coherent whole-cell transposition to be rejected. A second test removes the new field from serialized JSON and confirms legacy ProcessPlan data still deserializes, while the source-cell audit returns `MissingProcessCellBinding` rather than silently upgrading legacy evidence.

This is **source process-cell consistency**, not complete provenance: the cell is a plan field, not a cryptographically authenticated claim about a musician-authored motif. A caller needing tamper evidence must pin the complete plan, score, input intent/spec and engine revision in an immutable receipt. The current generator's cell is derived from the selected CompositionSpec and frozen intent; it is not proof of preservation of some separate user-authored motif that was never supplied to this API. The audit also rejects source-cell events with non-positive duration.

## November 2026 external benchmark and product context

The competitive landscape reinforces why this is a control-verification increment, not an audio-quality claim.

### Audio-generation preference

Artificial Analysis currently publishes separate instrumental and vocal v1.1 boards. Its method uses blind same-prompt, same-modality pairwise preference, at least ten seconds of listening per track, Bradley–Terry estimation and 95% confidence intervals; tracks are statically loudness-normalized to approximately -16 LUFS and capped at -1 dBTP before 320 kbps MP3 delivery.

Its current displayed instrumental board lists Suno v6 at 1148±22, Suno v6-mini at 1114±21, Mureka V9.5 at 1084±21, Mureka V9 at 1083±21, Lyria 3 Pro at 1080±21 and Lyria 3.5 at 1078±21. Its vocal board lists ACE Music v1 at 1160±21, Suno v6 at 1144±18, Suno v6-mini at 1132±18 and Mureka V9.5 at 1101±18. These are arena-specific preference estimates, not symbolic-control measurements; overlapping confidence intervals should not be presented as definitive rank separations.

Sources: [Artificial Analysis methodology](https://artificialanalysis.ai/methodology/music), [instrumental leaderboard](https://artificialanalysis.ai/music/leaderboard/instrumental), [vocal leaderboard](https://artificialanalysis.ai/music/leaderboard/vocals).

### Product control and editing

Suno Studio 2.0 now offers a browser-based production workflow with MIDI import/record/edit, piano roll, MIDI-to-audio workflows, automation, built-in effects and synths, and chat-driven plugin creation. Lyria RealTime exposes moment-by-moment steering, blended text prompts and direct key, tempo, density and brightness controls in a continuous 48 kHz stereo stream.

Muse should not claim that “controllability” alone is unique. The differentiator must be independently measured control correctness: whether a requested edit changes the intended musical variables, preserves protected material, produces a replayable event diff, and can be reversed.

Sources: [Suno Studio 2.0](https://help.suno.com/en/articles/13670529), [Google DeepMind Lyria RealTime](https://deepmind.google/models/lyria/lyria-realtime/).

### Symbolic long-range structure

CAST (Scientific Reports, 14 April 2026) is a direct comparison point. It separates macro-harmonic skeleton planning from micro-texture generation using MusicBERT and MuseFormer, reporting structural error 0.22 versus 0.58 for its baseline and chord accuracy of 96% on sequences up to 1,000 tokens. The paper's stated limitations are important: its evaluated setup is single-track piano and chord-progression-driven; complex counterpoint, atonality and multi-track timbral orchestration remain future directions.

This means a skeleton/texture split is not, by itself, a novel Muse claim. Muse's research bar should be higher: a typed multi-grammar plan, exact event identity, score-side independent verification, protected-edit semantics, and reproducible evidence receipts on multi-part tasks. These are target capabilities, not demonstrated comparative wins.

Sources: [CAST paper](https://www.nature.com/articles/s41598-026-46750-0), [CAST limitations](https://www.nature.com/articles/s41598-026-46750-0).

### Symbolic benchmark design

LilyBench (June 2026) jointly evaluates symbolic generation and understanding with a 200-prompt generation suite and ten understanding tasks. It triangulates compile rate, MusPy descriptor distributions (Jensen–Shannon similarity), and LilyBERT Fréchet Music Distance, and reports disagreement between descriptor and embedding metrics. Muse should use benchmark triangulation: a file compiling is not proof of form, a descriptor match is not proof of motif identity, and a preference vote is not proof of control fidelity.

Source: [LilyBench paper and benchmark](https://arxiv.org/abs/2606.08722).

## Recommended next benchmark programme

Keep five scorecards independent:

1. **Symbolic plan fidelity:** exact constraints and grammar obligations accepted/rejected, with adversarial mutation detection and false-accept/false-reject counts.
2. **Revision safety:** event-level correspondence F1; protected-event retention; edit locality; exact undo/redo; no unrequested mutation.
3. **Causal control:** same seed and frozen inputs, one declared control varied at a time; effect size on intended measures plus invariance on unrelated measures.
4. **Perceptual outcome:** blinded musician/listener judgments of process audibility, musicality and preference. Use the Artificial Analysis-style protocol for preference comparisons, but add task-specific questions for whether the requested process was actually heard.
5. **Collaboration quality:** compare live adaptive co-creation against a yoked replay of the same prior user actions and against a fixed-policy baseline; measure latency, user agency, accepted edits, recovery from misunderstanding and user-rated usefulness.

For deterministic structural invariants, the target should be 100% on frozen positive and negative fixtures. For open-ended generative behavior, preregister coverage/abstention, stratify by grammar and task, report confidence intervals, and do not optimize one headline scalar at the expense of protected-content failures.

## November 18 implementation update — complete ProcessAdditive score contract

The verifier now audits the current three-role realization rather than melody alone. Its serialized report records melody, expressive, harmony and bass event counts. It checks the plan-carried process cell, prefix contour, exact note pitches/durations, velocity and section-intensity bit patterns, phrase-start emphasis, harmony clock, bass anchor, and the no-inferred-PartId boundary. Any extra voice role or assigned part ID is rejected. The existing generator behavior is unchanged; this is an audit of its explicit current contract.

Twenty-one new tests are authored in addition to the three pre-existing ProcessAdditive tests. No test is compiled or run in this environment.

## Benchmark governance and rights constraints

SMEBench is licensed **CC BY-NC-SA 4.0**; component datasets and SoundFonts keep their original licenses. The paper identifies MidiCaps as CC BY-SA 4.0 and MoisesDB/Arachno SoundFont as CC BY-NC-SA 4.0, with other resources under their own terms. That dataset license is not a commercial-use grant. Luminous must not put these assets into commercial training or deliverables under the benchmark's non-commercial permission; any research evaluation must retain component-level license/permission records and the applicable restrictions.

SMEBench also states its data is predominantly Western popular/classical/orchestral/electronic/rock/pop and its SoundFonts are optimized for Western timbres. Its own limitation section notes the synthetic-to-real gap, only twelve MIDI-expressible edit operations, no vocal/production-effect/genre-transfer operations, audibility/filtering bias, and that a paired target is one plausible valid solution rather than the only correct output. Preserve the synthetic and real-audio strata separately, and musician-governed cultural cohorts must be evaluated separately rather than assuming SMEBench is culturally representative.

Sources:
- https://arxiv.org/abs/2610.36715
- https://huggingface.co/datasets/anonymous-account-submission/SME-Bench

The ICASSP 2027 Audio Editing Challenge states registration closed on 8 October 2026, the 500-example hidden test set per track opened on 10 November, and final submissions close on 25 November 2026 at 11:59 PM US Pacific. The Agent Track additionally requires its model components to have been publicly released before 1 November. This note does not assume Luminous is registered or eligible; use the public 2,000-example / 17,741-rubric MMAE development set for internal reproduction unless actual registration and credentials are confirmed.

Sources:
- https://audio-editing-challenge.github.io/
- https://2027.ieeeicassp.org/sp-grand-challenges/

## CAST comparison scope and a fair path to exceed it

CAST reports five-run means of structural error 0.22±0.02 and chord accuracy 0.96±0.01, improving on the MuseFormer baseline (0.58±0.03 and 0.64±0.02). Its data is GiantMIDI-Piano: 10,855 solo-piano works by 2,786 composers and about 38.7 million transcribed notes, with a filtered 7,236-work subset for the experiments. The paper explicitly identifies current limitations: chord-progression-driven skeletons, insufficient treatment of complex polyphonic counterpoint and atonal music, and single-instrument piano experiments without multi-track timbre interaction.

A fair comparison must first reproduce CAST's exact split, tokenization, metrics, train/test boundaries and five-run protocol. The first target is to match its reported structural error and chord accuracy; stretch targets are structural error below 0.22 and chord accuracy above 0.96, with confidence intervals and a no-leak held-out-composer split. Only after that matched baseline should we evaluate multi-part counterpoint, non-chordal/atonal structures, event lineage, time-varying process and protected edits as separate task families. Those latter tasks are a broader capability extension, not a numerical win on CAST's piano benchmark.

The CAST paper and its data availability statement:
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13237092/
- https://github.com/bytedance/GiantMIDI-Piano

## External industry and benchmark refresh — 2026-11-18

### Audio generation preference is not the same as control correctness

The current Artificial Analysis live instrumental v1.1 board displays Suno v6 1148±22, Suno v6-mini 1114±21, Mureka V9.5 1084±21, Mureka V9 1083±21, Lyria 3 Pro 1080±21, and Lyria 3.5 1078±21. The live vocal v1.1 board displays ACE Music v1 1160±21, Suno v6 1144±18, Suno v6-mini 1132±18, and Mureka V9.5 1101±18. The official method uses blind same-prompt, same-modality pairwise preference, a ten-second minimum listen per track, Bradley–Terry estimation, and 95% confidence intervals. It is an audio-preference benchmark, not a score-edit or source-preservation test.

Sources:
- https://artificialanalysis.ai/music/leaderboard/instrumental
- https://artificialanalysis.ai/music/leaderboard/vocals
- https://artificialanalysis.ai/methodology/music

### Music editing is a direct competitor capability

SMEBench (27 October 2026) has twelve instrumentation, articulation, dynamics and pitch operations, a balanced 3,600-example synthetic MIDI test set, and a smaller real-audio multitrack complement. Its paired targets explicitly separate source preservation, target agreement and textual compatibility; its analysis finds that common metrics can rank editors differently. It is a direct benchmark to consider for Muse's controlled editing adapters, subject to its research-use and component-license constraints.

Source: https://arxiv.org/abs/2610.36715

The ICASSP 2027 Audio Editing Challenge's public MMAE development set has 2,000 examples and 17,741 atomic rubrics. It reports Instruction Following Rate (IFR), Consistency Rate (CR), and Exact Match Rate (EMR); the official challenge page says existing systems remain below 5% EMR. On 10 November, the organizers released 500 hidden test examples per track and the submission SDK. Final submission is 25 November 2026 (11:59 PM US Pacific); registration closed 8 October and the Agent track has a 1 November model-release cutoff. The current task is to reproduce the public baseline and build a private Muse evaluation; participation eligibility cannot be inferred and must not be assumed.

Sources:
- https://audio-editing-challenge.github.io/
- https://2027.ieeeicassp.org/sp-grand-challenges/

### Symbolic generation and structure comparisons

CAST (Scientific Reports, 14 April 2026) reports structural error 0.22±0.02 and chord accuracy 0.96±0.01, against its MuseFormer baseline at 0.58±0.03 and 0.64±0.02, over five runs. It demonstrates the value of explicit skeleton guidance. Muse must compare on the paper's exact single-track task before claiming better results; future multi-part, counterpoint, atonal and event-identity tasks would be separate extensions, not directly comparable wins.

Source: https://www.nature.com/articles/s41598-026-46750-0

MIREX 2026 symbolic-generation subjective results report DAMN at 3.68±0.16 coherence, 3.39±0.18 structure, 3.18±0.16 creativity and 3.47±0.14 musicality; Anticipatory Music Transformer reports 3.47±0.20, 3.34±0.16, 3.34±0.16 and 3.61±0.17 respectively. The survey used eight prompts, twelve listeners and 38 ratings per model, with one sample selected from eight generated candidates. Treat this as a small subjective comparison, not a definitive general SOTA leaderboard.

Source: https://music-ir.org/mirex/wiki/2026:Symbolic_Music_Generation_Results

MIREX 2026's Music Structure Analysis task on 200 Harmonix test songs reports SongMamba HR3F 0.840 (best displayed) and SongFormer accuracy 0.802 (best displayed); the task is audio section-boundary analysis, not symbolic process generation. It can be a downstream perception/structure benchmark for a Muse audio-analysis path, but is not the same metric as CAST or ProcessAdditive adherence.

Source: https://music-ir.org/mirex/wiki/2026:Music_Structure_Analysis_Results

### Product frontier

Suno Studio 2.0 offers MIDI tracks and piano roll, velocity and pitch-bend editing, automation, effects, synths, and chat-driven clip/plugin creation. Lyria RealTime offers continuous streaming generation with weighted-prompt steering and BPM, density, key and brightness controls. These are strong product references; neither alone proves exact event preservation or an independently verified edit.

Sources:
- https://help.suno.com/en/articles/13670529
- https://deepmind.google/models/lyria/lyria-realtime/

## Validation status

- Source authored on a branch based on current `main` at `67939c904adcfafd246848b74cd53c52a2cca244`.
- Sixteen new Rust unit tests authored, including the 4,096-case deterministic holdout matrix and note-level expressive-envelope mutations.
- Cargo/rustc/rustfmt execution unavailable in this environment; compilation, formatting and test results are pending.
- No workflow pass, audio result, human preference result, or SOTA superiority is claimed.
