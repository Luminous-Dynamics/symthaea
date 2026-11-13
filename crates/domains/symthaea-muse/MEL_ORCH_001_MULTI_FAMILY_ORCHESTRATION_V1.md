# MEL-ORCH-001 — Instrument-family orchestration contract v1

**Status:** source-independent symbolic orchestration contract and synthetic known-answer example. It is not yet an end-to-end orchestral composer or a claim that every listed instrument has a qualified renderer.

## 1. Research question

Can Melothaea compose one coherent work whose independent musical parts are assigned to and realized by different instrument families—e.g. cello, piano, saxophone, French horn, harp, flute, clarinet, percussion and voice—while preserving musical structure, instrumental playability, and intentional interaction?

The target is not a maximal instrument checklist. The target is an arrangement whose instrumentation is selected because the parts need different registers, attacks, sustain profiles, articulation possibilities, harmonic density and expressive roles.

## 2. Architectural separation

The design separates five concepts:

1. **Musical part identity:** `PartId` answers which continuing line is being written. It is not inferred from register, role, MIDI channel or note overlap.
2. **Musical function:** a part may act as theme, bass ground, harmonic motion, counterline, ostinato, response, drone or another explicitly authored function. A function may change over time without changing the identity of the part.
3. **Instrument instance:** a specific performer/instrument instance realizes a part. Two independent violinists require two instance IDs; one piano may realize multiple parts only within its declared polyphony budget.
4. **Capability profile:** sounding MIDI range and maximum simultaneous-note budget are explicit inputs. They are not universal constants inferred from an instrument's family label.
5. **Performance/rendering:** notation transposition, articulation, breath/bow limits, sample-bank coverage, expressive performance and audio rendering are separate qualifications.

This avoids the common error of flattening the ensemble into one piano roll, one instrument preset, or four generic voice roles.

## 3. Implemented in v1

The Rust module `symthaea_music_theory::orchestration` provides:

- Nine broad families: keyboard, bowed strings, plucked strings, woodwinds, brass, percussion, free reed, voice and electronic.
- A named instrument vocabulary covering common orchestral and selected world instruments.
- Per-instance, caller-declared sounding ranges and polyphony budgets.
- Explicit `PartId -> instrument instance -> musical function` assignments.
- Fail-closed checks for missing/duplicate parts, unknown instruments, invalid profiles, unassigned notes, sounding-range violations, invalid rational event times and polyphony overcommitment.
- Exact rational-time overlap ordering for bounded score times; end events sort before start events at the same moment, so abutting notes do not consume two simultaneous-note slots.
- No automatic octave shifting, note dropping or implicit reassignment.

Run the synthetic known-answer example:

```sh
cargo run -p symthaea-music-theory --example orchestration_family_contract_benchmark
```

The example contains a four-family cello/piano/French-horn/harp arrangement and negative controls for out-of-range notes and overcommitted polyphony. It checks symbolic contract mechanics only.

## 4. Deliberate limits

- Score pitches are treated as **sounding pitches**. Written transposition conventions (e.g. transposing winds and horn notation) are not yet encoded.
- A caller supplies the range and polyphony for each performer instance. The module does not pretend that one range is correct for all players, instrument models, techniques or registers.
- No breath duration, bow-change, hand-span, pedal, harp-pedaling, tonguing, double-stopping, percussion-stick, articulation or idiomatic leap constraints are implemented yet.
- The validator does not generate an allocation and does not rewrite a score.
- The module's instrument vocabulary does not certify that every instrument has a high-quality sample bank or production-grade synthesis implementation.
- The current theory renderer still has a role-based ensemble path (melody/harmony/bass/counter) with a small style-selected instrument tuple. The new `PartId` allocation contract is not yet wired into that renderer.
- Several generic composer paths still emit `PartId::UNASSIGNED`. Do not guess persistent part identity from `VoiceRole`; instrument-aware generation must assign identity when it writes the part. Existing part-aware engines are a safer first integration target.

## 5. Next engineering gates

### O0 — Contract correctness

- 100% acceptance of known-valid fixtures.
- 100% rejection of planted missing assignments, duplicate IDs, invalid ranges, out-of-range notes, invalid event times and polyphony overcommitment.
- Deterministic JSON serialization and stable issue ordering.
- Property tests for rational-time comparison, note-end ordering and score scaling.

### O1 — Part identity at generation time

- Make selected score generators emit stable part IDs when the part is created.
- Preserve identity through transformations, ornaments, introductions, codas, repeats, returns and development.
- No post-hoc inference from register, role, instrument, channel or note overlap.
- 100% identity preservation on synthetic known-answer transformations.

Begin with engines that already author persistent parts (e.g. fugue, passacaglia and Renaissance polyphony) before changing the broad fallback composer.

### O2 — Part-aware realization

- Resolve each assigned instance to an explicit renderer instrument.
- Require exact supported mappings or return an explicit unsupported-instrument diagnostic; never silently substitute a different family.
- Route every part to a separate render voice/stem and preserve stable identity in MIDI and project exports.
- Record sample source, articulation, fallback status, renderer version and instrument profile digest.
- Test that an edit to one part leaves every protected part's symbolic events unchanged.

### O3 — Instrumental idiom and notation

Add capability layers incrementally, with evidence provenance per claim:

- sounding vs written pitch and instrument transposition;
- practical and extended range, including register-specific quality;
- polyphony and technique (e.g. piano/harp chordal capacity vs monophonic wind line);
- articulation availability and transition cost;
- breath and bow endurance, phrase breathing and bow-change opportunities;
- instrument-specific ergonomics (hand span, leaps, string crossings, pedal/pedaling);
- dynamic range and register-dependent balance;
- ensemble masking and doubling.

Do not promote an estimated or synthetic capability profile to a measured fact.

### O4 — Adaptive orchestration

The planner should first choose *what the music needs* (function, range, density, articulation, sustain, attack, spatial placement and interaction), then choose among eligible instrument instances. This is more general than mapping a style label to a fixed trio.

Support constrained edits:
- swap one instrument while preserving protected lines;
- thin or expand the ensemble without changing the theme;
- move a melody between families while adapting it idiomatically;
- assign counterlines to a contrasting timbre;
- double, harmonize, answer or deliberately leave a line alone;
- create alternative orchestrations as comparable candidates, not irreversible rewrites.

## 6. Benchmark suite proposal

Use paired positive/negative cases and publish raw task-level outcomes. Separate symbolic correctness, render correctness, and human judgement.

| Gate | Measurement | Initial target (aspirational) |
|---|---|---:|
| Part integrity | Missing, merged, duplicated or identity-swapped part events | 0 on deterministic tests |
| Range compliance | Notes outside the declared sounding range | 0 after validation |
| Polyphony | Over-budget simultaneous note demand | 0 after validation |
| Edit preservation | Protected-part note events changed by a localized edit | 0 symbolic changes |
| Instrument routing | Requested instrument instance routed to exact supported renderer | 100% or explicit unsupported |
| Stem completeness | Every assigned part exported as an independently addressable stem | 100% |
| Family coverage | Qualifying tasks across all nine families and their cross-family combinations | At least 30 cross-family task cells before broad claims |
| Playability | Expert adjudication of idiom/technique constraints | Establish baseline, then target >=95% on qualified task subset |
| Control fidelity | Correctly obeyed instrument, register, role and edit constraints | >=95% on frozen task suite |
| Human preference | Blind pairwise orchestration preference against matched baselines | Point estimate >=55%, with 95% CI excluding 50% |
| Revision efficiency | Human correction time and number of note/part edits | Demonstrable reduction vs matched baseline |

Targets are engineering aspirations, not measured results. A hard contract pass does not prove musicality. A human preference win does not prove playability or correct instrument control.

## 7. Evaluation protocol

Each benchmark item should freeze:

- score/task prompt and musical constraints;
- random seeds and generation settings;
- target instrument instances and capabilities;
- requested structure and part functions;
- renderer/sample-bank versions and asset digests;
- validation outputs and exported MIDI/stems;
- baseline model and human-produced reference where available.

Use at least three evaluation panels:
1. **Orchestrators / instrumentalists:** playability, idiom, register, breath/bow/hand technique, balance and part independence.
2. **Musicians / listeners:** musical coherence, emotional communication, interest, ensemble blend and preference.
3. **Machine checks:** score identity, range, polyphony, event/section integrity, exact controls, stem completeness, renderer routing and reproducibility.

Blind pairwise comparisons should randomize ordering, balance prompt/task cells, use confidence intervals, and publish enough sample counts to interpret uncertainty. Objective audio metrics such as FAD or embedding distances are secondary diagnostics, not a substitute for listening or playability review.

## 8. Industry comparison and differentiating claim

The current Artificial Analysis AA-Music-Instrumental v1.1 board (snapshot checked 9 November 2026) ranks general instrumental outputs by blind pairwise preference; its displayed leaders include Suno v6 (1148 ±22), Suno v6-mini (1114 ±21), Mureka V9.5 (1084 ±21) and Lyria 3 Pro (1080 ±21). This is a useful broad perceptual-quality reference, but it does not independently measure score-level instrument assignment, notational playability, part identity, editing preservation, or independently addressable instrument stems.

Commercial production tools have moved into multitrack workflows. Suno Studio 2.0 documents automation, effects, MIDI, a synth, plugin design and chat-directed generation/arrangement; Studio export documentation describes multitrack stems and MIDI extraction. That is a serious product benchmark for editability and production workflow. It is not evidence that generated parts are idiomatic orchestral notation.

Melothaea should compete on a different, complementary axis:
- a symbolic score that can be inspected and revised;
- persistent, identity-preserving parts;
- explicit instrument-family and performer-capability constraints;
- deterministic, auditable edits;
- independent MIDI/audio stems;
- evidence-qualified playability and expert review.

Then compete on listener preference as a separate axis. Do not claim superiority over the current general-purpose models until matched blind tests exist.

## 9. Research rationale

Duan, Yu and Briot's 2026 survey identifies musicality, consistency, controllability, representation and evaluation as continuing challenges across symbolic, audio and multimodal generation. It specifically calls for more unified representations, fine-grained control and stronger evaluation protocols.

Huang et al. (ISMIR 2025), in *Aligning Text-to-Music Evaluation with Human Preferences*, report that commonly used metrics including standard FAD can poorly track human preference in their studied setting; their proposed MAD metric correlates more strongly with human preference. This supports a multi-method evaluation design rather than optimizing one audio-distance score.

## 10. Data and governance

A real orchestration benchmark needs score-aligned multitrack material with instrument and part labels, not just complete mixed recordings. Every item needs provenance, rights status, permitted research/commercial uses, performer/community attribution where appropriate, and a correction/withdrawal path. Reuse the existing corpus rights registry and the musician-governed qualification programme; this document creates no parallel cultural-consent policy.

Do not infer that a public score, MIDI file, recording or sample is free to ingest or redistribute. Synthetic fixtures remain clearly labelled as synthetic.

## References

- Duan, W., Yu, Y. & Briot, J.-P. (2026). *Recent advances in music generation: methods, evaluation, and challenges*. Artificial Intelligence Review. https://doi.org/10.1007/s10462-026-11582-x
- Huang, Y. et al. (2025). *Aligning Text-to-Music Evaluation with Human Preferences*. ISMIR. https://arxiv.org/abs/2503.16669
- Artificial Analysis. *AA-Music-Instrumental v1.1 Leaderboard and Methodology*. https://artificialanalysis.ai/music/leaderboard/instrumental?tab=leaderboard and https://artificialanalysis.ai/methodology/music
- Suno. *Introducing Suno Studio 2.0*. https://help.suno.com/en/articles/13670529
- Suno. *Exporting from Studio*. https://help.suno.com/en/articles/13925249

## Claim boundary

The current implementation verifies a proposed symbolic part-to-instrument allocation against caller-supplied ranges and polyphony budgets. The synthetic example is not a generated song, a rendered orchestral recording, a playability study, an expert judgement, a commercial comparison result, or a SOTA claim.
