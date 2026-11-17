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
7. Uses a maximal-prefix cycle as the realized cell template and requires each other cycle's melody to match the corresponding exact pitch-and-duration prefix.

The return value is a small versioned report: cycle count, observed melody event count, and maximum prefix length. The error enum reports specific rejection causes.

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

Six new tests are present in `process_grammar.rs`:

- accept unmodified realizations for all four declared trajectory kinds;
- reject a plan with a false cycle count / contour;
- reject a mutation to a note's rhythm/content within a cycle;
- reject a pitch mutation that breaks the repeated cell prefix;
- reject a shifted cycle boundary;
- reject a score with a missing melody event.

Existing process tests remain in place. No test has been compiled or executed in the current environment. Do not describe these tests as passing.

## Benchmark plan

Report separate measurements; do not collapse them into a single music-quality score.

| Dimension | Metric | Qualification target |
|---|---|---|
| Deterministic positive fixtures | Exact plan-to-score audit acceptance | 100% |
| Frozen adversarial fixtures | Invalid score rejection | 100%; zero false semantic accepts |
| Generated holdout cohort | Exact adherence across trajectory kinds, seeds, and bar counts | 100% for the pinned deterministic generator |
| Mutation robustness | Detection rate by mutation class, plus false rejection rate | 100% detection on the frozen matrix; report every class separately |
| Coverage | Accepted examples / all examples; abstentions by reason | Report explicitly; never hide abstentions in accuracy |
| Runtime | p50 / p95 / p99 verifier latency, by event count | Establish measured baseline before target setting |
| Perceptual validity | Blind listening recognition of growth/release and musical preference | Separate preregistered human study; no score-audit proxy |

A useful first holdout matrix is all four process kinds × seeds 0–255 × 2, 4, 8 and 16 input bars, with engine revision pinned. Then add mutation families: missing/extra notes, pitch substitution, duration change, wrong cycle start, wrong contour declaration, gap, overlap, cross-boundary note, and trailing/leading events. Preserve each case's plan digest, score digest, expected disposition, actual disposition and verifier version.

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

## Important limitation: source motif identity is not yet bound

The v1 audit compares each cycle against the maximal-prefix cell **observed in the supplied score**. It does not receive the original `Motif`, a source-cell digest, or an authenticated generation receipt. A coherent pitch transformation applied to every cycle therefore remains internally prefix-consistent and is accepted by this audit.

A dedicated test now documents that behavior so this contract is not later overstated. This is intentional scope, not an accidental claim that motif preservation has been proven. The next stronger contract should bind the source cell explicitly (for example, a versioned source-cell digest plus a canonical pitch/rhythm representation) and independently verify each realized cycle against it. That work should reuse the source identity and event-mapping contracts rather than infer identity from `VoiceRole`, event order, or register.

Likewise, this verifier assumes the supplied plan and score are the inputs being qualified; it does not authenticate their origin. A caller that needs provenance must bind both to an immutable generation receipt and exact pinned engine/input digests.

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

## Validation status

- Source authored on a branch based on current `main` at `67939c904adcfafd246848b74cd53c52a2cca244`.
- Six new Rust unit tests authored.
- Cargo/rustc/rustfmt execution unavailable in this environment; compilation, formatting and test results are pending.
- No workflow pass, audio result, human preference result, or SOTA superiority is claimed.
