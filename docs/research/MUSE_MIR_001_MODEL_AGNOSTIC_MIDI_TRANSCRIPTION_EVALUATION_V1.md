# MUSE-MIR-001 — Model-agnostic MIDI transcription evaluation (V1)

**Parent research tracks:** [MEL-TIME-001 #10262](https://github.com/Luminous-Dynamics/symthaea/issues/10262), [perception grounding #10265](https://github.com/Luminous-Dynamics/symthaea/issues/10265)  
**Realization reference:** [MUSE-PERF-001 #11541](https://github.com/Luminous-Dynamics/symthaea/pull/11541)  
**Date:** 11 November 2026  
**Status:** research implementation; focused Rust tests authored, not yet compiled or executed

## Why this exists

Melothaea needs a reproducible way to compare its own transcription outputs and external audio-to-MIDI systems without putting any one vendor's weights, Python runtime, model API, or license inside the production Rust library.

V1 evaluates a model-produced Standard MIDI file against an exact part-aware MIDI artifact plus its sidecar manifest. It uses the MIDI file as the prediction interface, so a future MuScriptor/MT3/YourMT3+ adapter only needs to produce a MIDI file and identify the evaluation configuration. The evaluator does not download, run, or bundle any model.

## Implementation

Adds 'crates/domains/symthaea-muse/src/transcription_eval.rs', exported behind the existing 'theory' feature.

### MIDI parsing

- Supports Standard MIDI Type 0 (SingleTrack) and Type 1 (Parallel) with metrical PPQ timing.
- Rejects Type 2 (Sequential) and SMPTE time division rather than guessing their semantics.
- Integrates piecewise MIDI tempo events. An explicit tempo at tick zero overrides the MIDI default; conflicting explicit tempo values at the same tick are rejected.
- Captures NoteOn pitch, velocity, MIDI channel and the active General MIDI program.
- Treats NoteOn velocity zero as NoteOff.
- Reports orphan NoteOffs, unclosed NoteOns, tempo point count and repeated same-pitch/same-channel overlapping NoteOns.
- Pairs overlapping same-pitch notes FIFO within a track/channel. Standard MIDI 1.0 does not give a NoteOff a unique note-instance identifier; FIFO is a declared convention, not recovered ground truth.

### Reference integrity

Before scoring, V1 checks:
- the manifest schema and known digest-schema identifiers;
- SHA-256 syntax and exact MIDI-byte digest;
- the domain-separated manifest digest;
- Standard MIDI Type 1 structure and PPQ;
- tempo and meter metadata;
- exact NoteOn/NoteOff event multisets including track, channel, pitch and velocity;
- per-track General MIDI program changes;
- PartId/segment/instrument binding and segment timing;
- contiguous, unique score-scoped source-note ordinals.

The validator checks that the manifest and MIDI agree. The original symbolic Score is not passed to this API, so the evaluator cannot independently recompute score_sha256 from source score content. A digest is an integrity commitment, not a producer signature or proof of authorship.

### Event-level metrics

Defaults follow the documented mir_eval note-matching tolerances:
- onset difference ≤ 50 ms;
- pitch difference ≤ 50 cents;
- offset difference ≤ max(20% of reference duration, 50 ms).

Each prediction can match at most one reference note and vice versa. V1 uses maximum-cardinality bipartite matching and reports four distinct views:

| Metric | Match requirements |
|---|---|
| Onset + pitch | Pitch and onset |
| Full note | Pitch, onset and offset |
| Exact GM program + onset | GM program, pitch and onset |
| Exact GM program + full note | GM program, pitch, onset and offset |

The two GM-program metrics are stricter additions; they are not substitutes for the instrument-agnostic scores used by many AMT benchmarks. GM program is not a persistent PartId or a fine-grained instrument identity.

Timing reports include signed mean onset error plus mean, median, p95 and p99 absolute onset/offset errors. Offset errors are measured over onset-matched pairs even when the offset is outside the full-note acceptance tolerance, so a poor release does not disappear from the timing error summary merely because it failed full-note F1.

### Frame-level activity

V1 additionally reports instrument-agnostic pitch-frame F1 and exact-GM-program/pitch-frame F1 on a configurable grid, 10 ms by default.

Frame cells are half-open time intervals. A note activates every cell it overlaps. Repeated/overlapping notes of the same pitch collapse to one pitch/frame cell in the instrument-agnostic metric. This is a custom, explicit frame-occupancy view; do not claim direct numeric equivalence to another paper's Frame F1 until its frame hop, rounding, pitch mapping and evaluation code have been reproduced exactly.

### Velocity evaluation

MIDI velocity is captured, but velocity metrics are disabled by default. Some transcription models do not estimate dynamics yet still serialize a conventional placeholder velocity. The caller must explicitly set velocity_metrics_enabled=true only when the prediction pipeline genuinely estimates velocity. When enabled, the report includes velocity-aware full-note F1 and absolute velocity error summaries. Otherwise those fields are absent, not zero.

### Bounded work

V1 uses explicit limits to prevent malformed or adversarial MIDI from triggering unbounded work:
- maximum 20,000 parsed/reference notes;
- maximum 2,000,000 candidate note-pair edges;
- maximum 10,000,000 matching edge scans per matching pass;
- maximum 10,000,000 frame-activation scans and 2,000,000 unique activation cells per frame view.

Capacity exhaustion is an explicit error, not a partial score.

## Tests authored

Fifteen focused Rust tests currently cover:
1. exact MIDI prediction produces perfect instrument-agnostic and exact-program scores;
2. wrong GM program changes the exact-program metrics without hiding correct pitch/onset notes;
3. onset outside the 50 ms window reduces onset and full-note scores;
4. offset-only errors preserve onset F1 while lowering full-note F1 and remaining visible in offset error;
5. piecewise tempo conversion;
6. unclosed NoteOn is counted for onset matching but cannot match a full note;
7. wrong velocity is reported separately when velocity scoring is explicitly enabled;
8. exact frame-occupancy scores and same-pitch frame union;
9. reference MIDI byte substitution rejection;
10. re-sealed manifest relabeling of velocity rejection;
11. maximum-cardinality matching without double-counting duplicate candidates;
12. conflicting same-tick tempo rejection;
13. explicit non-default tempo at tick zero;
14. inclusive exactly-50-ms onset boundary;
15. velocity metrics absent by default.

These tests are authored, not compiled or run in the authoring environment.

## External baselines and what they mean

### MuScriptor — the current open multi-instrument comparator

The MuScriptor 1.3B full-pipeline model card reports the following results on its authors' held-out D_Test set of 372 multi-instrument tracks:

| Model | Onset F1 | Frame F1 | Offset F1 | Drums F1 | Multi F1 |
|---|---:|---:|---:|---:|---:|
| YourMT3+ baseline | 32.5 | 45.5 | 17.8 | 41.4 | 21.9 |
| MuScriptor 1.3B | **60.4** | **72.4** | **48.6** | **49.6** | **47.8** |

These are the authors' published results and are not a universal leaderboard. The model card identifies MuScriptor as approximately 1.3B parameters. It uses 5-second audio chunks, 16 kHz mono input and a 100 Hz mel-spectrogram frame rate; its 128 MIDI programs are grouped into 36 MT3 taxonomy groups. The tokenizer does not produce velocity, cannot represent two simultaneous notes of the same pitch and instrument, and treats drums as onset-only. Its documentation also warns that onset/offset precision is weaker in some styles (e.g. choral music) and that its training data skews toward pop and Western classical music.

**Protocol:** freeze a local MuScriptor version/checkpoint, inference options, instrument-conditioning choices, chunk policy, resampler, exact WAV bytes, MIDI outputs and adapter version. Run those MIDI outputs through the same evaluator as Symthaea outputs. Record whether instrument conditioning was supplied and whether the condition used ground-truth source knowledge; a conditioned run and an unconditioned run are different experimental conditions.

The MuScriptor code repository is MIT-licensed, but the model weights are CC BY-NC 4.0 with additional specific conditions. The model card requires users to hold the necessary rights to input music and use its outputs. Therefore:
- use the weights only in a rights-cleared, non-commercial research comparison under the actual terms;
- do not bundle or make the weights a production dependency;
- do not transcribe third-party music without required rights;
- keep any MuScriptor-derived evaluation artifact rights metadata attached.

Sources:
- Paper: https://arxiv.org/abs/2607.08168
- Official model card, scores, architecture, limits and terms: https://huggingface.co/MuScriptor/muscriptor-large
- Official code: https://github.com/muscriptor/muscriptor

### mir_eval — the note-event standard we should match precisely

The mir_eval documentation describes the 50 ms onset / 50-cent pitch criterion, a separate full-note test with the greater of 20% reference duration or 50 ms offset tolerance, and one-to-one bipartite matching. It also explains that these rules differ from historical MIREX implementations. V1 aims at the documented mir_eval defaults, but a pinned parity test against the actual Python library should still be run before claiming byte-for-byte or score-for-score parity.

Reference: https://mir-eval.readthedocs.io/latest/api/transcription.html

### Complementary corpora

- Slakh2100: synthetic multi-track stems with aligned MIDI; useful for controlled instrument mixtures and render-known-answer tests. https://www.slakh.com/
- MAESTRO: high-quality aligned piano audio/MIDI; piano-only, so it cannot establish multi-instrument performance. https://magenta.tensorflow.org/datasets/maestro
- PianoCoRe: piano score/performance resources; verify each dataset component's terms before use. https://huggingface.co/datasets/SyMuPe/PianoCoRe

Do not silently place non-commercial or share-alike materials into a commercial training corpus. Build a separately consented/commissioned multitrack test set for any commercial evaluation.

## Benchmark targets (proposals, not results)

A credible first external result is to reproduce MuScriptor's D_Test comparisons with exact model artifacts and the same annotation split. Then aim for a Symthaea/Melothaea transcription system that, at matched data and compute budgets:

- exceeds 60.4 onset F1 and 48.6 offset F1 under the identical D_Test protocol;
- exceeds 47.8 Multi-F1 without sacrificing onset, offset or drums;
- beats 72.4 Frame F1 only after reproducing the same frame-evaluation details;
- reports velocity separately on a dataset where velocity ground truth exists;
- shows gains on genre-held-out and instrument-held-out splits, not only random tracks;
- reports 95% confidence intervals, performance by instrument family, and failure rates on overlap/density strata.

Those are stretch targets for a future transcription model, not a claim that an evaluator alone improves model accuracy. The first engineering deliverable is a reliable, model-neutral measurement instrument.

## Beyond SOTA: the next research shifts

1. **Score-conditioned transcription.** When a trusted score exists, use it as a prior to disambiguate the audio without silently treating the score as acoustic proof. Report audio-only and score-conditioned metrics separately.
2. **Open-set transcription.** Support unknown instruments and abstention instead of forcing every source into 36 or 128 instrument buckets.
3. **Event identity and overlap.** Evaluate duplicate notes, same-pitch overlap, voice separation, velocity and articulation as independent dimensions. MIDI 1.0's lack of note-instance IDs must remain visible.
4. **Uncertainty that is calibrated.** Add event-level confidence and evaluate calibration, coverage-risk curves, selective precision/recall and abstention—not just F1.
5. **Independent acoustic adjudication.** Keep the transcriber distinct from the observer. Use audio-side onset/pitch evidence and an independent renderer/patch reference; neither should self-certify the other.
6. **Human utility.** A musician study should measure edit time, corrections per minute, voice/part preservation, and trust calibration. Higher transcription F1 alone does not prove a better creative workflow.

## Next implementation

1. Pin mir_eval in a parity harness and run fixture-by-fixture comparisons for onset match, full-note match, frame occupancy and overlap ratio.
2. Add average overlap ratio and independent onset-only / offset-only metrics.
3. Add a frozen Slakh2100 subset and a rights-cleared commissioned performance corpus.
4. Add an adapter manifest recording model commit/checkpoint hash, inference configuration, preprocessing, instrument condition, runtime/hardware, and exact input/output digests.
5. Add event-level confidence only when the upstream model supplies calibrated or calibratable confidence; do not invent confidence from F1.
