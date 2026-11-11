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
- Merges Type-1 absolute events in a deterministic `(tick, track index, event index)` order before applying channel program state and pairing notes; Type-0 stays within its single track.
- Captures NoteOn pitch, velocity, MIDI channel and the active General MIDI program. Note events retain a stable source-event ordinal so simultaneous events do not get re-ordered by pitch when building matching graphs.
- Treats NoteOn velocity zero as NoteOff.
- Reports orphan NoteOffs, unclosed NoteOns, tempo point count and repeated same-pitch/same-channel overlapping NoteOns.
- Pairs overlapping same-pitch notes FIFO across the merged channel stream. Standard MIDI 1.0 does not give a NoteOff a unique note-instance identifier; FIFO is a declared convention, not recovered ground truth. Simultaneous conflicting program changes from different tracks on the same channel are rejected rather than guessed.

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

Each prediction can match at most one reference note and vice versa. V1 ports the `mir_eval 0.8.2` Hopcroft–Karp matching traversal, including first-seen predicted-node order, and reports four distinct views:

| Metric | Match requirements |
|---|---|
| Onset + pitch | Pitch and onset |
| Full note | Pitch, onset and offset |
| Exact GM program + onset | GM program, pitch and onset |
| Exact GM program + full note | GM program, pitch, onset and offset |

The two GM-program metrics are stricter additions; they are not substitutes for the instrument-agnostic scores used by many AMT benchmarks. GM program is not a persistent PartId or a fine-grained instrument identity.

V1 also reports separate onset-only and offset-only F1 diagnostics that deliberately ignore pitch, matching the scope of the corresponding mir_eval functions. It reproduces `mir_eval 0.8.2`'s four-decimal (`N_DECIMALS = 4`) onset/offset distance rounding before inclusive threshold checks. Average Overlap Ratio (AOR) is reported over full-note matched pairs as the mean intersection duration divided by the union span; it is zero when there are no full-note matches, as in mir_eval.

Timing reports include signed mean onset error plus mean, median, p95 and p99 absolute onset/offset errors. Offset errors are measured over onset-matched pairs even when the offset is outside the full-note acceptance tolerance, so a poor release does not disappear from the timing error summary merely because it failed full-note F1.

### Frame-level activity

V1 additionally reports instrument-agnostic pitch-frame F1 and exact-GM-program/pitch-frame F1 on a configurable grid, 10 ms by default.

Frame cells are half-open time intervals. A note activates every cell it overlaps. Repeated/overlapping notes of the same pitch collapse to one pitch/frame cell in the instrument-agnostic metric. This is a custom, explicit frame-occupancy view; do not claim direct numeric equivalence to another paper's Frame F1 until its frame hop, rounding, pitch mapping and evaluation code have been reproduced exactly.

### Part separation — track topology is not the same as note accuracy

The part-separation diagnostic uses the existing one-to-one onset+pitch matches to identify which reference PartId each matched predicted note corresponds to. It then reports two different topologies:

- **Track purity:** group matched predictions by MIDI track. This reports whether multiple reference PartIds are mixed into the same physical MIDI track.
- **Track/channel-lane purity:** group matched predictions by (MIDI track, MIDI channel). This recognises Type-0 files that put several logical parts on one physical track but separate them by channel.

For both levels the report gives the number of prediction groups, number of mixed groups, number of matched notes residing in mixed groups and note-weighted purity (the fraction assigned to each group's most common reference PartId). It also reports reference-PartId coverage: the fraction of reference parts represented by at least one matched note.

This is a structural metric, not an instrument identity classifier. It says whether note events associated with distinct score PartIds were kept in distinct MIDI output tracks or channel lanes. It does not prove which acoustic instrument played the source recording. Only matched notes contribute to purity; unmatched extra notes are already counted as false positives by the note metrics and are not assigned a speculative PartId. A PartId can validly be spread across multiple tracks or lanes; the score is based on dominant-part purity rather than a strict one-track-per-part bijection.

### Velocity evaluation

MIDI velocity is captured, but velocity metrics are disabled by default. Some transcription models do not estimate dynamics yet still serialize a conventional placeholder velocity. The caller must explicitly set velocity_metrics_enabled=true only when the prediction pipeline genuinely estimates velocity. When enabled, the report includes velocity-aware full-note F1 and direct absolute MIDI-velocity error summaries. Otherwise those fields are absent, not zero.

This is a custom direct-velocity metric, not mir_eval.transcription_velocity: that module fits a linear scale/offset regression on matched velocities before thresholding normalized velocity error. We deliberately do not apply regression calibration by default because it can hide absolute expressive-dynamics scale errors.

### Bounded work

V1 uses explicit limits to prevent malformed or adversarial MIDI from triggering unbounded work:
- maximum 20,000 parsed/reference notes;
- maximum 64 MiB per MIDI input and 1,000,000 MIDI events;
- maximum 2,000,000 candidate note-pair edges;
- maximum 10,000,000 matching edge scans per matching pass;
- maximum 10,000,000 frame-activation scans and 2,000,000 unique activation cells per frame view.

Capacity exhaustion is an explicit error, not a partial score.

## Tests authored

Thirty-four focused Rust tests currently cover:
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
15. velocity metrics absent by default;
16. onset-only matching can identify a correctly timed event despite an incorrect pitch;
17. offset-only matching can identify a correct release despite an incorrect pitch and onset;
18. Average Overlap Ratio reports interval overlap for a tolerance-accepted full-note pair;
19. the upstream mir_eval Hopcroft–Karp adversarial graph yields the same deterministic pairing;
20. Type-1 note-on/note-off events across tracks pair in the shared channel stream;
21. a conductor-track program change applies to part-track notes on the same channel;
22. simultaneous conflicting program changes across tracks on the same channel abstain;
23. an empty reference and empty prediction are scored as a negative example;
24. false-positive notes are counted against an empty reference;
25. velocity tolerances above the MIDI 1.0 range are rejected.
26. distinct output tracks preserve two reference PartIds with perfect track and lane purity;
27. notes can be pitch/onset-correct yet still reveal contamination when two PartIds share one output track;
28. a Type-0 single-track file can preserve distinct parts on separate MIDI channels, yielding low track purity but perfect lane purity;
29. mixed PartIds on the same track/channel lane reduce lane purity;
30. a missing reference part reduces part coverage without penalizing the purity of correctly separated surviving parts;
31. the part-separation summary is included in the report commitment;
32. frame metrics are withheld when predicted notes have unclosed releases;
33. velocity error is still measured for onset+pitch matches when note offsets are wrong;
34. an empty reference is accepted as a valid all-false-positive evaluation case.

These tests are authored, not compiled or run in the authoring environment.

Empty-reference clips are valid benchmark items. They are essential for measuring false positives in silence, pauses, isolated noise and music with no target instrument; they are not discarded as malformed references.

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

### MulTTiPop — real-world pop stress test

The July 2026 MulTTiPop dataset contains 572 multitrack pop segments (3.5 hours), drawn from 374 songs and 263 artists. Its artist-separated test split contains 403 segments (152 minutes); its dev split contains 169 segments (61 minutes). The dataset spans 101 genres by its authors' artist-genre labeling and covers songs from 1939–2009, with no 2010s material.

The authors' published onset F1 results are low, which makes this a valuable robustness target rather than an easy leaderboard:
- MT3: 10.87 exact-instrument onset F1; 27.82 after harmonic/percussive reduction.
- YourMT3+: 11.51 exact-instrument onset F1; 28.74 after harmonic/percussive reduction.

The results use a 50 ms onset tolerance. Their exact-instrument score requires the same MIDI program, while their reduced task groups instruments as harmonic or percussive. The paper qualitatively reports inconsistent instrument programs, missing parts, broken line continuity, and reduced rhythmic nuance.

**Use constraints:** the authors explicitly say MulTTiPop is for evaluation, not model training, because it is small and its labels correspond to copyrighted commercial recordings. The authors also disclose a strong Western-pop bias. We should keep it as an evaluation-only, no-training benchmark; obtain audio through the dataset's permitted evaluation procedure and respect the underlying rights and platform terms. The paper says 49.1% of 1,164 candidate alignments were selected as acceptable; the resulting released set contains 572 accepted segments. Use the released artist-separated splits rather than rebuilding or cherry-picking alignment candidates.

**Immediate target:** reproduce MT3 and YourMT3+ on the published test split and exact/harmonic-percussive instrument definitions, then run a frozen MuScriptor checkpoint using the same audio segments and same evaluator. Do not compare its MuScriptor D_Test result directly to MulTTiPop. The first meaningful Melothaea achievement is to beat the published 11.51 exact-instrument / 28.74 harmonic-percussive YourMT3+ results with 95% confidence intervals, while reporting each score separately and retaining the test split untouched.

Source: https://arxiv.org/abs/2607.08756

### mir_eval — the note-event standard we should match precisely

The mir_eval 0.8.2 documentation/source specifies the 50 ms onset / 50-cent pitch criterion, a separate full-note test with the greater of 20% reference duration or 50 ms offset tolerance, four-decimal onset/offset distance rounding, and one-to-one Hopcroft–Karp bipartite matching. V1 ports the matching traversal and thresholds, including first-seen predicted-node insertion order. A pinned fixture-by-fixture parity run against the Python package is still required before claiming complete score-for-score parity on arbitrary datasets.

References: https://mir-eval.readthedocs.io/latest/api/transcription.html and https://github.com/mir-evaluation/mir_eval/tree/0.8.2

### October 2026 industry reports — a wider frontier than note F1

Two recent vendor-published studies add important, but non-independent, competitive evidence.

**Songscription / Play Anything (October 2026).** On 200 multi-instrument recordings from RWC, JazzSAMBA and MulTTiPop, the authors report a composite MIDI accuracy average of three mir_eval-derived metrics (onset F1, onset-offset F1 and frame F1): Songscription 52.2, Mirelo 45.1, MuScriptor 41.0, Klangio 40.4, YourMT3+ 39.8 and Basic Pitch 32.7. This is not an individual F1 score, and it is not the same test set or protocol as MuScriptor's 372-track D_Test. The same report says its systems received the same full recordings, no separated stems and no manual corrections; each product was shown in its best-scoring mode. Its multi-instrument sheet-music comparison uses a separate private set of 28 professional transcriptions and reports Songscription 61.1 vs Klangio 46.5, so do not merge that score with the MIDI metric.

**Klangio (October 22, 2026).** Its benchmark uses 130 one-minute clips across 13 sets, with ten clips per set. The reported raw-note F1 uses a 100 ms onset tolerance, averaged over nine aligned sets: Klangio 0.82, Songscription 0.77, AnthemScore 0.76, MuScriptor 0.73 and YourMT3+ 0.72. Its mean final-score F1 across all 13 sets is 0.67, 0.47, 0.43, 0.29 and 0.19 respectively. It reports the leading score-F1 result on 12 of 13 sets and emphasizes beat, barline, meter and key structure before notation. This is a vendor-produced evaluation and includes internal recordings/sets, so it is not an independent public leaderboard. The 100 ms raw-note metric is not directly comparable to this evaluator's default 50 ms mir_eval-style threshold.

The striking systems lesson is that **the score is not merely a container for transcribed notes**. Klangio reports a median 7 seconds per one-minute clip on one GPU versus 20 seconds for MuScriptor on the same GPU, but its comparison also says Songscription's 31-second browser time includes upload and queue time; that is not a matched latency test. It also reports MuScriptor default-tempo fallback on 117 of 130 of its clips. These are vendor-reported observations, not independently reproduced measurements.

Sources:
- Songscription / Play Anything, October 2026: https://playanything.ai/blog/state-of-the-art-music-transcription
- Klangio, October 22, 2026: https://klang.io/blog/music-transcription-benchmark-2026/

**Melothaea response:** build the benchmark as a scorecard vector, not one average:
1. note onset/full-note/frame metrics at standardized 50 ms tolerances;
2. beat/downbeat, tempo, meter, barline and measure-relative note placement;
3. track-to-part separation, voice continuity and instrument identity;
4. notation readability and edit distance (MUSTER / OMR-NED or exact pinned equivalents);
5. rendered playback fidelity and musician preference;
6. human correction time, edit locality and workflow trust.

The October vendor figures are useful to select baselines and failure cases. They are not substitute labels, nor valid direct comparators across different datasets, onset tolerances, best-mode policies or score-construction workflows.

### Dual score and playback fidelity — beyond note F1

A transcription can be symbolically close to a reference while producing a poor resynthesis, or can render close to the source while being awkward or inaccurate as notation. The August 2026 *A Dual Evaluation for Music Transcription* study formalizes these as separate dimensions. It evaluates notation with OMR-NED and playback with feature-alignment metrics, learned embeddings and human preference. Its piano study used 230 recordings (23 works, 30 performers, six composers), 106 participants and 3,180 pairwise judgments.

Among that study's automatic playback metrics:
- CLEWS segment-mean similarity reached Spearman ρ=0.971 and Kendall τ=0.891 against its human Bradley–Terry preference scores.
- Position-balanced Gemini 3.1 Pro ABX reached ρ=0.970 and τ=0.870.
- TWED with MFCC features reached ρ=0.924 and τ=0.790.
- DTW with Chroma CENS reached ρ=0.891 and τ=0.761.

CLEWS was also cheaper than the other learned/LLM evaluators in that study: its cloud-equivalent estimate was $0.37 per 1,000 comparisons, versus $5.10 per 1,000 for the paired Gemini ABX requests. These cost estimates are paper-specific, not universal service prices.

**Melothaea adaptation:** keep symbolic MIDI note-event scores as one axis. Add a separately versioned audio-render comparison track: render transcription MIDI with a frozen renderer/soundfont; compare against the exact source WAV using a deterministic DTW/TWED feature baseline first; add CLEWS or another embedding model only as a separately pinned evaluator; validate metric ranking against blinded musician A/B choices. Do not conflate source-audio similarity with MIDI-to-MIDI note accuracy. The study is piano-centric and its authors caution that transfer to ensembles, voice and non-Western genres is not established.

For full score-generation workflows, add notation-side OMR-NED or an equivalent MusicXML symbol metric separately. The Melothaea MIDI evaluator itself does not claim to measure engraving or readability.

Source: https://arxiv.org/abs/2608.04511

### Complementary corpora

- Slakh2100: synthetic multi-track stems with aligned MIDI; useful for controlled instrument mixtures and render-known-answer tests. https://www.slakh.com/
- MAESTRO: high-quality aligned piano audio/MIDI; piano-only, so it cannot establish multi-instrument performance. https://magenta.tensorflow.org/datasets/maestro
- PianoCoRe: piano score/performance resources; verify each dataset component's terms before use. https://huggingface.co/datasets/SyMuPe/PianoCoRe

Do not silently place non-commercial or share-alike materials into a commercial training corpus. Build a separately consented/commissioned multitrack test set for any commercial evaluation.

## Benchmark targets (proposals, not results)

A credible first external result is to reproduce MuScriptor's D_Test comparisons with exact model artifacts and the same annotation split. Then aim for a Symthaea/Melothaea transcription system that, at matched data and compute budgets:

- exceeds 60.4 onset F1 and 48.6 offset F1 under the identical D_Test protocol;
- exceeds 47.8 Multi-F1 without sacrificing onset, offset or drums;
- reproduces Songscription's 52.2 composite only after reconstructing its exact 200-recording set, metric aggregation and product settings; do not compare that composite to a single F1;
- reproduces Klangio's 100 ms raw-note F1 separately from the 50 ms standard scorecard, and measures barline/meter accuracy on the same 13-set corpus where access permits;
- beats 72.4 Frame F1 only after reproducing the same frame-evaluation details;
- reports velocity separately on a dataset where velocity ground truth exists;
- shows gains on genre-held-out and instrument-held-out splits, not only random tracks;
- reproduces MulTTiPop YourMT3+ (11.51 exact-program onset F1; 28.74 harmonic/percussive onset F1) before claiming a win on that test;
- reports symbolic metrics separately from playback-side scores and human preference correlation;
- reports 95% confidence intervals, performance by instrument family, and failure rates on overlap/density strata.

Those are stretch targets for a future transcription model, not a claim that an evaluator alone improves model accuracy. The first engineering deliverable is a reliable, model-neutral measurement instrument.

## Beyond SOTA: the next research shifts

1. **Score-conditioned transcription.** When a trusted score exists, use it as a prior to disambiguate the audio without silently treating the score as acoustic proof. Report audio-only and score-conditioned metrics separately.
2. **Structure-first transcription.** Infer beat/downbeat, tempo, meter and barline hypotheses before committing notes to measures. Keep the performance clock separate from the notated score clock, and report meter/barline confidence.
3. **Open-set transcription.** Support unknown instruments and abstention instead of forcing every source into 36 or 128 instrument buckets.
3. **Event identity and overlap.** Evaluate duplicate notes, same-pitch overlap, voice separation, velocity and articulation as independent dimensions. MIDI 1.0's lack of note-instance IDs must remain visible.
4. **Uncertainty that is calibrated.** Add event-level confidence and evaluate calibration, coverage-risk curves, selective precision/recall and abstention—not just F1.
5. **Independent acoustic adjudication.** Keep the transcriber distinct from the observer. Use audio-side onset/pitch evidence and an independent renderer/patch reference; neither should self-certify the other.
6. **Dual fidelity.** Measure symbolic note correctness and rendered-audio playback similarity independently; for notation output, measure score readability/notation distance as a third axis. Use CLEWS/TWED/DTW as candidate playback measures only after calibration against musician judgments on ensembles.
7. **Human utility.** A musician study should measure edit time, corrections per minute, voice/part preservation, and trust calibration. Higher transcription F1 alone does not prove a better creative workflow.

## Next implementation

1. Pin mir_eval in a parity harness and run fixture-by-fixture comparisons for onset match, full-note match, onset-only, offset-only, AOR and frame occupancy.
2. Add a frozen Slakh2100 subset and a rights-cleared commissioned performance corpus.
3. Add an adapter manifest recording model commit/checkpoint hash, inference configuration, preprocessing, instrument condition, runtime/hardware, license/rights basis, and exact input/output digests.
4. Add event-level confidence only when the upstream model supplies calibrated or calibratable confidence; do not invent confidence from F1.
