# Melothaea Phrase-Timing Curves v1

**Status:** new realization primitive and proposed experiment protocol; not a validated human-performance model.  
**Scope:** coordinated expressive timing for an ordered musical phrase.

## Why this exists

The current realization path humanizes notes independently. Even if each note's jitter is individually bounded, independent timing draws can change the relational timing that gives a phrase its shape. The production path documents that this can push adjacent monophonic notes into overlap, requiring a later de-overlap repair. A local note heuristic does not know whether a phrase should lean into a cadence, push through a pickup, or move a whole chord as one gesture.

This increment introduces a deterministic, piecewise-linear timing curve whose control points define signed onset displacements over normalized phrase position. It applies the curve atomically; malformed inputs, negative onsets, and reordered attacks are rejected without mutating the phrase. Chord tones at the same phrase position receive the same displacement.

The primitive does not infer expressive intent, train on performer recordings, pick a style, or guarantee that a curve sounds human. It is a controlled seam for experiments that compare phrase-coherent timing with the current independent-jitter baseline.

## Paradigm shift: expressive time as a field over phrase structure

Rather than treating expressive timing as independent noise added to every note, represent it as a function over musical time:

`delta_t = f(phrase_position, meter, role, intent, performer_profile)`

In this first increment, the caller supplies `f` through explicit control points. Later versions can fit the curve from licensed, aligned performance data and condition it on role, tempo, groove, articulation and phrase function. Keep slow phrase-level motion separate from fine residual microtiming; they answer different questions and should be ablated separately.

This is a research hypothesis, not an assertion that correlated curves are inherently more musical. The literature on phrase-level expressive performance reports that generated phrase-level timing/dynamics were not consistently preferred to expert or unexpressive alternatives, and discusses omitted cues such as articulation and vibrato. That is precisely why this path must be benchmarked rather than activated by assumption. See the [phrase-level expression study](https://pubmed.ncbi.nlm.nih.gov/31031671/) and the [critical review of expressive-performance models](https://www.frontiersin.org/journals/digital-humanities/articles/10.3389/fdigh.2018.00025/full).

## Implementation contract

- `TimingCurvePointV1` supplies normalized phrase position and signed displacement in milliseconds.
- `PhraseTimingCurveV1` requires at least two strictly ordered anchors spanning exactly position 0 through 1.
- The declared absolute offset bound is constrained to 0–250 ms; every anchor must respect it.
- Exactly one nondecreasing normalized phrase position is required per note.
- All proposed onsets are computed and validated before any note is mutated.
- Negative/non-finite starts, non-finite or invalid note values, and any new onset order inversion fail closed.
- Duration and velocity do not change in this function.
- Chord tones should carry the same phrase position so their starts move together.
- Monophonic line overlap prevention is still a separate invariant. The new primitive guarantees attack-order preservation, not a note-off contract; the caller must continue to check duration overlap and articulation after applying the curve.
- The function is available to callers but is **not yet wired into the production performance/export path**. The existing path stays unchanged until a controlled comparison justifies the integration.

## Immediate deterministic tests

1. Known control points produce the expected interpolated onset offsets while note pitch, duration and velocity remain unchanged.
2. Simultaneous chord tones with the same phrase position remain simultaneous.
3. A curve that would reorder consecutive attacks is rejected, with the entire input unchanged.
4. Malformed anchors and curves that would push an onset below zero are rejected atomically.
5. A deterministic property-style panel now exercises 128 generated curves across 17 event positions, asserting bounded displacements, unchanged pitch/duration/velocity, and ordered attacks. Follow-on work should scale this to thousands of valid/invalid curves and explicitly mutate every contract field.

The authored Rust tests on this branch have not yet been compiled or executed in this environment.

## Benchmark design

Run matched input scores through at least four arms:

- **A — quantized control:** no expression curve and no microtiming.
- **B — legacy realization:** current production per-note humanization, unchanged and version-pinned.
- **C — explicit curve:** phrase-level timing curve only, with the same renderer and same source notes.
- **D — fitted curve:** future curve parameters learned from training performers, then frozen before the held-out test.

Where relevant, add a fifth arm that uses independent jitter matched to the curve's displacement distribution. This isolates the value of cross-note dependency from the mere amount of timing variation.

### Synthetic invariants

- Neutral curve is an exact timing no-op.
- 100% of negative/non-finite/out-of-bound test cases rejected.
- 100% of order-inverting curves rejected without partial mutation.
- For a known piecewise-linear curve, interpolation error stays within a declared floating-point tolerance (start with 0.01 ms for synthetic anchors).
- Zero note reorderings, zero negative onsets, and zero mutation after errors in the deterministic property panel.

### Human-data diagnostics

Use the Groove MIDI Dataset (GMD) as an initial data source for the drum-specific branch: it provides tempo-aligned expressive human drumming, performance/session identifiers and a supplied train/validation/test split. It is not a guitar/bass proxy. Use instrument-specific datasets for other roles only under their stated license and splits. Source: [official GMD dataset documentation](https://magenta.withgoogle.com/datasets/groove).

The repository now has a read-only Rust profile adapter at `examples/groove_timing_benchmark.rs`. With the official MIDI-only archive extracted, run:

```sh
cargo run --release -p symthaea-muse --features theory --example groove_timing_benchmark -- \
  --dataset-root /path/to/groove --split test --styles 24 --seeds 2 \
  --output /path/to/new-groove-profile.json
```

The dataset root must contain `info.csv` and the MIDI files named by its `midi_filename` column. The official MIDI-only archive is about 3.11 MB; its SHA-256 is `651cbc524ffb891be1a3e46d89dc82a1cecb09a57c748c7b45b844c4841dcc1e9` (verify against the [official download page](https://magenta.withgoogle.com/datasets/groove)). The adapter reports human and generated distributions separately by drum role, then also breaks the human reference out by drummer and drummer/session so one high-volume performer cannot be mistaken for the entire human baseline. It reports sixteenth-grid residuals in both milliseconds and beat-normalized units, hashes the metadata and every read MIDI file, records exclusions, and emits `quality_qualified: false`. `--limit` is a deterministic early-row subset for smoke tests, not a representative sample. The generated baseline uses the unchanged production performance path; it does not apply the new phrase-timing curve. The adapter does not download the dataset, include source audio, align the different corpora as matched prompts, or prove superiority.

For each role and tempo/style stratum, compare:
- distributions of beat-relative timing residuals by subdivision and event role;
- lag-1 and longer-range autocorrelation of residual timing;
- conditional kick/snare/hat or melody/accompaniment timing relationships;
- phrase-final timing, pickup behavior and structural-boundary behavior;
- whether the model reproduces a held-out performer's timing without copying source phrases;
- offset magnitude and curve smoothness, jointly with articulation and velocity rather than alone.

Do not simply minimize MAE to a single human performance: expert players disagree, and expressive timing is intentionally variable. Report both distributional fit and task success. Train/test splits must separate performers or sessions, not neighboring measures from the same recording.

### Human preference

Use blinded matched-brief comparisons of the same score and renderer across the four arms. Rate groove/feel, phrase direction, articulation, instrument authenticity, musical intent and overall preference independently. Normalize loudness and keep the rest of the signal chain fixed.

Primary outcome: pairwise preference for phrase curves versus the legacy per-note baseline. Secondary outcomes: preference versus the quantized control, expert-rated suitability for the requested feel, and an automatic prediction model's calibration to held-out judgments.

A useful eventual target is a 60% preference rate for a *named task and instrument domain*, with the lower bound of a pre-registered 95% confidence interval above 50%, followed by replication. This is a future qualification target, not current evidence or a universal claim of superiority.

## Broader research direction

After this primitive survives deterministic checks and held-out dataset evaluation:

1. Learn a phrase-time curve from performer data, with uncertainty and source provenance.
2. Add explicit groove controls (laid-back/pushed, swing ratio, subdivision-specific displacement) instead of burying them inside a generic noise scale.
3. Coordinate parts with a shared temporal reference but role-specific displacements; a bass may sit with or deliberately behind the kick, while guitar strums may share one gesture.
4. Condition timing on phrase function and structural events (pickup, climax, cadence, fill boundary), preserving exact symbolic note identities.
5. Fit listener-preference prediction against human outcomes rather than directly optimizing an unvalidated aesthetics score.
6. Integrate only through an explicit experiment arm with a preserved baseline and identical rendering conditions.

Recent work reinforces the need for task- and listener-grounded evaluation: the 2025 [human-preference music generation benchmark](https://arxiv.org/abs/2506.19085) collected 15,000 pairwise comparisons, while the 2026 [CMI-RewardBench task](https://www.music-ir.org/mirex/wiki/2026%3AMusic_Evaluation_via_CMI-RewardBench) evaluates models' predictions of preference for tracks conditioned on compositional multimodal instructions. Neither provides evidence that this curve is better; they inform how we should test it.

## Related correction: performance calibration parser

This branch also hardens `examples/performance_calibration.rs`. Previously, the evaluation parser keyed open notes by pitch alone, shared that map across tracks, and replaced an earlier same-pitch note-on silently. It could pair events from different channels/tracks or corrupt durations in repeated-note cases. The new track-local parser keys by channel and pitch and rejects ambiguous re-triggers, orphan note-offs, unmatched note-ons, zero-duration events, PPQ outside the positive MIDI metrical range or with a fractional value, and tick overflow. Regression tests use synthetic MIDI events; they are authored but not executed here.

This keeps a future baseline comparison from being invalidated by the measurement harness itself.

## Claim boundary

Passing these tests would establish implementation invariants only. It would not establish expressive realism, expert-level musicianship, listener preference, or a superhuman performance result. Those claims require successful held-out evaluation and controlled listening evidence.
