# MUSE-MIR-009 — Barline and pickup-aware measure evaluation (V1)

**Parent:** [#11677 — MUSE-MIR-004 score-structure evaluation](https://github.com/Luminous-Dynamics/symthaea/issues/11677)  
**Design parent:** [#11678 — score-structure evaluation specification](https://github.com/Luminous-Dynamics/symthaea/pull/11678)  
**Implementation issue:** [#11705 — barlines and pickup-aware measure structure](https://github.com/Luminous-Dynamics/symthaea/issues/11705)  
**Date:** 12 November 2026  
**Status:** Research and interface specification only; no production implementation or test execution is claimed.

## 1. Why this is a separate measurement

Melothaea now has separate evaluators for note events and part topology (#11597 / #11660), explicit MIDI meter metadata (#11680), a within-measure note-phase diagnostic (#11681), and beat/downbeat timestamp evaluation (#11687 / #11696).

Those measurements do not establish whether a transcription reconstructed the notated measure structure.

The existing measure-phase diagnostic deliberately assumes that the measure origin is tick zero. It is useful for a first regular-meter diagnostic, but it cannot qualify pickup measures, incomplete first measures, irregular bars, or general mixed-meter structure. The beat-grid evaluator consumes timestamps in performance seconds. A performed downbeat is not, by itself, proof of a notated barline.

**MUSE-MIR-009 evaluates explicit notated barline positions and measure spans in score-beat coordinates.** It does not infer bars from a constant meter, and it does not promote beat-tracker output into score structure.

This is a narrow evaluation increment under MUSE-MIR-004, not a replacement for the wider score-IR work.

## 2. Paradigm: the measure lattice is explicit, not modulo arithmetic

A simplistic evaluator assumes:

    barline(n) = origin + n * meter_length

That assumption breaks for anacrusis, incomplete measures, mixed meter, irregular bars, changing grouping, and other score structures. It also makes tick zero an accidental authority.

V1 therefore represents a score as an ordered set of measure spans in a canonical score clock. Each span has an exact start and end in quarter-note beat coordinates. A measure may be regular, partial/pickup, or incomplete. Internal boundaries are derived from adjacent spans; the beginning and end of the complete score are not automatically counted as internal barlines.

The score-clock coordinate is independent of:
- MIDI ticks-per-quarter (PPQ);
- nominal tempo and tempo-map conversion to seconds;
- expressive performance timing, swing and rubato;
- rendered WAV sample positions;
- audio beat/downbeat annotations.

The representation must preserve these distinctions instead of converting every timing observation to one floating-point timestamp.

## 3. V1 normalized input contract

The evaluator consumes already parsed, provenance-bound score structure. It does not parse audio or MusicXML itself.

### 3.1 Coordinate

Each position is a reduced exact rational number of quarter-note beats:

    RationalBeatV1 { numerator: i64, denominator: u32 }

Validation requires:
- denominator greater than zero;
- canonical reduced form;
- checked arithmetic for comparisons, differences and span lengths;
- strictly increasing measure spans;
- no overlap or unexplained gap unless the schema explicitly represents one;
- no float-to-rational guessing.

Equivalent positions at PPQ 96, 480, 960 or 1920 must normalize to the same rational beat value.

### 3.2 Measure span

A proposed V1 measure record carries:
- a score-local measure ordinal;
- exact start and end quarter-note positions;
- a declared measure class: regular, pickup/partial, incomplete, or irregular;
- optional source measure label and source-format locator;
- optional numerator/denominator and grouping metadata when known.

The evaluator uses the explicit boundary sequence as primary evidence. Meter metadata is contextual evidence; it is not a substitute for explicit measure boundaries.

A pickup is a real partial measure span, not an unexplained offset that the evaluator silently removes. Its duration is scored separately. A silent/rest-only measure remains in the reference measure inventory.

### 3.3 Sequence and provenance

Reference and prediction inputs bind:
- schema and canonicalization versions;
- score-clock identifier;
- source format and parser/adapter identity;
- exact source artifact digest;
- ordered measure spans and internal barline positions;
- prediction mode: audio-only, score-conditioned, instrument-hinted, or human-corrected;
- declared unsupported, unknown, or absent metadata state.

The evaluator must distinguish an explicitly empty internal-barline list (for a one-measure excerpt) from an unsupported or missing boundary representation. Missing metadata must never be converted to a perfect empty score.

### 3.4 MIDI boundary

A MIDI time-signature event describes meter metadata. It does not, by itself, identify every notated barline, pickup, or irregular measure. A MIDI-only system may participate when it supplies a provenance-bound measure-boundary sidecar. Without such a sidecar, the result is Unsupported / Unknown rather than a fabricated regular measure lattice.

MusicXML or a canonical score IR can provide measure spans directly. Adapters must preserve source measure structure and explain any unsupported constructs.

## 4. Metrics — a scorecard vector, not one scalar

### A. Barline detection

Report precision, recall and F1 for internal barline positions. Use deterministic one-to-one maximum-cardinality matching under an exact rational quarter-note tolerance.

The primary matching implementation should exploit the one-dimensional ordered structure with a bounded two-pointer sweep. It must be proven against a small exhaustive oracle in tests. Do not construct a dense all-pairs matrix.

Proposed initial primary tolerance: 1/8 of a quarter-note beat (one 32nd-note duration in quarter-note units), with a preregistered sensitivity panel at 1/16, 1/8, 1/6 and 1/4 quarter-note beats. This is a proposed MUSE protocol, not a claim that these are universal perceptual thresholds or equivalent to vendor tolerances.

### B. Boundary displacement

On matched internal boundaries only, report:
- signed mean displacement;
- absolute mean, median, p95 and p99 displacement;
- maximum displacement;
- matched-boundary count.

Units are quarter-note beats. Do not convert this result to milliseconds using one global tempo.

### C. Measure-span fidelity

Match predicted and reference measure spans only when both endpoints satisfy the declared tolerance. Report:
- measure-span precision / recall / F1;
- exact span-length agreement;
- absolute span-length error distribution;
- measure-class agreement (regular / pickup / incomplete / irregular);
- missing, extra, unparseable and unmatched measure counts.

A correct barline F1 cannot hide a wrong measure inventory. Likewise, a matching meter label cannot hide a displaced barline.

### D. Pickup and partial-measure alignment

Report separately:
- whether the initial partial measure was declared;
- pickup duration error in exact quarter-note units;
- first complete-measure boundary error;
- whether the first full measure begins at the expected score position.

Do not count the piece start as a normal internal barline. Do not silently align the first full bar to zero.

### E. Structural coverage and abstention

Report the number of reference measures represented, including empty measures, as well as unsupported/missing boundary metadata. If a prediction cannot express measure structure, retain the note/beat scores but mark the barline dimension Unsupported. Do not score an unsupported dimension as zero or perfect.

No aggregate score may blend barline, beat, downbeat, note, meter, part-separation, or rendered-audio results.

## 5. Algorithm and resource limits

- Validate schema, canonical rational form, score-clock identity, digest bindings, monotonicity and measure-span consistency before scoring.
- Keep references and predictions in canonical sorted order; reject malformed order rather than silently sorting input artifacts.
- Use a deterministic two-pointer matching algorithm for sorted one-dimensional boundary positions. On each step, discard a point only when it lies outside the tolerance window of the earliest point on the other side; otherwise match the earliest valid pair. Pin tie-breaking and inclusive tolerance semantics.
- Use checked integer/rational arithmetic for comparisons and differences. Floating-point conversion is allowed only for report summaries after the match set is fixed.
- Bound total boundaries, measure spans, serialized input bytes and matching work.
- Domain-separate the report commitment and bind the exact reference, prediction, configuration, algorithm and output summaries.
- Replay validation must reject a changed source, changed boundary list, altered measure class, changed tolerance or altered report summary.

A report digest is an integrity commitment, not a digital signature or proof that an independent model actually ran. A public benchmark result should be signed by a separately managed runner identity in a later evidence layer.

## 6. Required synthetic fixtures and negative controls

1. Regular 4/4 and 3/4 scores with exact internal barlines.
2. A one-beat pickup followed by full 4/4 measures; first complete bar starts at beat 1, not beat 0.
3. Mixed meter such as 7/8, then 5/8, then 4/4.
4. An incomplete measure at the end of an excerpt.
5. Irregular measure span whose length cannot be derived from the surrounding meter label.
6. Empty and rest-only measures retained in the measure denominator.
7. Equivalent scores represented at PPQ 96, 480, 960 and 1920.
8. Boundary errors exactly at, just inside, and just outside the rational tolerance.
9. Missing and extra boundaries, repeated boundaries, and unordered input.
10. Correct beat/downbeat timestamps with incorrect notated barlines.
11. Correct notated barlines with intentionally wrong audio beat timestamps.
12. Explicit meter metadata but no barline sidecar: Unsupported, not an inferred score.
13. Changed pickup flag, altered measure span, tampered digest and re-sealed inconsistent report.
14. Empty single-measure excerpt with no internal boundary: NotApplicable / correct-empty only when the source explicitly declares the measure inventory.
15. Event-budget exhaustion and integer overflow.

The small exhaustive oracle should enumerate short sorted boundary sequences and verify that the production two-pointer matcher returns maximum cardinality and deterministic pairs.

## 7. Industry and research comparison

### Beat tracking is not barline reconstruction

MIREX 2026 separates audio beat tracking and downbeat estimation. Its beat-tracking results include GTZAN, SMC, osu2beat2025, Yamaha_JPop, Yamaha_Balanced and JSIR; the official results list separate F1, Cemgil, Goto, P-score and continuity metrics. These evaluate audio event locations, not notated measure-boundary recovery. See the [MIREX 2026 Audio Beat Tracking task](https://www.music-ir.org/mirex/wiki/2026:Audio_Beat_Tracking) and [results](https://music-ir.org/mirex/wiki/2026:Audio_Beat_Tracking_Results).

The MIREX split policy also matters: its task page states that the declared test datasets are not to be used for training, validation, model selection or parameter tuning. MUSE benchmark adapters must retain each dataset's current split and usage rules rather than treating public availability as training permission.

### Commercial score transcription

Klangio's October 2026 vendor-published comparison reports a mean bar F1 of 0.72 alongside score F1 and meter accuracy on a 130-clip, 13-set evaluation. This is a valuable product-level signal that readable score structure is a different capability from note-event transcription. It is not an independently reproduced leaderboard, and its precise tolerance, data construction and scoring protocol must be reproduced before a numeric comparison is claimed. See [Klangio's benchmark report](https://klang.io/blog/music-transcription-benchmark-2026/).

The public 2025 AMT Challenge results provide a controlled multi-instrument note-transcription comparator, but their note F1 is not a barline metric. Keep these results in their own scorecard row; never rank a barline score against a note F1 score.

### MUSE's competitive opportunity

MUSE should aim to beat shallow score exporters on *structural reliability*, not merely optimize one permissive F1:
- explicit pickup and mixed-meter correctness;
- no silent fallback to 4/4 or tick-zero measure origin;
- lower p95 boundary displacement;
- correct abstention when measure structure is not encoded;
- stable results across PPQ and equivalent source encodings;
- lower musician correction burden in a later blinded study.

Proposed engineering target for a rights-cleared confirmatory corpus: at least 0.80 barline F1 at the preregistered primary tolerance, while reaching 100% correctness on the synthetic contract fixtures and zero false-success outcomes on unsupported/missing metadata controls. The 0.80 is a research target, not a measured result or a directly comparable claim against Klangio's 0.72. We should not call this SOTA until the vendor protocol or an independent public protocol is reproduced.

## 8. Dataset and experimental design

Start with a small rights-cleared development corpus containing regular meter, pickup measures, mixed meter, incomplete excerpts and irregular bars. Keep synthetic tests separate from empirical results.

For confirmatory work:
- split by composition / work / composer / recording lineage, never by randomly sampled adjacent excerpts alone;
- retain source score, MusicXML or canonical IR, audio and annotation lineage as distinct artifacts;
- keep score-conditioned, audio-only, instrument-hinted and human-corrected modes in separate rows;
- include expert annotations and adjudication rules for measure structure;
- publish denominator counts, unsupported rates and parsing failures;
- report 95% confidence intervals clustered by composition / work / artist;
- pre-register the primary tolerance and sensitivity panel before opening confirmatory labels;
- do not train or tune on evaluation-only or rights-restricted material.

Musician correction time and perceived readability are important downstream outcomes, but they require their own blinded protocol and cannot be inferred from automated F1.

## 9. Implementation sequence

1. Add a self-contained Rust module for the normalized rational barline / measure-span contract, bounded matching and report commitment.
2. Add synthetic fixtures, including an exhaustive short-sequence matching oracle.
3. Add a MusicXML or canonical-score-IR adapter only when its source measure semantics are preserved; keep MIDI unsupported without an explicit sidecar.
4. Add the focused module to the pinned Muse workflow and verify formatting/tests at the exact PR head.
5. Build a rights-cleared empirical corpus and compare systems under the same score-clock contract.
6. Add musician correction-time/readability studies only after the score representation and annotation protocol are frozen.

Keep existing note F1, part purity, meter metadata, measure-phase and beat-grid reports byte-for-byte unchanged in their own modules. No implementation, CI, model accuracy, industry parity or SOTA result is claimed by this specification.

## References

- [MUSE-MIR-004 design specification](https://github.com/Luminous-Dynamics/symthaea/pull/11678)
- [MUSE-MIR-006 measure-phase diagnostic](https://github.com/Luminous-Dynamics/symthaea/pull/11681)
- [MIREX 2026 Audio Beat Tracking task](https://www.music-ir.org/mirex/wiki/2026:Audio_Beat_Tracking)
- [MIREX 2026 Audio Beat Tracking results](https://music-ir.org/mirex/wiki/2026:Audio_Beat_Tracking_Results)
- [Klangio, 2026 music transcription benchmark](https://klang.io/blog/music-transcription-benchmark-2026/)
- [mir_eval 0.8.2 beat evaluation reference](https://mir-eval.readthedocs.io/stable/api/beat.html)
