# VocalPerformance v3 — explicit lyric / phone / score alignment graph

**Date:** 7 November 2026  
**Status:** architecture proposal; v2 remains the production contract. No v3 implementation or benchmark result is claimed.

## 1. Why a new representation is needed

The current `VocalPerformance` v2 contract stores a sequence of `VocalSyllable` items. Each syllable has one primary note plus optional `melisma_notes`. The constructor aligns syllables to melody array positions and maps surplus notes to the final syllable's vowel.

This is a safe narrow case, but it does not express general lyric-to-score alignment. Examples include:
- multiple rapid syllables within one notated note or beat span;
- one syllable whose vowel spans tied notes, grace notes or an ornament;
- a pickup that begins before the nominal note onset;
- rests and phrase breaks as first-class score events;
- time signatures, tuplets and tempo changes;
- one pronunciation candidate or dialect variant aligned against the same score;
- local lyric or note revision with stable references to unaffected events.

The current constructor now rejects insufficient note counts instead of dropping words. This is a correctness improvement, but it does not solve alignment richness. Increasing the number of notes or changing the G2P algorithm cannot fix the limitation by itself.

## 2. Paradigm: pronunciation and alignment are separate hypotheses

Treat the performed line as a graph of evidence-bearing objects, not a single flat array:

1. **Text layer:** source text, Unicode normalization, grapheme spans, language and word boundaries.
2. **Pronunciation layer:** one or more candidate phone sequences with dialect, stress, confidence, provenance, and OOV status.
3. **Syllable layer:** nucleus, onset/coda phone membership, stress and timing priors.
4. **Score layer:** note/rest/tie/grace/ornament events with beat and time coordinates.
5. **Alignment layer:** explicit links between phone/syllable events and score events, including intervals and uncertainty.
6. **Expression layer:** phrase, breath, articulation, energy, vibrato, portamento and timing controls.
7. **Render/evidence layer:** backend capability declaration, exact input digest, output digest and per-event diagnostics.

A renderer receives a validated alignment graph. It must not silently re-run G2P, drop syllables, invent vowels or decide that an extra note is a melisma without an explicit alignment decision.

## 3. Proposed top-level v3 shape

The names below are conceptual, not committed Rust types or a frozen wire schema.

```text
VocalPerformanceGraph
  schema_version
  performance_id
  language_profile_id
  source_lyrics
  normalized_lyrics
  word_spans[]
  pronunciation_candidates[]
  selected_pronunciation_ids[]
  phoneme_events[]
  syllable_events[]
  score_events[]
  alignment_edges[]
  phrase_events[]
  expression_events[]
  provenance
```

### 3.1 Text and pronunciation

`WordSpan` should hold a stable ID, the source text, grapheme start/end offsets, normalized text, punctuation metadata, and language tag.

`PronunciationCandidate` should carry:
- candidate ID and word-span ID;
- ordered IPA phones;
- syllable segmentation and primary/secondary stress;
- dialect/variant label;
- source class (dictionary, G2P model, human-reviewed, user override);
- model/dictionary digest and version;
- confidence or calibrated uncertainty, if available;
- OOV/unsupported state and explanation.

Do not treat a numerical confidence as truth until it is calibrated against independent human phonetic references. Preserve multiple plausible variants for heteronyms and accent-sensitive pronunciations. A user correction is an explicit override with provenance, not a hidden mutation of the model output.

### 3.2 Phoneme and syllable events

`PhonemeEvent` should have stable ID, symbol, word ID, syllable ID, phoneme class, vowel/consonant status, stress inherited or explicitly assigned, natural-duration prior, and an optional realized time interval.

`SyllableEvent` should identify:
- word ID;
- ordered phone event IDs;
- exactly one designated nucleus event (or an explicitly modelled diphthong nucleus sequence);
- onset and coda phone groups;
- stress;
- phrase ID;
- time constraints and duration prior.

A consonant-only fragment is not silently turned into a vowel-bearing syllable. The alignment stage must either identify a legal pronunciation/syllabification or leave the fragment explicitly unsupported for the selected renderer.

### 3.3 Score events

Score events should include:
- note: pitch, onset, duration, velocity, beat location and stable score ID;
- rest: explicit silence span;
- tie: relationship between note events that sustain the same sounding pitch;
- grace note: short ornament with explicit attachment and timing policy;
- ornament: trill, turn, mordent, scoop, slide or other typed event;
- tempo/time-signature metadata and bar/beat coordinate.

Seconds alone are not enough for a music representation. Retain beat-domain coordinates and a tempo map so tempo changes can be evaluated and replayed deterministically.

### 3.4 Alignment edges

An `AlignmentEdge` links one phoneme or syllable event to one or more score events. It contains:
- source event ID and target score-event ID;
- interval within the score event;
- alignment role (attack, nucleus, coda, sustain, ornament, release, silence);
- hard/soft constraint class;
- confidence/evidence and optional solver rationale.

This supports both important directions:

**Fast syllabic passage:** one long notated note/beat span may be divided into several non-overlapping syllable intervals. Each syllable retains its own onset, nucleus and coda, rather than having its consonants stretched across the full note.

**Melisma:** one syllable nucleus may align to multiple notes, ties or ornaments. Onset and coda consonants are not repeated. The held-vowel policy is explicit in the graph; it is not inferred from “notes left over”.

Coarticulation can use overlapping transition windows, but these overlaps must be represented explicitly rather than violating the core event timing constraints.

## 4. Alignment solver: constrained optimization, not positional truncation

A first deterministic solver can be dynamic programming over ordered syllable/note sequences with explicit transition types:
- one syllable to one note;
- one syllable to multiple tied notes (melisma);
- multiple syllables to one note span (subdivision);
- pickup/anticipation;
- grace-note attachment;
- rest/phrase boundary.

Candidate objective terms may include:
- phone natural-duration priors;
- syllable stress versus metrical prominence;
- vowel nucleus placement relative to note attack;
- consonant anticipation and coda release;
- score onset/offset adherence;
- phrase boundary and breath constraints;
- melisma legality and continuity.

Hard constraints:
- no dropped or duplicated word/syllable/phone event;
- every supported syllable has exactly one nucleus definition;
- no consonant is sustained through melisma unless the selected language/phonetic event explicitly permits it;
- every alignment interval is finite, positive and contained in its score event;
- rests cannot acquire voiced phoneme coverage;
- every note event is either aligned, explicitly ornamental, or deliberately unvoiced/unassigned;
- every unsupported or ambiguous item is surfaced as a decision/error, never silently repaired.

The solver should emit a machine-readable explanation for each alignment decision. A human correction can override an edge and retain the original edge and evidence for audit.

## 5. v2 compatibility

Keep `VocalPerformance` v2 frozen until v3 exists and is tested. Build an explicit adapter:

- v2 syllable + primary note -> one syllable event + one score event;
- v2 melisma notes -> multiple score events aligned to the designated nucleus only;
- phrase/expression curves -> phrase/expression events;
- v2 inputs outside this restricted mapping fail conversion rather than being approximated.

The reverse v3-to-v2 adapter must declare its lossiness. Multi-syllable-per-note, complex ornaments, time-signature maps or richer pronunciation variants cannot be silently flattened into v2.

## 6. Benchmark plan

Use synthetic cases with known exact alignments first, then a stratified human-reviewed score/lyric corpus.

### Symbolic alignment
- word/phone/syllable coverage (missing, duplicated, reordered);
- syllable-to-score edge precision/recall/F1;
- note-to-lyric coverage;
- hard-constraint violation rate (target zero);
- melisma detection precision/recall;
- multi-syllable note subdivision correctness;
- rest and breath event placement.

### Temporal realization
- phone/syllable onset and offset MAE, median and p95;
- consonant anticipation error by manner/place;
- nucleus-to-note attack displacement;
- coda release error;
- melisma nucleus continuity and consonant re-articulation rate;
- phrase/breath boundary F1.

### Generalization
Hold out complete singers, songs, score patterns, meters, tempos, lyric words and pronunciation variants. Report dialect and language separately. ASR WER cannot be the sole pronunciation or alignment ground truth.

### Revision locality
Change one word, phone, note, phrase arc or breath. Measure target-region success, unaffected-region identity, collateral-change rate and deterministic replay. Preserve exact cached audio only where the renderer guarantees deterministic locality; otherwise measure aligned acoustic/perceptual deltas.

## 7. External comparisons

- **DiffSinger:** useful baseline for phoneme/duration/F0 conditioning, but compare the same supplied alignment inputs.
- **SoulX-Singer:** useful modern score/MIDI/F0-conditioned reference for naturalness, melody fidelity and zero-shot identity. Its output must be assessed under the same score/lyric cases.
- **MusicJudge:** useful independent assessment pattern for block-aligned lyric correctness plus pitch/rhythm fidelity.
- **Artificial Analysis AA-Music-Vocal:** a separate prompt-to-song task; do not compare its generated lyrics/music directly against the fixed-score Track A.

A competitive advantage would be higher exact score/lyric compliance, interpretable local edits and reproducible evidence at comparable perceptual quality—not a claim that symbolic controllability alone beats a commercial song generator in naturalness.

## 8. Implementation order and dependencies

1. Complete the G2P replacement evaluation in issue #10869.
2. Define a versioned v3 JSON schema and invariants without changing the v2 production path.
3. Implement deterministic synthetic alignment fixtures and the restricted v2 adapter.
4. Build the constrained alignment solver and its per-edge evidence output.
5. Adapt Kokoro and DiffSinger through capability-specific render adapters.
6. Reconcile the physical tract API in issue #10860 before adding it as a renderer.
7. Add v3-specific case-level provenance and artifact receipts; extend the model/voicebank tree manifest work in issue #10872.

## 9. Claim boundary

This is an architecture proposal. It does not establish that v3 exists, that the current Muse path supports arbitrary lyric/score alignment, or that any alignment solver has achieved human-level performance. The v2 production path remains intentionally narrower until v3 has a tested migration path and exact-head evidence.
