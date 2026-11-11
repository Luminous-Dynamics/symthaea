# ZA-CLICK-001 — synthetic multi-source click articulatory gesture compiler

**Status:** implementation draft; 13 authored symbolic contract tests; Rust CI pending.

**Data policy:** synthetic-only. No performer audio, voice identity, external speech corpus, model training corpus or participant permissions are used.

## 1. Why this needs a paradigm shift

A click is not just a phoneme symbol followed by a formant frame. A typical click gesture has a front oral closure and a simultaneous posterior closure. The tongue body changes the volume and pressure of the trapped oral cavity; release at the front produces an ingressive event. The larynx and velum can add voicing and nasal coupling through distinct mechanisms.

The three principal place families represented in this v1 are dental, post-alveolar and lateral. A study of isiZulu clicks reports differences in front-release characteristics across these places and substantial coarticulation with following vowel context. This means a static place label is not enough: the compiler needs a time-structured gesture with overlapping constraints.

The system must distinguish:
- **velaric oral airstream** for the click event;
- **pulmonic/laryngeal voicing** as an optional accompaniment;
- **velic nasal coupling** as another optional accompaniment;
- **post-release aspiration** as its own interval;
- **prenasal lead** as a separate pre-closure event;
- **vowel nucleus** as the carrier of the melodic and lexical-tone trajectory.

This is a multi-source articulatory representation, not a conventional single glottal source plus formant-filter frame.

## 2. Source grounding and limitations

- UCL phonetics teaching material describes the double closure, tongue-body rarefaction, front release and velaric ingressive airstream. It also distinguishes optional pulmonic voicing and nasal coupling: https://www.phon.ucl.ac.uk/courses/spsci/SSC_talking/material/week_03/sounds-of-the-worlds-languages.pdf
- *Coproduction and Coarticulation in IsiZulu Clicks* reports place-specific release behavior for dental, post-alveolar and lateral clicks and coarticulatory adaptation to following vowels: https://escholarship.org/uc/item/4vv5373d
- Mahura and Pascoe's 2026 Oxford Handbook chapter describes isiZulu as tonal, with 59 consonant phonemes including clicks and prenasalized consonants, and seven vowels: https://doi.org/10.1093/9780191964909.003.0077

The sources motivate an explicit gesture graph. They do not mean the generated timing fixtures reproduce the measured human articulations or validate any language output.

## 3. Implemented model

The crate lives under `tools/za-click-gesture-bench/`.

A `ClickGesture` includes:
- stable event and source-case IDs;
- click place and explicit orthographic-hint metadata;
- accompaniment profile;
- explicit airstream mechanism;
- front and posterior closure intervals;
- rarefaction interval;
- front release and ingressive burst intervals;
- optional laryngeal voicing;
- optional velic nasal coupling;
- optional post-release aspiration;
- optional prenasal lead;
- following vowel nucleus;
- the vowel's independent tone/F0 points and optional melody note.

A text timeline renderer makes overlapping events inspectable. It is not an audio synthesizer.

### Symbolic validator

The validator checks:
- at least 50% overlap of the shorter closure in the v1 synthetic profile;
- rarefaction contained inside the double-closure interval;
- release after closure/rarefaction;
- burst at/after release onset;
- correct velaric-ingressive mechanism;
- explicit, non-conflicting accompaniment fields;
- aspiration in the release-to-vowel gap;
- prenasal lead timing;
- tone points ordered and inside the vowel nucleus;
- no melodic F0 points on the click transient.

Every invalid/missing feature yields an explicit rejection code. Unknown or pulmonic-only airflow is not mapped to a guessed click mechanism.

## 4. Permission-free known-answer suite

The generator produces 18 symbolic cases:

**3 places × 6 accompaniment profiles**

| Axis | Levels |
|---|---|
| Place | Dental, post-alveolar, lateral |
| Accompaniment | Plain, aspirated, voiced, nasalized, prenasalized, voiced-nasalized |

All event times are synthetic profile values for testing order, interval overlap and event separation. They are not normative durations, IPA ground truth or native-speaker data.

## 5. Initial benchmark targets

These are engineering qualification targets, not external SOTA numbers.

| Benchmark dimension | Initial target |
|---|---:|
| Valid known-answer symbolic fixtures accepted | 100% |
| Missing/contradictory event controls rejected | 100% |
| Double-closure overlap constraints enforced | 100% |
| Rarefaction/release ordering violations detected | 100% |
| Airstream mechanism confusion detected | 100% |
| Voicing/nasal/aspiration distinctions preserved | 100% |
| Tone attached to click transient rejected | 100% |
| Stable event identities and manifest hashes | 100% |

The confirmatory expansion should use property-based generation over event-time permutations, overlapping intervals, boundary conditions and random gesture graph shapes. Split by generated profile family/seed, not random events from one phrase.

## 6. How this helps Muse compete

Current singing systems such as SoulX-Singer, YingMusic-Singer-Plus and other neural SVS models are primarily compared on synthesis quality, score/MIDI conditioning, lyric adherence and melody preservation. Those are important but do not demonstrate that a backend-neutral performance graph can represent a click's multi-source articulatory sequence or safely preserve the independent vowel melody/tone event.

Muse's opportunity is:
1. compile complex articulatory intent into explicit, stable gesture events;
2. keep click place separate from airstream/accompaniment;
3. keep the click transient separate from vowel pitch/tone;
4. support coarticulation as event dependencies rather than static token substitutions;
5. allow backends to declare which parts of the gesture they cannot realize;
6. verify the requested structural behavior and refuse unsupported synthesis claims.

This is not a claim that Muse currently surpasses neural singing systems on naturalness or language performance. It is a concrete capability/assurance track those systems' general preference scores do not measure.

## 7. Integration path

1. Adapt the gesture events into the proposed VocalPerformance v3 graph (#10876) as stable articulatory events and edges.
2. Add vowel-context coarticulation and consonant-vowel boundary dependency cones.
3. Add a separate click airstream renderer interface. Do not route click events through the conventional glottal pulse source or silently map them to noise bursts.
4. Keep the existing source-filter renderer's click capability explicitly unsupported until a backend demonstrates the required mechanism.
5. Add generated acoustic proxies only after the symbolic gesture model is stable; label them as proxy signals and avoid claiming realistic click acoustics.
6. If native-language validation eventually becomes a priority, create a separate, rights-aware evaluation track. It is not required to make progress on the compiler.

## 8. Explicit claim ceiling

ZA-CLICK-001 currently validates a symbolic, synthetic gesture representation and temporal constraints only. It does not establish:
- physically faithful click acoustics;
- native isiZulu or isiXhosa pronunciation;
- correct lexical tone realization;
- singing intelligibility or naturalness;
- cultural authenticity or human preference;
- production capability of the legacy formant renderer.

The current renderer is a legacy source-filter backend and does not model the click's velaric oral airstream. A correct compiler should expose that capability gap rather than paper over it.

## 9. Validation record

The first implementation contains 13 authored tests. An exact-head Rust 1.96 workflow is included for formatting, tests, Clippy and deterministic manifest generation. No local Rust toolchain was available when authoring, so no compile/test/Clippy pass is claimed until CI produces exact-head evidence.
