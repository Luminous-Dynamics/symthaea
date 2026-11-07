# MEL-VOCAL-001 — Music-aware singing phoneme alignment evaluation

**Status:** executable scoring primitive proposed; audio recognition and human validation are not claimed.  
**Date:** 7 November 2026.  
**Implementation:** tools/mel-vocal-alignment-eval.

## Research problem

Symthaea already represents a vocal performance in terms of lyrics, phonemes, syllables, note assignments, melisma, pitch curves, and phrase-level expression. Its singing release gate also checks score-relative pitch, onset/duration, render integrity, and listener ratings.

Those are useful but incomplete. We need an independent answer to the following questions:

- Were the intended words and phonemes actually sung?
- Were consonants, vowel nuclei, and syllable boundaries realized at useful times?
- Did a consonant cluster collapse, disappear, or become an invented vowel?
- Was a melisma sustained as one vowel, or did the system unintentionally re-articulate it?
- Did a recognizer hallucinate words during breath, silence, vibrato, or instrumental leakage?

We need a music-aware evaluation layer that links lexical truth, phone alignment, and musical timing without forcing every expressive performance to match a rigid waveform template.

## External landscape (7 November 2026 snapshot)

### Whole-song vocal generation: Artificial Analysis

Artificial Analysis AA-Music-Vocal v1.1 is a useful production-quality comparator. Its 7 November snapshot lists Suno v6 at 1,144 ±18 Elo, Suno v6-mini at 1,132 ±18, Mureka V9.5 at 1,101 ±18, Mureka V9 at 1,063 ±18, and Lyria 3.5 at 1,050 ±18. Its protocol compares blind same-prompt tracks, with listening samples and loudness normalization.

This is an end-to-end text-to-song comparison. It does not directly measure phone accuracy under a supplied score, so it should not be treated as an SVS benchmark. Source: https://artificialanalysis.ai/music/leaderboard/text-to-music

### Score-conditioned singing generation: SoulX-Singer

SoulX-Singer is a high-priority open-model candidate to reproduce. Its public code advertises both F0-conditioned and score/MIDI-conditioned synthesis, zero-shot singer conditioning, and Mandarin, English, and Cantonese. The associated SoulX-Singer-Eval suite separates singing aesthetics (SingMOS-Pro and Sheet-SSQA), spectral distance (MCD), ASR-derived WER/CER, speaker similarity, and melodic FFE/GPE/VDE.

The paper and repository make this a strong candidate for controlled reproduction, not evidence that it outperforms Symthaea locally. Its own documentation notes that generated score/lyric preprocessing may need manual alignment review. Sources:
- https://github.com/Soul-AILab/SoulX-Singer
- https://github.com/Soul-AILab/SoulX-Singer-Eval
- https://arxiv.org/abs/2602.07803

### Singing-specific phoneme alignment: ISMIR 2026

Kim and Lee's *Vocal Music under Phoneme-Conditional Analysis* (ISMIR 2026) provides a particularly useful alignment reference. Its released benchmark reports a validation set of 47 sung NUS-48E recordings with 25,157 manually reviewed phone boundaries. The selected blended aligner reports 22.8 ms median onset error, 74.6% of onsets within 50 ms, and 90.5% within 100 ms. The paper's comparison also reports that Montreal Forced Aligner produced a usable alignment for only 8 of 47 sung recordings in that experiment, despite being close on read speech. This is strong evidence that speech forced alignment cannot be assumed to transfer to expressive singing.

The aligner pipeline compares an MMS CTC pass with an IPA-native model plus spectral-flux snapping, then blends the results. The benchmark is useful because it retains manually checked phone boundaries and tests actual singing. It is not a universal leaderboard: the sample is small, its phone sequence is supplied/curated, and its domain and annotation conventions need to be reproduced before direct comparison.

Source and code: https://github.com/gillosae/phoneme-conditional-singing-analysis

**Evaluator change in this iteration:** MEL-VOCAL-001 now reports median absolute phone-onset error and the fraction of exact phone-identity pairs within 50 ms and 100 ms. These figures are conditional on exact symbol, word, syllable and vowel/consonant identity matches; they are not a joint score that hides transcription mistakes. To reproduce the ISMIR metric fairly, first run a fixed-reference-phone alignment track on the same manually annotated corpus, then report the joint recognition-plus-alignment result separately.

**Proposed stretch target, not an observed result:** on a speaker/song-disjoint singing alignment set using the same annotation convention, aim for median onset error <=20 ms, >=80% within 50 ms, and >=95% within 100 ms, while preserving high recall (do not achieve low timing error by rejecting difficult phones). Report results separately for vowels, stops, fricatives, word boundaries, melisma, vibrato, breathy singing, glissando and falsetto.

### Phoneme-conditioned musical identity: beyond generic voice similarity

The same ISMIR 2026 work asks whether language-specific vocal identity remains measurable in singing when melody, singer and genre are held constant. It matches marker phones to non-marker controls within the same song, then tests articulatory effects (formants, spectral tilt, H1-H2, HNR), melodic deviation (F0), rhythmic adaptation (duration), melisma, and timbral properties. On a 5,024-track set gated from 9,331 aligned tracks, it reports 85.5% balanced accuracy in nine-way language classification under artist-grouped folds.

This suggests a research direction for Muse: do not only ask whether the voice sounds like the same singer. Also measure whether its language-specific articulatory and prosodic identity survives when singing on different notes, in different registers, and across styles. Use within-singer / within-song matched controls where possible so the language effect is not confounded with singer, melody, or production. This is a separate language-characterization study, not a pronunciation-accuracy score.

### Music-aware singing quality assessment: MusicJudge

Samsung Research's MusicJudge (Interspeech 2026) is especially relevant because it combines lyric correctness with pitch/rhythm fidelity at meaningful music-block boundaries. Its published evaluation reports ranking agreement with expert judges on SwaraLyrics (Spearman 0.683; Kendall tau 0.499). This demonstrates why lyric-only WER, score-only pitch error, and audio-only MOS should not be collapsed into one universal quality measure.

Source: https://arxiv.org/abs/2606.26451

### Singing-quality predictors: SingMOS-Pro

SingMOS-Pro is a useful auxiliary predictor and benchmark reference. Its published corpus contains 7,981 Chinese and Japanese singing clips from 41 systems across 12 datasets, with at least five professional ratings per clip and separate lyric, melody, and overall-quality annotations.

Its language distribution does not automatically qualify an English or African-language voice. It should be evaluated as an auxiliary screen, not a release authority.

Sources:
- https://github.com/South-Twilight/SingMOS
- https://arxiv.org/abs/2510.01812

### Whole-song open baseline: ACE-Step 1.5

ACE-Step 1.5 is a useful alternative whole-song baseline, using an LM planner plus diffusion transformer for lyric-conditioned complete-song generation. It can help compare end-to-end vocal production, but it is not equivalent to a score-conditioned singer: melody control, singer consistency, pronunciation, and per-language capability must be separately tested on the exact model.

Source: https://github.com/ace-step/ACE-Step-1.5

## Paradigm shift: evaluate the singer as a musical-linguistic performer

The evaluator should not ask one model whether the output is good. It should assemble independent evidence across four linked but distinct layers:

1. **Lexical truth:** transcript, words, language, script, and explicit pronunciation variants.
2. **Phonetic realization:** phone identity, voicing, articulation, onset, offset, and uncertainty.
3. **Musical realization:** note map, beat map, pitch, duration, phrase boundaries, expressive timing, and melisma.
4. **Perceptual judgment:** intelligibility, naturalness, expressive fit, timbral identity, emotional fit, and listener preference.

The Rust crate is deliberately narrow: it scores the second layer when a recognizer/aligner supplies phone events. It also checks the relationship between a sustained vowel and its declared melisma note span. It does not infer the events from audio and does not claim to implement MusicJudge.

## Benchmark design

### Track A — controlled score-conditioned synthesis

Start with the existing concealed 60-case Muse vocal set (20 phrase challenges crossed with three registers/styles). Freeze:

- exact lyric and language;
- canonical phone sequence and pronunciation profile;
- symbolic note sequence and score digest;
- voice/backend/checkpoint identity;
- sample rate, source audio digest, and render configuration;
- phrase, register, technique, and challenge labels in the organizer-only manifest.

Add explicit phone-reference boundaries for a stratified subset first. Include plosives, fricatives, affricates, nasals, diphthongs, consonant clusters, word-final consonants, sustained vowels, melisma, phrase releases, and deliberate silence.

### Track B — independent recognizer/aligner challenge

Run at least two distinct systems on the exact same dry vocal stems. Retain their raw outputs, versions, prompts, sample-rate transforms, and confidence values. Score them independently rather than selecting one as ground truth.

Metrics:
- phone error rate (substitutions + deletions + insertions);
- exact-phone onset and offset MAE/p95; exact identity includes phone symbol, word index, syllable index, and vowel/consonant class;
- duration MAE on exact phone matches;
- per-phoneme confusion and deletion rates;
- voiced/unvoiced confusion on vowel/consonant regions;
- melisma vowel-span coverage;
- unsupported-phone and low-confidence coverage;
- uncertainty calibration where a predictor exposes confidence.

ASR word error rate remains a separate lexical metric. It cannot stand in for phonetic boundary accuracy.

### Track C — musical alignment and expressivity

Join phone events to the exact performance plan and note map. Evaluate:
- consonant anticipation relative to the notated vowel onset;
- vowel onset against note onset with explicitly permitted transitions;
- melisma coverage across note changes;
- vibrato and portamento without penalizing valid pitch trajectories;
- phrase breath, release, and rest handling;
- timing relative to beat/tempo with tempo- and technique-specific tolerances.

Keep the audio sample-frame clock separate from the monotonic runner clock. Do not mix compute latency with musical event timing.

### Track D — naturalness and artist value

Use concealed, loudness-matched clips and at least five independent listeners as an initial pilot minimum, then set confirmatory sample size from a power analysis. Rate dimensions separately:
- lyric intelligibility;
- naturalness;
- melodic fidelity;
- expressive plausibility;
- emotional fit;
- identity consistency;
- artifact severity;
- musical preference.

Include licensed human-sung anchors and strong model baselines. Use ITU-R BS.1534-3 (MUSHRA) or an explicitly justified derivative where appropriate, with hidden reference and anchor material. Report the adaptation because the original recommendation addresses intermediate audio quality, not every singing-specific construct. Do not use a single weighted composite as a substitute for dimension-level results. Source: https://www.itu.int/rec/R-REC-BS.1534

## Candidate comparison portfolio

| Candidate | Task fit | Main advantage | Main caveat |
|---|---|---|---|
| Current Kokoro singing path | Speech model adapted to singing | Available, low setup burden | Singing is outside its primary task; measure note/phoneme degradation |
| In-tree DiffSinger worker | Score/phoneme-conditioned SVS | Explicit phone, duration, pitch controls | Current mainline worker/production docs need reconciliation; English-only map |
| Functional vocal tract | Physics-inspired articulatory synthesis | Interpretable anatomy, motor gestures, acoustic state | Deterministic tract tests do not establish natural singer quality |
| SoulX-Singer | Zero-shot, score- or F0-conditioned SVS | High-priority modern open baseline | Large model/data footprint; verify exact weights and data terms |
| ACE-Step 1.5 | Lyrics-conditioned complete-song generation | Strong open whole-song baseline | Not equivalent to score-conditioned SVS; melody/phoneme control must be tested |
| Suno / Mureka / Lyria | Commercial full-song baseline | Current blind-preference production frontier | API black boxes; cannot assume editable phone/score control |

## Initial engineering targets

These are proposed test-contract targets, not observed results:

- 100% deterministic correctness on synthetic alignment fixtures, including same-symbol wrong-word, wrong-syllable, and vowel-class negative controls.
- Zero accepted malformed, out-of-range, or non-finite phone events.
- 100% expected melisma spans covered on the synthetic positive control; zero coverage on the deliberately truncated negative control.
- Timing metrics must be undefined (not zero) when there are no exact phone matches.
- Each language profile must have an explicit phone inventory and no silent fallback.
- At least two independent recognition/alignment sources before any recognizer becomes a release authority.
- No production gate promotion based solely on an ASR model, a MOS predictor, or the generator's own evaluator.

Real acoustic thresholds for onset/offset error, phone error rate, and intelligibility should be calibrated on licensed, speaker-disjoint, song-disjoint data and frozen before confirmatory testing.

## Immediate implementation sequence

1. **This branch:** dependency-free Rust sequence scorer, strict event validation, phone edit accounting, exact-match timing, melisma coverage, synthetic positive/negative controls.
2. Re-anchor the open vocal-hardening PR #7389 on current main rather than merging its 212-commit/194-behind branch wholesale. Cherry-pick/reconcile its listener/private pack separation, score-blind F0 diagnostics, input validation, and language registry.
3. Add a versioned sidecar contract to bind audio digest, aligner model, phone map, raw prediction digest, and scorer version.
4. Add a two-system offline evaluation runner for the existing frozen 60-case corpus.
5. Reproduce a small SoulX-Singer score-conditioned baseline and run the exact same score/lyrics through current Kokoro and DiffSinger.
6. Conduct a language-specific Afrikaans pilot only after G2P/phone inventory and licensed held-out speaker data are available. Keep isiZulu, isiXhosa, and Sesotho explicitly unverified for singing until real evidence exists.
7. Run blind listening with human-sung anchors before promoting a backend or making SOTA claims.

## Claim boundaries

- The scorer measures supplied event alignment only; it does not prove that event predictions are correct.
- A passing phoneme score does not prove naturalness, emotional expression, voice identity, or cultural authenticity.
- A score-conditioned pitch pass does not prove pronunciation.
- A strong ASR result does not prove musical quality.
- A high singing-MOS predictor score does not replace blinded human judgment.
- Research-only or non-commercial data must never enter a commercial training corpus without an independently documented rights basis.
