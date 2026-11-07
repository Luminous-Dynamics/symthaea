# Vocal phoneme coverage and South African singing research

**Date:** 7 November 2026  
**Status:** implementation and benchmark plan; no new singing-quality or multilingual-capability result is claimed.

## Findings addressed by this patch

The functional tract singer existed in src/voice/functional_singing.rs but was not declared by src/voice/mod.rs. Production examples import it as symthaea::voice::functional_singing, so it was not actually part of the compiled voice module graph.

The same source mapper used a default-to-AX rule for unknown IPA. That silently transformed unsupported phones into schwa. The in-tree DiffSinger worker independently defaulted unknown IPA to ax, defaulted absent language IDs to zero, duplicated source-phone duration for composite mappings, and could carry the final consonant across melisma.

This patch:
- exposes the functional tract singer behind the existing singing feature;
- rejects unsupported IPA in the Rust functional singer;
- rejects non-English functional-singer language profiles until a reviewed profile exists;
- moves the DiffSinger IPA map into a dependency-light helper module;
- rejects unsupported IPA and missing language IDs instead of inventing defaults;
- preserves ordered multi-token mappings and divides source-phone duration across mapped tokens;
- sustains the last explicitly marked vowel across melisma; consonant-only syllables cannot acquire an implicit schwa;
- requires the voicebank's silence token explicitly.

## Research framing: benchmark different tasks separately

There is no single leaderboard that answers whether a system is a good singer under a prescribed score.

### Whole-song generation

Artificial Analysis AA-Music-Vocal v1.1 ranks whole-song generation using blind pairwise preferences. Its November 2026 snapshot places Suno v6 at 1,144 ±18 Elo, Suno v6-mini at 1,132 ±18, Mureka V9.5 at 1,101 ±18, and Lyria 3.5 at 1,050 ±18. Its protocol supplies prompts and the models generate lyrics and vocals; it does not test a supplied lyric/phoneme/note contract.

Sources:
- https://artificialanalysis.ai/music/leaderboard/vocals
- https://artificialanalysis.ai/methodology/music

### Score-conditioned singing synthesis

SoulX-Singer and SoulX-Singer-Eval provide a stronger controlled-singing baseline. The evaluation suite reports singing MOS predictors, spectral distortion, ASR WER/CER, speaker similarity, and melodic FFE/GPE/VDE. Its GMO-SVS collection contains 802 samples; its zero-shot set contains 100 segments from 50 unseen speakers in Mandarin and English. This is a much better match for Muse's score-conditioned task than a whole-song arena, but it does not establish African-language singing coverage.

Sources:
- https://github.com/Soul-AILab/SoulX-Singer
- https://github.com/Soul-AILab/SoulX-Singer-Eval
- https://arxiv.org/abs/2602.07803

### Singing quality assessment

SingMOS-Pro provides 7,981 singing clips from 41 systems across 12 datasets, with at least five ratings per clip and distinct lyric, melody, and overall-quality annotations. It should be used as an auxiliary predictor and cross-benchmark comparison, not as a substitute for blinded human evaluation in languages or styles outside its data.

Sources:
- https://arxiv.org/abs/2510.01812
- https://doi.org/10.1109/ICASSP55912.2026.11460899

MusicJudge (Interspeech 2026) proposes block-aligned assessment that combines lyric correctness with pitch/rhythm fidelity. This reinforces the need to evaluate lyrics and music jointly without collapsing all evidence into one score.

Sources:
- https://research.samsung.com/research-papers/Listening-Like-a-Judge-A-Music-Aware-Framework-for-Automatic-Singing-Performance-Evaluation
- https://arxiv.org/abs/2606.26451

## Paradigm shift: qualify controllability, not just audio preference

The research benchmark should distinguish at least five tasks:

1. **Whole-song generation:** prompt-to-song preference, arrangement, vocal/instrumental balance, and overall listener preference.
2. **Score-conditioned singing:** exact lyric, score, rhythm, phrase, and pronunciation control.
3. **Voice conversion and identity conditioning:** consented speaker identity consistency without confusing timbre similarity with pronunciation or naturalness.
4. **Articulatory and physiology synthesis:** interpretable phone-to-gesture-to-acoustic behavior and generalization to held-out phonetic contexts.
5. **Multilingual singing:** language-specific pronunciation, phonotactics, tone/stress, code-switching, and speaker-reviewed intelligibility.

The same held-out lyric/score set should be sent to every score-conditioned backend. A separate prompt-to-song track should compare commercial systems. Do not compare a commercial self-written three-minute song against a phoneme-controlled one-phrase stem as though the tasks were identical.

## Proposed release and research measurements

### Phoneme contract

- Unsupported-phone rejection: 100% of deliberately unsupported symbols rejected.
- No fallback substitution: zero unknown-phone-to-schwa or unknown-language-to-ID-zero mappings.
- Composite phone preservation: exact ordered phone expansion and conserved source duration.
- Melisma: one sustained vowel across note changes; no repeated consonant token; consonant-only melisma rejected.
- Provenance: phone-map version, language profile, voicebank digest, and source-audio digest attached to every evaluation run.

### Singing performance

Keep the existing 60-case Muse gate as a controlled baseline and expand only for measured failure modes. Add independent alignment references for a stratified subset, including plosives, fricatives, affricates, nasals, diphthongs, consonant clusters, word-final consonants, phrase releases, and melisma.

Report:
- phone error rate and per-phone confusion/deletion;
- exact-phone onset/offset MAE and p95;
- lyric WER/CER;
- score-independent pitch and voicing;
- score-aligned pitch error, onset, and note-duration error;
- melisma vowel coverage;
- breath/rest hallucination;
- naturalness, intelligibility, expressive plausibility, identity, artifact severity, and preference.

At least two independent recognition/alignment systems should be used as cross-checks before any recognizer participates in a release gate. Human annotators remain necessary for the reference subset and confirmatory listening.

## South African language direction

A promising public speech resource is African Next Voices / Swivuriso: a South African multilingual speech collection described as 3,000+ hours across isiZulu, isiXhosa, Sesotho, Setswana, Sepedi, isiNdebele, and Tshivenda.

**Important rights boundary:** its published dataset terms prohibit text-to-speech, voice cloning, and voice synthesis use. It must not be used to train or fine-tune a singing or voice model under those terms. Any future speech/ASR or linguistic use must separately follow the dataset's access, consent, and governance conditions.

Sources:
- https://capstone.aims.ac.za/datasets/african-next-voices/
- https://www.dsfsi.co.za/za-african-next-voices/

The singing programme therefore needs a separate licensed and consented collection, with explicit authorization for:
- speech and singing capture;
- model training and evaluation;
- voice identity representation;
- commercial use, where applicable;
- redistribution and derivative weights, where applicable.

Build the language capability registry by task, not by language-count marketing. Text normalization, G2P, ASR, translation, speech generation, singing generation, code-switching, and community review each need their own evidence and status. isiZulu, isiXhosa, Sesotho, and the remaining languages should stay unverified for singing until speaker-reviewed held-out evidence exists.

## Next experiments

1. Compile and run the newly exposed functional singer on the exact singing feature head after the repository workspace blocker is resolved.
2. Run the dependency-free DiffSinger phone-contract tests and pinned Rust formatting checks.
3. Render identical English score/lyric cases through Kokoro, DiffSinger, and the physical tract backend; do not assume comparable quality before the outputs are measured.
4. Run SoulX-Singer on a small, rights-compatible subset of the same score-conditioned benchmark.
5. Prepare a licensed Afrikaans singing pilot, followed by carefully reviewed isiZulu and isiXhosa pilots.
6. Keep the vocal-tract identity/gesture path as a research renderer until blind listening and held-out acoustic evidence justify a production claim.

## Claim boundary

This document and patch establish stricter input handling and expose the existing functional renderer. They do not establish human-level naturalness, SOTA performance, a trained Symthaea voice, or supported African-language singing.
