# Vocal phoneme coverage and South African singing research

**Date:** 7 November 2026  
**Status:** implementation and benchmark plan; no new singing-quality or multilingual-capability result is claimed.

## Findings addressed by this patch

The shared VocalPerformance constructor and validator now explicitly reflect the current English-only Muse G2P frontend. They reject non-English language tags, unsupported graphemes, digits, internal punctuation, tokens that produce no G2P syllables, and serialized performances whose declared syllable count no longer matches their lyrics. Boundary punctuation around supported words is accepted. The constructor also rejects a melody with fewer notes than the lyric syllable count rather than silently dropping trailing lyrics. Surplus melody notes continue to be represented as melisma. These checks prevent silent corruption; they do not prove the letter-fallback frontend produced a linguistically correct pronunciation. The remaining G2P coverage/accuracy problem is tracked separately in issue #10869.

The physical-tract singing prototype in `src/voice/functional_singing.rs` was not declared by `src/voice/mod.rs`. I temporarily exposed it behind the `singing` feature to test its actual integration contract. Exact-head compilation then showed that it references a set of physical renderer types and controller methods absent from the current `symthaea-vocal-tract` crate API (including `GesturePlanner`, `BranchedWaveguideV2`, `PhysicalTractFrame`, `phoneme_latent`, and `latent_output`). The module was therefore removed from the compiled module graph again, and the prototype file was restored unchanged. It is not a working or production-integrated backend in the current tree.

The mismatch is substantive: `docs/VOCAL_PHYSIOLOGY_ARCHITECTURE.md` and `docs/FUNCTIONAL_VOCAL_TRACT.md` describe an intended / experimental physical pipeline, while the checked-in crate root currently exports the HDC encoder/controller/FEP pipeline, formant frames, and speech G2P. The separate cognitive-physical-voice example also imports APIs that are not exported by the crate root. This needs its own API reconciliation and exact-head compile work; this PR will not introduce compatibility stubs or imply that the waveguide implementation is integrated.

The remaining active DiffSinger worker had a separate and real input-contract defect: it defaulted unknown IPA to `ax`, defaulted missing language IDs to zero, duplicated source-phone duration for composite mappings, and could carry the final consonant across melisma.

This patch:
- moves the DiffSinger IPA map into a dependency-light helper module;
- rejects unsupported IPA and missing language IDs instead of inventing defaults;
- preserves ordered multi-token mappings and divides source-phone duration across mapped tokens;
- sustains the last explicitly marked vowel across melisma; consonant-only syllables cannot acquire an implicit schwa;
- requires the voicebank's silence token explicitly;
- rejects a non-English performance language in the current English-only DiffSinger worker;
- hardens `VocalPerformance` construction and validation (language mismatch, unsupported orthography, unpronounceable tokens, lyric/syllable mismatch, dropped lyric syllables, invalid note/phoneme/expression values, overlapping melismas and malformed timelines);
- adds a Muse G2P phone-inventory fixture so new source phones cannot silently lack a DiffSinger mapping;
- removes the unintegrated physical singer from the comparative release gate;
- adds exact per-case melody/performance receipts and BLAKE3 digests for serialized performance inputs and rendered WAV artifacts;
- records and repairs six source-only workspace paths and a stale `sha2` lockfile entry needed for a truthful feature compile.

No claim is made that the physical waveguide singer is exposed or buildable by this patch. The comparison gate now covers only the integrated Kokoro path and optional DiffSinger worker. Its `run_manifest.json` records the source revision, voice ID, optional worker path, per-case seed/lyrics/style/score notes, and serialized VocalPerformance digest. Each blind clip carries the exact WAV-file digest and its performance-input digest. The manifest does not yet attest the full voicebank/model-weight tree; that needs a separate provisioned-model manifest before claiming complete model provenance.

## G2P replacement candidates and isolated spike — 7 November 2026

The current Muse English G2P is a small dictionary plus ASCII letter fallback. Its strict input checks prevent dropped characters and missing syllables, but do not prove correct pronunciation. Issue #10869 tracks the actual replacement.

### Candidate A — Piper Plus G2P 0.5.3

Piper Plus G2P 0.5.3 (26 October 2026) is a Rust crate documented as MIT-licensed and IPA-first. Its language backends include English (rule-based, CMUdict-derived), Japanese, Chinese, Korean, Spanish, French, Portuguese and Swedish. The direct output API includes IPA tokens and prosody; its separate encoder supports strict unknown-token behavior.

This makes it a good *engineering candidate* for a reproducible spike, not yet a quality winner. Its English backend is itself rule/dictionary based. It does not cover isiZulu or isiXhosa, and a supported language label does not imply good singing pronunciation or dialect adaptation.

An isolated probe has been added under `tools/g2p-spike/`, with fixed cases for common words, clusters, irregular -ough spellings, heteronyms, silent letters, contractions, multi-syllabic words, punctuation and OOV candidates. The dedicated workflow `.github/workflows/g2p-research-spike.yml` compiles/tests the exact pinned crate version and uploads a TSV of candidate IPA/prosody outputs bound to the exact source revision. This spike does not alter the production G2P path.


### First exact-head spike observations

The G2P workflow completed successfully on source revision `dea2b8fd14dc10c029d34106b586ac56dff25e09`. Its artifact (`piper-plus-g2p-spike-...`) was downloaded and inspected. These are raw outputs, not a human-verified accuracy score.

Observed examples:
- **"quick queen strength sixth":** the current Muse fallback emits `ʌ ɪ k` for "quick" and `k ʌ ɛ n` for "queen", because the grapheme fallback drops unsupported `q` and treats `u` as a vowel. Piper Plus emits `k w ɪ k` and `k w iː n`, which are plausible English pronunciations.
- **"though through rough tough cough bough":** Muse's fallback repeats a largely spelling-driven sequence across these irregular words. Piper Plus produces distinct candidates (`ð oʊ`, `θ ɹ uː`, `ɹ ʌ f`, `t ʌ f`, `k ɑ f`, `b aʊ`). These look more useful but still require accent-specific review.
- **"colonel":** Muse emits a spelling-like sequence (`k ɒ l ɒ n ɛ l`); Piper Plus emits a candidate closer to the common US pronunciation (`k ɜː n ə l`).
- **Contextual heteronym "read":** in "I read the book yesterday and I will read it tomorrow", Piper Plus emits the same `r ɛ d`-like sequence for both occurrences. The current Muse output also fails to distinguish them. This proves that simply swapping in a CMUdict-derived G2P is not enough for contextual pronunciation.
- **OOV behavior:** for nonsense input such as "xqzpl flurble", Piper Plus returns a long spelling-derived phone sequence instead of an explicit unknown/OOV status. A non-empty output is not evidence that the word was understood.
- **Prosody output:** Piper Plus emits stress markers as separate IPA tokens (for example `ˈ` and `ˌ`) and long-vowel markers separately in some cases. Any adapter needs a normalized, versioned token contract and tests for tie/bar/length/stress markers; do not feed raw debug tokens to a singing voicebank.

No corpus-level PER/WER was calculated: a manually reviewed reference lexicon and normalization policy do not yet exist. The artifact does, however, confirm that the current Muse letter fallback has concrete failure modes and that Piper Plus has its own contextual/OOV limitations.

**Decision:** retain Piper Plus as a comparative candidate, not a production phonemizer. The higher-value architecture is a pronunciation lattice: multiple candidates from a lexicon and one or more G2P frontends, context-sensitive variant selection, calibrated confidence/OOV detection, user/community corrections, and a frozen selected pronunciation plan consumed unchanged by score alignment and renderers. This turns uncertainty into an explicit system state rather than a silent substitution.


### Candidate B — CMU Pronouncing Dictionary

CMUdict is an established US-English pronunciation lexicon. Its upstream project states commercial and research use are unrestricted, requests attribution, and warns that errors, omissions and inconsistencies remain. It is useful as a supplemental lexicon/reference candidate, but must preserve variant pronunciations and regional limitations rather than treat one entry as universal truth.

### Candidate C — pure-Rust eSpeak NG port

The pure-Rust eSpeak NG port exposes text-to-phoneme output, and may be useful as a broad coverage comparison oracle. It is GPL-3.0-or-later. It should not be included in a commercial product distribution without an explicit license review; the current isolated spike does not depend on it.

### Benchmark protocol

For each fixture, compare current Muse, Piper Plus and any legally approved reference frontends against independently human-reviewed phones, syllable boundaries and stress. Report:
- phone error rate and per-phone confusion/deletion;
- word pronunciation accuracy, variant selection and heteronym accuracy;
- syllable-boundary F1 and lexical stress accuracy;
- OOV detection precision/recall and false-confidence rate;
- coverage by word-frequency band, dialect and accent;
- duration/stress transfer to singing score alignment.

Do not use a library's own output as ground truth for scoring that library. For ambiguous words such as *read* and *lead*, the reference must encode context-conditioned acceptable variants. OOV words should produce an explicit candidate/uncertainty state, never a silent guessed pronunciation.

### Sources

- Piper Plus G2P 0.5.3 documentation and feature/license details: https://docs.rs/crate/piper-plus-g2p/latest
- CMUdict project and license/accuracy notice: https://github.com/cmusphinx/cmudict
- Pure Rust eSpeak NG API and GPL-3.0-or-later license: https://github.com/eugenehp/espeak-ng-rs


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
- Score timeline: note/rest gap preservation, syllable and melisma overlap rejection, and explicit silence-token coverage.
- Provenance: exact source revision, score/performance digest, rendered WAV digest and explicit language/backend metadata attached to each run. Voicebank/model-weight tree digest remains a separate required receipt.

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

1. Reconcile the unintegrated physical-tract prototype with the actual `symthaea-vocal-tract` API; create one canonical module graph and compile its example/renderer before exposing it.
2. Run the dependency-free DiffSinger phone-contract tests and pinned Rust formatting checks.
3. Render identical English score/lyric cases through Kokoro, DiffSinger, and the physical tract backend; do not assume comparable quality before the outputs are measured.
4. Run SoulX-Singer on a small, rights-compatible subset of the same score-conditioned benchmark.
5. Resolve issue #10869: replace the small dictionary plus letter fallback with a versioned, uncertainty-aware pronunciation plan and a held-out lexical benchmark. Prepare a licensed Afrikaans singing pilot, followed by carefully reviewed isiZulu and isiXhosa pilots.
6. Keep the vocal-tract identity/gesture path as a research renderer until blind listening and held-out acoustic evidence justify a production claim.

## Claim boundary

This document and patch establish stricter input handling and expose the existing functional renderer. They do not establish human-level naturalness, SOTA performance, a trained Symthaea voice, or supported African-language singing.
