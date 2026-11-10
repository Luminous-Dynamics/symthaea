# Vocal phoneme coverage and South African singing research

**Date:** 10 November 2026  
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
- hardens `VocalPerformance` construction and validation (language mismatch, unsupported orthography, unpronounceable tokens, lyric/syllable mismatch, dropped lyric syllables, invalid note/phoneme/expression values, overlapping melismas, explicit melisma rests, and malformed timelines);
- adds a Muse G2P phone-inventory fixture so new source phones cannot silently lack a DiffSinger mapping;
- removes the unintegrated physical singer from the comparative release gate;
- adds exact per-case melody/performance receipts and BLAKE3 digests for serialized performance inputs and rendered WAV artifacts;
- records and repairs six source-only workspace paths and a stale `sha2` lockfile entry needed for a truthful feature compile.

No claim is made that the physical waveguide singer is exposed or buildable by this patch. The comparison gate now covers only the integrated Kokoro path and optional DiffSinger worker. Its `run_manifest.json` records the source revision, voice ID, optional worker path, per-case seed/lyrics/style/score notes, and serialized VocalPerformance digest. Each blind clip carries the exact WAV-file digest and its performance-input digest. The manifest does not yet attest the full voicebank/model-weight tree; that needs a separate provisioned-model manifest before claiming complete model provenance.


## Capability-qualified DiffSinger score timeline — 8 November 2026

The local DiffSinger worker now consumes a frozen `VocalPerformance` through a deterministic phone/timeline flattener in `scripts/diffsinger_phonemes.py`. The flattener emits ordered (phone, duration, F0) events, explicit `SP` silence events for absolute score gaps, and explicit melisma vowel events. It validates finite note timing and refuses overlaps. Consonant anticipation is represented by moving onset consonants before the target note onset and shortening the preceding vowel by the same amount; if there is no safe preceding vowel to borrow from, it fails rather than shifting or deleting a note.

The strict workflow now includes deterministic tests for:
- phone-duration conservation within a note and coda;
- absolute score-gap insertion as an `SP` event;
- consonant anticipation with duration borrowed from the preceding vowel;
- one-vowel melisma with coda released after the final melisma note;
- overlap rejection when borrowing would violate the minimum preceding vowel duration;
- invalid language/profile rejection.

This closes an important distinction between “worker produces some audio” and “worker follows the notated timeline.” It remains a score-conditioned synthesis path, not a claim of naturalness or SOTA. Vibrato, arbitrary continuous pitch curves, phrase expression, breath events, and true local editing remain unsupported by the current worker and are reported as such in its capability manifest.

## Executable backend capability receipts

The comparative release gate now requires every renderer to declare all 20 controls in `SingingBackendCapabilities`, checks renderer identity against the manifest, applies the score-conditioned render policy before synthesis, and records the complete manifest plus BLAKE3 digest with each objective clip. The private backend key also binds that digest to the blind clip.

The gate includes only currently integrated renderers: Kokoro, the formant-vocoder baseline, and optional DiffSinger. The physical waveguide prototype remains excluded pending issue #10860. A renderer cannot silently pass a score-conditioned comparison by ignoring an undeclared control. Approximations and conditional behavior are visible in the report.


## G2P replacement candidates and isolated spike — 7 November 2026

The current Muse English G2P is a small dictionary plus ASCII letter fallback. Its strict input checks prevent dropped characters and missing syllables, but do not prove correct pronunciation. Issue #10869 tracks the actual replacement.

### Candidate A — Piper Plus G2P 0.5.3

Piper Plus G2P 0.5.3 (26 October 2026) is a Rust crate documented as MIT-licensed and IPA-first. Its language backends include English (CMUdict JSON lookup, ARPAbet-to-IPA rules, function-word destressing and suffix-based morphology), Japanese, Chinese, Korean, Spanish, French, Portuguese and Swedish. The Rust English source notes that truly OOV words without a morphology match produce no output. That makes OOV detection an explicit benchmark dimension, not a reason to trust all returned tokens. The direct output API includes IPA tokens and prosody; its separate encoder supports strict unknown-token behavior. The spike now enables `bundled-dicts` and uses `EnglishPhonemizer::new_bundled()` so CI does not rely on a preinstalled CMU dictionary file.

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

- Piper Plus G2P 0.5.3 documentation and feature/license details: https://docs.rs/crate/piper-plus-g2p/0.5.3
- English implementation (CMUdict lookup, IPA mapping, OOV behavior): https://docs.rs/piper-plus-g2p/0.5.3/src/piper_plus_g2p/english.rs.html
- CMUdict project and license/accuracy notice: https://github.com/cmusphinx/cmudict
- Pure Rust eSpeak NG API and GPL-3.0-or-later license: https://github.com/eugenehp/espeak-ng-rs



### Pronunciation lattice prototype

An isolated schema/validator now exists at `tools/pronunciation-plan/`, with exact-head checks in `.github/workflows/pronunciation-plan-contract.yml`. It represents each source-text word span with multiple candidates, candidate provenance (source/provider/version/digest), normalized IPA phones, syllabification, stress, explicit review/OOV/unsupported state, and a selection receipt. Its renderer-facing `validate_renderable()` allows only explicitly selected candidates. This is a contract prototype; it does not perform G2P or contextual selection. The intended integration is: G2P systems propose candidates, a context/lexicon/human layer resolves them, and every renderer consumes the same frozen pronunciation artifact without making a new pronunciation decision.


## Integrated formant-vocoder baseline

The compiled voice tree already contained a real three-stage path: Muse singing bridge (lyrics/notes to sung phoneme events), formant-frame generation, and the root FormantVocoder. It was callable through a low-level `sing_to_pcm` helper but did not implement the shared `SingingVoiceEngine` contract, so it could not participate in the capability-qualified comparative gate.

A new `FormantSingingEngine` adapter now:
- consumes a frozen VocalPerformance and verifies that the current English Muse G2P reproduces its exact IPA/vowel/stress sequence;
- rejects phone or note-binding mismatches rather than silently re-running a different pronunciation;
- checks chronological non-overlapping note events;
- re-times irregularly spaced formant frames onto a fixed 200 Hz frame clock and inserts explicit silent frames for leading, internal and trailing rests;
- declares each of the 19 singing controls as native, approximated, conditional or unsupported;
- explicitly marks melisma and fixed vibrato as approximations and voice identity, local edits, breathiness, phrase-expression controls, and custom portamento as unsupported.

This is a **conventional formant/source-filter baseline**. It is not the missing waveguide/gesture physical renderer tracked in issue #10860, and it does not establish human-level naturalness. It is useful because it adds a third integrated renderer to the identical-case release gate without pretending that an uncompiled prototype is functional.

## New G2P frontier: user-generated text

A recent benchmark directly changes how we should approach lyric normalization. UGTPHON, published in Findings of EMNLP 2026, studies noncanonical user-generated text in English, Vietnamese and Korean. It reports canonical/noncanonical G2P gaps reaching 66.8 PER points in its studied settings, and a retrieval-augmented staged approach that reduces this gap by up to 83% with a model below 0.5B parameters. The key lesson is architectural: lexical normalization and pronunciation should be explicit stages, with retrieval and compositional inference, rather than asking a single phonemizer to infer every intent from surface spelling.

Source: https://aclanthology.org/2026.findings-emnlp.1020/

The isolated G2P spike now contains an additional UGT suite for abbreviations, initialisms, noncanonical spellings, repeated-letter elongation, slang and digit/letter mixtures. Its review rows are intentionally pending. This lets us compare raw candidate behavior while requiring independent, contextual adjudication before scoring canonical pronunciation. No global rewrite such as `u -> you` is permitted without context and an explicit selected hypothesis.

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

## Phrase-aware performance authoring

The shared `VocalPerformance` model now offers `from_melody_with_phrase_specs` alongside the compatibility `from_melody` and `from_melody_with_phrase_ends` constructors. Each `VocalPhraseSpec` declares an exclusive syllable end, breath-before and breath-after durations, and an independent start/peak/release expression profile for energy, breathiness and tension.

These are authoring controls, not a claim that the renderers synthesize breaths. The 20-control capability schema separates `PhraseBreathTiming` from `AbsoluteOnsetAndRests`: explicit musical rests are represented in note timelines and SP/silent events, whereas breath-before/after metadata is currently unsupported by Kokoro, DiffSinger and FormantSingingEngine. A phrase-expression qualification track should therefore require the former metadata at the plan level and only award rendering credit after acoustic breath-event evidence exists.

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

## 2026 multilingual phonology benchmark update

Two recent EMNLP 2026 benchmarks materially change how the G2P replacement should be evaluated.

### mPhonBench — cross-language phonological competence

mPhonBench covers 10 tasks, 58 languages, and 19 scripts. Its reported findings are relevant to the South African roadmap: mid-sized language models can match or exceed expert systems on some G2P tasks, but underrepresented languages still suffer a performance penalty. Few-shot examples help partially; generic phonological descriptions alone are not reliably exploited.

Source: https://aclanthology.org/2026.emnlp-main.2456/

**Implication:** avoid a language-count target. A language is only supported for a task after word-level, phoneme-level, and dialect-specific held-out evaluation. The registry should explicitly record text normalization, G2P, code-switching, ASR, speech synthesis, singing synthesis, and community review separately.

### UGTPHON — canonical and non-canonical text are different tasks

UGTPHON evaluates English, Vietnamese, and Korean user-generated text. Its authors report a canonical-to-non-canonical pronunciation error gap of up to 66.8 PER points for existing systems; a retrieval-augmented, staged compositional method with fewer than 0.5B parameters reduced that gap by up to 83% in their experiments.

Source: https://aclanthology.org/2026.findings-emnlp.1020/

**Implication:** contractions, abbreviations, slang, names, dialect spellings, and code-switches should not be handled by a single spelling-to-phone fallback. Introduce a normalization/retrieval stage that retains the original surface form and context, proposes canonical-form hypotheses, and then asks the pronunciation lattice to resolve them. Retrieval results remain candidates, never ground truth.

### Singing annotation and quality evaluation

STARS (Findings ACL 2025) unifies singing transcription, phoneme alignment, note transcription, vocal-technique labels, and global style annotation. It is a useful reference for building the annotation pipeline, while the independent gold subset still needs human adjudication and singer/song-disjoint splits.

Source: https://aclanthology.org/2025.findings-acl.781/

SingMOS-Pro provides 7,981 clips from 41 systems and 12 datasets with at least five professional ratings per clip, including lyric, melody, and overall-quality dimensions. It should be used to calibrate and screen quality predictors, not as a replacement for blind human comparisons on our own languages and control tasks.

Source: https://arxiv.org/abs/2510.01812/

### Revised benchmark protocol and proposed gates

Keep these as separate tracks; do not collapse them into a single score.

| Track | Core measures | Proposed qualification target |
|---|---|---|
| Contextual G2P | PER, exact word-variant accuracy, heteronym selection, per-phone confusion | Report per language/dialect and by word-frequency/OOV stratum; target ≥95% exact accuracy on reviewed in-domain English pilot, with no silent fallback |
| Selective prediction | Coverage versus selective PER, OOV precision/recall, Brier score and calibration | At an explicitly chosen coverage point, target ≤1% error among accepted high-confidence predictions; ambiguous/OOV words must remain unresolved |
| Score/phone compliance | Missing/duplicated/reordered phones, event coverage, rest leakage, melisma legality | Zero hard symbolic-contract violations on synthetic controls and reviewed fixed-score test cases |
| Temporal realization | Phone onset/offset median and p95 error, note FFE/GPE/VDE, coda-release error | Initial target: phone-boundary p95 ≤30 ms and zero overlap/rest violations; publish per-phone and per-register distributions |
| Expressive control | One-factor-at-a-time control-response matrix, intended-change success, collateral change | Target ≥90% directional success for supported controls, with <5% collateral change outside the target region |
| Perceptual quality | Blinded naturalness, intelligibility, expression, identity, artifacts and pairwise preference | Non-inferiority to the strongest reproducible score-conditioned baseline on shared cases before claiming a quality advantage |

These are proposed qualification gates, not existing results or published SOTA. The 95% G2P target applies only to the initial reviewed English in-domain pilot, not to all accents or South African languages. The 30 ms boundary target is an engineering target to be validated against human annotator uncertainty and frame/sample resolution.

A credible superiority claim should require:
1. exact reproduction of at least one score-conditioned external baseline on the same held-out lyrics, score, language, singer and voice-reference cases;
2. higher hard-constraint compliance and better controllability on the same cases;
3. no statistically meaningful loss in human-rated naturalness or intelligibility;
4. confidence intervals, per-case receipts, failures, abstentions and exclusions;
5. separate reports for prompt-to-song systems and fixed-score singing systems.

For South African languages, start with language-specific reviewer agreements and a small licensed, consented gold corpus. Do not infer isiZulu/isiXhosa singing competence from multilingual speech-model results or from a public dataset whose terms prohibit synthesis use.

## Next experiments

1. Reconcile the unintegrated physical-tract prototype with the actual `symthaea-vocal-tract` API; create one canonical module graph and compile its example/renderer before exposing it.
2. Run the dependency-free DiffSinger phone-contract tests and pinned Rust formatting checks.
3. Render identical English score/lyric cases through Kokoro, DiffSinger, and the physical tract backend; do not assume comparable quality before the outputs are measured.
4. Run SoulX-Singer on a small, rights-compatible subset of the same score-conditioned benchmark.
5. Resolve issue #10869: replace the small dictionary plus letter fallback with a versioned, uncertainty-aware pronunciation plan and a held-out lexical benchmark. Prepare a licensed Afrikaans singing pilot, followed by carefully reviewed isiZulu and isiXhosa pilots.
6. Keep the vocal-tract identity/gesture path as a research renderer until blind listening and held-out acoustic evidence justify a production claim.

## Claim boundary

This document and patch establish stricter input handling and expose the existing functional renderer. They do not establish human-level naturalness, SOTA performance, a trained Symthaea voice, or supported African-language singing.

### Formant phrase-expression experiment

The integrated FormantSingingEngine now interpolates phrase expression points over the frame timeline. It maps energy to a relative frame-energy scale, breathiness to reduced periodic voicing plus modest resonance widening, and tension to resonance-bandwidth narrowing. This is an intentionally approximate source-filter control mapping—not a calibrated vocal-fold biomechanical model. Tests assert the numerical transformation on a synthetic frame, while perceptual effects still require listening and acoustic evaluation. Phrase breath-before/after metadata is not rendered as an inhalation and remains unsupported.


## Multilingual phonology and singing datasets — 8 November 2026

A new EMNLP 2026 benchmark, **mPhonBench**, evaluates text-only and speech-augmented models across 10 phonological tasks, 58 languages and 19 scripts. Its authors report that mid-sized open models can match or exceed specialist G2P systems on some tasks, while underrepresented languages still incur a performance penalty. This changes the G2P strategy: compare dictionary/rule systems and multilingual models against a broad phonological task suite, not only a small English pronunciation test. The released benchmark's language manifest must be inspected before claiming isiZulu or isiXhosa coverage; the paper abstract alone does not verify either language.

GTSinger is a valuable singing-alignment/control reference corpus: 80.59 hours, 20 singers, nine languages, six labelled techniques, manual phoneme-to-audio alignments, realistic music scores and 16.16 hours of paired speech. Its currently published language set is Chinese, English, Japanese, Korean, Russian, Spanish, French, German and Italian; it does not supply the South African language coverage required by this programme. Critically, its dataset licence is **CC BY-NC-SA 4.0**, not a commercially permissive licence. The dataset card also includes an additional agreement clause concerning responsibility/indemnity for M4Singer audio and representation of authority to accept the terms. Do not use this corpus to train or validate a commercial/proprietary model without separate legal review and an appropriate rights basis. It may be useful only for work that is genuinely compatible with the non-commercial/share-alike obligations and all additional terms.

SoulX-Singer-Eval provides a more directly comparable zero-shot evaluation protocol: 100 singing segments from 50 distinct speakers (25 Mandarin, 25 English), with two segments per speaker and manually annotated melody. Its README describes academic-consent Mandarin data and English segments sourced from Mixing Secrets. This is a useful open baseline, but it does not cover South African languages and is not a substitute for rights-cleared commercial voice data.

### Updated research implications

1. **Use mPhonBench for broad phonological ability, not as a South African singing gold set.** It can test G2P, phonological reasoning, language/script generalization and word-level phone recognition; we still need language-community reviewed references and actual singing data.
2. **Use GTSinger only inside its rights boundary.** It is exceptionally useful for studying phoneme timing, technique control, and score-to-audio alignment, but its non-commercial licence conflicts with unrestricted commercial deployment. Do not mix its training material into a commercial model by default.
3. **Build a separately consented South African singing benchmark.** Start with English, Afrikaans, isiZulu, isiXhosa and Sesotho as separate profiles, but do not turn a language name into a coverage claim. For each language, specify dialect/region, orthography, phoneme inventory, tone/stress/prosody conventions, code-switching policy and a reviewer/adjudication plan.
4. **Split by singer, song, lyric and pronunciation lemma.** A singer-held-out test alone is insufficient if lyrics, melody or recording sessions leak across splits. Maintain a separate held-out set of names, loanwords and code-switching cases.
5. **Keep speech and singing labels distinct.** mPhonBench-style spoken word G2P is necessary but does not prove sung syllabification, consonant timing, melisma behavior, or score alignment.

Sources:
- mPhonBench, EMNLP 2026: https://aclanthology.org/2026.emnlp-main.2456/
- GTSinger dataset card: https://huggingface.co/datasets/GTSinger/GTSinger
- GTSinger dataset licence: https://huggingface.co/datasets/GTSinger/GTSinger/blob/main/dataset_license.md
- GTSinger NeurIPS 2024 paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html
- SoulX-Singer-Eval protocol: https://github.com/Soul-AILab/SoulX-Singer-Eval


### South African speech resource clarification

The **CSIR SAMA Speech Corpus Manual Datasets** and the **NCHLT Speech Corpus** are distinct collections and must not be described as one resource. The CSIR SAMA SADiLaR record lists orthographically transcribed broadband speech across Afrikaans, isiXhosa, isiZulu, Sepedi, Sesotho and Tshivenḓa, with a **CC BY 4.0** licence. NCHLT is a separate corpus covering all 11 official languages, described in the ML-SUPERB data catalogue as approximately 10 minutes of speech per language, with **CC BY 3.0**. Neither is a singing dataset; NCHLT's per-language duration is only a smoke-test resource, not evidence of production G2P coverage.

- CSIR SAMA corpus metadata and licence: https://repo.sadilar.org/items/e44a1b0c-34c5-4957-9560-4266ca17c7d2
- NCHLT description, per-language coverage and licence: https://multilingual.superbbenchmark.org/challenge-interspeech2025/data_description
- NCHLT speech-corpus project and phone-set/dictionary resources: https://sites.google.com/site/nchltspeechcorpus
