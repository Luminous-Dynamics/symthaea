# Symthaea Voice: Multilingual and Interspecies Roadmap

**Status: research plan and capability audit, 2026-10-12.** This document distinguishes existing behavior from target capabilities. It does not claim that Symthaea currently speaks every language, produces click consonants, or translates animal communication.

## Findings from the current code

- The custom functional singing path maps a bounded English IPA subset to its ARPAbet-like tract-planner inventory. The mapper has no click symbols and rejects unsupported IPA instead of substituting a different sound.
- The DiffSinger bridge only accepts an explicit English language ID and a fixed English phone map. Unknown languages and phonemes are rejected.
- The Kokoro path uses a Misaki/Kokoro phoneme vocabulary designed around English phones in the current Rust integration. That is not evidence of all-language coverage.
- These paths are one part of the voice stack. The existence of speech recognition, a multilingual translation model, or a bioacoustic classifier elsewhere in the system would not by itself prove end-to-end spoken fluency or animal translation.

## Can Symthaea eventually speak every human language?

It is a reasonable long-term goal to provide useful speech in as many human languages as possible. “Every language” is not an achievable claim without an explicit coverage registry and per-language evidence. UNESCO documents more than 8,000 languages in its atlas, with roughly 7,000 still in use; data, orthography, dialect variation, and community access are uneven. [UNESCO language overview](https://www.unesco.org/en/multilingualism-linguistic-diversity) reports the current landscape.

Existing models establish that large-scale coverage is technically possible, but not universal or uniformly high-quality:

- Meta's [Massively Multilingual Speech project](https://ai.meta.com/research/publications/scaling-speech-technology-to-1000-languages/) reported ASR models for 1,107 languages, TTS models for 1,107 languages, and language identification for 4,017 languages. This is a strong starting point for adapters and baselines, not a claim of native-quality speech in every language.
- Meta's [SeamlessM4T model card/documentation](https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/README.md) lists 101 speech-input languages, 96 text-input/output languages, and 35 speech-output languages. Its different task counts illustrate why input-language coverage must not be conflated with output-language coverage.
- [Coqui XTTS v2 documentation](https://docs.coqui.ai/en/latest/models/xtts.html) lists 16 supported TTS languages in its current documentation. It is an example of a multi-language voice model, not an all-language solution.

### Recommended architecture

1. **A language capability registry.** Key every tested capability by BCP-47 tag (and where necessary a regional/dialect profile), script, model/checkpoint digest, licence, and supported tasks. Track text normalization, ASR, translation, G2P, TTS, code-switching, and streaming as separate axes.
2. **Model adapters rather than a single universal phoneme map.** Reuse suitable multilingual ASR/translation/TTS models where the licence, platform, privacy, and latency are acceptable. Keep Symthaea's native symbolic and articulatory path as a research backend, not as an unproven replacement for every language model.
3. **Language-specific phonology.** Add an explicit grapheme-to-phoneme module, phoneme inventory, allophony, stress/tone, timing, and coarticulation profile per language. IPA is a notation, not an audio generator; merely accepting an IPA symbol does not produce its correct acoustics.
4. **Evidence-gated language onboarding.** A newly requested language stays `experimental` until text coverage, G2P accuracy, phoneme coverage, ASR WER/CER, number/name/negation comprehension, pronunciation review by qualified speakers, and voice quality have recorded results. Preserve dialect and speaker variation; do not make one reviewer a gatekeeper for an entire community, and do not force community data contributions.
5. **Learning loop.** Accept consented, licensed text/audio pairs or corrections; freeze a training/evaluation split by speaker and source; train or adapt the relevant model; run regression tests on already supported languages; attach model and corpus digests; promote only the specific task/language that met its declared criterion. Avoid evaluating with training clips.
6. **Coverage without false equivalence.** Publish per-language tiers such as `unverified`, `experimental`, `automated-checks-passed`, and `community-reviewed`. Never collapse “the model emits audio” into “the language is supported well.”

### Immediate multilingual baseline: model licence and data limits

A snapshot checked on **2026-10-12** narrows the initial adapter plan:

- Meta's [MMS TTS model card](https://huggingface.co/facebook/mms-tts) lists 1,107 language models and a **CC-BY-NC 4.0** licence. Treat it as a research comparator only unless legal review establishes that the exact intended use is compatible or a separate commercial permission applies. Do not silently treat broad language coverage as a commercially reusable dependency, and do not infer quality from the language count.
- [Common Voice Scripted Speech 27.0 — Zulu](https://mozilladatacollective.com/datasets/cmu610aue00jtnq07hnb00xio) is listed under CC0-1.0 but currently has 223 clips (0.34 hours) from 9 speakers, only 29 validated clips, and 3 clips in its listed test split. The dataset page explicitly forbids attempting to identify speakers and re-hosting or re-sharing the dataset.
- [Common Voice Scripted Speech 27.0 — Xhosa](https://mozilladatacollective.com/datasets/cmu60x5rv00ilo1070jbpgna2) is listed under CC0-1.0 but has 35 clips (0.06 hours) from 5 speakers, only 13 validated clips, and 4 clips in its listed test split. Its page states the same no-speaker-identification and no-rehosting restrictions.

These Common Voice snapshots can exercise ingestion, decoding, tokenizer/G2P wiring, and a few reproducible smoke cases. Their tiny validated/test partitions are **not sufficient evidence of robust ASR, dialect coverage, voice quality, or synthesis capability**. Do not train or promote a model based on these clip counts alone. The next responsible step is to pin permitted artefacts and metadata, build a language-aware pipeline smoke test, then gather additional consented/licensed audio and a held-out-speaker evaluation set with language-community input. The registry intentionally keeps Afrikaans, isiZulu, and isiXhosa unverified; the existence of these datasets does not alter that status.

### Candidate ASR baseline: BuzzASR Afrikaans

A particularly relevant publicly documented candidate is [BuzzASR/afrikaans](https://huggingface.co/BuzzASR/afrikaans). Its model card declares an MIT licence, describes it as a language-specialized fine-tune of Whisper large-v3 that retains Whisper's tokenizer, and reports **normalized WER 4.21% on Common Voice 25** versus **22.36% on FLEURS**. It is a useful comparator, not a Symthaea result. The two reported WERs differ sharply, which is a concrete reminder that dataset and domain change the result; metrics are reproduced here only as model-card-reported claims, not independently verified measurements. The associated [BuzzASR paper](https://arxiv.org/abs/2609.09554) describes a broader suite of 102 language-specific models.

The current hypothesis is to export a fixed BuzzASR snapshot, attempt the documented [whisper.cpp fine-tuned-model conversion path](https://github.com/ggml-org/whisper.cpp/blob/master/models/README.md#fine-tuned-models), and run it through a pinned local worker. Conversion must be validated before any result is trusted: the converted model, converter commit, tokenizer/config files, quantization mode and runtime build must all be pinned and hashed. A Q5-quantized artifact is a **separate candidate** from F16; compare both on identical clips and don't assume equal quality. Do not automatically fetch model weights from the runtime or CI.

Important evaluation caveat: BuzzASR reports training on FLEURS and Common Voice 25. A Common Voice 27 run is newer, but that fact alone does **not** prove there is zero overlap of contributors or audio across releases. The scorer below records exact local artifacts and held-out splits, but it cannot establish that a candidate model never saw the same audio or source text during training. Record the model's training-corpus provenance separately; treat overlap as `not_assessed` unless it has been independently checked. Do not use the reported Common Voice 25 score as an unbiased, apples-to-apples baseline for a Common Voice 27 result.

### Reproducible WER/CER scorer

I added `scripts/voice_asr_score.py` plus `scripts/test_voice_asr_score.py`, wired into the CI `muse` job. The scorer itself runs on the Python standard library; model inference remains a separate local adapter/runtime step. It requires an intake report, the exact local archive and test metadata, a regular-file model export, and one prediction for every test row. The prediction JSON pins locale, dataset ID/version, archive SHA-256, test.tsv SHA-256, model ID, exact 40-hex model repository revision and declared licence.

For the report it computes micro-averaged WER and CER, per-reference-length buckets and deterministic 95% percentile-bootstrap confidence intervals. Text normalization is explicitly versioned (`NFKC`, Unicode case-folding, punctuation-to-space, whitespace collapse) and preserves diacritics. It records an SHA-256 manifest of every model-export file. Reports contain only aggregate metrics and hashes—not transcripts, clip paths or speaker tokens—and use exclusive-create semantics to prevent overwrite. Empty ASR output counts as deletions rather than being silently dropped; missing, duplicate or unknown predictions fail the evaluation.

Example, after producing a complete local prediction JSON using an approved adapter:

```sh
python3 scripts/voice_asr_score.py \\
  --dataset-root "$CV_ROOT/af" \\
  --archive "$CV_ARCHIVE" \\
  --intake-report "$PRIVATE_DIR/af-dataset-intake.json" \\
  --predictions "$PRIVATE_DIR/af-asr-predictions.json" \\
  --model-root "$PRIVATE_MODEL_EXPORT" \\
  --locale af \\
  --model-id BuzzASR/afrikaans \\
  --model-revision "$MODEL_REVISION_40_HEX" \\
  --model-license MIT \\
  --mode smoke \\
  --report "$PRIVATE_DIR/af-asr-score.json"
```

The predictions file has schema version 1, the pinned dataset/model metadata described above, and a `predictions` array of objects with exactly `path` and `text`. It contains the private transcripts and must remain in a private directory. `--mode smoke` is exploratory only. `--mode heldout-evaluation` additionally requires the intake audit's speaker-disjoint mode, at least 100 test clips and at least 10 test speakers. Neither mode alone promotes the language registry. No model was downloaded or run during this change, and no WER/CER result is being claimed for Symthaea.

### First runnable dataset-intake gate: Afrikaans ASR

The first additional-language *pipeline smoke target* is **Afrikaans** (`af`). This is a prioritization choice for a locally relevant Latin-script language and an available public ASR dataset, not a claim that Afrikaans is already supported or that this corpus has sufficient quality for production. Mozilla announced Common Voice Scripted Speech v27.0 on **September 18, 2026**, covering 295 languages overall; that global figure is not a per-language coverage metric ([release announcement](https://discourse.mozilla.org/t/common-voice-scripted-speech-v27-0-and-spontaneous-speech-v5-0-released/149490)). The current dataset index lists Afrikaans as a CC0-1.0 ASR corpus; use the exact locale page and release metadata, not just the index, when pinning a run ([Mozilla Data Collective datasets](https://mozilladatacollective.com/datasets)).

I added `scripts/voice_dataset_intake.py` and its dependency-free tests in `scripts/test_voice_dataset_intake.py`, and wired the tests into CI. The tool:

- Requires a language tag, exact dataset release and ID, source page URL, an operator-supplied licence identifier, an archive digest, and explicit acknowledgment of Common Voice's no-speaker-identification and no-rehosting/redistribution restrictions.
- Audits `train.tsv`, `dev.tsv`, and `test.tsv` for required columns, nonempty text, duplicate clip paths, safe relative paths, nonempty referenced audio files, and stable metadata SHA-256 digests. It rejects symbolic-link dataset roots/audio roots/files and an empty or symlinked archive.
- Reports only aggregate row/speaker counts and pairwise speaker-overlap counts; it does not print speaker tokens, transcripts or per-clip paths. The hashed Common Voice `client_id` is used only for overlap detection, never for attempting identification.
- Has two explicitly different modes: `ingestion-smoke` checks whether the local dataset is structurally usable but never claims ASR quality; `heldout-evaluation` additionally requires pseudonymous speaker IDs on all rows, zero cross-split speaker overlap, at least 100 test clips, and at least 10 test speakers. These are a basic exploratory-data floor, not a statistically validated or production-release threshold.

Example command after the corpus is acquired lawfully and extracted into a local directory (set the exact dataset ID/URL from its page; keep the output under a private directory):

```sh
python3 scripts/voice_dataset_intake.py \\
  --dataset-root "$CV_ROOT/af" \\
  --archive "$CV_ARCHIVE" \\
  --locale af \\
  --dataset-version cv-corpus-27.0-2026-09-11 \\
  --dataset-id "$CV_DATASET_ID" \\
  --source-url "$CV_SOURCE_URL" \\
  --license-id CC0-1.0 \\
  --acknowledge-usage-restrictions \\
  --mode ingestion-smoke \\
  --report "$PRIVATE_DIR/af-dataset-intake.json"
```

The example was **not executed** here: no Common Voice archive or licensed audio was downloaded into the repository. The report pins the supplied archive digest and each split-metadata digest, but the operator must still verify the exact source-page metadata and local dataset provenance. When creating a speaker-disjoint evaluation partition, preserve only the dataset's provided pseudonymous IDs for assignment; do not infer real identity, and do not publish the audio or metadata export. This dataset gate is a prerequisite to an ASR adapter experiment, not the adapter experiment itself.

## Can she produce click consonants?

**Not through the current custom singing path, and not reliably in the current DiffSinger bridge.** The five core click place symbols in the IPA are `ʘ` (bilabial), `ǀ` (dental), `ǃ` (post-alveolar), `ǂ` (palatoalveolar), and `ǁ` (alveolar lateral). These are non-pulmonic, velaric-ingressive articulations with a different pressure/closure/release mechanism from ordinary English stop consonants. See the [International Phonetic Association chart](https://www.internationalphoneticassociation.org/IPAcharts/IPA_charts_EI/IPA_charts_EI.html) and [University of Newcastle IPA consonants guide](https://teaching.ncl.ac.uk/ipa/consonants-non-pulmonic.html).

An English fallback that spells the letter “x” as `ks` is not an isiXhosa or isiZulu click implementation: those languages use orthographic click letters and combinations whose pronunciation depends on the language's phonology. Current click symbols should fail closed, not be approximated as `k`, `t`, `s`, or a generic burst.

### Click-support engineering path

- Add a typed phoneme event such as `Click { place, phonation, nasalization, aspiration, release }`, distinct from the English ARPAbet table.
- Implement a transient constriction/release event in the tract or waveguide renderer. The target must represent closure geometry, oral cavity pressure/volume change, release burst, voicing/nasal accompaniment, and context-dependent transitions. Static vowel formants alone are insufficient.
- Add language-specific G2P for isiXhosa, isiZulu, and other relevant languages; test actual contrastive words and clusters instead of isolated click symbols only.
- Measure burst timing, spectral shape, contrast identification and word comprehension against naturally produced recordings, while respecting speaker consent and language community expertise.
- Do not switch the default English mapper to a lossy approximation. Click-enabled output becomes a separately qualified capability with explicit test coverage.

The new regression tests intentionally assert that the five symbols are rejected until a click-specific renderer is implemented.

## Can she speak to animals?

**She can be developed into a tool for studying and participating in limited, evidence-based interspecies interaction. She cannot currently be said to translate animal language or hold human-like conversations with animals.**

A promising starting point is animal-sound understanding, not imagined sentence translation. The open [NatureLM-audio project](https://github.com/earthspecies/NatureLM-audio) and its [capability guide](https://projects.earthspecies.org/naturelm-audio/1.0/prompting_guide.html) describe sound detection/classification, call-type tasks and audio captioning. The project's own guidance separates core evaluated tasks from experimental use. These are ways to build and compare bioacoustic baselines, not proof of decoding animal semantics.

### A rigorous, welfare-first development sequence

1. **Perception before translation.** Begin with one species and well-defined tasks: call detection, vocalization type, individual identity where validated, and correlations with independently observed context/behavior. Combine audio with timing, environment and (where appropriate and consented) non-invasive posture/video context. Keep model predictions distinct from observed facts.
2. **Test semantic hypotheses.** Use held-out individuals, locations and time periods; compare against simple acoustic and context-only baselines; report uncertainty and confounders. A caption such as “alarm-like call” is not the same as evidence that an animal intended a particular sentence.
3. **Build an interaction model.** Map a small set of validated signals or learned cues to safe responses: provide space, offer a learned choice, stop an action, or let a human caregiver respond. Require an observable outcome and a cautious interpretation; allow a no-response/unknown result.
4. **Output should be species-appropriate.** If playback is justified, use documented naturalistic calls or cues with the correct species and context, controlled volume, duration and stop conditions—not Symthaea's human voice pretending to be an animal. Don't synthesize or play distress/alarm calls without specialist approval.
5. **Ethics and welfare are hard constraints.** Use animal-ethics/veterinary expertise, minimize disturbance, never induce distress to obtain labels, and stop experiments on stress/avoidance signals. Maintain a clear human review path and never treat uncertain model interpretations as permission to manipulate an animal.

A 2025 open-access review of ethical issues around AI-mediated interspecies communication stresses the risk of misinterpreting signals and the need for welfare, autonomy and privacy safeguards ([Küçükuncular, *AI and Ethics*](https://link.springer.com/article/10.1007/s43681-025-00828-z)). A 2025 linguistics review emphasizes that human–animal interaction is multimodal and that human categories of sound do not always fit animal vocalizations ([Mondémé, *Language & Communication*](https://www.sciencedirect.com/science/article/pii/S0271530925000680)).

## First milestones

- **M1 — Honest capability manifest:** implemented as `scripts/voice_language_registry.json`, validated by `scripts/validate_voice_language_registry.py`, with evidence-gate regressions in `scripts/test_voice_language_registry.py` and execution wired into the CI `muse` job. The initial manifest has four human-language tags and 32 task records; only three English code paths are marked `implementation_present`. All other task records remain `unverified`, including Afrikaans, isiZulu and isiXhosa. No language is marked production-qualified.
- The registry uses BCP 47-shaped language tags based on [RFC 5646](https://www.rfc-editor.org/rfc/rfc5646); the local validator checks tag syntax and case-insensitive uniqueness, not the full IANA subtag registry or canonicalization rules.
- Model and dataset evidence is deliberately explicit. A dataset being publicly downloadable does not imply every speech recording has the same licence or provenance; for example, [Mozilla Common Voice terms](https://commonvoice.mozilla.org/dav/terms) describe CC0 contributions and contributor assurances. Pin the exact dataset version/digest and record its licence rather than relying on its project name.
- The validator checks evidence shape only. It does not contact GitHub to verify that a run passed, nor does it measure actual language quality. Promotion to a higher status still requires independent review of the linked artifacts and exact commit.

- **M2 — Human-language adapter baseline:** compare an appropriate multilingual ASR/translation/TTS stack on a small set of high-value languages, starting with English, isiZulu, isiXhosa, Afrikaans and additional languages chosen with speakers. Keep translation, recognition and generation scores separate.
- **M2 — Human-language adapter baseline:** compare an appropriate multilingual ASR/translation/TTS stack on a small set of high-value languages, starting with English, isiZulu, isiXhosa, Afrikaans and additional languages chosen with speakers. Keep translation, recognition and generation scores separate.
- **M3 — Click prototype:** construct a click-specific transient event in isolation and test generated audio against recorded exemplars before placing it inside connected speech.
- **M4 — Animal sound baseline:** choose one species/task with a usable ethical dataset, pin a model and dataset, and evaluate on held-out recordings. No “translation” label until independently validated.
- **M5 — Controlled interaction trial:** only after M4, collaborate with relevant animal-behavior experts on minimal-risk, observable-outcome trials.

## Decision rule

Treat every output as a measured capability, not a universal identity claim. The system may learn language-specific skills and animal-signal associations through data, models, and feedback, but only the task and domain supported by evidence should be advertised. Unknown language, phone, species, or communicative intent must remain explicitly unknown rather than being guessed into fluent-sounding output.
