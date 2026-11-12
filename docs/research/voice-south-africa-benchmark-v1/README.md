# South African speech benchmark cohorts v1

**Status:** metadata-only cohort registry. No audio, transcript labels, speaker identifiers, access tokens or derived embeddings are copied into this repository.

Parent research: [#10913 — VOICE-SA-001](https://github.com/Luminous-Dynamics/symthaea/issues/10913). Language-learning programme: [#9607](https://github.com/Luminous-Dynamics/symthaea/issues/9607).

## Why a cohort registry comes first

South African speech resources do not form one interchangeable dataset. They include read speech, scripted telephone speech, unscripted/domain speech, broadcast/code-switched conversation and text-only synthetic code-switching. Their licences, speaker populations, label quality, splits and permitted uses differ.

The registry at `cohorts-v1.json` keeps those distinctions explicit. Every corpus must be pinned to an immutable revision and exact source files before an evaluation adapter is written. A dataset card or aggregate licence does not automatically clear every underlying source recording for model training, commercial use, voice synthesis, cloning or public redistribution.

## Core external references

### African-wide baselines

- **PazaBench (2026):** evaluates 52 ASR models across 39 African languages, harmonizing six public corpora and reporting WER, CER and RTFx. It is the broad comparative reference; the released Paza models described by Microsoft currently cover six Kenyan languages and must not be mistaken for a South African model release. [Paper](https://www.microsoft.com/en-us/research/publication/pazabench-large-scale-asr-benchmark-for-low-resource-languages-in-africa/) · [Project](https://labs.ai.azure.com/innovations/paza/)
- **AfriVox (EACL 2026):** tests speech LLMs and conventional ASR/AST on African languages and accents, including domain-specific and unscripted speech. It reports substantial variation between nominally supported languages. [Paper](https://aclanthology.org/2026.eacl-long.122/)

### South African monolingual / domain cohorts

- **NCHLT Auxiliary:** a combined collection describing more than 1,420 hours across the 11 South African spoken-language resources and roughly 3,400 speakers. Pin original language source collections and official splits; do not assume the combined mirror changes the source licence. [Dataset card](https://huggingface.co/datasets/dsfsi-anv/multilingual-nchlt-dataset)
- **Lwazi:** scripted telephonic speech for nine South African languages. Useful as a distinct channel/domain cohort, not as a substitute for natural conversation. Check per-language original release licences and splits. [Dataset card](https://huggingface.co/datasets/dsfsi/lwazi-asr-corpus-compressed)
- **African Next Voices / Swivuriso:** a large community-oriented corpus/project for seven South African languages with scripted and unscripted domains. Its visible access terms prohibit TTS, voice cloning and voice synthesis; this registry therefore treats it as ASR-only and requires acceptance of exact current terms. [Dataset page](https://capstone.aims.ac.za/datasets/african-next-voices/) · [Project](https://www.dsfsi.co.za/za-african-next-voices/)

### Code-switch evaluation

- **Five-language soap-opera corpus:** 26.9 hours of English mixed with isiZulu, isiXhosa, Setswana and Sesotho, with ad-lib/spontaneous examples. The Stellenbosch catalogue states research-only use. Treat it as an evaluation cohort only unless the rights holder confirms otherwise. [Repository record](https://hdl.handle.net/20.500.12185/545) · [Interspeech 2018 baseline paper](https://www.isca-archive.org/interspeech_2018/ylmaz18c_interspeech.html)
- **AfriVox-v1/v2:** separate releases with different metadata and terms; the cards report African language coverage including several South African languages. Preserve the version and annotation-method distinction. [v1](https://huggingface.co/datasets/intronhealth/afrivox) · [v2](https://huggingface.co/datasets/intronhealth/afrivox-v2)
- **AfroCS-xs:** text-only, human-validated synthetic code-switching for Afrikaans, Sesotho, isiZulu, Yoruba and English. It may support a text/translation diagnostic; it is not audio or ASR ground truth. [Paper](https://aclanthology.org/2025.acl-long.1601/)

## Evaluation design

Keep five lanes separate:

1. **Read / scripted monolingual:** exact transcript and orthography; report WER and CER.
2. **Telephonic scripted:** channel robustness, noise/codec and named-entity preservation.
3. **Unscripted / domain speech:** source- and speaker-disjoint evaluation by domain.
4. **Natural code-switch:** mixed-language WER/CER, language-span and switch-boundary detection, and intra-word/morphological switches where annotated.
5. **Full-duplex conversation:** add timestamps, overlap, interruptions and addressee behavior; a final offline transcript score does not qualify streaming interaction.

For every cohort, report per-language scores and sample counts. Preserve strict orthographic scoring and a separately declared normalization view. Do not macro-average away a weak language or treat unsupported profiles as zero-error.

## Baseline matrix

Where accessible and licensed, compare exact dated revisions of Whisper, MMS, XLS-R, W2v-BERT, suitable CTC/encoder-only systems, Paza models where the language is supported, and speech-native models/APIs. Include a pinned local baseline. Keep acoustic frontend, text normalization, language model/lexicon, decoding budget, compute and latency conditions explicit.

Symthaea-specific comparisons should be staged:
- DSP-only acoustic features;
- current offline direct-mel HDC frame encoder;
- live encoder after the parity fix in #10912;
- shared HDC–CfC candidate in #10842 only after modular baselines are stable.

No improvement is attributed to HDC unless it beats DSP and multilingual foundation baselines at matched compute/data budget.

## Mandatory admission gates

- Pin exact dataset repository revision, file digests, split and licence/terms.
- Record source domain, language tag, script/orthography, channel, speaker namespace, annotation source and consent/provenance.
- Check train/test overlap at speaker and source-program/document level.
- Keep research-only, non-commercial, gated and TTS-restricted cohorts out of unauthorized training and synthesis.
- Do not publish raw audio, speaker metadata or restricted transcripts in PR fixtures.
- Native speakers must adjudicate normalization, code-switch boundaries, dialect and high-consequence entities.
- Unknown language, unsupported profile and uncertain switch boundaries remain explicit; no English fallback.
- Any South African speech claim names the exact languages, varieties, domains and cohort versions it passed.

## Nonclaims

This registry is not a model, dataset ingestion, permission grant, benchmark result or evidence that Symthaea supports any South African speech variety. It is the admission and reproducibility scaffold for #10913.
