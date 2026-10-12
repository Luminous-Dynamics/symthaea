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

- **M1 — Honest capability manifest:** publish current per-language and per-species tasks and unknowns; CI protects the English/click boundary.
- **M2 — Human-language adapter baseline:** compare an appropriate multilingual ASR/translation/TTS stack on a small set of high-value languages, starting with English, isiZulu, isiXhosa, Afrikaans and additional languages chosen with speakers. Keep translation, recognition and generation scores separate.
- **M3 — Click prototype:** construct a click-specific transient event in isolation and test generated audio against recorded exemplars before placing it inside connected speech.
- **M4 — Animal sound baseline:** choose one species/task with a usable ethical dataset, pin a model and dataset, and evaluate on held-out recordings. No “translation” label until independently validated.
- **M5 — Controlled interaction trial:** only after M4, collaborate with relevant animal-behavior experts on minimal-risk, observable-outcome trials.

## Decision rule

Treat every output as a measured capability, not a universal identity claim. The system may learn language-specific skills and animal-signal associations through data, models, and feedback, but only the task and domain supported by evidence should be advertised. Unknown language, phone, species, or communicative intent must remain explicitly unknown rather than being guessed into fluent-sounding output.
