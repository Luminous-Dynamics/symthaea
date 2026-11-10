# Muse singing synthesis: control-first benchmark and research roadmap

**Date:** 8 November 2026  
**Status:** research protocol and engineering targets, not a claim of benchmark victory  
**Scope:** Muse, the shared vocal-performance contract, the pronunciation-plan research crate, the formant-vocoder baseline, Kokoro singing, and the local DiffSinger bridge.

## Executive direction

The opportunity is not to build another opaque text-to-song generator and compete only on a single naturalness score. It is to build an **evidence-bearing, score-conditioned singing-performance compiler**:

1. Resolve text into a reviewed pronunciation plan, retaining dialect, context, variant, uncertainty and provenance.
2. Align syllables and phonemes to an explicit beat-domain score, with rests, phrase boundaries, melisma, onset anticipation and coda release represented rather than inferred at render time.
3. Compile that score into a backend-neutral performance representation with explicit F0, duration, energy, breathiness, tension, articulation and vibrato controls.
4. Render through backends whose declared capabilities match the controls they actually consume.
5. Measure the output against the same frozen pronunciation and musical targets, then use blind human evaluation for qualities that objective metrics cannot establish.
6. Preserve local-edit scope and provenance so a corrected word or note does not require regenerating and revalidating an entire song.

This is more like a compiler toolchain than a black-box generator. The representation is the contract. A renderer cannot quietly rerun G2P, invent a schwa, discard a word, flatten rests, or claim a control that it ignores.

**Current engineering state:** the branch has a standalone evidence-bearing pronunciation plan, beat-domain score and explicit score-to-syllable alignment contract, independent G2P reference scorer, isolated Piper Plus G2P experiment, stricter DiffSinger phone/timing bridge, formant-vocoder score-conditioned baseline, and blind human-listening release-gate scaffold. The formant backend now consumes the performance's shared log-frequency portamento and vibrato target; the renderer applies its own short attack/release envelope, while objective evaluation reads the same pitch target from VocalPerformance.target_pitch_at. This is code-level control coverage, not proof of human-level singing quality.

## Paradigm shifts

### 1. From text-to-waveform generation to compilation of an evidence-bearing performance

The source object should preserve:
- exact source text and lexical spans;
- one or more candidate pronunciations per word;
- source/version digest and source type for each candidate;
- contextual and dialect variant rationale;
- selected candidate plus a review/selection receipt;
- explicit OOV, unsupported or needs-review states;
- a beat-domain score, tempo map, explicit rests and sung events;
- syllable-to-note intervals, phrase ranges and melisma nucleus identity;
- backend capability manifest and exact renderer/model/voicebank revisions.

The tools/pronunciation-plan crate is the first implementation of the pronunciation and score-alignment portion. Its validation requires every non-whitespace source character to be covered by a lexical span, every selected word to have a receipt, and every sung note interval to be assigned exactly once. Its beat-domain score carries tempo changes explicitly and converts to seconds only at the renderer boundary. The root formant-vocoder backend now has opt-in resolved-plan and beat-domain render methods that consume this selected pronunciation/alignment contract without rerunning Muse G2P. This is an initial source-filter adapter, not a general or neural score-conditioned singer.

### 2. From one pronunciation string to a contextual pronunciation lattice

Homographs, dialects, contractions, names, code-switching and language-specific syllabification make a single IPA string an unsafe foundation. Retain candidate hypotheses and their provenance, then resolve them by context and review.

Piper Plus G2P 0.5.3 is an isolated research candidate, not a production dependency. Its English backend returns character-level IPA tokens with separate stress markers and parallel prosody metadata. The adapter reconstructs only word-level raw IPA and stress evidence; it does not pretend that each Unicode character is a phone, infer syllable boundaries, or infer OOV status. The independent reference scorer accepts reviewed pronunciation variants, so heteronyms and dialect alternatives can be represented.

The committed G2P JSONL fixtures are synthetic contract tests only. They are not a human-reviewed linguistic gold set and must never be reported as evidence of G2P accuracy.

### 3. From note lists to a beat-domain score with interval ownership

Seconds are an implementation coordinate; beats and a tempo map are the musical source of truth. The score contract should make rests explicit, integrate tempo changes that occur inside notes, and assign each sung interval to one and only one syllable. A syllable may span several notes for melisma; several syllables may occupy disjoint portions of one long note. Neither case permits overlap, unassigned intervals or a melisma that crosses a rest.

The research crate tests these cases, including tempo changes within events, explicit rests, melisma phone identity and full interval coverage.

### 4. From backend names to per-control capability contracts

“Kokoro”, “DiffSinger” and “formant” do not imply equivalent musical control. Every backend must declare whether each control is native, approximated, conditional or unsupported, with a rationale. A missing declaration is unsupported by default.

Current engineering direction:
- **Formant-vocoder v2 path:** deterministic compatibility baseline; current Muse G2P must exactly match the frozen phone sequence. It consumes per-syllable log-frequency portamento, vibrato onset/rate/depth, note pitch, approximate duration, phrase expression and explicit timing.
- **Formant-vocoder resolved-plan path:** opt-in renderer method consumes a selected PronunciationPlan plus VocalScoreAlignment or BeatDomainScore. It does not rerun G2P, validates exact score interval coverage, preserves explicit rests, subdivides long notes across syllables and sustains a designated vowel across melisma. It uses a strict allowlist of the existing formant table, fixed consonant/nucleus timing heuristics and 200 Hz frame grid. It currently does not synthesize breath events, continuous pitch/expression controls or identity conditioning. It is a new build/test integration path, not a validated quality result.
- Neither formant path is a neural singer, personalized identity system, full physiological tract solver, or production winner on naturalness.
- **Kokoro singing:** uses a natural speech voice source and retimes/retunes syllable and vowel material; coarticulation and speech naturalness are useful, but many fine controls remain approximations or unsupported. It should not be described as a singing-native model.
- **DiffSinger:** local ONNX worker accepts explicit phones and score events and uses a licensed local voicebank. It still does not consume every performance control; arbitrary continuous F0 curves, phrase curves, tension, breathiness, energy and localized edits remain unsupported or approximated.
- **Research-only physical tract:** do not present an unintegrated waveguide/gesture prototype as a selectable production backend.

### 5. From one-way synthesis to a closed score/audio verification loop

A renderer should emit audio plus an evidence receipt. An independent analyzer should estimate actual voiced/unvoiced intervals, pitch track and note error, phoneme and word alignment, phrase and note onset/duration, rest leakage and acoustic artefacts, plus uncertainty or unmeasurable regions.

The analyzer must compare against the frozen performance plan, not rerun a different frontend and silently change the target. Human ratings then assess naturalness, intelligibility, expressiveness, identity consistency and control fidelity. The score, renderer, analyzer and listener bundle should all identify the same input revision and artifact digests.

The current singing_quality module measures pitch, voiced/unvoiced errors, onset, duration, clicks, clipping, DC and cleanliness. Its target-pitch calculation now delegates to the shared performance contract so vibrato and portamento are not evaluated against a stale or different curve. The resolved-plan renderer uses a selected candidate/score graph directly, so its future evaluation must use that same plan as ground truth rather than the v2 legacy G2P target.

### 6. From global regeneration to local, bounded edits

The long-term editing primitive should be a patch on the performance representation: replace one word's selected pronunciation; change one syllable's note assignment; alter one vibrato onset/depth/rate; change one phrase's expression arc or breath; or change a single voice/technique segment.

Each edit should identify its dependency cone and which audio regions must be rerendered and reevaluated. The release receipt should prove that unaffected regions are byte-identical where the backend supports deterministic local rendering, or acoustically unchanged within a specified tolerance otherwise. This can be a meaningful industry differentiator: correctness and edit locality rather than only full-clip quality.

### 7. From a single “good voice” to rights-aware, community-governed voice profiles

Voice identity must be an explicitly authorized asset with consent record, license terms, source hash, permitted uses and revocation/expiry conditions. Community-reviewed language inventories and pronunciation rules must be stewarded by speakers and language experts. No voice profile or language model should be silently inferred from unconsented data. Voice identity similarity should only be evaluated on authorized references.

## Industry and research reference points

The figures below come from different datasets and evaluation setups. They are context, not a common leaderboard on which Muse has already been measured.

| Reference | Reported result | What it tells us | Comparability warning |
|---|---:|---|---|
| SOTA2 “Singing Voice Synthesis on Singing” evaluation set, updated May 2026 | In-house SVS 2026.05: quality score 4.45; SingMOS 4.148; WER 0.118; clarity 4.64; naturalness 4.46; expressiveness 4.55 | A high external reference for perceptual quality and lyric transcription on that evaluation set | Underlying evaluation set, model access and full scoring protocol are not identical to Muse's tests |
| SOTA2 OpenCpop leaderboard | FM-Singer: F0 RMSE 25.2, MCD 2.703; VISinger2 NF: 25.5 / 2.939 / MOS 3.569 | Reference for score-to-audio pitch and spectral distortion on OpenCpop | Dataset, F0 unit/normalization and model protocol differ from the SOTA2 singing evaluation set |
| SOTA2 M4Singer/OpenCpop out-of-domain benchmark | iSTFTNet + UnivHD: PESQ 2.9, MCD 2.25, F0 RMSE 38.79; ground-truth PESQ 4.5 | Out-of-domain generalization and waveform quality deserve separate evaluation | Not directly comparable to OpenCpop in-domain or the singing evaluation set |
| SOTA2 SCSBench English singing | SoulX-Singer English WER 0.129; GMO-SVS English WER 0.149 | Useful external lyric-recognition reference for English singing | Different corpus and transcription protocol from the “Singing” evaluation set |
| STARS (Findings of ACL 2025) | Unified singing transcription, phoneme-audio alignment, note transcription/timing, vocal technique and global style annotation | Supports a hierarchical annotation/verification loop instead of evaluating only the final waveform | Annotation model results are not an SVS MOS score |
| Phoneme-level energy control (IEEE ASRU 2025) | Reported over 50% reduction in energy-sequence MAE versus baseline/energy-predictor approaches | Directly supports explicit phoneme-level dynamics as a controllability primitive | The reported task and trained model are not our current formant baseline |

Commercial prompt-to-song systems and score-conditioned singing systems also serve different jobs. A prompt-to-song model may have excellent overall sound and arrangement but not expose exact IPA, word-level uncertainty, syllable-to-note interval ownership, deterministic rests or reproducible local edits. Traditional score-conditioned SVS offers more symbolic control but often has model- and dataset-specific interfaces. Muse should benchmark the overlap honestly and focus its differentiation on exact controllability, editability, provenance and multilingual review—not claim that a current prototype beats those systems in perceived quality.

## Proposed benchmark suite

The suite has four layers. A release must pass the integrity and score-control layers before human quality scores can promote it.

### A. Pronunciation and linguistic correctness

| Metric | Initial engineering target | Strong target | Measurement requirement |
|---|---:|---:|---|
| Canonical-phone PER on independently reviewed test words | ≤ 8% | ≤ 5% | Canonical phone tokens; accepted variants; language/dialect profile fixed |
| Exact word pronunciation accuracy | ≥ 90% | ≥ 95% | Context-sensitive variants; no character-level proxy |
| Syllable boundary F1 | ≥ 0.90 | ≥ 0.95 | Only when reference and system expose syllable boundaries |
| Lexical stress accuracy | ≥ 0.90 | ≥ 0.95 | Stress values reviewed independently; report primary/secondary separately |
| OOV hallucinated-pronunciation rate | ≤ 2% | ≤ 1% | Confirmed OOV cases; abstain or request review rather than guess |
| Source coverage / dropped graphemes | 100% | 100% | Every non-whitespace grapheme is covered or explicitly unsupported |
| Contextual variant selection | ≥ 90% | ≥ 97% | Heteronyms and multi-pronunciation words with sentence context |

Do not publish PER from the Piper character stream until a language-specific canonicalizer has mapped it to reviewed phone units. Character error rate on raw IPA is a diagnostic, not canonical PER.

### B. Musical score and control fidelity

| Metric | Initial engineering target | Strong target | Measurement requirement |
|---|---:|---:|---|
| Score-event ownership | 100% | 100% | Every sung event covered exactly once; every rest explicit and unbound |
| Note pitch median absolute error | ≤ 25 cents | ≤ 15 cents | Estimate F0 from rendered audio; report voiced-frame coverage and octave errors |
| Stable-note pitch p95 | ≤ 40 cents | ≤ 30 cents | Exclude attack/release and transition windows; report them separately |
| Transition/portamento pitch p95 | ≤ 150 cents | ≤ 75 cents | Compare actual F0 against the shared per-note target curve |
| Vibrato-rate absolute error | ≤ 0.5 Hz | ≤ 0.25 Hz | Estimate rate on sustained voiced windows with vibrato enabled |
| Vibrato-depth absolute error | ≤ 10 cents | ≤ 5 cents | State whether depth is peak or peak-to-peak |
| Vibrato onset error | ≤ 50 ms | ≤ 25 ms | Compare estimated onset to onset fraction and note duration |
| Portamento trajectory RMSE | ≤ 40 cents | ≤ 20 cents | Compare framewise cents trajectory, not only note endpoints |
| Note onset median absolute error | ≤ 30 ms | ≤ 20 ms | Separate first onset after rest from connected legato transitions |
| Note duration absolute error | ≤ 80 ms | ≤ 30 ms | Report note- and phrase-level results |
| Rest leakage | ≤ −35 dB | ≤ −45 dB | RMS during explicit rests relative to adjacent voiced RMS |
| Event/word coverage | 100% | 100% | No lost lyric words, skipped syllables, invented phones or unassigned events |

The current objective gate is an initial protection layer, not the strong target. It presently uses a 25-cent median pitch bound, 40-cent stable p95, 150-cent transition p95, 30 ms onset, and 80 ms total-duration bounds. Tighten these only after the analyzer and reference corpus are qualified; do not loosen them to turn a failing run green.

### C. Perceptual quality, intelligibility and identity

| Metric | Initial engineering target | Strong target | Measurement requirement |
|---|---:|---:|---|
| Naturalness MOS | ≥ 4.0 / 5 | ≥ 4.2 / 5 | Blind randomized clips; human confidence intervals |
| Lyric clarity MOS | ≥ 4.2 / 5 | ≥ 4.5 / 5 | Native/qualified listeners; phrase and word comprehension |
| Expressiveness MOS | ≥ 4.0 / 5 | ≥ 4.3 / 5 | Declared emotion/style prompt and independent rubric |
| Control-MOS | ≥ 4.2 / 5 | ≥ 4.5 / 5 | Listener evaluates whether requested note, dynamics and technique were followed |
| Authorized identity consistency | ≥ 4.0 / 5 | ≥ 4.2 / 5 | Only consented reference voice and approved similarity protocol |
| Sung-lyric WER | ≤ 0.15 | ≤ 0.10; stretch ≤ 0.05 | At least two independent ASR systems plus human transcription adjudication |
| Artifact-free MOS | ≥ 4.2 / 5 | ≥ 4.5 / 5 | Noise, clicks, clipping, phase artefacts and consonant/vowel discontinuities |

PESQ, MCD, spectral distance and learned MOS predictors are useful diagnostics, but no one metric is a complete proxy for singing naturalness. MOS and control-MOS need blind human review. Report listener count, qualification, randomization, confidence intervals and inter-rater reliability. Treat model-based MOS as secondary, not release authority.

### D. Runtime, robustness and reproducibility

| Metric | Initial target | Strong target |
|---|---:|---:|
| Render real-time factor on reference CPU | ≤ 1.0 | ≤ 0.5 |
| Peak memory and model footprint | Report exact values | Fit target device class with declared margin |
| Explicit failure for unsupported language/phone | 100% | 100% |
| Reproducible input and model receipts | 100% | 100% |
| Local edit isolation | No unrequested word/score edits | Unchanged regions byte-identical when deterministic, otherwise below an agreed audio-delta bound |
| Crash-free render corpus | 100% | 100% |
| Exact source/voicebank/model/score digests | Present on every receipt | Independently verifiable end-to-end |

Always report RTF by backend and device, model load time separately from per-phrase render time, and CPU/GPU utilization. A real-time target is not meaningful without hardware, sample rate, phrase length and warm/cold state.

## Dataset and evaluation protocol

### Initial Muse control corpus

Start with a small, licensed and deliberately difficult test set before scaling:

- **Five language profiles:** English, isiZulu, isiXhosa, Sesotho and Afrikaans. Give each an explicit orthographic/phonological profile version; do not silently treat a language name as a complete dialect specification.
- **At least 100 short phrases per profile** in the first gold release, with lexical spans, context, accepted phone variants, syllabification, stress and source evidence. Expand to 500+ phrases per language as community review capacity permits.
- **Score challenges:** single-note syllables, consonant clusters, long held vowels, medial and final melisma, onset anticipation, coda release, explicit rests, syncopation, tempo changes inside notes, octave changes, portamento, vibrato and phrase breaks.
- **Linguistic challenges:** heteronyms, inflection and derivation, contractions, proper names, loanwords, high/low-frequency items, dialect variants, code-switching, click consonants and phones absent from the current English voicebank.
- **Voice challenges:** multiple authorized voice profiles across ranges/registers; low/high tessitura, soft/loud dynamics, breathy/pressed/neutral styles, sustained vowels and fast consonant sequences.

### Annotation and splits

Use two independent language-qualified annotators for every gold pronunciation, followed by adjudication for disagreements. Preserve accepted alternatives instead of forcing a single prestige form. Store reviewer evidence and source digests, not personal reviewer identities, in the pronunciation artifact. For audio data, separately record singer consent, dataset license, allowed synthesis/derivative uses and voicebank/model training restrictions.

Split by singer, song and lexical content to prevent leakage. Publish at least an in-domain split, held-out lyric/word split, held-out singer split where licenses permit, dialect/profile transfer split, and deliberately OOV/proper-name challenge split.

Do not use a G2P system's own output to create its reference labels. Do not put gold test pronunciations into a model's lexicon or tuning corpus. The evaluation runner should fail when candidate and reference case IDs or profile IDs differ.

For South African language evaluation, community review is essential. African Next Voices/Swivuriso should not be used for TTS, cloning or singing synthesis under the published restrictions noted in the existing research documentation; obtain appropriately licensed and consented singing data instead.

### Human listening

A pilot can begin with five or more qualified listeners per clip set to detect obvious problems. A release claim should target at least 20 independent ratings per system/condition, with more listeners for small expected differences. Blind the renderer, randomize presentation order, use repeated anchor clips, and include authorized human-singer references. Analyze listener and phrase variation (for example with mixed-effects models) and publish 95% confidence intervals. Keep exploratory tuning data separate from confirmatory evaluation.

### Ablation and comparison design

Compare all backends on the same pronunciation plan and score:
1. formant-vocoder baseline;
2. Kokoro singing adapter;
3. local DiffSinger voicebank;
4. authorized human-singer reference.

Run ablations for no explicit phrase boundary versus phrase-aware expression/breath; fixed vibrato versus per-syllable vibrato onset/rate/depth; no portamento versus log-frequency portamento; no consonant anticipation/coda handling versus explicit onset/coda schedule; raw text re-G2P at render time versus frozen pronunciation plan; implicit timing versus beat-domain score with explicit rests; and global render versus localized edit/re-render.

Report deltas and failure cases by language, phone class, tessitura, note duration, tempo, expression and backend. A single aggregate MOS or PER can hide a complete failure on a language or phoneme subgroup.

## Promotion gates

- **Gate 0 — Contract integrity:** all source spans, selected pronunciations, score intervals, rests and backend capabilities validate. No silent fallback.
- **Gate 1 — Controlled baseline:** pitch, note timing, vibrato/portamento, rest and waveform integrity metrics pass on synthetic and human-reference fixtures. Synthetic fixtures prove code contracts only.
- **Gate 2 — Linguistic gold set:** independently adjudicated pronunciation set is complete for the claimed profile; PER, word accuracy, OOV abstention and stress/syllable metrics meet targets.
- **Gate 3 — Blind human quality:** MOS, clarity, expression, control and identity targets meet lower confidence bounds on consented test material.
- **Gate 4 — Operational readiness:** reproducible receipts, runtime, device footprint, failure reporting, licensed voice assets, privacy and local-edit regression pass.

Do not promote a language as supported solely because its Unicode text passes parsing or because an English phone fallback emits something. A language becomes supported when its phone inventory, phonotactics, syllabification, stress/tone and score alignment are reviewed and evaluated for that profile.

## Immediate engineering work

1. Keep the canonical target_pitch_at implementation as the single score-pitch reference for both renderer and waveform evaluator. Add analyzer tests that deliberately perturb F0, vibrato rate/depth and portamento to prove errors are detected.
2. Add waveform-level vibrato-rate/depth and note/rest leakage metrics to singing_quality, with missing or unvoiced windows reported as unmeasurable—not silently zero.
3. Extend VocalPerformance/alignment integration so a selected PronunciationPlan can be compiled without rerunning Muse G2P. Keep the current English constructor as a compatibility adapter until a profile-aware compiler is proven.
4. Add a shared backend-control test matrix: for each control, change only that control and verify the expected audio/feature delta while unrelated target regions remain stable.
5. Complete the independent multilingual gold set with language-community reviewers and rights-cleared singing references.
6. Benchmark on public licensed corpora where the official splits and scoring protocols can be reproduced; keep SOTA2 figures as contextual reference until exact reproduction is complete.
7. Use a blind confirmatory protocol before any “better than SOTA” statement.

## Current conclusion

Muse has an emerging control-first architecture and a stronger evidence boundary than a generic text-to-waveform wrapper: the pronunciation lattice refuses invented outputs, score alignment owns event intervals and rests, backends declare actual capability limits, and the shared pitch target gives the formant renderer and evaluator the same definition of note pitch, vibrato and portamento.

It has **not yet demonstrated state-of-the-art singing quality**. The immediate advantage is a better-defined engineering target and more inspectable control. The next decisive evidence is a completed independent gold corpus, objective pitch/phoneme/control measurements on real renders, and a blinded human comparison on a common test set.

## References

- Pan et al., “Synthetic Singers: A Review of Deep-Learning-based Singing Voice Synthesis Approaches,” Findings of IJCNLP-AACL 2025, https://aclanthology.org/2025.ijcnlp-long.24/
- Guo et al., “STARS: A Unified Framework for Singing Transcription, Alignment, and Refined Style Annotation,” Findings of ACL 2025, https://aclanthology.org/2025.findings-acl.781/
- Ryu, Shin and Kim, “Controllable Singing Voice Synthesis using Phoneme-Level Energy Sequence,” IEEE ASRU 2025, DOI: https://doi.org/10.1109/ASRU65441.2025.11434790 ; arXiv: https://arxiv.org/abs/2509.07038
- SOTA2 Research, “Singing Voice Synthesis on Singing (evaluation set),” https://www.sota2.com/research/sota/singing-voice-synthesis-on-singing-evaluation-set
- SOTA2 Research, “Singing Voice Synthesis on OpenCpop,” https://www.sota2.com/research/sota/singing-voice-synthesis-on-opencpop
- SOTA2 Research, “Singing Voice Synthesis on M4Singer and OpenCpop (OD),” https://www.sota2.com/research/sota/singing-voice-synthesis-on-m4singer-and-opencpop-od
- SOTA2 Research, “Singing Voice Synthesis,” task overview, https://www.sota2.com/research/task/singing-voice-synthesis


---

## Research update — 10 November 2026

This addendum records newly available work. It expands the research agenda; it does not change the status of the existing results or establish a Muse SOTA claim.

### New result: classifier-guided diffusion for joint pronunciation and pitch

Li, Liu and Lu's *Singing Voice Synthesis via Guided Diffusion* was published in the Journal of the Audio Engineering Society on 10 November 2026 (74(11), 811–824; DOI: https://doi.org/10.17743/jaes.2026.0303).

The published abstract describes an unconditional diffusion prior guided by two external classifiers: a phoneme classifier for pronunciation and a pitch classifier for musical alignment. It also describes gradient-conflict projection to reconcile competing classifier objectives, and compares guidance in noisy and clean feature spaces. The abstract does not report numerical results, named public test sets, or an independent reproduction; those details must be checked in the full paper before treating this as a proven improvement.

**Muse research hypothesis:** retain the evidence-bearing performance compiler as the source of truth, then explore classifier guidance as an optional acoustic refinement stage. The phoneme and pitch objectives should remain separately observable and separately gated. Gradient conflict must not be "resolved" by silently relaxing an exact-phone or score constraint.

Proposed ablation matrix on one frozen lyric/pronunciation/score corpus. The isolated research utility at `tools/vocal-guidance-lab` now accepts externally exported phoneme and pitch gradients, checks their dimensions and provenance, measures original and projected cosine similarity, and records when the two-objective projection leaves no joint guidance direction. Its fixture is synthetic mathematics only; it is not a diffusion implementation or an audio-quality result.

1. Unguided acoustic baseline.
2. Phoneme guidance only.
3. Pitch guidance only.
4. Naive joint phoneme + pitch guidance.
5. Joint guidance with gradient-conflict projection.
6. Clean-space versus noisy-space guidance.
7. Guidance-scale sweep, including the no-guidance point.

Report canonical PER and word accuracy, voiced-frame F0 error in cents, note-onset/duration error, rest leakage, lyric WER, naturalness and control-MOS, runtime, and memory. Additionally record the cosine similarity and norm ratio of the phoneme and pitch guidance gradients by diffusion step. Plot the pronunciation–pitch–naturalness Pareto frontier rather than reducing the result to one weighted score. Every run must use the same frozen plan, same score, same data split, and exact model/checkpoint digests.

### New architecture reference: factorized speech/singing conditioning

UniVoice (Zheng et al., arXiv:2606.05852, 4 June 2026) factorizes content, melody and timbre conditions and uses modality-appropriate encoders with a shared generation backbone. Its abstract reports 30,000 hours of speech plus 35,000 hours of singing, and singing PER of 16.22% against 24.72% for the cited Vevo1.5 unified baseline on its own evaluation setup.

This is not directly comparable to the SOTA2 results above. It is nevertheless a useful architectural reference. Muse already separates lexical/pronunciation, score, and voice identity concepts in its compiler representation. The opportunity is to preserve that separation through model interfaces, cache keys, conditioning channels, and evaluation—not to imitate a large-model training recipe without the data or compute budget.

### New open-system comparison: zero-shot singing and code-switching

SoulX-Singer's 2026 report describes MIDI- or melody-conditioned controllable synthesis in Mandarin, English and Cantonese, trained on more than 42,000 hours of vocal data, with a separate zero-shot evaluation benchmark. Treat it as a strong external comparison candidate, not as a direct score until the evaluation corpus, voice rights, preprocessing, sample rate, and scoring protocol match.

UniVocal (ACL 2026, https://aclanthology.org/2026.acl-long.1452/) introduces speech–singing code-switching synthesis and SCSBench. Its described approach infers vocal mode from text context and uses a cent-token representation to plan prosody. This suggests a valuable Muse challenge: a performance where speech, rap, and singing transitions are explicitly annotated, including language switches within a phrase. For Muse, mode inference should be optional; an explicit author-controlled mode track remains the deterministic reference.

### Data stewardship is a benchmark property, not paperwork

Cotton et al., *Revising research practices for singing data collection*, AI & Society (published 12 July 2026), analyzes power imbalances between singing-data contributors and data collectors, including contributors' lower control over access and downstream use despite greater exposure to non-consensual use harms.

Accordingly, add data governance to the qualification record: consent scope, allowed model-training and synthesis uses, voice identity scope, retention/withdrawal process, derivative-use rules, and who can authorize each use. An asset with unknown or incompatible rights is ineligible for model training or synthesis, even when its technical quality is excellent.

### Revised next experiments

| Priority | Experiment | Decision it should enable | Promotion rule |
|---|---|---|---|
| P0 | Phrase-boundary pitch reset and breath timing | Prove that a phrase break does not inherit an unintended portamento from the preceding phrase | Exact target at phrase onset; breath intervals do not receive voiced pitch scoring |
| P0 | Independent phoneme/pitch analyzer perturbation tests | Prove that intentional phone and pitch defects are detected independently | Negative controls must fail their corresponding metric |
| P1 | Guided-diffusion multi-objective ablation | Test whether external linguistic and pitch guidance improves the Pareto frontier | No hard constraint regression; blinded human review; reproducible checkpoints |
| P1 | SoulX-Singer and UniVoice reference adapters | Compare modern score-conditioned and unified systems with Muse on matched inputs | Same held-out score/lyrics, language profile, sample format and independent references |
| P1 | Speech–rap–sing mode-switch corpus | Measure continuity and control across vocal modes and code-switches | Every source interval has a mode label or explicit unknown state; no inferred mode is silently treated as author intent |
| P2 | Rights-cleared South African singing corpus | Qualify English, isiZulu, isiXhosa, Sesotho and Afrikaans singing profiles | Community review, explicit consent and task-specific rights before model use |
| P2 | Constrained local edit benchmark | Prove that changing one word or note does not corrupt protected regions | Collateral drift reported separately from target-edit success |

### References added

- Li, Sijia; Liu, Shiguang; Lu, Wenhuan. “Singing Voice Synthesis via Guided Diffusion.” *Journal of the Audio Engineering Society*, 74(11), 811–824, November 2026. https://doi.org/10.17743/jaes.2026.0303
- Zheng et al. “UniVoice: A Unified Model for Speech and Singing Voice Generation.” arXiv:2606.05852, 4 June 2026. https://arxiv.org/abs/2606.05852
- Meng et al. “SoulX-Singer: Towards High-Quality Zero-Shot Singing Voice Synthesis.” arXiv:2602.07803, 2026. https://arxiv.org/abs/2602.07803
- Shi et al. “UniVocal: Unified Speech-Singing Code-Switching Synthesis.” ACL 2026. https://aclanthology.org/2026.acl-long.1452/
- Cotton et al. “Revising research practices for singing data collection.” *AI & Society*, 2026. https://doi.org/10.1007/s00146-026-03205-4
