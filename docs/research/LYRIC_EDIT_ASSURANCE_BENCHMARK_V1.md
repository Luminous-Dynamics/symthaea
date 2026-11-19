# VMEA-001 — Task-aware lyric-edit assurance benchmark

**Date:** 10 November 2026  
**Status:** standalone rubric evaluator and synthetic contract tests; no real audio or SOTA result claimed  
**Owner:** https://github.com/Luminous-Dynamics/symthaea/issues/11503

## Research question

Can an editor execute a requested lyric transformation while preserving melody, voice identity, vocal quality, timing and protected accompaniment—and can every axis be evaluated without hiding a failed operation behind an aggregate preference score?

## Competitive context

LyricEditBench (arXiv:2603.24589) has 7,200 test examples covering Partial Substitution, Full Substitution, Deletion, Insertion, Translation and Code-Mixing, in Chinese and English. It uses independent melody and timbre references and reports PER, speaker similarity (SIM), F0 Pearson correlation (F0-CORR), Vocal Score (VS), plus human naturalness and melody MOS. The dataset card declares CC BY-NC-SA 4.0. Its test construction samples language/gender categories and technique strata; its published paper reports the benchmark task definitions and results.

YingMusic-Singer uses an optional timbre reference, a melody-providing singing clip and modified lyrics without manual alignment. The paper reports better lyric PER, F0-CORR and Vocal Score than Vevo2 across the six task types in its experiments, while Vevo2 has advantages on speaker similarity in some settings. These are task-specific published comparisons, not a universal industry ranking.

The ICASSP 2027 Audio Editing Challenge adds a broad audio-editing comparison. Its MMAE public development set has 2,000 examples and 17,741 atomic rubrics. On November 10, the organizers released two separate 500-example hidden-rubric test tracks; only input audio and instructions are public, while evaluation rubrics remain private until official results are finalized.

The newly released official test-set cards report:
- Track 1 (Single Model): AuK baseline IFR 40.33%, CR 74.70%, EMR 7.60%.
- Track 2 (Agent): router + DSP + SAM-Audio + AuK baseline IFR 44.13%, CR 76.09%, EMR 8.00%.

These are different tracks and must not be compared as if they were a matched model ablation. The organizer's broad challenge overview describes prior systems as below 5% EMR on the MMAE benchmark. Do not substitute that general statement for the new held-out track cards. Also do not mix the held-out test scores with the public development baseline repository's locally evaluated results: its README reports Track 1 on 1,003 single-complexity examples at IFR 42.12%, CR 75.78%, EMR 7.58%, and Track 2 on the 2,000-example full public set at IFR 44.06%, CR 74.63%, EMR 7.45%. These are different evaluation scopes, locally judged with Qwen3-Omni-30B-A3B-Instruct, and are not direct comparisons to the new held-out Track 1/Track 2 cards.

The official challenge page states that each rubric is judged three times by a frozen Qwen3-Omni model with shuffled answer choices and majority voting. Overall EMR is the primary ranking metric, with IFR then CR as tie-breakers. Challenge registration closed October 8; final confirmation and submission credentials were distributed to already registered teams. A new participant cannot assume entry eligibility after the deadline. Public data development is separate from official leaderboard participation.

The official MIT-licensed SDK requires Python 3.11+ and validates a submission directory containing submission.jsonl plus an audio directory. Each nonempty UTF-8 JSONL line must contain exactly two string fields, id and audio_path. All 500 official IDs for the selected track must appear exactly once; audio paths must be unique, canonical relative POSIX paths ending in .wav. The manifest is limited to 1 MiB.

Every referenced WAV must be a nonempty little-endian RIFF/WAVE PCM file at 16 kHz, mono, 16-bit integer PCM, and no longer than 300 seconds. The SDK validates RIFF framing and actual PCM bytes, not merely a filename or header claim. Symlinks, path traversal, unreferenced files, links and non-USTAR archive extensions are rejected.

The archive is uncompressed USTAR, with submission.jsonl as the first member, one audio file for every manifest entry, correct tar padding and terminators, and a 2 GiB limit that includes headers and padding. The SDK fetches the current server policy before validation, so the supported policy and expected IDs are authoritative at submission time. The upload service requires an organizer-issued evaluation URL and track-specific secret key; uploading does not itself start evaluation.

Architecture consequence: Muse should produce a clean, content-addressed Rust export manifest and preserve the exact output WAVs, then hand the package to the official SDK validator as the final submission gate. Do not duplicate the organizer's RIFF/USTAR validator or claim our local VMEA scorer reproduces hidden official rubrics. The local adapter must not store, log or commit competition secrets.

## Adjacent 2026 research and what Muse should borrow

### Segment-level control: SegTune (ACL 2026)

SegTune introduces local musical descriptions aligned to song segments and broadcasts them into corresponding time windows while a global prompt maintains broad stylistic coherence. It also predicts sentence-level timestamps for lyric-to-music alignment and proposes segment-alignment and vocal-consistency metrics. This validates segment-scoped expression control as a live direction. Muse should represent those descriptions as typed, time-bounded intents attached to stable performance-graph IDs, not free-form text alone. Local expression requests must carry an explicit dependency cone and preservation guards for neighboring segments.

### Full-song generation: SongBloom (IEEE TASLP, June 2026)

SongBloom combines semantic discrete tokens with acoustic latent patches generated by diffusion, and reports full-song generation up to 150 seconds with a multi-stage training and preference-optimization pipeline. This is a relevant full-song synthesis baseline, but not a direct baseline for a bounded local singing edit. Compare its quality and structure only under a matching generation task; use a fixed melody/source and edit operation to compare edit systems.

### Independent lyric/music evaluation: MusicJudge (Interspeech 2026)

MusicJudge aligns semantically meaningful lyric blocks and combines lyric correctness with pitch/rhythm fidelity. The authors report Spearman correlation 0.683 with human expert rankings on their 120-performance judged subset. Its multi-signal block matching and singing-aware transcription are useful independent evaluator candidates. It is not a preservation/locality evaluator: its global key policy deliberately accommodates transposition, so Muse still needs score-relative F0 and explicit protected-region measurement. Keep MusicJudge output distinct from the official rubric judge and blinded listener results.

### Speech/singing mode switching: UniVocal and SCSBench (ACL 2026)

UniVocal introduces speech-singing code-switching: transitions between speech and singing are inferred from text context rather than supplied as explicit mode tags. Its SCSBench separates implicit, explicit and mixed mode-switching scenarios, and the paper reports state-of-the-art results on that benchmark. This is a relevant comparator for Muse's speech-to-singing or spoken-to-sung phrase boundary handling, but it is **not** itself a multilingual-language code-switch benchmark. Language identity switches (for example, isiZulu ↔ English) and vocal-mode switches (speech ↔ singing) must be different labels and evaluation tracks.

The SOTA2 SCSBench pages report UniVocal WER 5.83 / SIM 65 / UTMOS 4.36 for the implicit subset, and WER 10.90 / SIM 65.2 / UTMOS 4.36 for the mixed subset. These are benchmark-page results under that benchmark's metric definitions; do not compare them numerically with LyricEditBench PER/F0-CORR/SIM or claim they qualify South African language performance. Use the official UniVocal paper and SCSBench definitions for any reproduction.

Sources: https://aclanthology.org/2026.acl-long.1452/ and https://www.sota2.com/research/sota/speech-quality-evaluation-on-scsbench-mixed

### Multi-aspect song quality: SongBench and SingMOS-Pro

SongBench decomposes song quality into seven dimensions—vocal, instrument, melody, structure, arrangement, mixing and musicality. SingMOS-Pro covers 7,981 clips from 41 models and 12 datasets, with at least five experienced annotators per clip and separate lyric, melody and overall ratings. These are valuable quality and perceptual-reference tracks. They do not by themselves establish whether a requested local edit was executed or whether unrelated waveform content was preserved.

**Implication:** Muse should combine segment-scoped symbolic intent, backend-neutral performance graph edits, independent block-aligned lyric/music analysis, VELB locality receipts, and separate blinded quality judgments. No one metric or model should become the sole judge of its own output.

## Paradigm: two-sided edit contracts

A complete edit must independently pass:
1. Instruction realization — requested words/phonemes, operation, target scope, language direction and code-switching.
2. Preservation — melody, singer/timbre, vocal score/quality, timing/rhythm, protected regions and accompaniment where measured.

The scorer never sums PER, F0-CORR, SIM, MOS or locality into a single quality score. Each metric keeps its own unit, evaluator and accepted interval. Same-timbre cases must bind the same melody/timbre reference digest; cross-timbre cases must bind distinct digests, so mislabeled conditions are rejected rather than pooled.

Each interval receives a three-way decision:
- PASS: the full interval is within its accepted region;
- FAIL: the full interval is outside it;
- INDETERMINATE: it crosses a decision boundary.

Indeterminate and missing evidence never count as success. Exact Match requires all required instruction and preservation rubrics to pass.

## VMEA-001 implementation

The Rust evaluator in tools/lyric-edit-assurance-eval:
- supports six LyricEditBench operations and a separate general-audio-edit family;
- binds source, melody and timbre references, lyric payloads, system and evaluator artifact digests;
- preserves singer/song/session, language, technique and same-/cross-timbre mode;
- checks exact rubric coverage, units and interval provenance;
- retains unsupported, failed and abstained runs in the denominator;
- computes local IFR, CR and strict EMR, per-case results and group metrics;
- requires complete predeclared operation × language × timbre-mode groups;
- distinguishes its local rubric calculation from the official MMAE judge.

## Published YingMusic-Singer results: what the numbers actually say

YingMusic-Singer is the strongest directly relevant published melody-guided lyric-edit comparison in this protocol. It conditions on an optional timbre reference, an existing singing clip as melody reference and modified lyrics, without manual phoneme/MIDI alignment. The paper evaluates six task types across Chinese and English, with both cross-timbre Melody Control and same-timbre Sing Edit conditions.

Selected cross-timbre rows from its published Table 2 illustrate why Muse must measure identity separately:

| Condition | Model | PER ↓ | SIM ↑ | F0-CORR ↑ | Vocal Score ↑ |
|---|---|---:|---:|---:|---:|
| Chinese partial substitution | Vevo2 | 0.1378 | 0.6462 | 0.8471 | 1.3578 |
| Chinese partial substitution | YingMusic | 0.0192 | 0.6543 | 0.9364 | 2.0779 |
| Chinese translation | Vevo2 | 0.4409 | 0.6115 | 0.7678 | 1.3061 |
| Chinese translation | YingMusic | 0.0881 | 0.5791 | 0.9378 | 1.9372 |
| English partial substitution | Vevo2 | 0.3352 | 0.6357 | 0.8794 | 1.0340 |
| English partial substitution | YingMusic | 0.0685 | 0.6078 | 0.9355 | 1.5054 |
| English code-mixing | Vevo2 | 0.5132 | 0.6325 | 0.8927 | 1.0925 |
| English code-mixing | YingMusic | 0.2668 | 0.6076 | 0.9389 | 1.5060 |

These examples show substantial lyric and melody gains, alongside identity-similarity losses in some cross-timbre conditions. The paper reports PER/F0-CORR/Vocal Score improvements over Vevo2 across all six task families in both evaluation settings, but does not establish a universal win on every metric. We should preserve that distinction in any competitive claim. The same-timbre rows also show that identity similarity remains a difficult dimension even when melody and lyric metrics improve.

The paper's benchmark uses PER from singing-trained Qwen3-ASR-1.7B converted to phonemes (tone markers removed), SIM from WavLM-large speaker-verification embeddings, F0-CORR from RMVPE framewise F0 contours, and Vocal Score from VocalVerse2. It additionally reports human N-MOS and M-MOS from 30 listeners on 120 sampled outputs. Muse should pin evaluator versions and report these separately, then supplement them with exact target-region/locality, neighboring-phone, onset/offset, protected-region and preservation receipts. ASR/phoneme errors on code-switching need human-reviewed audit samples because the paper itself notes potential ASR hallucination inflation on mixed-language PER.

### Practical model comparison

The YingMusic-Singer model is approximately 727.3M parameters (453.6M conditional flow-matching backbone, 156.1M VAE and 117.6M melody extractor) and the paper reports training on 8×A800 80GB GPUs with distributed training and bf16. This is a significant reproduction cost. Our near-term strategy should be:
1. use its released inference code/weights as an external baseline where hardware, dependencies and component licences permit;
2. first reproduce a small, pinned subset of its published test conditions and metrics;
3. focus Muse's new research on bounded edits, explicit dependency/locality guarantees, joint identity–melody–lyric preservation and correction evidence rather than trying to retrain a 727M system from scratch;
4. only pursue training once matched pilot data identifies a gap that cannot be solved through control, editing or integration.

LyricEditBench is CC BY-NC-SA 4.0. Do not use the dataset in commercial model training, product validation or redistribution without explicit rights clearance. The paper says its own training used internally licensed singing tracks, not the benchmark test set. The published benchmark's rights do not automatically grant rights to its underlying GTSinger recordings or any source data. Keep the public benchmark as a research-only external comparison unless a separate rights review establishes otherwise.

### Separate whole-singing synthesis reference

The SOTA2 *Singing Voice Synthesis on Singing (evaluation set)* page currently displays its May 2026 in-house system at Quality Score 4.45, SingMOS 4.148, Sheet-SSQA 2.534, WER 0.118, Clarity 4.64, Naturalness 4.46 and Expressiveness 4.55. Its page says it was updated six months ago. This is a separate singing-generation evaluation, not a LyricEditBench result and not a directly comparable lyric-edit score. Use it only as a reference for the broader naturalness/intelligibility programme, with the evaluation set, model conditions and metric definitions independently verified before comparison.

## Required experiment matrix

Compare the following on paired sources and requested tasks:
- YingMusic-Singer-Plus;
- Vevo2 where input/control conditions are supported;
- Muse's currently supported renderer path;
- Muse bounded VocalEditPatch path once #11356 is implemented;
- no-op and wrong-operation negative controls.

For each system, separate:
- all six edit operations;
- Chinese and English (and future language extensions only after qualified phoneme references);
- same-timbre and cross-timbre conditions;
- technique/register/SNR strata where annotations are available.

Report PER, SIM, F0-CORR, VS, human N-MOS and M-MOS, plus Muse-specific locality, neighboring-phone timing, phrase seam and accompaniment preservation. An improvement in lyric intelligibility must not mask a loss in timbre or melody. Report operation-level results and the worst supported stratum alongside macro/micro averages.

## Statistical design

- Use matched system/source/task cases.
- Preserve source singer/song/session IDs; cluster confidence intervals at source singer/song/session level, not audio frames.
- Keep calibration/development and confirmation manifests distinct and independently audit sample overlap.
- Freeze thresholds per metric/profile before confirmation.
- Blind and randomize human comparisons; include ties, neither-acceptable, repeat pairs and separate judgments for naturalness, melody adherence and operation success.
- Preserve judge-human disagreement and all failed, unsupported, empty and abstained runs.
- Keep the published LyricEditBench test set separate from any licensed/consented confirmation set.

## Proposed targets

Development targets, not current results:
- 100% provenance completeness and rubric coverage;
- zero indeterminate rubrics counted as passes;
- zero exact-match credit for no-op, wrong-operation, unsupported or failed outputs;
- at least 90% EMR on the subset of task operations declared supported as a first engineering gate, progressing toward 95%;
- zero protected-property violations in the fully qualified cohort;
- publish all six operations and all declared strata separately;
- improve exact match against matched baselines without reducing IFR or CR.

Do not select universal PER, SIM or F0-CORR limits from the published benchmark table. Freeze each metric's threshold in its own unit using independent calibration data.

## South African language coverage: do not substitute speech corpora for sung references

The separate [SA-SING-001 programme (#11549)](https://github.com/Luminous-Dynamics/symthaea/issues/11549) begins from the six language profiles audited in #11154 (Afrikaans, South African English, Sesotho, Setswana, isiXhosa and isiZulu), but those are pronunciation starting points, not qualified singing languages.

Current resource findings:
- The SADiLaR CSIR SAMA resource contains orthographically transcribed broadband podcast speech in several South African languages. It has a CC BY 4.0 catalogue record but is Level-0 verified and described as harvested/automatically segmented podcast speech. It may support speech-side orthography/phonetic evaluation after item-level and upstream broadcast-rights review; it is not singing data.
- The Lwazi ASR Hugging Face card is internally inconsistent: its language table says CC BY 3.0 while its narrative says CC BY 2.5. Do not ingest until the authoritative licence is resolved.
- Swivuriso / ZA-African Next Voices explicitly prohibits TTS, voice cloning, voice synthesis and any use intended to replicate/generate human voices. Exclude it from Muse singer training, timbre-reference training and generation. Only separate speech-recognition or permitted phonetic-evaluation tasks may be considered under its access terms.
- Lwazi III isiZulu TTS is listed under CC BY-NC-SA 4.0. This is speech synthesis material, not sung performance; do not infer commercial or cross-identity singing rights from it.

No clearly rights-cleared public South African singing set was identified in this source review. A small performer-consented evaluation-only pilot is therefore the correct first qualification step. South African Sign Language is the country's 12th official language, but is a distinct signed modality; do not force it into an acoustic F0/phoneme score. Keep signed musical performance as a separate future modality.

## Rights and eligibility

LyricEditBench is CC BY-NC-SA 4.0. Do not use it in commercial training, redistribute it, or claim commercial validation without additional rights clearance; inspect the underlying GTSinger recording rights separately. YingMusic's code/weights are generally described as CC BY 4.0 except Stable Audio-derived VAE components under the Stability AI Community License.

The official MMAE competition uses private submission credentials and had an October 8, 2026 team-registration deadline. Building a local adapter or evaluating public development data does not establish official entry eligibility. Do not call this scorer an official MMAE evaluator without a verified mapping to organizer rubrics and the frozen official judge.

## Related work

- #11356 renderer-neutral VocalEditPatch
- #11354 VELB-001 waveform locality
- #11401 VCRB-001 control-response matrix
- #11402 versioned acoustic observations
- #11425 independent singing F0 / VPTB-001
- #11456 interval-aware assurance / VIA-001
- #11482 bounded correction trajectories / VCCR-001
- arXiv:2603.24589 and LyricEditBench dataset card
- ICASSP 2027 Audio Editing Challenge / MMAE
