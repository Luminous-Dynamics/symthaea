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

These are different tracks and must not be compared as if they were a matched model ablation. The organizer page also describes earlier MMAE systems as below 5% EMR; the newly released track-specific baselines are the appropriate comparison points for this challenge test release. The official score separates Instruction Following Rate, Consistency Rate and Exact Match Rate. Challenge registration closed October 8; public development evaluation and access to the released test inputs are separate from official leaderboard eligibility.

The official SDK repository specifies Python 3.11+, exactly 500 sample IDs for the selected track, one nonempty 16 kHz mono 16-bit PCM WAV per ID, at most 300 seconds per file and a 2 GiB total upload package. It requires organizer-issued evaluation URL and track-specific secret key; uploading does not itself start evaluation. Any local validator must prepare/validate output artifacts but cannot bypass the official SDK or credentials.

## Paradigm: two-sided edit contracts

A complete edit must independently pass:
1. Instruction realization — requested words/phonemes, operation, target scope, language direction and code-switching.
2. Preservation — melody, singer/timbre, vocal score/quality, timing/rhythm, protected regions and accompaniment where measured.

The scorer never sums PER, F0-CORR, SIM, MOS or locality into a single quality score. Each metric keeps its own unit, evaluator and accepted interval.

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
