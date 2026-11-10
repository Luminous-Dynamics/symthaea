# VCRB-001 — Vocal Control Response Matrix Benchmark

**Date:** 10 November 2026  
**Status:** standalone evaluator and protocol proposal; synthetic contract tests only  
**Scope:** score-conditioned vocal rendering and measurable response to explicit performance controls

## Research question

When a musician changes one performance control, does the audio change in the intended way while other acoustic properties remain stable?

The API accepting a parameter is not evidence of control. A plausible-sounding output is not evidence that the parameter was followed. A target metric improving while other musical properties drift is not successful controllability.

VCRB-001 formalizes a one-factor intervention and its side effects as a measured response matrix.

## Relation to existing work

- **VCEB-001** defines the overall control, fidelity, identity, editing and quality benchmark. VCRB supplies an executable one-factor control-response measurement primitive for its control track.
- **VELB-001** checks whether a localized audio edit changes the target waveform region while preserving protected waveform regions. VCRB instead asks whether the intended acoustic property changed and whether other measured properties were disturbed. Use both when evaluating a local edit.
- **Issue #11357** covers explicit phoneme-level energy control and a matched reproduction of the ASRU 2025 result. VCRB does not implement that renderer control; it can measure its realized energy response once implemented.
- **VocalPerformance v3** supplies explicit pronunciation, score, syllable, note and phrase identities from which analyzers can identify target and collateral observations.

## Paradigm shift: control is an intervention, not a parameter

Represent each request as a controlled intervention:

**requested control change → measured acoustic response**

The requested control is one input perturbation. Each measured acoustic property is one output dimension.

For each control:

- freeze lyrics, pronunciation selection, score, tempo, voice identity, backend artifacts and all other controls;
- perturb exactly one requested control;
- run matched baseline/intervention repetitions;
- measure the target response and a common set of collateral dimensions;
- compare the signed target response against a preregistered interval;
- compare every off-target response against independently calibrated tolerances.

The result is a control-response matrix. Diagonal cells represent intended control response; off-diagonal cells expose coupling. Do not collapse the matrix into one unqualified score.

## Candidate acoustic dimensions

The initial metric vocabulary should include only independently measurable, versioned quantities:

| Dimension | Candidate measurement | Important guard |
|---|---|---|
| Note pitch | cents error against frozen score | Neighboring note pitch and transition |
| Vibrato | onset ms, rate Hz, depth cents | Mean F0 and phrase timing |
| Timing | phone/note onset and offset ms | Adjacent phone duration and rest leakage |
| Dynamics | aligned phone/phrase energy in declared dB domain | Pitch, articulation, neighboring energy |
| Articulation | phone duration, closure/release and consonant cues | Phone identity and syllable timing |
| Voice quality | calibrated breathiness/tension proxy | Identity, F0, intensity |
| Melisma | designated vowel continuity and note-span coverage | Consonant placement and explicit rests |
| Identity | embedding-based screen plus blinded listener ratings | Register and style changes |

An acoustic proxy is not automatically a perceptual construct. In particular, a breathiness classifier score is not proof of audible breathiness, and speaker-embedding similarity is not proof of identity continuity or consent.

## Executable contract

The Rust crate at tools/vocal-control-response-eval is dependency-free and consumes measurements from external, versioned analyzers.

A control specification declares:

- a stable control ID and requested input delta/unit;
- one target metric and a signed expected output-delta interval;
- a minimum fraction of repetitions that must land in that interval;
- at least two matched repetitions;
- explicit absolute-delta tolerances for every collateral metric.

A matrix:

- requires one trial per declared control;
- requires identical source-case identity and ordered matched-pair IDs across control rows;
- requires a common monitored metric universe and units;
- rejects missing, extra, duplicated, non-finite or unit-inconsistent observations;
- rejects a target interval that includes zero;
- reports mean, sample standard deviation, extrema and maximum absolute delta per metric;
- reports empirical response gain without treating it as a universal quality score.

The evaluator does not read WAV files, infer intervals, extract acoustic features, render audio, or judge naturalness. Those remain explicit external responsibilities.

## Experimental protocol

### 1. Calibration split

Measure no-op repeatability for every backend, analysis pipeline and metric. Set collateral tolerances on this split only. Keep deterministic and stochastic backends distinct. Record the exact voice/model artifact digest, backend manifest digest, source take, score, pronunciation plan, render settings, random seed, preprocessing and analyzer revision.

### 2. Confirmatory control split

For each source performance, test one control at a time. Useful starter interventions include:

- vibrato depth +10 cents;
- vibrato onset +100 ms;
- one note +2 semitones;
- one phrase energy −3 dB;
- one phone energy +2 dB;
- one syllable extended across three notes;
- one explicit 200 ms rest.

These are proposed stimuli, not accepted thresholds. Expected target-response intervals and collateral tolerances must be calibrated and frozen before the confirmatory run.

### 3. Negative controls

Include:

- no-op request (must fail target compliance);
- wrong-direction change (must fail);
- target achieved with an off-diagonal collateral shift (must fail);
- control request accepted by API but ignored by renderer (must fail);
- missing target measurement or metric unit mismatch (must fail closed);
- unrequested whole-phrase or whole-song change (must fail relevant collateral guards).

### 4. Human confirmation

Blind listeners should separately rate whether the requested difference is audible and appropriate, whether the voice remains coherent, and whether musical/linguistic integrity is preserved. Objective analyzer scores do not replace this panel.

## Proposed qualification targets

These are research targets, not observed results:

| Measure | Initial qualification target | Strong target |
|---|---:|---:|
| Target response in expected interval | ≥90% of matched repetitions | ≥95% |
| Controls with all collateral guards satisfied | ≥90% | ≥95% |
| Wrong-direction/no-op detection | 100% of synthetic controls | 100% of held-out challenge controls |
| Required metric coverage | 100% | 100% |
| Provenance completeness | 100% of scored trials | 100% |
| Human-perceived control compliance | ≥80% agreement | ≥90% agreement |

Do not publish a single composite score without the underlying control rows, metric definitions, backend capability support levels, and confidence/uncertainty reporting. These initial percentages are internal engineering gates, not SOTA values.

## Competitive positioning

Existing score-conditioned singing benchmarks emphasize combinations of lyric, pitch, timing, naturalness and style. STARS is a useful hierarchical annotation framework. SoulX-Singer is a high-priority open score/MIDI/F0-conditioned baseline. MusicJudge is a complementary source for block-aligned lyric and musical assessment. SOTA2 provides task-specific synthesis comparisons; Artificial Analysis AA-Music-Vocal measures end-to-end prompt-to-song preference and should remain a separate track.

The research opportunity is to publish a control-compliance matrix that makes intended response and unintended coupling visible under a fixed score and pronunciation plan. I did not identify a directly comparable, standardized public leaderboard for this complete intervention-and-cross-coupling protocol in the sources reviewed. That is a reason to publish a transparent protocol—not evidence that no prior work exists.

## Acceptance boundary

Passing synthetic evaluator tests proves only that the scorer enforces its input and arithmetic contract. It does not establish that a renderer follows any control. A renderer claim requires held-out audio, independent analyzers, calibrated tolerances, exact artifact receipts, negative controls and blinded listening. A SOTA claim additionally requires matched baselines, licensed data, preregistered splits, adequate repetitions and independently reproducible results.

## References

- STARS: A Unified Framework for Singing Transcription, Alignment, and Refined Style Annotation, Findings of ACL 2025. https://aclanthology.org/2025.findings-acl.781/
- Controllable Singing Voice Synthesis using Phoneme-Level Energy Sequence, ASRU 2025. https://doi.org/10.1109/ASRU65441.2025.11434790
- SoulX-Singer, score/MIDI/F0-conditioned singing voice synthesis. https://arxiv.org/abs/2602.07803
- MusicJudge, block-aligned lyrical and musical assessment. https://arxiv.org/abs/2606.26451
- SOTA2 singing synthesis evaluation set. https://www.sota2.com/research/sota/singing-voice-synthesis-on-singing-evaluation-set
- Artificial Analysis music benchmark methodology. https://artificialanalysis.ai/methodology/music
