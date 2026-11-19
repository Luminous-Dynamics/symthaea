# Muse: Evidence Stack and Competitive Benchmark Programme

**Date:** 2026-11-19  
**Status:** Research programme and prospective targets; no new composer or listener results are claimed here.

## Executive thesis

The strongest opportunity is not to build another system that generates plausible music from a prompt. It is to build a **musical collaborator whose proposals are inspectable, localized, reversible, and evidence-qualified**.

The unit of progress should be a musical intention with a traceable outcome:

`Intent -> Resolve -> Plan -> Candidate -> Verify -> Preview -> Accept/Reject -> Reconcile`

The symbolic score is the authoritative editable object when source-authored musical structure exists. Rendered audio is a separate realization with its own provenance, preservation tests, and human evaluation. Neither should silently stand in for the other.

This programme therefore has six non-substitutable evidence layers:

1. **Intent and scope:** what the user asked to change, which musical entity it refers to, and what is allowed to move.
2. **Symbolic execution:** the score, parts, materials, relationships, and grammar plan produced by the composer.
3. **Causal control:** whether a specified control causes the preregistered response, including interactions, invariances, and cross-seed stability.
4. **Preservation and transaction correctness:** whether protected material and all out-of-scope content remain invariant, and whether failure is atomic.
5. **Audio realization:** whether score-to-audio realization is complete, reproducible, and faithful to the symbolic plan.
6. **Human relevance and agency:** whether listeners and qualified musicians perceive the requested effect, consider it musically useful, and retain authorship and veto power.

A system must not pass one layer by borrowing evidence from another. A score hash does not prove audio fidelity; a preference score does not prove preservation; a symbolic control response does not prove that the musical instruction was fulfilled.

## 1. Paradigm shifts

### 1.1 From prompt-to-track to intent-to-verified musical transaction

Treat generation and editing as two sides of one transaction protocol. For editing, freeze the source score and digest, resolve a target to source-authored PartId/material/relationship identity, specify permitted operations and protected scopes, then build a candidate copy. Verify the candidate independently before it can be accepted. Ambiguous targets, missing identity, unsupported operations, and stale source digests must fail closed.

This is a stronger contract than simply regenerating a complete track from a revised prompt.

### 1.2 From style labels to measurable causal control contracts

A style label or declared supported axis is a hypothesis. The MUSE-CAUSAL-001 3^4 design tests valence, arousal, energy, and bars over frozen styles and seeds. Its exact-axis negative controls are especially important: a supposedly unsupported control should not alter canonical musical content under matched conditions.

Positive control response and negative-control invariance are separate results. A supported axis may be flat, non-monotonic, or interact with another axis; do not force every attribute into a monotonic response.

### 1.3 From one score to a persistent musical event world

The long-term representation should preserve:
- persistent part identity;
- source-authored event and material lineage;
- time-indexed roles and relationships;
- grammar/section plans and obligations;
- performance intent and realization provenance;
- renderer/instrument capability evidence.

Part identity must not be reconstructed from register or VoiceRole. Event identity must not be invented after composition by hashing similar notes. If a generator cannot provide the required identity, the editor must abstain from claims that depend on it.

### 1.4 From one quality score to a vector of independently falsifiable claims

Report distinct scores for target attainment, preservation, coherence, perceptibility, preference, collaboration agency, reproducibility, latency, and cost. Do not hide a failed preservation constraint behind a high preference score or a low-quality render behind a correct score diff.

### 1.5 From a single model to an evidence-gated creative system

Use the best available generator, symbolic planner, renderer, and specialist tools as interchangeable candidate producers. Keep the authority boundary outside any single generative model:
- the model proposes;
- the plan constrains;
- the executor realizes;
- independent verifiers test;
- the human accepts, rejects, or revises.

This architecture can improve as models and renderers change without changing the preservation contract.

## 2. Current external benchmark landscape

### 2.1 ICASSP 2027 Audio Editing Challenge / MMAE

The official challenge is the most immediate audio-editing comparator:
- public MMAE development set: 2,000 examples and 17,741 atomic rubrics;
- hidden challenge set: 500 previously unreleased examples for each of the Single Model and Agent tracks;
- metrics: Instruction Following Rate (IFR), Consistency Rate (CR), and Exact Match Rate (EMR);
- rankings use EMR first, then IFR and CR;
- the official page reports existing systems below 5% EMR on MMAE, illustrating the difference between plausible output and complete editing.

The current challenge baselines cover an end-to-end single-model editor and an agentic tool-orchestration system. The agent baseline uses a language-model router plus DSP tools, source separation and generative editing. This is directly relevant to the audio realization and editing stages, but not a direct score-native transaction benchmark.

**Access constraint as of 2026-11-19:** registration closed on October 8; hidden test submissions opened November 10 and freeze November 25, 2026. The Agent Track model/version release cutoff was November 1. A team needs its participant identity and submission key; do not assume a new team can join after registration closure. We can still use the public MMAE material for development and design a reproducible local adapter, but no challenge submission or rank is claimed.

Official sources:
- https://audio-editing-challenge.github.io/
- https://audio-editing-challenge.github.io/leaderboard/
- https://2027.ieeeicassp.org/sp-grand-challenges/

### 2.2 SMEBench: paired symbolic-target music editing

SMEBench is the more directly useful bridge for score-aware editing. Its paper describes a controlled MIDI-edit benchmark with 12 instrumentation, articulation, dynamics, and pitch operations; a 3,600-example test set; paired source and target audio; edit metadata; and a smaller real-audio complement. It separates source preservation, target agreement, and textual compatibility, and reports that different metrics can rank systems differently.

This matches the Melothaea strategy more closely than a generic generation leaderboard. The integration should preserve the benchmark's distinctions rather than collapse everything into a single preference measure.

Required local benchmark fields:
- immutable source score/audio digests;
- task instruction and operation class;
- paired target or independently adjudicated acceptable target set;
- target region/source identity;
- protected content regions;
- edit result and raw output digests;
- target, preservation, and instruction-compatibility rubric outcomes.

Paper: https://arxiv.org/abs/2610.36715

### 2.3 YuE2: direct symbolic score-editing baseline

YuE2 is not just a full-song generation comparator. Its public technical report includes a controlled score-editing evaluation based on the first generated score for each of 192 WildSongBench prompts, two seeds per applicable condition, and 3,844 full-song recordings including unedited controls. It reports these automatic edit-adherence point estimates:

| Edit | Metric | Reported score |
|---|---|---:|
| Melody | Target-pitch accuracy | 84.17 / 100 |
| Harmony | Target-chord agreement | 79.54 / 100 |
| Rhythm | Relative onset accuracy | 73.43 / 100 |
| Key | Weighted key score | 90.58 / 100 |
| Tempo | Acc2 | 95.68 / 100 |

These are useful direct baselines for *target adherence*, not a unified measure of quality or preservation. The report also provides SongBench quality, phoneme error rate and character error rate for edited and unedited outputs. It explicitly notes that each edit regenerates the whole recording: score-level preservation does not imply unchanged waveform, voice, or timbre. The published table reports point estimates and does not claim statistical significance or human preference.

This changes the comparison plan: Muse/Melothaea should not claim to be the first score editor or rely on the existence of a symbolic plan as differentiation. It should target stronger operation-specific adherence **and** exact protected-score preservation, then independently qualify audio preservation.

Candidate matched-task targets, pending pilot and power calibration:
- Melody target-pitch accuracy: at least 90 / 100.
- Harmony target-chord agreement: at least 85 / 100.
- Rhythm relative-onset accuracy: at least 85 / 100.
- Key weighted score: at least 95 / 100.
- Tempo Acc2: at least 98 / 100.
- Protected symbolic content: 100% exact preservation for the explicitly protected scope.
- Unauthorized symbolic mutations: zero.

These targets apply only if we reproduce the same task definitions, source set, sampling/seed budgets, rendering protocol and metric implementation. They are not comparable to a single candidate result if YuE2 uses different sampling, seed averaging, or selection. The public WildSongBench model leaderboard separately uses different candidate-selection budgets, including best-of-8; always disclose that distinction.

Sources:
- YuE2 paper: https://arxiv.org/abs/2609.33757
- YuE2 public score-editing benchmark protocol/results: https://github.com/MaxPlaksin/YuE_Music/blob/main/docs/benchmarks.md
- WildSongBench results and candidate budgets: https://map-yue2.github.io/

### 2.4 Suno Studio: production-workflow comparator

Suno Studio 2.0 is a product/workflow comparator, not a like-for-like score-preservation benchmark. Its August 2026 release describes a production environment with automation, effects, MIDI, an integrated synth, user-designed plugins, and a conversational interface for generating clips and arranging songs. Subsequent release notes describe improved MIDI-to-audio transcription, creating audio stems from MIDI via the chat interface, improved downbeat detection, clip time-stretching, and editable fade curves.

Muse should not claim parity with a full DAW or production environment until it has equivalent editing, arrangement, MIDI, stems and automation workflows. The defensible differentiator is a stronger semantic transaction layer: source-authored score identities, localized edit scope, protected invariant checks, independent orchestration/render capability evidence, reproducible receipts and reversible candidates.

Sources:
- Suno Studio 2.0 announcement: https://help.suno.com/en/articles/13670529
- MIDI and Studio release notes (September 17, 2026): https://suno.com/release-notes
- Timing, clip-stretch and fade-curve update (October 8, 2026): https://suno.com/release-notes/studio-updates-oct26

### 2.5 General music-generation and evaluation research

The 2026 survey on recent music generation methods and evaluation identifies persistent difficulties in musicality, consistency, controllability, representation, and evaluation. It finds no single sufficient metric and recommends combining distinct objective methods with human listening and task-based evaluations. This supports a multi-scoreboard programme rather than optimizing one embedding score.

Survey:
- https://link.springer.com/article/10.1007/s10462-026-11582-x

The 2025 *Survey on the Evaluation of Generative Models in Music* also emphasizes that system-output quality and system-use/collaboration need separate evaluation, including engineering, musicological, and HCI perspectives:
- https://doi.org/10.1145/3769106

### 2.4 Direct symbolic-planning competitors

Systems such as YuE2 show that editable melody/chord planning and agentic score revision are now part of the competitive landscape. Melothaea should not claim that symbolic planning or conversational musical revision is unique. Its differentiation must instead be tested through persistent identities, exact localized plans, protected invariants, atomic candidate application, renderer qualification, reversible diffs, and evidence-qualified abstention.

Compare only matched capabilities. A new complete recording from a revised score is not the same task as a localized edit that preserves an existing performance. Disclose sample count, candidate count, best-of-k selection, model/tool budget, and selection metric.

## 3. The benchmark stack

| Scoreboard | Core question | Primary measurements | Evidence source |
|---|---|---|---|
| Intent resolution | Did we resolve the exact requested entity and scope? | selector accuracy, ambiguity detection, unsupported-operation abstention | source IDs + human/adjudicated labels |
| Symbolic control | Did the control alter the intended symbolic property? | registered response contrasts, interactions, cross-seed dispersion | 3^4 factorial output |
| Negative controls | Did an unsupported axis stay invariant? | 27/27 matched semantic-digest slice counts | canonical semantic digest |
| Edit locality | Did only authorized content change? | zero unauthorized event/field mutations; protected digest preservation | source/candidate score diff |
| Instruction fulfilment | Did the candidate achieve the requested task? | operation-specific target attainment and textual compatibility | paired target/rubrics |
| Audio consistency | Was unrelated audio preserved? | CR, target/non-target audio measurements, hidden rubric results | rendered audio + independent rubrics |
| Musical quality | Is the result musically coherent and useful? | dimensioned musician/listener ratings, preference, calibration | blinded human study |
| Agency | Did the human retain control and authorship? | veto/undo correctness, acceptance, intervention and correction rate | interaction traces + participant study |
| Reproducibility | Can the exact candidate and evidence be reconstructed? | exact source/head, manifest, artifact and semantic digests | machine-readable receipts |
| Production cost | Is collaboration practical? | end-to-end latency, localized/full render ratio, cost per accepted edit | execution receipts |

## 4. Prospective targets

These are engineering targets to test, not achieved results. Set noise floors and acceptance thresholds before opening confirmation data.

### 4.1 Deterministic and structural correctness
- 100% expected case identities present exactly once.
- 100% deterministic replay for a frozen source commit, manifest, style, seed and factor cell.
- 100% rejection of malformed or mutated manifests.
- 100% explicit disposition of failed, missing and not-applicable measurements.
- 0 false success labels for queued, skipped, incomplete, or not-applicable evidence.
- 100% exact canonical invariance for every registered unsupported axis across the 27 matched slices, unless a source-level support contract is revised before confirmation.
- 0 unauthorized symbolic mutations outside the plan's permitted scope.

### 4.2 Control effectiveness
- At least 90% discovery-to-confirmation agreement for preregistered effect classifications on the held-out style/seed panel.
- For supported axes, require a preregistered, style-appropriate response target rather than merely any digest change.
- Report linear, quadratic, pairwise and higher-order terms separately; three-/four-way terms remain diagnostics until independently confirmed.
- Report per-style and per-seed results. Do not let a pooled average hide unsupported styles or fragile seeds.

### 4.3 Editing and preservation
- 100% known-answer target/selector decisions on the deterministic conformance corpus.
- 100% preservation of protected PartId/material/event scopes in accepted symbolic edits.
- 0 silent instrument, renderer, or operation substitutions.
- 100% explicit abstention if stable target identity or required capability evidence is unavailable.
- Report both exact symbolic preservation and rendered-audio consistency; never infer one from the other.

### 4.4 Human relevance and agency
- Pilot first to estimate variance, rater dependence, and task difficulty; then preregister powered confirmatory sample sizes.
- Use clear known-answer controls (valid, sham/no-op, wrong direction, wrong axis, protected-region mutation) and balanced blinded presentation.
- Treat ≥90% correct discrimination on unambiguous target-vs-sham controls and ≥95% rejection of clear wrong-direction changes as candidate targets after pilot/power calibration.
- Report musicians, general listeners, and tradition-specific practitioners separately.
- Keep confidence calibration, correction effort, authorship, veto, undo, and acceptance separate from overall preference.

## 5. How to beat SOTA without making false comparisons

### 5.1 Win where contracts matter

Do not start by claiming superiority in raw audio generation quality. First demonstrate a capability the audio generators do not automatically guarantee:
- exact source and candidate binding;
- stable part/material identity;
- typed edit scope and preconditions;
- protected-scope proof;
- atomic candidate/reject semantics;
- deterministic replay and receipts;
- renderer/instrument feasibility evidence;
- calibrated abstention.

A localized edit that changes only its permitted musical scope can be a meaningful advantage even when another model generates more impressive audio from scratch.

### 5.2 Then bridge to audio

Once the symbolic transaction is stable:
1. render the source and candidate with the same frozen renderer, patch set, and settings;
2. record source and candidate audio digests;
3. measure target-region change and non-target preservation separately;
4. test stems, timing, loudness, and renderer fallback;
5. run paired human evaluation;
6. compare on SMEBench/MMAE tasks using matched candidate budgets and task subsets.

### 5.3 Avoid leaderboard traps

- Never compare best-of-8 to a single candidate without labeling selection and cost.
- Never optimize on the hidden confirmation set.
- Never substitute CLAP/FAD/preference for exact target and preservation checks.
- Never remove failed generations or abstentions from denominators.
- Never report symbolic validation as audio evidence.
- Never claim cultural authenticity from one global listener panel.
- Never compare unlike tasks (full-track generation, mixture source extraction, score edit, and localized waveform inpainting) as if they shared an outcome contract.

## 6. Immediate execution order

1. Qualify the standalone case generator and result analyzer at exact head.
2. Compile and run the source-pinned composer adapter tests after workspace manifest resolution.
3. Run a small, explicitly labelled smoke composition block before any full sweep.
4. Run discovery only after the adapter has passed exact-source, deterministic replay, and output-schema checks.
5. Freeze response thresholds from discovery and run the held-out confirmation cohort.
6. Advance score-native editing using source-authored PartId/material identity and protected-scope receipts.
7. Build the SMEBench/MMAE audio adapter only after the renderer output and task-preservation contract can be independently measured.
8. Start a small randomized human pilot, preregister the instrument, and tune the confirmation protocol only from pilot evidence.

## 7. Current evidence boundary

At the time of writing:
- the 3^4 design, case exporter, contrast engine, analyzer, composer adapter, and qualification workflows are authored;
- the 15,228 planned composer executions are still prospective;
- the exact-head Rust and workflow runs are pending;
- no real composer effect, exact unsupported-axis pass, SMEBench score, MMAE score, listener preference, or cultural-validity result is claimed.

The central thesis is simple: **Muse should not merely produce music that sounds plausible. It should be able to explain what it intended to change, prove what it actually changed, show what it preserved, and let the human decide whether the result is musically right.**
