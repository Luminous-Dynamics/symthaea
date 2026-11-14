# MEL-MR-EVAL-038 — Metamorphic invariance and musical equivariance for edits

**Date:** 14 November 2026  
**Status:** Initial pair-contract implementation; exact-head qualification pending  
**Tracking:** [Issue #12083](https://github.com/Luminous-Dynamics/symthaea/issues/12083)  
**Stacked research:** [EVAL-037 / PR #12050](https://github.com/Luminous-Dynamics/symthaea/pull/12050)

## Why this is needed

Paired-reference benchmarks test selected input/output pairs. They do not fully establish whether an editor remains reliable when the same musical task is represented differently, when musical coordinates change, or when a transformation invalidates the task evidence.

MMAE has 2,000 public examples and 17,741 atomic rubrics across audio modalities and complexity levels. The official Agent baseline combines an LLM router, DSP, SAM-Audio and AuK; it reports IFR 44.06%, CR 74.63% and EMR 7.45% on the full 2,000-example scope. The Single Model baseline is reported on a 1,003-example subset and is not directly comparable. See the [official challenge](https://audio-editing-challenge.github.io/) and [baseline results](https://github.com/Audio-Editing-Challenge/Audio-Editing-Challenge-Baseline).

The AACL-IJCNLP 2026 AudioEditSurvey distinguishes acoustic, semantic, instance and composite edits and calls for separate measures of edit success and non-target preservation. It lists MMAE, ZoME-Bench and MelodiaEdit as complementary rather than interchangeable resources. See [AudioEditSurvey](https://github.com/MM-Speech/AudioEditSurvey).

## Paradigm

For source x, instruction i, editor E and transformation T:

- **Invariant:** a semantics-preserving nuisance transform leaves the operation, instruction and validity envelope unchanged.
- **Equivariant:** a musical coordinate transform changes the instruction/envelope under an explicit property map; the edit should transform consistently.
- **NonInvariant:** a transform changes task semantics, so invariance is not expected.
- **AbstainExpected:** a transform removes the event or evidence needed to evaluate the task.

The output evaluator should compare symbolic and acoustic properties separately. A renderer change can preserve symbolic score semantics while changing waveform; raw waveform equality is not a universal acceptance condition.

## Implemented pair contract

- Pair identity, dataset revision, benchmark track, base/transformed samples, common source lineage and matching split.
- Base/transformed source bytes, instructions and EVAL-037 validity-envelope commitments.
- Transform family, immutable transform version, implementation digest, parameter digest and transform receipt digest.
- Four relation classes: Invariant, Equivariant, NonInvariant and AbstainExpected.
- Invariant pairs require exact instruction and envelope commitment equality, plus an explicit property mapping.
- Equivariant pairs require a property mapping and bind both transformed task commitments.
- NonInvariant and AbstainExpected pairs require an explicit reason.
- Domain-separated pair digest.
- Fourteen authored Rust tests cover invariant success/drift, equivariant map requirements, non-invariant/abstention reason, lineage/split substitution, mutable versions, duplicate sample identity and digest sensitivity.
- Four metadata-only fixtures cover gain invariance, transposition equivariance, fixed-A440 non-invariance and target-crop abstention. Unit tests include known-answer invariant and transposition-equivalent observations, a wrong-transposition negative control, preservation failure and non-invariant no-score behavior.
- Dedicated Rust 1.96 isolated harness and exact-head workflow.

## What is not implemented

The pair contract does not execute transforms, verify transform receipts against local bytes or run an editor. The first observation evaluator now compares independently produced operation/preservation reports using numeric affine, categorical and boolean property maps; it calculates per-property residuals, binds observations and mappings, replay-verifies the assessment commitment, and prevents a relation pass when either side failed its operation or preservation gate. The evaluator still does not compare raw waveforms or produce perceptual judgments. The fixture contains declarations and synthetic digests only. No audio assets or real editing results are included.

## Evaluation protocol enabled

Report invariant task-success agreement, equivariant property error after applying the declared map, protected-invariant violations, instruction/selector drift, nuisance sensitivity, false invariance on negative controls, abstention precision/recall, worst-group performance and added render cost. Use source-lineage-clustered intervals; transformations are repeated measures, not independent compositions.

Metamorphic consistency is not correctness by itself: a system that consistently makes the wrong edit can score well on relation agreement. Pair this with EVAL-037 validity obligations and EVAL-021 blinded human calibration.

## Prospective targets (not results)

- At least 99% relation correctness on deterministic symbolic invariant/equivariant fixtures.
- Zero false invariant acceptance on deterministic non-invariant and abstain controls.
- At least 95% relation coverage on supported held-out transformation pairs.
- No more than 5 percentage points of operation-success degradation under held-out benign nuisance transformations.
- No acoustic cross-renderer equivalence claim until renderer-specific human calibration.
- Confidence intervals by composition lineage and transform family.

## External comparison

Suno Studio 2.0 integrates chat, MIDI, effects, automation and plug-in design; Moises Studio provides live collaboration, editable stems and visible/reversible agent actions; Google Lyria RealTime supports streaming generation with runtime BPM, density, brightness, scale, and mix controls. These make generic control and collaboration weak differentiators. The defensible Melothaea moat is operation-specific correctness, preservation, causal diagnosis, reversible edits, calibrated abstention and reproducible evidence. Sources: [Suno Studio](https://help.suno.com/en/articles/13670529), [Moises Studio](https://moises.ai/newsroom/product-announcements/moises-launches-studio-collaborative-workspace/), [Lyria RealTime](https://ai.google.dev/gemini-api/docs/realtime-music-generation).

No SOTA superiority, human study, benchmark score or broad cultural-generalization claim is made.
