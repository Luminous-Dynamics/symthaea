# MEL-HEAR-001 — MusicListenBench v1.0 adapter

**Status:** synthetic-first Rust evaluator; awaiting exact-head qualification  
**Date:** 26 November 2026  
**Issue:** #13703

## 1. Why FLIP/STAY matters

Music understanding requires more than detecting that two waveforms differ. A collaborator should notice a change in the queried musical property while ignoring changes in room acoustics, EQ, hiss or other sound-only nuisance factors that preserve that property.

MusicListenBench v1.0 evaluates paired questions across four tasks: Melody, Harmony, Timbre and Rhythm.

Each source item has a CLEAN / FLIP / STAY triad:
- CLEAN supplies the reference relation.
- FLIP changes the queried musical property, so its expected relation must invert.
- STAY changes the acoustic presentation but preserves the queried musical property, so its expected relation must remain the same as CLEAN.

An audio-ignoring model can score 50% by choosing a relation without listening. A system that reacts to every sound change can appear sensitive on FLIP while failing STAY. Both rates must therefore remain separately visible.

## 2. Input contract and semantic normalization

A strict run capsule pins MusicListenBench ID/version/split, an immutable dataset-manifest revision digest, and the official scorer revision digest. Every triad carries item/pair IDs, source-work lineage digest, task, FLIP/STAY factors, model/prompt/runtime/decoding digests, and three versioned variants. All triads in one capsule must share the same inference identities.

Each variant includes the exact audio SHA-256, full question and SHA-256 of the exact UTF-8 question bytes, semantic interpretations of option A and option B (Same or Different), expected semantic relation, and model response (ChoiceA, ChoiceB, Unparseable or Abstain).

The evaluator scores the **decoded semantic relation**, not the raw A/B letter. It checks that A and B represent different semantic options, the expected relation appears in one of the options, CLEAN and STAY expectations agree, and FLIP opposes them. This handles item wordings that reverse option order without changing the underlying truth.

Task-specific FLIP factor admission:
- Melody → NoteIdentity;
- Harmony → ChordThirdQuality;
- Timbre → InstrumentFamily;
- Rhythm → PulseRate.

Admitted STAY factors are Room, EQ and Hiss for the task set; RelativePitchPreservingTransposition is admitted only for Melody and Harmony. These factor labels are task-scoped metadata, not proof that the audio transformation was actually applied or perceptually preserved. That decision remains with #13260 and the task-specific transformation owners.

The validator rejects duplicate item/pair/variant IDs, duplicate audio artifact digests within a triad, invalid SHA-256 syntax, question-digest substitution, mismatched identities, incomplete semantic relations and incompatible factors.

## 3. Metrics

For each CLEAN / FLIP / STAY variant, the report retains item count, correct and incorrect outcomes, answered coverage, unparseable count and abstention count. Unparseable and abstained responses are errors in the primary accuracy denominators; they are not dropped.

For a set of triads:
- CLEAN accuracy = correct CLEAN / all CLEAN.
- FLIP miss rate (MR) = incorrect FLIP / all FLIP.
- STAY false-flip rate (FFR) = incorrect STAY / all STAY.
- CLEAN-correct-conditioned miss rate and false-flip rate are computed only among triads whose CLEAN answer was correct.
- Acc_FS = 1 - (MR + FFR)/2.
- Joint exact rate is the fraction where all three answers are correct.

The report also exposes A/B answer counts and A-rate among answered outputs, task slices, FLIP-factor miss rates, STAY-factor false-flip rates, source-lineage count, and whether the proposed 30-lineage confirmatory reporting floor is met. No bootstrap interval is calculated in V1.

## 4. Synthetic fixture

The committed fixture has four triads (one per task) and two source lineages. It deliberately contains:
- one valid option-order reversal on a FLIP variant;
- one incorrect CLEAN response;
- one unparseable FLIP response;
- distinct task-compatible nuisance factors.

Expected synthetic results:
- CLEAN accuracy: 3/4 = 0.75;
- FLIP miss rate: 1/4 = 0.25;
- STAY false-flip rate: 0/4 = 0;
- Acc_FS = 0.875;
- CLEAN-correct-conditioned FLIP miss rate: 1/3;
- CLEAN-correct-conditioned STAY false-flip rate: 0;
- joint CLEAN+FLIP+STAY exact rate: 2/4 = 0.5;
- source lineages: 2, below the proposed 30-lineage confirmatory floor.

These values validate evaluator plumbing only. They are not MusicListenBench model results.

## 5. Competitive interpretation

MusicListenBench's reported rendered benchmark finds that several general-purpose and open audio LLMs are near the 50% ignore-audio floor; its reported strongest commercial untrained result is 74.2% Acc_FS, while a small 12-listener panel is reported at 87.0%. A Qwen2.5-Omni GRPO experiment reaches 99.0% clean accuracy but 86.0% Acc_FS with a 25.7% STAY false-flip rate. On the real-recording track, model performance is much weaker and same-chord strum pairs remain difficult.

These are external paper/leaderboard figures, not Melothaea results. The public leaderboard states that submitted results are self-reported and not yet reproduced by the maintainers. Reproduce the official scorer and baseline conditions before making a direct ranking claim.

Suno Studio 2.0 and ElevenLabs Music v2.5 already provide MIDI, chat-driven production, inpainting and section-level composition capabilities. Generic editing or MIDI support is not a unique value proposition. A stronger differentiation would be a musician-governed collaborator whose perception preserves the distinction between musical and sound-only change, explains uncertainty and lets the musician steer or veto edits.

## 6. Prospective targets (not achieved)

The existing issue #13703 proposes staged targets. For the rendered track, an ambitious qualification goal is:
- Acc_FS ≥ 90%;
- MR ≤ 5%;
- FFR ≤ 15%;
- FFR_c ≤ 10%;
- CLEAN accuracy ≥ 95%;
- no task/nuisance slice below 80% Acc_FS;
- joint exact rate ≥ 85%.

For real recordings, begin with a separate ≥75% Acc_FS target and explicitly cap same-performance/same-chord false distinctions. These targets need preregistered calibration/validation splits and a source-work-level power analysis; they are not yet validated thresholds.

## 7. Claim boundary

This adapter calculates the declared metrics from supplied triad metadata and model responses. It does not invoke a model, fetch the 7.47 GB benchmark audio, authenticate those audio bytes, verify transformation production, adjudicate human perception, or establish human-level musical understanding. Keep rendered and real-recording results separate. No training or post-training optimization is part of V1.

## References

- Roy, Mehrish & Herremans (22 Nov 2026), MusicListenBench: Can Audio LLMs Hear the Difference Yet? https://arxiv.org/abs/2611.29375
- Official benchmark / leaderboard: https://amaai-lab.github.io/MusicListenBench/
- Official Hugging Face dataset: https://huggingface.co/datasets/amaai-lab/MusicListenBench
- Suno Studio 2.0: https://help.suno.com/en/articles/13670529
- ElevenLabs Music v2.5 inpainting: https://elevenlabs.io/docs/eleven-api/guides/how-to/music/inpainting
