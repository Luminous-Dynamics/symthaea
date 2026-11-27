# Symthaea Counterfactual Audiovisual Grounding Protocol v0.1

**Date:** 2026-11-27  
**Status:** evaluation protocol proposal; no Symthaea AV score claimed  
**Parent:** polyembodiment and integrated interaction benchmark plan

## 1. Research question

Does the agent use visual conversational evidence when that evidence is necessary, ignore irrelevant visual changes, and combine audio and video without confusing synchronization with understanding?

VideoFDB identifies captioning collapse and visual-stream ignorance as failure modes: a model may describe the user's appearance instead of participating in conversation, or produce essentially the same output with audio-only and audiovisual inputs. The proposed protocol turns these into controlled, counterfactual tests.

Reference: [VideoFDB project and paper](https://research.nvidia.com/labs/amri/projects/video-fdb/).

## 2. Core experimental design

Construct matched interaction items with frozen audio and controlled visual variants.

| Condition | Audio | Video | Purpose |
|---|---|---|---|
| AV-positive | same audio | visual cue present | test correct use of visual evidence |
| AV-counterfactual | same audio | cue removed or changed | test causal sensitivity |
| AV-irrelevant-control | same audio | irrelevant appearance/background change | test invariance to irrelevant pixels |
| Audio-only | same audio | absent | estimate audio-only baseline |
| Video-only | absent | same visual cue | isolate visual evidence |
| AV-shuffled | same audio | temporally misaligned visual stream | test synchronization sensitivity |
| Human/oracle annotation | same media | gold labels | establish attainable reference |

Every counterfactual pair must preserve the audio waveform and audio timestamps exactly unless the trial explicitly tests audiovisual timing. Visual substitutions must preserve duration, frame rate, resolution, and encoding profile where feasible.

## 3. Task families

- **Turn and floor state:** visual evidence changes whether the partner has yielded, is preparing to speak, or continues to hold the floor.
- **Backchannel interpretation:** distinguish a brief nod or visual acknowledgment from an actual floor-yield event.
- **Gaze and attention:** detect a task-relevant gaze shift without describing the user's face or clothing.
- **Gesture and affect display:** recognize a task-relevant nonverbal cue only when the annotation supports it; do not infer hidden emotional states from appearance.
- **Object/scene references:** respond to a visible shared referent when the interaction explicitly makes it relevant.
- **Temporal alignment:** detect a cue at its correct time and reject the same cue shifted outside the response window.

Do not use appearance, identity, demographic, or biometric inference as a substitute for interaction understanding.

## 4. Metrics

### 4.1 Visual Causal Contribution (VCC)

Measure the gain from audiovisual input over the audio-only baseline on examples where the visual cue is necessary:

[
VCC = Acc(AV_{positive}) - Acc(AudioOnly)
]

Report per task family and per cue type. A positive average is not sufficient if one cue class is ignored.

### 4.2 Counterfactual Sensitivity (CS)

For matched pairs where the visual evidence changes the correct decision, measure the proportion of trials where the output changes in the direction required by the changed evidence.

A model should be sensitive to relevant visual changes, not maximally sensitive to any pixel change.

### 4.3 Irrelevant Visual Invariance (IVI)

For matched pairs where only irrelevant appearance or background content changes, measure the proportion of trials where task-relevant decisions remain stable.

### 4.4 Visual-Stream Ignorance (VSI)

Measure cases where the audiovisual model gives the same task decision as the audio-only model despite an annotated, decision-relevant visual cue.

### 4.5 Captioning Collapse Rate (CCR)

Measure task-focused interactions where the response describes appearance rather than addressing the conversational goal. Blind graders and explicit category labels must distinguish appropriate scene description from irrelevant appearance commentary.

### 4.6 Temporal grounding

Report:
- visual cue detection recall and false-positive rate;
- cue-to-decision latency;
- sensitivity to controlled audio/video offset;
- missed-cue duration;
- stale-frame influence after the cue has ended;
- interruption and turn-decision cost.

## 5. Causal attribution safeguards

- Keep the audio byte-identical across visual counterfactual pairs.
- Hash and publish the paired media manifests.
- Use identical prompt, system policy, model version, decoding, client, and network conditions within each pair.
- Randomize pair order and hide the condition labels from graders.
- Include visual no-op controls, temporal-shuffle controls, and audio-only controls.
- Pre-register exclusions and invalid-pair rules.
- Report every pair, not only aggregate accuracy.
- Do not treat correlation between a visual token and a response as proof of causal use; the controlled substitution must change the evidence while preserving confounders.
- Keep model/provider latency and client-heard latency separate.
- Use clustered confidence intervals by speaker/session, not by individual frame.

## 6. Proposed targets (not results)

Initial research goals:
- >= +10 percentage-point VCC on decision-relevant visual-cue subsets versus the audio-only baseline;
- >=90% counterfactual sensitivity on clear, held-out visual-cue pairs;
- >=95% irrelevant visual invariance on no-op appearance/background substitutions;
- <=5% captioning collapse rate on task-focused interactions;
- >=90% correct rejection of temporally misaligned visual cues;
- no material regression in audio-only task accuracy, turn timing, or interruption handling.

These targets require a pilot, sample-size analysis and fixed annotation policy. They are not claims about current Symthaea performance.

## 7. Baselines and industry comparison

Run matched conditions for:
- audio-only Symthaea;
- audiovisual Symthaea;
- a simple visual heuristic baseline where applicable;
- an open full-duplex audiovisual model;
- an appropriate hosted full-duplex model;
- human annotations as an upper reference.

Use VideoFDB's official perception task where compatible, but report this counterfactual suite separately. Its aggregate score does not replace causal attribution tests. Use InteractionBench for timing/task/grounding dimensions and StreamEval for following/synchronization, without merging their scores.

## 8. Integration with Symthaea EGCI

This protocol should become an evidence-gated benchmark lane:

1. freeze media pairs and annotation manifest;
2. present audio/video event digests to the agent;
3. record the decision candidate, deadline, decision and confidence;
4. bind the decision to the observed input windows;
5. replay the same audio with altered visual evidence;
6. evaluate whether the changed decision is justified;
7. verify that stale, unauthorized or privacy-restricted frames do not enter the evidence window;
8. publish per-pair results and exact run manifest.

Symthaea proposes and records a decision; the evaluator controls the counterfactual pair and independently computes correctness.

## 9. Machine-readable paired-item contract (2026-11-27)

Two companion artifacts define a reproducible manifest format:

- `SYMTHAEA_AV_COUNTERFACTUAL_ITEM_SCHEMA_V1_2026-11-27.json`: Draft 2020-12 schema for condition types, audio/video payload digests, timing profiles, cue annotations, task family, and expected decisions.
- `SYMTHAEA_AV_COUNTERFACTUAL_ITEM_FIXTURES_V1_2026-11-27.json`: one six-condition turn-yield example plus four negative cross-condition cases encoded as deterministic JSON Patch mutations against the positive item.

The six conditions include AV-positive, AV-counterfactual, irrelevant visual control, audio-only, video-only, and temporally shuffled AV. The same baseline audio digest is declared for all audio-bearing conditions. Video conditions share an identical timing profile in the positive fixture.

JSON Schema validates record shape. A paired-item verifier must separately compare audio digests, video timing profiles, cue intervals against clip duration, and the requirement for decision-relevant visual annotations. Those cross-condition checks are intentionally not misrepresented as JSON Schema capabilities.

The fixture manifest contains hashes and annotations only; it does not include raw audio/video, identity, or biometric data. The schema/fixtures have been authored and should next be run through an independent Draft 2020-12 validator and paired-invariant verifier.

## 9. Qualification status

This protocol is a proposed benchmark lane. No paired dataset, annotation set, runtime harness, counterfactual run, or Symthaea score has been produced by this document.
