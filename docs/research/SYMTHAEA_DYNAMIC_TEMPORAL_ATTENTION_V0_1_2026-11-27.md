# Symthaea Dynamic Temporal Attention for Embodied Conversation v0.1

**Date:** 2026-11-27  
**Status:** research hypothesis and evaluation protocol; no result claimed  
**Parent:** Symthaea Polyembodiment v0.1

## Executive hypothesis

A conversational embodiment should not spend the same visual-perception budget on every moment.

We propose **event-triggered temporal foveation**: maintain a privacy-preserving, low-cost temporal context; increase sampling only around candidate interaction boundaries; then return to the baseline rate once the event window closes.

The hypothesis is not "more frames are always better." It is that **well-timed frames may be more informative than uniformly sampled frames at the same compute and privacy budget**.

This is especially relevant to Symthaea because its proposed embodiment interface is continuous and full-duplex, while the renderer must remain responsive on local and modest hardware.

## Why investigate this now

VideoFDB v1.4 (2026-11-25) supplies a current, task-relevant reason:

- The benchmark contains 237 expert-annotated dyadic clips spanning 11 conversational dynamics.
- It separates perception from generation and evaluates dialogue, gaze, face, and body signals.
- The paper reports that common AV2A approaches often process video at around 1 FPS, which can miss nonverbal events unfolding over 1–2 seconds.
- Increasing the controllable sampling rate for one tested system beyond 2 FPS degraded performance, including fluency, consistent with a cross-modal attention-budget problem.
- Cascaded speech-to-avatar systems were unable to emit nonverbal responses during the user's turn and incurred substantial timing deficits.

These observations motivate a matched-budget experiment. They do not establish that event-triggered sampling will work; that is the research question.

Official references:
- https://research.nvidia.com/labs/amri/projects/video-fdb/
- https://research.nvidia.com/labs/amri/projects/video-fdb/leaderboard.html
- https://arxiv.org/abs/2605.30256

## Proposed control loop

1. **Low-cost context:** process only explicitly permitted local signals. The camera remains off by default. When enabled, use a low-rate local context pass; do not retain raw frames by default.
2. **Candidate trigger:** detect a possible conversational boundary from audio timing (voice activity onset/offset, overlap, pause) or from a permitted visual-change detector. The trigger is a candidate, not a claim about intent or emotion.
3. **Bounded temporal burst:** raise sampling rate for a short, fixed window around the trigger. Start with a capped burst no higher than 4 FPS for the experiment; the tested system's performance above 2 FPS is a reason to constrain and measure this, not assume a gain.
4. **Temporal integration:** aggregate the short window into typed observations with explicit confidence and clock identity. A future CfC/LTC adapter may be tested as a predictor of event timing, but the first baseline should be a simple deterministic trigger.
5. **Decision and expression:** emit a typed interaction event only when the evidence threshold is met; otherwise preserve WAIT / listening state or mark the observation inconclusive.
6. **Decay and discard:** end the burst promptly, discard raw frames unless separate retention consent exists, and publish a trace receipt containing timings and feature provenance.

The detector must not infer gaze, facial affect, or user intention from audio alone. Each signal must identify its actual sensor modality and source.

## Experiment design

### Frozen conditions

| Condition | Visual sampling policy | Purpose |
|---|---|---|
| T0 | No visual stream | Audio-only / privacy baseline |
| T1 | Uniform 1 FPS | Low-rate baseline |
| T2 | Uniform 2 FPS | Current practical-rate comparison |
| T3 | Uniform 4 FPS | Higher-rate compute/attention baseline |
| T4 | Event-triggered 0.5–4 FPS with bounded bursts | Proposed policy |
| T5 | Oracle event windows at the same average frame budget as T4 | Upper bound, never deployable result |

T4 must be matched to T1–T3 by **average frame count, total processed pixels, and model/token budget**, not just nominal FPS. T5 is an oracle upper bound using ground-truth annotations and must never be reported as an online system result.

### Trigger conditions

Separate and report:
- audio-only triggers;
- visual-only change triggers;
- fused triggers;
- disabled visual path.

Pre-register the trigger threshold, burst length, cooldown, maximum number of windows, and handling of overlapping triggers before inspecting test outcomes.

### Required metrics

**Event perception**
- per-dynamic recall and precision across the VideoFDB taxonomy;
- event detection delay relative to annotated event onset;
- false triggers per conversation minute;
- calibration of trigger confidence;
- missed-event duration.

**Conversation behavior**
- turn-taking / turn-onset alignment;
- nonverbal backchannel appropriateness;
- pause and interruption handling;
- event-conditioned response timing;
- generation rubric scores reported separately from perception.

**Resource cost**
- average and peak frame rate;
- total frame count and pixel count;
- model input tokens or equivalent attention cost;
- CPU/GPU time, RAM, thermals, and energy per conversation minute;
- p50/p95/p99 event-to-decision and event-to-visible-cue latency.

**Privacy and user control**
- sensor activation only after explicit permission;
- raw-frame retention default = zero;
- permission revocation latency;
- auditability of every capture window;
- task completion in camera-disabled mode.

## Preregistered success criteria (targets, not results)

The policy should produce a Pareto improvement, not merely a higher aggregate score.

Initial target:
- at least 0.15 points improvement on the prespecified perception aggregate versus uniform 1 FPS at a matched compute budget; **or**
- remain within 0.05 points of uniform 2 FPS while using at least 20% fewer processed frames and lower measured energy.

In either case:
- no material degradation in generation or conversational fluency;
- no worse false-trigger rate than the preregistered bound;
- report all 11 dynamics, not only the aggregate;
- confidence intervals clustered by source clip and participant;
- preserve all invalid, excluded, and failed runs.

The margins are engineering targets chosen for this experiment, not values derived from the VideoFDB paper. Pilot results must be used to validate feasibility and power calculations before a confirmatory study.

## Baselines and ablations

1. Fixed-rate sampling at 1, 2, and 4 FPS.
2. Audio-only turn detector with fixed visual windows.
3. Visual-change trigger without audio cues.
4. Fused event trigger.
5. Fused trigger with CfC/LTC temporal prediction.
6. Ablation without temporal prediction.
7. Ablation with a fixed, non-adaptive attention budget.
8. Oracle event-window upper bound.

All methods receive identical allowed source material and the same held-out split. No method may use annotation timestamps at inference time except the oracle condition.

## Safety and privacy constraints

- Camera and microphone use remain explicit user permissions.
- Local processing is the default target; remote transmission needs separate consent.
- No raw media is retained by default.
- Never infer emotional state as fact from facial movement, gaze, or prosody.
- A missing or inconclusive observation is not neutral intent.
- A renderer must not convert a weak trigger into a strong interpersonal signal.
- Visual and audio sampling budgets must be user-configurable and visible.
- On disconnect or permission revocation, bursts stop and the UI reports that the sensor stream is unavailable.

## Integration sequence

1. Implement a deterministic trace reader for pre-annotated event windows.
2. Evaluate fixed-rate policies with matched frame budgets.
3. Implement audio-only and visual-change triggers as separate baselines.
4. Implement fused bounded bursts.
5. Test CfC/LTC prediction only as an ablation after the simple trigger baseline is frozen.
6. Connect the selected policy to the semantic Embodiment Event API.
7. Run VideoFDB-compatible scoring only under the official dataset and evaluation rules.
8. Run a separate consented user study for real-time comfort, comprehension, and perceived intrusiveness.

## What would count as a contribution

A defensible contribution would be a reproducible demonstration that event-triggered sampling improves temporal perception or reduces compute at matched quality, while preserving full-duplex timing and privacy.

Do not claim a novel field-wide technique solely from implementing an adaptive sampler. Novelty would need to be established by a literature review and a matched comparison against existing event-camera, adaptive frame-sampling, and active-perception methods.

## Status

- Research hypothesis: documented.
- Dataset/benchmark rationale: tied to VideoFDB v1.4.
- Trigger implementation: not implemented.
- Dataset evaluation: not run.
- Symthaea superiority: not established.


## Related work: event cameras are not adaptive frame sampling

A second, more radical path is a **neuromorphic event-sensor adapter**.

Conventional adaptive sampling chooses when to request the next full RGB frame. Event cameras instead emit asynchronous per-pixel brightness-change events. Recent reviews describe their high temporal resolution, sparse representation, and potential low-latency/low-power benefits, while also identifying limitations: event streams lack absolute intensity, are noise-sensitive, and require distinct processing algorithms and representations.

References:
- Cimarelli et al., *Hardware, Algorithms, and Applications of the Neuromorphic Vision Sensor: A Review* (Sensors, 2025): https://doi.org/10.3390/s25196208
- Cazzato and Bono, *An Application-Driven Survey on Event-Based Neuromorphic Computer Vision* (2024): https://doi.org/10.3390/info15080472
- *Neuromorphic computing for robotic vision: algorithms to hardware advances* (Communications Engineering, 2025): https://www.nature.com/articles/s44172-025-00492-5

### Proposed two-sensor architecture

- **Event stream:** detect motion/change boundaries at high temporal resolution.
- **RGB frame stream:** provide semantic context only at explicit, bounded sampling points.
- **Temporal integrator:** aggregate irregular event bursts and RGB observations into typed event candidates.
- **Policy gate:** require permission and adequate evidence before issuing any outward cue.

The event sensor is a trigger and temporal signal, not a substitute for semantic interpretation. No user emotion, intent, identity, or consent should be inferred from pixel changes.

This is a future hardware track, not a dependency for the first implementation. A normal RGB camera with bounded event-triggered frames remains the near-term experiment. A DVS adapter should be tested separately against both frame-based and hybrid sensor baselines; the device's power and resolution claims must be measured on the actual selected hardware.
