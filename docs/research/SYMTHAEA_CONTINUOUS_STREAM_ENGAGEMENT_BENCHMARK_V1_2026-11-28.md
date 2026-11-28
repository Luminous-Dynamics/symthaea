# Symthaea continuous-stream engagement benchmark v1 — 2026-11-28

## Research basis

DIVE (October 20, 2026) evaluates continuous audiovisual interaction across 525 audio/video samples and six tasks spanning proactive intervention, continuous engagement and interaction coordination. It separates response quality, participation decisions, response-window rates and reaction latency. The reported leading average response-window rates were at or below 53.77% despite sub-second average reaction latency.

This suggests a missing diagnostic in isolated question-answer benchmarks: **did the agent notice the right opportunity and participate within the useful window?**

The new source evaluator is inspired by DIVE's separation of participation from quality and timing. It is not a reproduction of DIVE's dataset, labels or scoring.

## Implemented contract

Source: `crates/domains/symthaea-soma/src/embodiment_stream_engagement.rs`.

Each opportunity binds:
- stream and opportunity IDs;
- source monotonic clock domain;
- event-start and response-deadline timestamps;
- opportunity kind;
- whether the agent was already speaking;
- a source-event digest.

Opportunity kinds cover proactive visual cues, incoming new information, listener backchannels, other-directed speech, ambient speech and human resolution.

Expected behavior is derived from the opportunity:
- proactive visual cue → intervene;
- new information during output → adapt output;
- backchannel / other-directed / ambient speech → continue if speaking, otherwise remain silent;
- human resolution → stop if speaking, otherwise remain silent.

Each opportunity requires exactly one prediction. Response/adjustment outputs require an independently graded quality score. The evaluator reports:
- response-window rate (only correct action by the deadline counts);
- late actions and missed actionable opportunities;
- overall action accuracy;
- silence preservation and false intervention;
- continuation through non-interrupting overlap;
- stop compliance after human resolution;
- response quality;
- median and p95 response latency.

Undefined denominators and response-latency statistics remain `None`, not zero. Missing predictions and duplicate IDs fail closed. The evaluator consumes annotation and model-output records; it does not infer opportunities from raw media.

## Proposed qualification targets (not results)

| Metric | Target |
|---|---:|
| Response-window rate on actionable opportunities | >=75% |
| p95 response reaction latency | <=500 ms |
| Correct continuation through backchannels | >=98% |
| Correct continuation through other-directed/background speech | >=95% |
| Correct silence preservation when not addressed | >=98% |
| Stop compliance after human resolution | >=98% |
| Independently graded response quality | >=4.0/5 (map to [0,1] only after rubric validation) |

Targets must be assessed per opportunity family and under continuous stream conditions; do not hide poor response-window performance behind low average latency or high response quality.

## Additional baselines

- DIVE for continuous response windows and intervention timing.
- VideoFDB for dyadic audiovisual perception/generation.
- Full-Duplex-Bench v1.5 for interruption, backchannel, other-directed and ambient overlap.
- InteractionBench for task, timing, whisper/volume and long-context grounding.
- τ-Voice for grounded task completion under realistic noise and accents.
- Duplex-MPE for multi-party addressee and human-resolution behavior.

Do not pool leaderboard scores across different stimuli, model input modalities, scoring units or network boundaries. Run matched input conditions where possible and record p50/p95/p99 end-to-end latency.

## Evidence boundary

Twelve source-level tests are authored but have not been compiled or executed. No DIVE sample, licensed benchmark data, raw audio/video, annotated continuous stream or model prediction was loaded. No response-window score or real-time latency measurement exists. Benchmark dataset licenses must be checked independently; evaluation permission does not imply model-training or media-derivative permission.
