# Symthaea Embodiment Benchmark Plan v0.1

**Date:** 2026-11-27  
**Status:** preregistration proposal; targets are aspirational and unmeasured  
**Parent:** SYMTHAEA_POLYEMBODIMENT_ARCHITECTURE_V0_1_2026-11-27.md

## 1. Research question

Can one cognitive system express the same interaction state through multiple embodiment forms with lower latency, greater user comprehension and agency, stronger epistemic fidelity, and better local-device efficiency than a conventional single human-avatar pipeline?

The comparison must separate:
1. the quality of the underlying cognitive/communication policy;
2. the semantic event contract;
3. the renderer and animation policy;
4. the end-user experience.

A renderer must not receive extra information unavailable to the baseline. An avatar cannot be scored as better because it was given more context.

## 2. Baselines

| Baseline | What it establishes | Fair comparison rule |
|---|---|---|
| Text-only Symthaea | cognitive/communication baseline without animation | same event and response content |
| Static canonical avatar | value of any visual presence | same identity and messages |
| Single humanoid puppet/avatar | conventional embodied interface | same model, speech and inputs |
| Cascaded speech-to-avatar pipeline | non-full-duplex commercial-style pipeline | same audio content and hardware where possible |
| Full-duplex AV agent | current frontier reference | compare only on compatible benchmark protocol |
| Multi-form Symthaea | proposed system | same source events, multiple renderers |

External systems are references, not presumed competitors on every dimension. Record model, version, access date, API tier, serving region, hardware, and any undisclosed components. Report unavailable measurements as unavailable rather than estimating them.

## 3. Benchmark tracks

### E0 — Event-contract correctness

**Purpose:** Prove that presentation is causally bound to the events that authorize it.

Fixtures cover:
- valid ordered events;
- duplicate event IDs and sequence numbers;
- cross-session and stale events;
- event expiry;
- missing input chunks;
- out-of-order delivery and replay;
- superseding/cancel events;
- unknown event kind and schema version;
- invalid confidence and privacy class;
- renderer crash/restart and reconnect;
- mixed clock domains and mismatched caller clock identity;
- per-session event-budget exhaustion;
- duplicate cancellation and cancellation of an unknown event;
- action-specific control payloads containing forbidden extra fields;
- privacy-labelled cues and redaction of generic accessibility text.

**Hard gate:** 100% expected outcomes on deterministic positive and negative controls; zero accepted stale, replayed, cross-session, mixed-clock, over-budget, or causally backdated events. Invalid traces are excluded from quality averages and reported separately.

### E0.5 — Consent and presentation authorization

**Purpose:** Prove that semantic events cannot directly authorize an output.

The renderer boundary must apply the default-deny policy in the
embodiment_policy.rs module. A permit is a scoped decision, not proof of renderer
execution, consent authenticity, or physical safety.

Deterministic positive controls cover an explicitly granted form/channel/
destination with a live session and clock. Negative controls cover missing or
expired grants, mismatched session/clock, ungranted forms/channels/destinations,
LocalSensitive output to shared/remote sinks, Restricted data, cancellation,
pause, channel revocation, invalid intensity, grant-ID reuse after replacement, and grant-history capacity exhaustion.

**Hard gate:** zero unauthorized permits across the complete deterministic
negative-control corpus; 100% expected allow/deny outcomes; no semantic or
provenance change from motion reduction or intensity clamping. A renderer must
re-check revocation/cancellation immediately before output.

Targets (not results):
- local pause/revocation effective before the next renderer update;
- p99 policy decision latency below 1 ms at 10,000 decisions/s on named hardware;
- zero sensor activations caused by a presentation grant;
- 100% traceability from source cue to policy decision and observed renderer outcome.

Consent authentication, sensor capture permission, remote data export, and
physical actuator safety are separate contracts and must not be inferred from
this module. See
[Embodiment Consent and Presentation Policy](SYMTHAEA_EMBODIMENT_CONSENT_POLICY_V0_1_2026-11-27.md).

### E0.6 — Presentation outcome receipt correctness

**Purpose:** Prove that authorization, submission, observed output and confirmed
cessation are distinct events.

Run the receipt ledger against the positive/negative lifecycle fixtures and
fault-injection traces. Enforce the permitted transition graph, single terminal
outcome, attempt identity, exact grant identity, event/grant/effective expiry,
observer-channel compatibility, and per-clock monotonicity. Verify every
ledger digest and previous-digest link.

**Hard gate:**
- 100% expected transition outcomes on frozen fixtures; final policy authorization and adapter-acknowledged submission are separate records;
- zero accepted invalid transitions or second terminal outcomes;
- authorization does not count as submission; a submitted receipt requires adapter acknowledgement; missing output observation remains OutcomeUnknown, never Observed or Stopped;
- Stopped requires a named observer reporting a stopped output state;
- receipt chain verification succeeds for every unmodified accepted trace;
- source and output timestamps are never compared across unrelated clock domains.

Targets (not results):
- p99 final policy-recheck-to-Authorized-record overhead below 1 ms on named hardware; report adapter acceptance latency separately;
- p95 submission-to-observation latency below 100 ms for the local visual renderer;
- cancellation-to-observed-stop p95 below one render update;
- zero raw-media data in receipt records;
- 100% records carry policy/adapter/build identity and both event and grant expiry.

Report observer type, adapter build, clock domains, unknown outcomes, dropped
observations, terminal failure types, and resource cost. An adapter callback
alone is not independent observation; external observer independence must be
qualified separately.

### E1 — Full-duplex conversational behavior

Use VideoFDB's published protocol and dataset if its terms permit. Do not claim a VideoFDB leaderboard result unless the official evaluation path and artifact requirements are followed.

Report separate perception and generation scores. Preserve the four communication channels and the event families: dialogue/turn-taking, gaze/attention, face/expression, and body/gesture.

Project targets (not results):
- matched full-duplex perception: >= 4.0/5, >= 92% TOR alignment, and <= 150 ms median TOR latency;
- matched full-duplex AV2AV generation: >= 4.35/5, >= 60% TOR alignment, and <= 900 ms median TOR latency;
- cross-architecture stretch (not an architecture-matched claim): >= 4.50/5 generation, >= 82% TOR alignment, and <= 800 ms, exceeding the current cascaded generation reference;
- compare to the best matching full-duplex architecture/modality, not merely the highest overall row;
- report every rubric axis, sample count, architecture, modality, weight availability and model/provider status.

The current VideoFDB v1.4 snapshot is dated 2026-11-25 and lists 13 total entries: 10 full-duplex and 3 cascaded. Tavus Griffin Lite leads the full-duplex perception bucket at 3.85/5 and 90.63% TOR alignment / 200 ms median latency; it leads full-duplex generation at 4.20/5 and 48.78% / 1,200 ms. Anam Cara-5 Preview leads overall perception at 3.87/5 and generation at 4.44/5, but is cascaded (perception 87.88% / 920 ms; generation 80.00% / 920 ms). The targets above are intentionally above the matched full-duplex reference; the separate cross-architecture stretch is explicitly labelled as such. Use official splits, preserve per-sample outputs, and do not tune on the evaluation set.

### E2 — Turn-taking and response timing

Adapt the project's MEL-RTI-001 evaluator carefully:
- retain WAIT, RESPOND, CONTINUE and YIELD semantics where appropriate;
- use the monotonic runner clock for decision and effect latency;
- use the source media clock for event alignment;
- never infer missing chunks as WAIT;
- score classification and effect timing separately.

Speech-specific labels must be defined independently from musical labels such as cadence, pickup, and solo handoff.

Metrics:
- per-action precision, recall and F1;
- macro-F1 and balanced accuracy;
- turn-onset alignment;
- interruption handling;
- backchannel timing;
- false activation during WAIT;
- premature yield during CONTINUE;
- p50/p95/p99 decision and effect latency;
- calibration and confidence intervals clustered by conversation/session.

Target: >= 90% turn-onset alignment on the frozen project test set after human annotation validation. A universal millisecond success window must not be invented; calibrate the permitted windows to each annotated conversational event.

### E2.5 — Voice-only interaction quality and client timing

Use InteractionBench as a complementary voice-only track, not a substitute for
VideoFDB's audio-visual perception/generation protocol. It measures timing,
spoken-task accuracy, whisper/volume control, and groundedness across 13
scenarios (221 items per system in the published study).

The November 2, 2026 study reports 34–38 correct spoken items out of 50
(68–76%) for the three hosted systems and the turn-based cascade; the six
full-duplex systems have timing and spoken-task scores that are related but
must be reported separately. The strongest published timing score is 86% for
GPT-Live on the client clock (its server-clock reconstruction drops to 56%).
Gemini Live whispers on 8/10 explicit requests, but only 2 of 35 whispered
requests across all systems led to a quieter reply. Hosted-system groundedness
falls between 67% and 88%. The benchmark reports dimensions separately; these
figures are not a combined system score and are not universal rankings.

Symthaea stretch targets on the same frozen protocol:
- >= 80% spoken-task accuracy;
- >= 90% conversational timing;
- >= 90% explicit whisper compliance, >= 90% user-volume understanding, and
  >= 80% appropriate output-volume adaptation to whispered users;
- >= 92% groundedness probes;
- response latency measured at the client output device, with p50 <= 1.0 s and
  p95 <= 1.5 s on a named network/hardware profile.

As an additional operational reference, Anam's vendor-published 2026-10-16
full-pipeline study reports 1.24 s median / 1.84 s P95 response latency across
4,703 measured turns and 60 participants. It is a vendor-published comparison,
not an independent academic benchmark. Match provider pipeline, network,
turn-boundary rules, and client measurement before asserting a win.

Record source-media, model, client-output, and network clocks separately.
Report early starts, wait-through-pause rate, interruption yield, backchannel
robustness, p50/p95/p99 latency, and groundedness. Server-only timestamps cannot
replace user-heard timing.

References:
- https://arxiv.org/abs/2611.02430
- https://avera-labs.github.io/InteractionBench/
- https://anam.ai/tools/latency-benchmark

### E3 — Cross-embodiment semantic consistency

Replay identical, frozen event traces through every renderer.

Measure:
- expected event-to-expression mappings;
- omitted and unsupported events;
- contradictions between forms;
- time alignment across renderers;
- identity continuity across transitions;
- preservation of user controls and privacy settings.

Target: 100% deterministic fixture mapping; >= 95% human-rated semantic agreement between forms on held-out interactions; zero false claims of verification, certainty, consent or action.

### E4 — User comprehension and agency

Run a preregistered, randomized, within-subject study comparing text-only, static avatar, single puppet, and multi-form presentation.

Primary outcomes:
- correct interpretation of turn state and uncertainty;
- recall of key information;
- task completion and error recovery;
- perceived control and ability to interrupt;
- unwanted anthropomorphic inference;
- cognitive load and distraction;
- preference by task and accessibility need.

Target: non-inferior task comprehension versus text-only (pre-register a 5 percentage-point non-inferiority margin), while improving at least one prespecified interaction outcome. Do not optimize solely for liking or anthropomorphic attachment.

Recruitment, sample-size calculation, exclusions, consent, privacy, and analysis plan must be fixed before collecting confirmatory data. Pilot data are not confirmatory evidence.

### E5 — Responsiveness and renderer performance

Measure:
- source-event-to-render-receipt latency;
- source-event-to-first-visible-frame latency;
- p50/p95/p99 frame time;
- dropped and duplicated frames;
- stale cue duration;
- transition time between forms;
- cancellation latency;
- CPU/GPU/RAM and thermal behavior;
- battery and energy per interaction minute.

Proposed engineering targets:
- local event-to-first-visible-frame p95 <= 100 ms;
- cancellation effective by the next render update;
- 60 Hz desktop target and 30 Hz mobile minimum target;
- < 1% dropped frames over a 15-minute reference workload.

Run separately on named hardware and power profiles. A target achieved on one device does not generalize to all devices.

### E6 — Accessibility and comfort

Test keyboard-only, touch, screen-reader, reduced-motion, low-vision, hearing-accessible, and no-audio paths.

Measure:
- task success by access mode;
- time and error rate;
- discoverability of pause/cancel;
- motion discomfort and distraction;
- clarity of nonverbal cues without colour or sound.

Hard requirement: every essential state also has a text or accessible semantic representation. No critical warning may depend solely on colour, animation, gaze, or audio.

### E7 — Privacy, security and robustness

Fault injection:
- permission revoked mid-session;
- camera/microphone unavailable;
- event stream disconnected;
- stale cached scene;
- renderer crash;
- malformed or malicious event;
- replayed event;
- oversized asset or decompression bomb;
- untrusted asset manifest;
- attempted actuator command without permission.

Hard gate:
- no sensor starts without explicit permission;
- raw audio/video retention is disabled by default;
- no actuator command without a separately authorized, typed action receipt;
- all invalid event cases fail closed;
- renderer failure cannot grant additional authority to the cognitive system.

## 4. Scoring policy

Do not collapse all outcomes into one weighted score during early development. Publish a dashboard with:
- hard safety/causality gates;
- full-duplex perception;
- full-duplex generation;
- turn-taking and latency;
- cross-form consistency;
- user comprehension and agency;
- accessibility;
- resource use.

A system failing a hard safety or causality gate is not rescued by high animation quality.

For comparisons, publish paired per-sample outputs, dataset/version hashes, prompt and policy versions, model/provider versions, hardware, and analysis scripts. Use bootstrap confidence intervals clustered by user/session; report effect sizes and correction for multiple comparisons. Preserve failed and invalid runs.

## 5. SOTA comparison strategy

### Current references

- VideoFDB v1.4 (2026-11-25): Anam Cara-5 Preview leads perception overall at 3.87; generation overall is 4.44. Other leading entries include Tavus Griffin Lite, MAUI, and Atmanity Causeur. Scores and timing are rubric-specific.
- MetaHuman Animator: strong real-time/offline human performance animation from video, mobile capture, or audio; use as a human-rig quality reference.
- VRM 1.0: relevant for portable humanoid puppet assets, not abstract forms.
- glTF 2.0.1: relevant for portable runtime 3D scene assets and animation channels.

### Where Symthaea should try to win

1. **Polyembodiment:** same meaning across non-human, puppet, graph, swarm and physical forms.
2. **Epistemic fidelity:** rendered confidence and verification states trace to typed source evidence.
3. **Full-duplex responsiveness:** timely listening, backchannels, interruption recovery and cancellation.
4. **Local efficiency:** useful interactions on mobile and modest GPUs, with measured energy and thermal profiles.
5. **Accessibility:** understandable without photoreal facial cues, colour, audio or high-motion animation.
6. **User sovereignty:** transparent, reversible form choice and explicit sensor/actuator permissions.
7. **Reproducibility:** pinned assets, policies, datasets and event traces, plus independent replay.

Do not state "better than SOTA" until a matched protocol demonstrates it. Use external leaderboards for the claims they actually measure, and our own benchmark for the orthogonal contributions.

## 6. Experiment execution plan

| Phase | Work | Exit evidence |
|---|---|---|
| 0 | inventory and event contract | frozen schema, positive/negative fixtures |
| 1 | renderer-neutral Rust module | exact test results, no renderer dependency |
| 2 | Living Constellation renderer | deterministic trace-to-frame receipts |
| 3 | one puppet + one nonhuman form | cross-form semantic agreement |
| 4 | streaming audio/full-duplex | VideoFDB-compatible report and turn timing |
| 5 | mobile integration | Pixel 8 Pro profiling and privacy evidence |
| 6 | user study | preregistered analysis and accessibility results |
| 7 | physical Hearth endpoint | separate actuator safety and human-factors gate |

## 7. Reproducibility requirements

Every benchmark run should bind:
- source commit and working-tree state;
- schema, policy, renderer, asset and dataset digests;
- hardware, OS, GPU/driver, display resolution and power mode;
- model/provider identifiers and serving configuration;
- raw per-sample results and scorer version;
- exclusions, failures and invalid traces;
- exact evaluation command and environment.

Do not infer a passing build or benchmark from an authored test, queued CI run, skipped workflow, or synthetic demonstration. Mark outcomes as design target, implementation, locally tested, exact-head CI tested, hardware tested, or independently validated.

## 8. Research references

- NVIDIA VideoFDB project: https://research.nvidia.com/labs/amri/projects/video-fdb/
- VideoFDB leaderboard v1.4: https://research.nvidia.com/labs/amri/projects/video-fdb/leaderboard.html
- VideoFDB paper: https://research.nvidia.com/labs/amri/publication/mazumdar2026videofdb/
- Epic MetaHuman Animator: https://dev.epicgames.com/documentation/en-us/metahuman/metahuman-animator
- Epic real-time MetaHuman animation: https://dev.epicgames.com/documentation/en-us/metahuman/realtime-animation
- VRM 1.0: https://github.com/vrm-c/vrm-specification/tree/master/specification/VRMC_vrm-1.0
- Khronos glTF: https://registry.khronos.org/glTF/
- VMBench: https://arxiv.org/abs/2503.10076
- Internal starting point: docs/research/MEL-RTI-001-STREAMING-TURNTAKING-SCORER-V1.md

## 9. Current status

No Symthaea embodiment benchmark scores are claimed by this plan. No renderer implementation, VideoFDB run, user study, or hardware qualification is included in this documentation-only proposal. The immediate implementation task is the typed event contract and deterministic replay harness.


## 10. Competitive snapshot refresh — VideoFDB v1.4 (2026-11-25)

The current official leaderboard lists 13 full-duplex systems and 10 cascaded systems in the architecture filters. Results must be compared within the relevant architecture and modality, not by sorting a single overall column.

### Perception

| Entry | Architecture | Overall | TOR alignment / median latency |
|---|---|---:|---:|
| Anam Cara-5 Preview | Cascaded, closed, AV2AV | 3.87/5 | 87.88% / 920 ms |
| Tavus Griffin Lite | Full-duplex, closed, AV2AV | 3.85/5 | 90.63% / 200 ms |
| MAUI + MiniCPM-o 4.5 | Full-duplex, closed, AV2A | 3.83/5 | 83.58% / 760 ms |
| Atmanity Causeur | Full-duplex, closed, AV2A | 3.82/5 | 88.06% / 1,400 ms |
| MiniCPM-o 4.5 | Full-duplex, open, AV2A | 3.40/5 | 73% / 720 ms |

### Generation

| Entry | Architecture | Overall | TOR alignment / median latency |
|---|---|---:|---:|
| Anam Cara-5 Preview | Cascaded, closed, AV2AV | 4.44/5 | 80.00% / 920 ms |
| Tavus Griffin Lite | Full-duplex, closed, AV2AV | 4.20/5 | 48.78% / 1,200 ms |
| Atmanity Causeur + Atmee v2 | Cascaded, closed, AV2AV | 4.16/5 | 62.75% / 1,200 ms |

Leaderboard timing is task-specific TOR alignment and median latency, not an end-to-end application latency guarantee. The top perception aggregate and the strongest full-duplex timing result belong to different entries. The highest generation aggregate is from a cascaded entry; it must not be presented as evidence that cascaded systems solve full-duplex nonverbal behavior.

VideoFDB reports 237 dyadic clips across 11 nonverbal conversational dynamics and separates perception from generation. Its project report also identifies the risk of low-rate frame sampling missing fast events, while raising the sampling rate globally can overload cross-modal attention. These observations motivate the event-triggered temporal-attention proposal; they do not prove that the proposed policy will outperform uniform sampling.

Official snapshot:
- https://research.nvidia.com/labs/amri/projects/video-fdb/leaderboard.html
- https://research.nvidia.com/labs/amri/projects/video-fdb/
- https://arxiv.org/abs/2605.30256

### Consequence for our target

Do not set "beat 3.87/4.44" as one undifferentiated goal. Report at least:
- matched full-duplex AV2AV perception and generation;
- matched AV2A perception;
- perception and generation separately;
- TOR alignment and latency separately;
- per-dynamic scores and sample counts;
- open/closed, full-duplex/cascaded, and model/provider status.

Our competitive objective is a Pareto improvement across task quality, timing, cross-form consistency, privacy, accessibility, and local compute—not a single aggregate score.


### E8 — Event-triggered temporal attention

Compare uniform 1, 2 and 4 FPS with a bounded adaptive 0.5–4 FPS policy and a ground-truth oracle upper bound. Match average frame count, processed pixels and attention/compute budget. Report each condition separately for audio-only, visual-only and fused triggers.

Metrics:
- recall/precision for each of VideoFDB's 11 conversational dynamics;
- trigger delay and missed-event duration;
- false triggers per conversation minute;
- TOR alignment and median/p95 event-to-decision latency;
- average/peak frame rate, pixel count, model input budget, energy and thermals;
- permission, raw-frame retention and revocation behavior.

Initial Pareto target: improve the perception aggregate by >= 0.15 points versus uniform 1 FPS at matched compute, or remain within 0.05 points of uniform 2 FPS while reducing processed frames by >= 20%, with no material generation/fluency regression. These are proposed margins, not evidence-based guarantees. Preregistered power analysis and pilot feasibility must precede confirmatory evaluation.

See `SYMTHAEA_DYNAMIC_TEMPORAL_ATTENTION_V0_1_2026-11-27.md` for the full protocol. Oracle annotations are prohibited at inference time and only establish an upper bound.


### E9 — HEARTBEAT Challenge 2027 compatibility

HEARTBEAT is a complementary external protocol, not a substitute for VideoFDB:

- Track I Reciprocal Audio Turn-Taking: WAIT, RESPOND, CONTINUE, YIELD; offline action/content and online speech with timestamped logs.
- Track II Streaming Proactive Understanding: online audio-visual and audio-only perception, comprehension, and reasoning; responses must be timely and relevant.
- The published release contains 132,571 samples (127,371 train, 5,200 development), approximately 3,103 hours, and about 599.83 GB packed.
- Every Track II input chunk must be logged, including chunks with no response; response timestamps cannot be backdated.
- The published rules require end-to-end models and disallow cascaded pipelines and agent-based systems. This may constrain Symthaea's modular architecture; competition eligibility must be assessed separately before entry.
- The evaluation environment advertises 8 × A100 80GB GPUs. Local development and official leaderboard execution must not be conflated.

The existing MEL-RTI-001 evaluator's WAIT / RESPOND / CONTINUE / YIELD vocabulary is a direct semantic starting point. A HEARTBEAT adapter must preserve its official chunk semantics, online/offline split, and evaluation rules; it must not relabel missing chunks as WAIT or backdate output.

**Schedule as published:** evaluation leaderboard opens December 2, 2026; both leaderboards close December 22; final container submissions are due December 25. Dates are UTC 23:59.

Initial research target:
- build a trace adapter and schema-mapping tests against public protocol examples;
- evaluate only on the official development split once data access, licensing, and resource constraints are checked;
- keep Track I and Track II results separate;
- verify model eligibility and end-to-end constraints before considering a competition submission.

References:
- https://heartbeat-challenge.github.io/
- https://github.com/heartbeat-challenge/HEARTBEAT_challenge_baseline/
- Internal evaluator: MEL-RTI-001, docs/research/MEL-RTI-001-STREAMING-TURNTAKING-SCORER-V1.md

No HEARTBEAT dataset has been downloaded, no official baseline has been run, and no leaderboard score is claimed.


## 11. New benchmark track — event-triggered temporal attention

The VideoFDB sampling findings motivate a new track described in
`SYMTHAEA_DYNAMIC_TEMPORAL_ATTENTION_V0_1_2026-11-27.md`.

Compare fixed 1, 2 and 4 FPS with a bounded event-triggered 0.5–4 FPS policy and an oracle upper bound, all matched by average frame count, pixel count, and compute budget. Report per-dynamic recall, event detection delay, false triggers/minute, TOR alignment, latency, compute, energy, and privacy behavior.

This is a research hypothesis. The trigger, sampler, dataset run, and any superiority claim are not implemented or measured yet.
