# Symthaea Embodiment Competitive Baselines and Differentiation v0.1

**Date:** 2026-11-27  
**Status:** research snapshot; external baselines are published/vendor-reported; Symthaea has no comparative result yet

## Executive position

Do not try to win first on photorealistic faces. That is where established real-time avatar stacks have mature capture, rendering, custom-identity, and cloud-delivery infrastructure.

The stronger research opportunity is to make embodiment **semantically faithful, full-duplex, permissioned, cross-form, accessible, and independently auditable**. A beautiful avatar that interrupts, invents confidence, reveals sensitive context, or continues after a user cancels is not a better conversational partner.

This is a differentiation strategy, not evidence of superiority. Symthaea needs matched evaluation before making a comparative claim.

## Current reference landscape

| Reference | What it demonstrates | What it does not establish |
|---|---|---|
| NVIDIA VideoFDB v1.4 | Full-duplex audio-visual perception and generation scores, timing, 10 full-duplex and 3 cascaded entries (13 total) | General intelligence, consent, privacy, accessibility, multi-form semantic invariance |
| Tavus conversational-video stack | Integrated LLM, speech, perception, rendering and WebRTC pipeline; tools, memory and RAG product capabilities | Independent evidence that the stack preserves epistemic provenance or supports a renderer-independent semantic identity contract |
| Epic MetaHuman | Real-time character animation from mono camera, supported mobile device or audio input in Unreal Engine 5.6+ | Full-duplex conversational cognition or event-level authority/consent semantics |
| Anam latency benchmark | Client-measured full-pipeline response and connection latency across avatar vendors | Independent academic validation; results are vendor-published and provider defaults differ |
| InteractionBench (2026) | Same-audio evaluation of timing, spoken-task accuracy, whisper/volume control and long-context groundedness | Full audio-visual perception/generation or multi-form renderer semantics |
| HEARTBEAT 2027 | Streaming proactive understanding and reciprocal audio turn-taking under a strict online protocol | Eligibility of a cascaded or agent-based Symthaea architecture; official rules require end-to-end submissions |

### VideoFDB v1.4 — current full-duplex AV reference

NVIDIA's leaderboard snapshot is dated November 25, 2026 and lists 13 entries total: 10 full-duplex and 3 cascaded.

Perception:
- Tavus Griffin Lite is the leading listed full-duplex entry: 3.85/5 overall, 90.63% turn-onset-response alignment, 200 ms median latency.
- Anam Cara-5 Preview is the top overall entry at 3.87/5, but it is cascaded; timing is 87.88% / 920 ms.

Generation:
- Tavus Griffin Lite is the leading full-duplex entry: 4.20/5, 48.78% timing alignment, 1,200 ms median latency.
- Anam Cara-5 Preview leads overall at 4.44/5 and 80.00% / 920 ms, but is cascaded.

The primary comparison for Symthaea's full-duplex ambition is the full-duplex bucket. The cascaded leader is a useful system-level reference but not an architecture-matched SOTA claim. VideoFDB uses 237 dyadic clips over 11 nonverbal dynamics and scores perception separately from generation. The benchmark reports failure modes such as visual-stream ignorance and captioning collapse; it also finds cascaded pipelines structurally limited in emitting nonverbal cues during the user's turn.

Source: https://research.nvidia.com/labs/amri/projects/video-fdb/leaderboard.html

### MetaHuman 5.8 — capture, animation and scale

Epic's current MetaHuman 5.8 release notes expand the comparison surface:
- real-time facial animation can consume RTSP video streams through Live Link;
- experimental MetaHuman Crowd tooling scales from tens to thousands of characters;
- single-camera body capture is an offline solve, marked experimental/editor-facing, and is not a runtime/player-facing body-animation feature;
- realtime facial animation remains available through audio or compatible mono-camera/mobile sources.

This is a strong animation, capture and content-production baseline. It still does not establish a permissioned semantic event contract, full-duplex conversational reasoning, or cross-form semantic invariance. The strategic choice is therefore not to race MetaHuman on face fidelity first; it is to make the source event, policy decision and presented meaning independently inspectable.

Sources:
- https://dev.epicgames.com/documentation/en-us/metahuman/metahuman-5-8-release-notes-in-unreal-engine
- https://dev.epicgames.com/documentation/en-us/metahuman/realtime-animation

### Commercial avatar delivery

Epic documents real-time MetaHuman animation from a mono camera, supported mobile device, or audio source, requiring Unreal Engine 5.6+ and the MetaHuman Live Link plugin. This is a strong character-animation and capture baseline. It is not the same task as full-duplex conversational reasoning or event-authorized output.

Source: https://dev.epicgames.com/documentation/en-us/metahuman/realtime-animation

Tavus advertises an integrated conversational-video system combining LLM, TTS, STT/ASR, WebRTC and perception, plus conversation memory, knowledge retrieval and function calling. Its product page also describes 1080p video, 24 kHz audio, audio-only and video modes, dynamic memory, guardrails and tool calling. That breadth is the product baseline Symthaea must respect: a renderer-only demo will not compete with an integrated service. These are vendor product claims, not independent benchmark results.

Source: https://www.tavus.io/pricing

Anam's vendor-published October 16, 2026 latency study reports a full-pipeline response latency of 1.24 s median and 1.84 s P95 for Cara-4.1 Express, from 1,173 measured turns in that provider row; the wider study included 4,703 turns, 926 connections and 60 participants. Its time-to-first-frame was 1.42 s median and 2.97 s P95. A separate shared LiveKit pipeline table reports different response latencies (2.12 s median and 3.64 s P95 for Anam); do not mix that setup with the provider-default full-pipeline comparison. These are vendor-published operational references, not independent academic results.

Source: https://anam.ai/tools/latency-benchmark

### Full-Duplex-Bench v3 — tool-using conversation

Full-Duplex-Bench v3 adds a task-completion lane that complements InteractionBench's timing, spoken control and groundedness probes. The published aggregator snapshot describes 100 examples across 79 scenarios, 12 speakers and four domains. Its available table reports GPT Realtime 1.5 at 60% strict pass@1, 6.36 s first response and 6.89 s task completion; Gemini Live 3.1 at 54%, 3.95 s and 4.25 s. A cascaded Whisper/GPT-4o/TTS baseline reaches 45% and takes 10.12 s to complete the task.

The table is a fixed snapshot transcribed from the benchmark paper, not a continuously updated universal ranking. Use it as a separate task-capability lane; do not combine its scores with VideoFDB or InteractionBench.

Source: https://benchlm.ai/voice-benchmarks/full-duplex-bench-v3

### Meta Muse Realtime Avatar — direct real-time embodiment comparator

Meta's September 23, 2026 technical post describes Muse Realtime Avatar as a streaming system sharing speech tokens between voice generation and avatar video generation. It reports:
- 448×768 portrait video at 25 FPS;
- approximately 870 ms from the end of the user's turn to the first byte of the synchronized voice-and-video response;
- 12 concurrent video-generation sessions on one GB200;
- a causal chunked Diffusion Transformer with a rolling video-latent context;
- a distilled two-step student versus a 120-evaluation teacher, described as a 60× reduction in model evaluations per chunk.

Meta also reports a preference comparison against Runway Characters and HeyGen LiveAvatar, with raters preferring Muse overall and on most dimensions; its mannerism comparison with Runway was not statistically distinguishable from parity.

These are Meta's own technical and preference-study claims, not an independent cross-provider benchmark. Its 870 ms metric is from end of turn to first synchronized response byte, not the same metric as Anam's client-measured response-start median or VideoFDB's turn-onset alignment. Keep all three separate.

**Strategic implication:** visual realism and real-time avatar generation are now mature, highly optimized competition areas. Symthaea should not attempt to outspend these systems on photorealistic video generation. Its research advantage should be a verifiable interaction contract: user-controlled presentation, causal turn state, cross-form semantic invariance, calibrated uncertainty, privacy-minimized renderer inputs, and truthful observation receipts.

Source: https://research.meta.ai/blog/bringing-your-muse-to-life

### InteractionBench — important voice-only complement

The November 2, 2026 InteractionBench study covers 13 scenarios and reports 1,519 graded runs across seven systems. It evaluates conversational timing, spoken-task accuracy, whisper/volume control and long-conversation groundedness separately. The three hosted systems and the turn-based cascade answered 34–38 of 50 spoken items (68–76%); the strongest client-clock timing score is 86% for GPT-Live, while the server-clock reconstruction is 56%. Gemini Live whispered on 8/10 explicit requests; only 2 of 35 whispered requests across all seven systems received a quieter reply. Hosted-system groundedness ranges from 67–88%. These are benchmark-specific observations, not universal provider rankings.

The key methodological finding is that server-reported latency can be misleading: client-side timing included network delay, and a server-clock correction changed GPT-Live's timing aggregate from 86% to 56%. Symthaea should measure at the user's output device and preserve both source and client clocks.

Source: https://arxiv.org/abs/2611.02430

### StreamEval — long-horizon interactive video and instruction following

StreamEval v1 (paper posted October 20, 2026) evaluates 300 scripted sessions lasting 15–300 seconds across five tasks: avatar dialogue, avatar performance, world generation, instruction editing and reference editing. It reports ten task-specific metrics in four dimensions: task execution, state consistency, visual fidelity and audiovisual coordination. Its headline finding is important: high visual quality and identity retention do not imply reliable execution of timed requests or sustained state consistency.

Current public leaderboard snapshot:
- Avatar dialogue: Vidu S2 Avatar scores 93.98 overall, but that total includes a +5 custom-avatar bonus; content-only quality overall is 88.98. Tavus CVI Full scores 86.65 overall with 94.44 A/V sync, 99.95 stability, 73.91 dialogue and 89.89 identity.
- Avatar performance: Vidu S2 Avatar scores 81.71 overall including the +5 bonus (76.71 content-only), with 99.60 A/V sync, 97.25 stability, 36.90 Following and 86.12 identity. LiveTalk 1.3B scores 75.85 overall including the bonus (70.85 content-only), with 71.27 quality, 86.44 A/V sync, 96.06 stability, 16.59 Following and 83.89 identity.

Do not compare overall numbers without removing the custom-avatar bonus and preserving task separation. Following scores are still low even where sync and visual stability are high; this is a strong signal that instruction execution is not solved by making an avatar look convincing.

**Symthaea implication:** build a separate polyembodiment task lane. Send timed commands and consent changes during a live stream, then evaluate whether the same semantic state is followed and preserved across all seven forms. Measure command success, time-to-follow, state preservation, semantic-digest invariance, A/V alignment, stale-output duration, privacy/permission compliance and recovery after interruption. Report dialogue and performance separately; never fold them into one score.

StreamEval is a public benchmark and its scores are a current external reference, not a matched Symthaea result. Its scripted request-following task is related to—but not a substitute for—Symthaea's explicit consent and receipt protocol.

Sources:
- https://arxiv.org/abs/2610.28402
- https://streameval.ai/
- https://github.com/StreamEval/StreamEval

### TurnBench — turn-taking quality, with a scoring warning

TurnBench reports a 30-hour dual-channel held-out corpus with 154 dialogues, 106 actors, six conversation styles and triple annotation (Fleiss' κ = 0.78). The current v1 leaderboard shows different trade-offs: Kuroko is 0.983 / 0.065 FPR / 619 ms; OpenYap Turn is 0.978 / 0.077 / 298 ms; p99lab turn-1-mini is 0.975 / 0.075 / 595 ms; Neosapience Nunchi v2 is 0.959 / 0.064 / 254 ms; FD-VAD is 0.826 / 0.049 / 1,025 ms.

The maintainers explicitly document that v1 can reward multiple and late predictions and ranks by recall, allowing simple VAD rules to perform deceptively well. They describe a planned v2 using per-frame probabilities and fixed hidden decision points.

Symthaea must not optimize v1 recall alone. Use the official v1 scorer only as a diagnostic, always publish FPR and latency, and build a one-decision-per-hidden-event internal evaluation while preparing to adopt v2.

Source: https://turnbench.sesame.com/

## Where Symthaea should differentiate

1. **One identity, many forms:** all renderers receive the same versioned semantic cue and provenance. A puppet, constellation, humanoid avatar, or instrument may differ in visual grammar, never in the event's meaning.
2. **Presentation as a permissioned effect:** events propose expression; a default-deny policy grants a bounded form/channel/destination/expiry/intensity. Sensor capture and physical actuation are separate permissions.
3. **Causal turn-taking:** WAIT, RESPOND, CONTINUE and YIELD are timestamped state transitions, not animation labels guessed from a final transcript.
4. **Content-bound rendering:** each presentation permit binds the full canonical cue with a deterministic digest; scene compilers reject payload substitution even when event IDs and semantic categories match. Renderer frames carry the digest but not raw evidence-reference identifiers.
5. **Epistemic fidelity:** measured, derived, inferred, proposed, and presentation-only states remain distinct. Uncertainty is rendered as uncertainty; a visual glow is never treated as evidence.
6. **Local-first graceful degradation:** text, static, reduced-motion and low-power modes are first-class; the core event and policy logic should run without a hosted avatar API.
7. **Outcome receipts:** distinguish permitted, submitted, observed, stopped and failed. Never equate an authorization decision with a successful presentation.
8. **User agency:** pause, cancel, form selection, privacy and motion controls override queued aesthetic or model preferences.

## Benchmark portfolio and proposed win conditions

These are targets, not results.

| Track | Proposed target | Why it matters |
|---|---|---|
| VideoFDB matched full-duplex perception | >=4.0/5, >=92% TOR alignment, <=150 ms median TOR latency | Exceed the current full-duplex reference on score and timing |
| VideoFDB matched full-duplex generation | >=4.35/5, >=60% TOR alignment, <=900 ms median TOR latency | Improve on 4.20/5, 48.78%, 1,200 ms full-duplex reference |
| Cross-architecture AV2AV stretch | >=4.50/5, >=82% TOR alignment, <=800 ms | Exceed the current cascaded generation row, but report as cross-architecture only |
| InteractionBench voice track | >=80% spoken tasks, >=90% timing, >=85% whisper/volume, >=92% groundedness | Exceed the best published dimension scores in the tested set |
| Client-heard response latency | p50 <=1.0 s and p95 <=1.5 s on a frozen network/hardware profile | Beat the vendor-reported 1.24 s / 1.84 s operational reference |
| Semantic invariance | 100% identity/provenance preservation across seven forms | Prevent visual form from rewriting cognitive state |
| Consent/security | Zero unauthorized permits in deterministic negative controls; revocation before next render tick | Make agency an executable property |
| Renderer performance | p95 event-to-first-visible-cue <=100 ms on named desktop/mobile hardware | Keep expressive cues synchronous with conversation |
| Accessibility | >=95% comprehension of essential event states without colour, motion, audio, camera or gaze | Ensure equivalent access rather than decorative fallback |
| Temporal attention | Match 2 FPS perception within 0.05 points using >=20% fewer frames, or beat 1 FPS by >=0.15 points at matched budget | Demonstrate a true quality/compute Pareto improvement |

Never combine these into one headline score. Report per-track scores, confidence intervals, p50/p95/p99, resource use, failure rates and architecture.

## Execution order

1. Compile and qualify the event, projection and policy core.
2. Add an independently observable presentation receipt protocol.
3. Build the first procedural Living Constellation renderer; no sensors, network or actuators.
4. Add a Puppet Theatre adapter over the same cue fixtures; run cross-form invariance tests.
5. Implement a client-clocked full-duplex trace harness and InteractionBench-compatible voice suite.
6. Evaluate VideoFDB only through the official dataset and scorer.
7. Run consented user studies for comprehension, agency, privacy comfort and non-assimilation.
8. Add camera/event-sensor adapters only after their permissions, retention and energy costs are separately tested.

The near-term win is not "more realistic". It is *more faithful, more interruptible, more private, more accessible, and measurably lower-latency* without changing what the system actually knows.
