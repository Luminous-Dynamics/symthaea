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
| NVIDIA VideoFDB v1.4 | Full-duplex audio-visual perception and generation scores, timing, 13 full-duplex and 3 cascaded entries | General intelligence, consent, privacy, accessibility, multi-form semantic invariance |
| Tavus conversational-video stack | Integrated LLM, speech, perception, rendering and WebRTC pipeline; tools, memory and RAG product capabilities | Independent evidence that the stack preserves epistemic provenance or supports a renderer-independent semantic identity contract |
| Epic MetaHuman | Real-time character animation from mono camera, supported mobile device or audio input in Unreal Engine 5.6+ | Full-duplex conversational cognition or event-level authority/consent semantics |
| Anam latency benchmark | Client-measured full-pipeline response and connection latency across avatar vendors | Independent academic validation; results are vendor-published and provider defaults differ |
| InteractionBench (2026) | Same-audio evaluation of timing, spoken-task accuracy, whisper/volume control and long-context groundedness | Full audio-visual perception/generation or multi-form renderer semantics |
| HEARTBEAT 2027 | Streaming proactive understanding and reciprocal audio turn-taking under a strict online protocol | Eligibility of a cascaded or agent-based Symthaea architecture; official rules require end-to-end submissions |

### VideoFDB v1.4 — current full-duplex AV reference

NVIDIA's leaderboard snapshot is dated November 25, 2026 and lists 13 full-duplex and 3 cascaded entries.

Perception:
- Tavus Griffin Lite is the leading listed full-duplex entry: 3.85/5 overall, 90.63% turn-onset-response alignment, 200 ms median latency.
- Anam Cara-5 Preview is the top overall entry at 3.87/5, but it is cascaded; timing is 87.88% / 920 ms.

Generation:
- Tavus Griffin Lite is the leading full-duplex entry: 4.20/5, 48.78% timing alignment, 1,200 ms median latency.
- Anam Cara-5 Preview leads overall at 4.44/5 and 80.00% / 920 ms, but is cascaded.

The primary comparison for Symthaea's full-duplex ambition is the full-duplex bucket. The cascaded leader is a useful system-level reference but not an architecture-matched SOTA claim. VideoFDB uses 237 dyadic clips over 11 nonverbal dynamics and scores perception separately from generation. The benchmark reports failure modes such as visual-stream ignorance and captioning collapse; it also finds cascaded pipelines structurally limited in emitting nonverbal cues during the user's turn.

Source: https://research.nvidia.com/labs/amri/projects/video-fdb/leaderboard.html

### Commercial avatar delivery

Epic documents real-time MetaHuman animation from a mono camera, supported mobile device, or audio source, requiring Unreal Engine 5.6+ and the MetaHuman Live Link plugin. This is a strong character-animation and capture baseline. It is not the same task as full-duplex conversational reasoning or event-authorized output.

Source: https://dev.epicgames.com/documentation/en-us/metahuman/realtime-animation

Tavus advertises an integrated conversational-video system combining LLM, TTS, STT/ASR, WebRTC and perception, plus conversation memory, knowledge retrieval and function calling. That breadth is the product baseline Symthaea must respect: a renderer-only demo will not compete with an integrated service.

Source: https://www.tavus.io/pricing

Anam's vendor-published October 16, 2026 latency study reports a full-pipeline median response latency of 1.24 s and P95 of 1.84 s for Cara-4.1 Express, from 4,703 turns and 926 connections across 60 participants. It reports a 1.42 s median time to first frame. Its full-pipeline comparisons use provider defaults and include network time; treat these numbers as an operational reference, not an independent benchmark result or like-for-like model comparison.

Source: https://anam.ai/tools/latency-benchmark

### InteractionBench — important voice-only complement

The November 2, 2026 InteractionBench preprint tests 221 items per system over 13 scenarios and 1,519 graded runs. It measures conversational timing, spoken-task accuracy, whisper/volume control and long-conversation groundedness. In its tested set, the best dimension scores were 72% spoken-task accuracy, 86% timing, 75% whisper/volume control, and 88% groundedness; the best groundedness coherence score was 86/100. These are small benchmark-specific results, not universal vendor rankings.

The key methodological finding is that server-reported latency can be misleading: client-side timing included network delay, and a server-clock correction changed GPT-Live's timing aggregate from 86% to 56%. Symthaea should measure at the user's output device and preserve both source and client clocks.

Source: https://arxiv.org/abs/2611.02430

## Where Symthaea should differentiate

1. **One identity, many forms:** all renderers receive the same versioned semantic cue and provenance. A puppet, constellation, humanoid avatar, or instrument may differ in visual grammar, never in the event's meaning.
2. **Presentation as a permissioned effect:** events propose expression; a default-deny policy grants a bounded form/channel/destination/expiry/intensity. Sensor capture and physical actuation are separate permissions.
3. **Causal turn-taking:** WAIT, RESPOND, CONTINUE and YIELD are timestamped state transitions, not animation labels guessed from a final transcript.
4. **Epistemic fidelity:** measured, derived, inferred, proposed, and presentation-only states remain distinct. Uncertainty is rendered as uncertainty; a visual glow is never treated as evidence.
5. **Local-first graceful degradation:** text, static, reduced-motion and low-power modes are first-class; the core event and policy logic should run without a hosted avatar API.
6. **Outcome receipts:** distinguish permitted, submitted, observed, stopped and failed. Never equate an authorization decision with a successful presentation.
7. **User agency:** pause, cancel, form selection, privacy and motion controls override queued aesthetic or model preferences.

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
