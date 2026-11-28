# Symthaea polyembodiment competitive strategy and qualification roadmap — 2026-11-28

## Executive strategy

Do not try to win by building the most photorealistic avatar or the highest-quality isolated talking-head clip. Compete on **causal task usefulness per unit of latency, compute, energy and risk**.

The distinctive product is a coherent cognitive/communicative identity that can be rendered through multiple embodiments while preserving semantic meaning, obeying revocable permissions, allocating visual attention according to decision value, and producing independently checkable evidence that outputs were actually presented or stopped.

The moat is not one generator. It is the end-to-end contract across:

1. canonical semantics and epistemic status;
2. authorization and privacy policy;
3. temporal action schedule;
4. renderer-specific presentation;
5. independent output observation;
6. counterfactual evaluation and reproducible receipts.

The programme should remain modular: an external speech/avatar generator may be used as a replaceable actuator, while Symthaea owns the policy, decision, continuity, grounding and evidence contracts.

## Industry snapshot (checked 2026-11-28)

### VideoFDB v1.4 — audiovisual perception and generation

The official NVIDIA/David AI board was updated November 25, 2026. It separates Perception from Generation and distinguishes full-duplex from cascaded architectures.

| Track / architecture | System | Overall | Timing |
|---|---|---:|---|
| Perception, best listed overall, cascaded | Anam Cara-5 Preview | 3.87/5 | TOR 87.88%, median 920 ms |
| Perception, strongest listed full-duplex | Tavus Griffin Lite | 3.85/5 | TOR 90.63%, median 200 ms |
| Generation, best listed overall, cascaded | Anam Cara-5 Preview | 4.44/5 | TOR 80.00%, median 920 ms |
| Generation, strongest listed full-duplex | Tavus Griffin Lite | 4.20/5 | TOR 48.78%, median 1,200 ms |

These are not like-for-like architecture comparisons. MAUI configurations include an unreleased external controller; one timing result is offline simulation and excludes generation latency. Preserve track, architecture, sample count, modality and latency methodology in every comparison. A stronger overall score does not automatically mean a better interactive system.

### InteractionBench — same-input task, timing, paralinguistic and grounding evaluation

InteractionBench (November 2, 2026) evaluates 221 items across 13 scenarios and seven systems (1,519 graded runs) using the same recorded user audio. It jointly measures conversational timing, spoken-task accuracy, whisper/volume control, and groundedness over conversations longer than twenty turns. In the reported setup, a turn-based cascade reached 38/50 spoken tasks; hosted systems passed 67–88% of groundedness probes, compared with 14–35% for open full-duplex systems. Client-heard timing includes network effects, and the paper documents a roughly 1.5-second server/client timing difference for one hosted system.

This gives Symthaea a valuable multi-axis external baseline. Reproduce the protocol and its exact system/transport boundary before comparing numbers; do not mix server-reported latency with client-heard latency.

### Duplex-MPE — multi-party floor control

Duplex-MPE (September 25, 2026) contains 2,000 three- or four-person interaction scenarios, paired with explicit and implicit addressing. Its four measures distinguish fresh response initiation, answer accuracy, silence preservation, and stopping when a human resolves a request. It reports that systems can be overly eager: frequent speech does not necessarily mean correct response or appropriate silence. This is a critical gap in dyadic benchmarks, where the system may not need to identify who was addressed.

**New Symthaea challenge:** require the agent to identify speaker/addressee scope, avoid responding to implicit requests not directed to it, preserve silence, and stop when another human resolves the question. Track false activation and silence preservation separately from response recall.

### TurnBench — turn transfer and interruption

The official TurnBench v1 page shows Kuroko at 0.983 end-of-turn recall / 0.065 FPR / 619 ms median and OpenYap Turn at 0.978 / 0.077 / 298 ms. But the benchmark authors explicitly document a v1 scoring flaw: repeated late predictions can obtain high recall, a prediction can be detected after the other speaker begins, late fallbacks up to three seconds are accepted, and ranking uses recall rather than a joint latency/false-positive objective. They are preparing a v2 with fixed hidden decision points and threshold-free metrics.

Do not optimize against v1 recall alone. Use v1 only as a legacy reference; publish the full recall/FPR/latency surface and build an internal v2-style replay protocol. Include Duplex-MPE-style three-/four-speaker explicit-versus-implicit addressing and stop-on-human-resolution conditions.

### StreamEval — streaming avatar performance

The listed Vidu S2 Avatar avatar-performance overall is 81.71, but that includes a +5 customizability bonus; its content-only quality-overall is 76.71. Its reported A/V sync is 99.60 and stability 97.25, while following is only 36.90. The strategic lesson: excellent synchronization and stability do not imply strong action/task following. Compare component metrics and exclude non-quality bonus points when claiming quality superiority.

### DIVE — continuous audiovisual engagement

DIVE (October 20, 2026) evaluates continuous audiovisual streams rather than isolated timestamped questions. It has 525 audio/video samples, six tasks across proactive intervention, continuous engagement and interaction coordination, with separate response quality, participation decisions, response-window rates and reaction latency. The paper reports sub-second average response latency for leading real-time systems, yet response-window rates remain at or below 53.77%.

This exposes a gap between quick answers and sustained participation. A system can answer fast when it chooses to answer and still miss most opportunities to intervene or adapt to newly arriving information.

### $\\tau$-Voice — grounded task execution under realistic audio

The ICML 2026 $\\tau$-Voice benchmark contains 278 grounded tasks combining multi-turn conversations, domain policies, environmental interaction and realistic audio. In its reported evaluation, voice agents achieved 31–51% task completion in clean conditions and 26–38% under realistic noise and diverse accents, compared with 85% for the evaluated text reasoning baseline. These are study-specific results, not a universal modality gap, but they demonstrate the need to measure how much of the underlying reasoning capability survives real-time voice interaction.

### Full-Duplex-Bench v1.5 — overlapping speech behavior

The benchmark separates user interruption, backchannels, speech directed at others and background speech, with both behavior rates and stop/response latency. Realtime-Venus reports responding to 75% of user interruptions while continuing through 97% of backchannels, 88% of other-directed speech and 86% of background speech.

Symthaea should not optimize interruption recall alone: it must respond to true interruptions while preserving continuation through non-interrupting overlap. Report the four behavior families and end-to-end stop latency separately.

### ActAvatar — temporally anchored action control

ActAvatar (CVPR 2026) reports phase-aware cross-attention with globally and temporally anchored action prompt segments, plus progressive audio-visual alignment to reduce interference between text-driven action control and audio-driven lip synchronization. This validates the value of explicit temporal phases, but it is a talking-avatar generation method rather than a complete permissioned full-duplex agent. Symthaea's action timeline should complement, not claim to replace, learned generation models.

## Paradigm shifts worth pursuing

### 1. From avatar-first to semantics-first polyembodiment

A single canonical cue should drive every renderer. Each embodiment may change geometry, timing style, animation and accessibility, but not event identity, epistemic status, privacy class, recipient, intent or authorization. Build cross-form metamorphic tests rather than comparing only screenshots.

**Success criterion:** the semantic digest and task decision remain invariant across every form; a renderer is unable to manufacture semantic edges or confidence values.

### 2. From animation clips to causal action contracts

Speech, visemes, gaze, gesture, expression, posture and spatial cues should be scheduled as typed phases on one monotonic source clock. Barge-in, user stop, policy revocation and expiry are first-class events. Cancellation requested, adapter accepted, device stopped and independently observed stopped must remain separate states.

**Success criterion:** a user interrupt is a causal stop event with measured end-to-end p50/p95/p99, not just a flag in an animation system.

### 3. From reactive saliency to decision-theoretic sensing

Do not allocate extra frames because they are visually surprising. Estimate the lower confidence bound on expected task-loss reduction, subtract latency and staleness penalties, and divide by source pixel/energy cost. Sensor permissions remain hard gates outside the optimizer.

**Success criterion:** lower decision regret than surprise-only, fixed-FPS and random ROI baselines under matched source-pixel, latency and energy budgets.

### 4. From audiovisual correlation to counterfactual grounding

A system must change its decision when a relevant visual cue is removed or altered, while remaining invariant to irrelevant face, clothing, background and lighting changes. Audio digest and timing must remain identical across the counterfactual pair. Independently graded task relevance must not be supplied by the model under test.

**Success criterion:** causal visual contribution and counterfactual sensitivity rise without increasing appearance-captioning collapse or irrelevant-visual sensitivity.

### 5. From adapter acknowledgement to independent output evidence

An ECS submission is not proof of display; a device API acknowledgement is not proof that the sound stopped; a renderer receipt is not an independent observer. The compositor/device observer must bind the observation to session, surface/device, frame/action, clock, cue digest, grant and policy version.

**Success criterion:** zero unauthorized/post-expiry output; every Observed/Stopped claim has independent evidence and explicit uncertainty handling.

### 6. From leaderboard averages to assurance profiles

Publish a Pareto surface across task success, fluency, grounding, latency, false positives, interruption, compute, energy, privacy and failure recovery. A scalar score can hide catastrophic weaknesses. Safety and authorization are hard gates, not weighted terms.

## Proposed benchmark scorecard

All values below are **targets, not results**. Targets should be frozen before a held-out evaluation; tune on development data only.

| Area | Public baseline / comparison | Symthaea target |
|---|---|---|
| VideoFDB perception | Best listed 3.87/5 cascaded; 3.85/5 full-duplex | >=4.0/5, TOR >=93%, median <=150 ms; publish full-duplex and cascaded separately |
| VideoFDB generation | 4.44/5 cascaded; 4.20/5 full-duplex | >=4.35/5, TOR >=60%, median <=900 ms; compare same architecture/track |
| Turn transfer | TurnBench v1 0.983 recall at 0.065 FPR, but v1 is known flawed | On fixed hidden decision points: EOT and interruption recall >=0.98 at FPR <=0.05; median <=200 ms; p95 <=500 ms |
| Streaming avatar following | StreamEval leader following 36.90; sync 99.60 | Following >=45; sync/stability >=98; show quality-only and following separately |
| Spoken task execution | InteractionBench cascade: 38/50 in its evaluated setup | >=45/50 under the same licensed protocol, with latency and groundedness reported independently |
| Multi-party control | Duplex-MPE: 2,000 three-/four-speaker scenarios; scores response initiation, answer accuracy, silence preservation and stopping | >=95% silence preservation; >=98% stop-on-human-resolution; <=2% false activation on implicit/unaddressed requests; report answer accuracy separately |
| AV causal grounding | Existing public leaderboards mostly score perception/generation rubrics | Visual causal contribution >=+10 pp over audio-only on visual-required items; counterfactual sensitivity >=95%; irrelevant visual invariance >=98%; captioning collapse <=3% |
| Cross-form semantics | No directly equivalent public benchmark | 100% cue digest preservation; >=95% human agreement on meaning across all seven forms |
| Action interruption | Reported as separate timing/turn-taking metrics across public work | p95 audio stop <=150 ms; p95 visual stop <=100 ms; zero post-revocation submissions |
| ROI value | No direct comparable leaderboard | >=5 pp decision-quality lift over surprise-only at matched pixels; <=10% regret versus oracle; include energy and latency |
| Calibration | ECE alone is insufficient | ECE <=0.05 per preregistered task family; Brier, calibration slope/intercept, risk-coverage and cluster-bootstrap intervals reported |
| ROI lower-bound calibration | Caller-supplied LCBs are not sufficient evidence | 95% Wilson lower coverage bound >= nominal LCB confidence on >=100 independent held-out speaker/session clusters per task family |
| Output evidence | Usually not a first-class public avatar metric | 100% of Observed/Stopped claims independently sourced; zero fabricated completion receipts |

For public benchmark claims, use the official leaderboard's exact task definition and licensed evaluation route. Do not compare a cascaded model's generation score to a full-duplex model's perception score, and do not compare a score that includes customization bonus to an unbonused quality metric.

## Evaluation protocol

### Baselines

For each task family, include:
- audio-only or text-only baseline;
- best available full-duplex baseline;
- best available cascaded baseline;
- multi-party speech baseline evaluated on explicit versus implicit addressing (Duplex-MPE);
- continuous engagement/intervention baseline (DIVE);
- realistic-audio, grounded task-completion baseline ($\\tau$-Voice);
- overlap-specific full-duplex behavior baseline (Full-Duplex-Bench v1.5);
- sparse visual baseline (1 FPS);
- fixed-rate visual baseline (2 FPS and 4 FPS where affordable);
- surprise-only ROI;
- random ROI;
- calibrated EVI ROI;
- oracle ROI (diagnostic ceiling, not deployable).

### Ablations

Remove one element at a time:
- no policy/permit binding;
- no temporal action schedule;
- no compositor/device observer;
- no visual counterfactual training/evaluation;
- no calibration verifier;
- no event-triggered foveation;
- no EVI score (surprise-only);
- single renderer versus seven-form consistency.

Ablations may exist only in isolated evaluation harnesses. Do not weaken production gates.

### Statistics

- Split by speaker, session and source capture, not by adjacent frames.
- Report sample counts, denominators, confidence intervals, clustered bootstrap intervals and failure categories.
- Report p50/p95/p99 latency and wall-clock end-to-end paths.
- Measure compute, energy and source pixels at the same boundary.
- Preregister thresholds and hold out a challenge set; use mutation tests against the evaluator itself.
- Separate missing/unknown outcomes from negative outcomes.
- A skipped workflow, missing check context or queued test is not a pass.

## Implementation sequence

1. Complete exact-head build/test of existing contracts once a Rust runner is available; do not rebase onto moving main until changes are understood.
2. Wire SceneGraphV2 geometry into an Authorized ECS submission, keeping the v1 path intact.
3. Add runtime action scheduler and independent audio/device stop receipts.
4. Produce consented, licensed, speaker/session-disjoint AV and multi-party challenge sets ([#14160](https://github.com/Luminous-Dynamics/symthaea/issues/14160)) plus verified calibration reports.
5. Integrate calibration-verified EVI into ROI candidate selection behind a disabled-by-default experiment flag.
6. Run matched-budget ablations and issue a benchmark passport with commits, configs, model digests, data licenses, metrics and raw denominator counts.

## Current evidence boundary

Source contracts, docs and authored tests exist on the research branch. They are not yet compiled or executed in this environment; no runtime rendering, audio actuation, calibrated dataset, energy profile or benchmark score has been produced. The competitive targets are an evaluation plan, not performance claims.


## New benchmark protocol connections (2026-11-28 refresh)

- [VideoFDB v1.4 leaderboard](https://research.nvidia.com/labs/amri/projects/video-fdb/leaderboard.html) — latest November 25 table; retain separate perception/generation tracks and full-duplex/cascaded strata.
- [StreamEval leaderboard](https://streameval.ai/) and [paper](https://arxiv.org/abs/2610.28402) — preserve component scores for following, synchronization, stability, identity and quality; exclude the custom-avatar bonus when comparing content quality.
- [InteractionBench paper](https://arxiv.org/abs/2611.02430) and [site](https://avera-labs.github.io/InteractionBench/) — same-input timing, spoken-task, whisper/volume and long-context grounding protocol.
- [DIVE paper](https://arxiv.org/abs/2610.28405) — continuous-stream engagement, intervention windows and reaction timing.
- [$\\tau$-Voice ICML 2026](https://proceedings.mlr.press/v306/ray26a.html) — grounded task completion with full-duplex audio, policies and realistic acoustic conditions.
- [Full-Duplex-Bench v1.5](https://github.com/vtrinh-nvidia/Full-Duplex-Bench) — interruption, backchannel, other-directed speech, ambient speech and stop latency.
- [Duplex-MPE paper](https://arxiv.org/abs/2609.31948) — multi-party interaction, explicit/implicit addressee and silence/stopping conditions.
- [TurnBench paper](https://arxiv.org/abs/2608.25218) — dyadic turn-taking reference; interpret the currently documented v1 ranking limitation conservatively.

The InteractionBench and Duplex-MPE data/code usage rights must be checked through the benchmark provenance/license gate before inclusion in any commercial model training or derived-media workflow. External evaluation rights do not imply training or redistribution rights.
