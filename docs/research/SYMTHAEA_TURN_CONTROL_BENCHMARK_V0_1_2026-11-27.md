# Symthaea Turn Control and Reciprocal Interaction Benchmark v0.1

**Date:** 2026-11-27  
**Status:** benchmark protocol and targets; no Symthaea result claimed

## 1. Research objective

Evaluate turn-taking as a real-time control decision, not a post-hoc transcript label or an animation choice.

Symthaea's turn-state vocabulary is:

- WAIT — keep listening; do not take the floor.
- RESPOND — begin a response when the floor is available.
- CONTINUE — retain the floor while producing a coherent response.
- YIELD — stop or relinquish the floor when the partner takes it or asks to speak.

A decision must be causal, timestamped, bounded to one active transition, and revocable by user control. The renderer may visualize a decision but must not infer it from visual style or animation state.

## 2. External baseline: TurnBench

The TurnBench page currently describes:
- a 30-hour held-out dual-channel dyadic conversation corpus;
- 154 dialogues, 106 actors, six conversation styles;
- three independent annotators per dialogue and Fleiss' kappa of 0.78;
- a 104-hour training corpus and 7.3-hour development split;
- end-of-turn and interruption events with event times.

Its published v1 leaderboard currently includes Kuroko at 0.983 recall / 0.065 false-positive rate / 619 ms median latency, and Neosapience Nunchi v2 at 0.959 / 0.064 / 254 ms.

**Important benchmark caveat:** TurnBench itself documents a v1 scoring weakness. Multiple predictions can be emitted for one pause, late fallbacks up to three seconds can still count, and the ranking is based on recall. Simple voice-activity rules can consequently rank highly without making a single timely decision. The maintainers describe a planned v2 with per-frame probabilities and fixed hidden decision points.

Therefore:
- use v1 as a diagnostic track only and always report false-positive and latency columns;
- do not claim state-of-the-art from recall-only v1 rank;
- prepare an evaluation adapter for v2 when released;
- maintain a fixed-decision, one-action-per-event internal evaluation in the interim.

Source: https://turnbench.sesame.com/

## 3. Symthaea evaluation protocol

### 3.1 Frozen input

Use the same permitted TurnBench split and annotation version for all compared systems. Freeze audio, timestamps, channel layout, resampling, VAD settings, model versions, thresholds and post-processing configuration.

Do not leak test labels, future audio, future chunks, future video frames, or annotator corrections into the decision path. Every prediction must include the latest input timestamp consumed, the output timestamp, the decision, confidence/calibration value (if available), and a stable decision ID.

### 3.2 Decision cardinality and causality

For each hidden decision point:
- emit at most one action for that event;
- no backdating;
- no prediction based on samples after the decision timestamp;
- no repeated late fallback predictions for the same pause;
- missing input, unavailable clock, or uncertain authority must produce WAIT/abstain, not an invented turn transfer.

The primary scorer must evaluate fixed hidden decision points or use the official v2 protocol. A second diagnostic scorer may report event detection, but cannot replace the hidden-point score.

### 3.3 Conversation-style stratification

Report each of the six styles separately:
1. Casual;
2. Task-oriented;
3. Instructional;
4. Collaborative;
5. Argumentative;
6. Narrative.

Also stratify by overlap, backchannel density, interruption type, speaking rate, channel bleed, echo, noise, language/accent when the corpus terms permit it, and network/chunk loss.

Do not average away a failure that occurs mainly in backchannel-dense casual dialogue or emotionally charged disagreement.

### 3.4 Cost-sensitive control utility

A false interruption is not equivalent to a missed decorative backchannel. Report a pre-registered cost matrix and the full confusion matrix.

Proposed internal cost starting point (to be validated with users):
- false RESPOND while the partner holds the floor: cost 5;
- false YIELD while Symthaea should continue: cost 4;
- missed floor transfer: cost 3;
- late but correct response: cost 2, scaled by delay;
- missed optional backchannel: cost 1;
- WAIT/abstain under genuinely ambiguous input: cost 0 for safety, but count coverage separately.

Report utility at several cost ratios (1:1, 2:1, 5:1) so conclusions do not depend on one hand-picked weight.

## 4. Metrics

Primary:
- event precision and recall for end-of-turn and interruption;
- false interruption rate per conversation minute;
- false-positive rate at fixed hidden decision points;
- median, p90 and p95 decision latency;
- duplicate/late action rate;
- floor-transfer timing error;
- calibration (Brier score, log loss and reliability curve when probabilities are available);
- abstention/coverage versus error;
- cost-sensitive interaction utility.

Secondary:
- WAIT/RESPOND/CONTINUE/YIELD transition validity;
- time to stop speech after YIELD;
- rate of talking over an active user;
- successful recovery after interruption;
- backchannel appropriateness;
- CPU/GPU/RAM, model and output frame budget, and energy per interaction minute.

Latency must be measured at the client output boundary, preserving both source and client clock domains. Never subtract raw monotonic timestamps from unrelated domains.

## 5. Proposed targets (not results)

The following are ambitious internal targets for a frozen, architecture-matched evaluation, not claims about TurnBench's current official leaderboard:

| Metric | Target |
|---|---:|
| End-of-turn recall | >= 0.97 |
| Interruption recall | >= 0.95 |
| False-positive rate | <= 0.04 |
| False interruptions | <= 0.05 per conversation minute |
| Median decision latency | <= 200 ms |
| P95 decision latency | <= 500 ms |
| Duplicate/late decisions | 0 |
| Per-style recall floor | >= 0.92 |
| Calibration | report Brier/ECE; improve over matched VAD and semantic baselines |
| Unauthorized or stale action | 0 |

The target deliberately aims to improve on the recall/false-positive/latency trade-off rather than only beating the highest recall number. Human floor-transfer timing is a useful reference, but TurnBench's event-latency metric is not identical to the human overlap statistic; do not claim direct equivalence.

## 6. Baselines and ablations

Compare:
- energy/VAD threshold baseline;
- VAD plus fixed hangover;
- causal acoustic model;
- semantic turn detector;
- Symthaea's event/state controller;
- oracle upper bound using gold labels (evaluation only; never inference).

Ablate:
- acoustic evidence only;
- language/semantic context only;
- joint model;
- fixed versus adaptive decision threshold;
- with/without uncertainty abstention;
- with/without hysteresis;
- with/without conversation-style conditioning.

Keep compute and context budgets matched. Report ablations separately instead of only publishing the best system.

## 7. Connection to embodiment and receipts

The turn decision and the visual cue remain different layers:

1. The turn controller emits a typed WAIT/RESPOND/CONTINUE/YIELD event.
2. Event validation checks session, sequence, source, timestamp and expiry.
3. The canonical cue is bound into the presentation permit digest.
4. The policy authorizes a chosen visual/audio output channel.
5. The receipt ledger separates Authorized, Submitted, Observed and Stopped.
6. The renderer draws only the authorized cue.

A visual pulse cannot count as a RESPOND decision. A YIELD animation cannot count as proof that speech stopped. A missing clock or missing input cannot be silently converted into a confident action.

## 8. Next experiments

1. Implement a trace adapter for official TurnBench v1 submissions and preserve the official output timestamps.
2. Add a fixed-hidden-point local evaluator with one-decision cardinality and cost-sensitive scoring.
3. Build conversation-style-specific confusion and latency reports.
4. Test a causal WAIT/RESPOND/CONTINUE/YIELD controller using frozen clips before live microphone access.
5. Only after offline qualification, evaluate streaming with explicitly granted microphone permission and a local-only default.
6. Re-run on TurnBench v2 when its hidden decision protocol is published.

No TurnBench data has been downloaded in this increment. No official or local Symthaea benchmark score is claimed.
