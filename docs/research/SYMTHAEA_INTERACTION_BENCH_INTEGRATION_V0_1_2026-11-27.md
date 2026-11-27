# Symthaea Integrated Interaction Benchmark Protocol v0.1

**Date:** 2026-11-27  
**Status:** protocol proposal; no Symthaea score has been measured  
**Purpose:** evaluate interaction as a causal system without hiding trade-offs inside one composite score

## 1. Research rationale

Most real-time voice and avatar evaluations isolate turn-taking, instruction following, paralinguistic control, grounding or rendering. This misses coupled failure modes: a system may detect a pause but interrupt too early, execute a spoken request but fail to stop when the user changes it, preserve lip sync while ignoring an instruction, or produce a plausible answer unsupported by the conversation.

InteractionBench is a useful recent reference because it measures four dimensions on the same recorded user audio and the same systems: conversational timing, spoken-task accuracy, whisper/volume behavior, and groundedness over conversations longer than twenty turns. Its published snapshot reports 221 items per system across 13 scenarios, seven systems and 1,519 graded runs. The authors report that the turn-based cascade scored 38/50 on spoken tasks but had the second-lowest timing score; hosted systems passed 67–88% of groundedness probes versus 14–35% for open full-duplex models; only Gemini Live whispered on most requests (8/10). The study also identifies a measurement caveat: client-heard timing includes network delay, while one hosted system's server clock differed by about 1.5 seconds, changing its reported timing score materially.

These findings motivate a multi-lane Symthaea evaluation; they do not establish that any one system is universally better. InteractionBench has 13 scenarios and a limited system sample, uses some system-specific instructions, and includes an LLM judge for parts of groundedness. Replication should preserve these distinctions and limitations.

## 2. Design principle: one trace, separate scorecards

Use a shared interaction trace as the unit of analysis. Any `unverified_claims` object in a trace is self-reported metadata only; the verifier ignores it and it must never be treated as measured benchmark outcome. A separate scorer must calculate task and quality results from frozen labels and observations. A trace records:
- source audio/video chunk timestamps and arrival times;
- canonical semantic event, source clock and event sequence;
- turn candidate, confidence, decision and decision timestamp;
- authorized request and active grant/permit version;
- action/response submission and receipt phase;
- first audible output and first visible frame timestamps;
- pause, interruption, correction, cancellation, expiry and revocation;
- final observation and outcome.

Every metric remains in its own scorecard. Do not construct one headline score until a preregistered utility function is justified and independently reviewed. A high visual or fluency score must never compensate for unauthorized output, false claims, missed cancellation, or stale output after revocation.

## 3. Evaluation lanes

### A. Causal reciprocal turn control

Use TurnBench for public comparison where applicable, but add a hidden-decision-point internal protocol:
- exactly one decision per eligible decision opportunity;
- no late fallback predictions, backdating or multiple guesses;
- retain no-decision/abstention as a first-class result;
- evaluate end-of-turn, interruptions, user backchannels, mid-sentence pauses, overlapping speech and deliberate silence;
- stratify by six conversation styles and by noise / packet jitter.

Report end-of-turn recall, interruption recall, false-positive rate, false interruption events per minute, decision latency p50/p95/p99, calibration error, abstention coverage, and the cost-weighted confusion matrix.

Proposed target (not a result): end-of-turn recall >=0.97, interruption recall >=0.95, FPR <=0.04, false interruptions <=0.05/minute, median decision latency <=200 ms, p95 <=500 ms, zero duplicate/late decisions, and per-style recall floor >=0.92.

### B. Spoken task execution and floor control

Mirror InteractionBench task classes without copying its protected recordings: answerable spoken questions, countdown completion, corrections requested within a time window, short keyword/backchannel behavior, and stay-quiet-until-asked constraints. Add multi-step updates, interrupted requests, explicit corrections and task cancellation.

Measure exact and partial task success, correct floor behavior, completion latency from input availability, correction/interruption recovery, duplicate/stale action count, unauthorized action count and evidence receipt completeness.

Proposed target: >=95% valid authorized task success on the held-out target suite; zero unauthorized side effects; zero duplicate/stale commits; cancellation effective by the next qualified output update, with p95 stop latency <=100 ms on a named reference device. Report client-heard and server timing separately.

### C. Paralinguistic control

Evaluate explicit requests to whisper, lower or raise speaking volume, slow down, or return to a prior voice mode. Score generated audio, not just text claims.

Measure request-conditioned mode accuracy, objective loudness/voicing measures, response intelligibility, transition time, and consistency across audio devices and noise levels. Distinguish a requested whisper from an inference about the user's emotional state. Do not infer sensitive traits from volume, pitch or speaking style.

Proposed target: >=95% correct execution of explicit voice-mode requests, with acoustic acceptance ranges frozen before evaluation. Establish thresholds with a listening study and calibrated capture chain; do not copy another benchmark's whisper threshold without validating microphones and codecs.

### D. Long-context groundedness

Use at least 22-turn conversations with scripted facts, updates, constraints, speaker attribution, intentionally absent information and false-premise probes. Include state changes that require the agent to retract an earlier answer.

Measure exact fact/update recall, constraint preservation, speaker attribution, unsupported assertion rate, correct abstention, correction after contradiction and calibration. Separate extractive memory from inference; label which source supports each answer.

Proposed target: >=95% accuracy on answerable held-out probes, <=1% unsupported factual assertions, and >=95% correct abstention on deliberately unanswerable probes. Publish per-probe traces and judge/human agreement; no LLM-judge-only headline.

### E. Cross-form semantic invariance

Replay the same canonical cue into all seven registered forms. Compare semantic-key and canonical-cue digest invariance, event order and causal turn state, privacy/accessibility equivalents, epistemic status/evidence representation, and human-rated meaning/task comprehension.

Proposed target: 100% same-cue digest preservation; >=95% blinded human semantic agreement; zero false displays of verification, certainty, consent or completed action. Human ratings should be paired by cue and form and reported with confidence intervals.

### F. Presentation authority and observed output

This is Symthaea's distinctive additional lane. Inject grant expiry, event expiry, revocation, policy replacement, privacy downgrade, reduced intensity, reduced-motion, pause, cancel, clock loss and renderer failure during steady-state output and cross-form transitions.

Measure accepted output after authorization loss (must be zero), rendered frames after effective expiry (must be zero), time from revocation/cancel to last visible frame, old/new scene overlap while each permit is live, output after source or target permit failure, receipt consistency from Authorized to Submitted to independently Observed and Stopped, and privacy/raw-evidence leakage.

Hard gate: zero unauthorized commands and zero post-expiry frames in all deterministic fault-injection cases. Do not call an ECS enqueue acknowledgement a visible-frame observation. Only a qualified compositor/device observer may write Observed or Stopped receipts.

## 4. Experimental controls

Freeze each dataset split, scenario manifest, event schema, decision policy, model/provider version, prompt, voice/asset package, renderer build and scorer. Bind each result to the exact source commit and dirty-tree state; schema/asset/grant-policy/dataset digests; OS, CPU, GPU, driver, resolution, audio interface, power mode and thermal state; client and server timestamps with clock-domain metadata; sample-level outputs, errors, retries, exclusions and invalid traces.

Run local and hosted baselines under clearly reported network conditions. Randomize system order where feasible. Use paired evaluation on identical input recordings, confidence intervals clustered by conversation, and correction for multiple comparisons. Keep a clean development set and held-out test set. Do not tune thresholds on the final test set.

## 5. Relationship to external benchmarks

| Benchmark | Use in Symthaea | Important limitation |
|---|---|---|
| InteractionBench | shared-trace timing, spoken tasks, whisper/volume, 20+ turn groundedness | 13 scenarios; mixed client/server timing and system-specific instructions; retain per-lane results |
| TurnBench | end-of-turn and interruption decisions | v1 permits late/multiple predictions and recall-heavy scoring; supplement with causal hidden decision points |
| VideoFDB | dyadic audiovisual perception/generation, nonverbal dynamics, turn-on-response alignment | compare full-duplex/cascaded and AV2AV/AV2A separately; perception is not generation |
| StreamEval | streaming avatar dialogue/performance, state following and A/V coordination | custom-avatar bonus affects Overall; report content-only metrics and Following separately |
| Symthaea fault-injection suite | executable consent, cross-form invariance, revocation and output receipts | new internal benchmark; require independent review and observer qualification |

External datasets, evaluation code and recordings must be used under published licenses and access conditions. This protocol does not redistribute benchmark data.

## 6. Why this may create a differentiated result

The target is not to beat every model on every dimension. It is to demonstrate a reproducible Pareto advantage in causal interaction; reliable task execution and groundedness; semantic identity across non-human and human-like forms; explicit and revocable presentation authority; privacy minimization and accessibility; local responsiveness and energy use; and independently observed output receipts.

A benchmark result is publishable only when the exact evaluated head, environment, scorer and raw per-sample outputs are available. Authored tests, synthetic fixtures and architecture claims are not leaderboard evidence.

## 7. Current implementation status (2026-11-27)

- Renderer-neutral event, projection, policy, receipt, scene, turn-control, continuity and interaction-trace verification modules are present.
- The isolated Bevy adapter now integrates the continuity planner with separate source/target scene groups, explicit alpha blending and live permit rechecks for both groups during transitions.
- Superseding an unfinished transition uses a hard cut; a denied target clears both groups rather than restoring prior output.
- Fifteen adapter tests and 128 Soma core tests are authored (143 total), none compiled or executed.
- The causal verifier has a Rust CLI that accepts a single trace or a positive/negative fixture bundle. Structural JSON Schema validation remains separate.
- No Rust build/test, Bevy display/runtime, compositor observation, InteractionBench, TurnBench, StreamEval or VideoFDB evaluation has been executed.
- No claim of SOTA superiority, consciousness, sentience or subjective experience is made.

## 8. References

- InteractionBench paper: https://arxiv.org/abs/2611.02430
- InteractionBench site: https://avera-labs.github.io/InteractionBench/
- InteractionBench evaluation code: https://github.com/avera-labs/InteractionBench
- TurnBench: https://turnbench.sesame.com/
- VideoFDB: https://research.nvidia.com/labs/amri/projects/video-fdb/
- VideoFDB leaderboard: https://research.nvidia.com/labs/amri/projects/video-fdb/leaderboard.html
- StreamEval: https://streameval.ai/


## 9. Cross-event causal verifier invariants

The JSON Schema checks event shape. A separate trace verifier must check relationships that JSON Schema cannot express across array entries. At minimum it must reject:
1. duplicate or regressing event sequence numbers within a session/clock stream;
2. regressing monotonic event timestamps, or chunk arrival earlier than its recorded source timestamp;
3. turn decisions emitted after their deadline, backdated, or duplicated for one candidate;
4. presentation without a preceding matching authorization for the same session, semantic event, digest, grant, privacy class, channel, destination and form; compare the source-clock sample against the authorized effective expiry;
5. actions without a preceding authorization sequence and an explicit action-kind scope in the actuator permit;
6. visible frames at or after grant/event effective expiry, following revocation, or following a stop/cancel observation;
7. an event ID reused with a different canonical cue digest;
8. a grounding answer whose cited support sequence is absent or future-dated; semantic relevance of the support requires a separate scorer or human review;
9. Submitted treated as Observed without a non-empty observer identifier and a matching active output/frame sequence; cryptographic observer qualification, independence and frame-digest attestation remain external;
10. an output-stop receipt that precedes the final visible frame or is absent after a required stop.

The Rust verifier is implemented in `crates/domains/symthaea-soma/src/embodiment_interaction_trace.rs`. It enforces event sequence and per-clock monotonicity, source-clock expiry, one decision per candidate, immutable cue digest and privacy class, monotonic policy versions, action-kind scopes, matching presentation authorizations, and output stop closure. Nineteen tests are authored, but none is compiled or executed.

The CLI supports either a single trace or the full positive/negative fixture bundle:

```sh
cargo run -p symthaea-soma --bin verify_interaction_trace -- docs/research/SYMTHAEA_INTERACTION_TRACE_FIXTURES_V1_2026-11-27.json
```

This command is authored but has not been run. The verifier is not a structural JSON Schema validator and does not prove observer independence, hardware attestation, semantic correctness of grounding evidence, or task success. Trace-supplied `unverified_claims` are not trusted or scored by the verifier. It rejects causally invalid samples rather than silently dropping them.
