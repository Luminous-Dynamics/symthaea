# Symthaea multi-party interaction control benchmark v1 — 2026-11-28

## Motivation

Dyadic turn-taking is necessary but not sufficient for a useful conversational agent in a group. The system must distinguish direct address from speech directed to another person, handle implicit requests, preserve silence, recognize when an addressee is ambiguous, and stop speaking when another human resolves the request.

This lane is inspired by Duplex-MPE (2,000 three-/four-speaker scenarios) and complements InteractionBench, VideoFDB and TurnBench. It is not a reproduction of either external benchmark.

## Implemented contract

Source: `crates/domains/symthaea-soma/src/embodiment_multiparty_control.rs`.

Scenario labels are typed as:
- DirectToAgent
- DirectToOther
- ImplicitToOther
- GroupBroadcast
- Ambiguous

Decision classes are Respond, ContinueSpeaking, StaySilent, StopSpeaking and AskClarifying.

The verifier requires three or four distinct opaque participant IDs, a registered agent and current speaker, valid addressing/addressee consistency, an event digest and a nonzero response deadline. No names, raw audio, transcript, speaker biometric or face identity are stored.

The scorer requires exactly one prediction per scenario and independently graded answer correctness for every Respond/ContinueSpeaking action. It keeps separate:
- action accuracy;
- addressed-response rate;
- silence-preservation rate;
- false-activation rate on direct/implicit requests to another person;
- stop-on-human-resolution rate;
- answer accuracy;
- deadline-met rate;
- median and p95 decision latency.

A missing prediction is an error rather than a silent pass. Duplicate scenarios and predictions are rejected. This is an evaluator/manifest contract only: it does not detect the speaker, addressee, intent or resolution state from audio/video and has not loaded an external dataset.

## Proposed challenge set

Build consented, human-annotated, three-/four-person conversations with speaker/session-separated development, calibration and hidden test partitions. Include:
1. explicit requests to the agent;
2. requests addressed to a different person;
3. implicit requests where another participant is the intended addressee;
4. group broadcasts and shared-floor questions;
5. ambiguous/noisy addressee cues;
6. human resolution before the agent replies;
7. human resolution while the agent is already speaking;
8. backchannel-heavy interruptions and overlapping speech;
9. cross-lingual / code-switching and accessibility-mediated addressing;
10. speaker changes, room acoustics and network-delay perturbations.

Every test item should annotate the intended addressee and human-resolution state separately from the transcript. Adjudication should use at least two independent annotators plus tie-breaking; the model under test must not grade itself.

## Proposed qualification targets (not results)

| Metric | Initial target |
|---|---:|
| Action accuracy | >=95% |
| Response rate on direct agent/group requests | >=95% |
| Silence preservation for direct/implicit requests to others | >=98% |
| False activation on unaddressed requests | <=2% |
| Stop compliance after human resolution | >=98% |
| Answer accuracy when a response is appropriate | >=90% |
| p95 decision latency | <=300 ms |
| Incorrect response after confirmed human resolution | 0 in safety-critical challenge cases |

Report precision/recall by addressing type and speaker/session, not only aggregate accuracy. Include cluster-bootstrap confidence intervals and confusion matrices. A model should not improve silence by abstaining from all requests; report addressed response rate and answer accuracy alongside silence preservation.

## Hard research questions

- Can addressee inference use conversational structure, prosody, gaze and gesture without overfitting speaker identity?
- How should the system represent uncertain addressee beliefs and choose between silence and clarification?
- Can the policy optimize expected interruption cost asymmetrically, making false activation more expensive than a brief delayed response?
- Can one canonical floor-control state drive speech, viseme, gesture and spatial embodiments without one renderer inventing intent?
- Does performance survive a change in group size, room acoustics, culture, language and network topology?

## Evidence boundary

Twelve source-level tests are authored. They have not been compiled or executed. The evaluator is not an audio perception model, and no consented multi-party dataset, external benchmark score or runtime latency result exists. Dataset licensing and participant consent must be checked independently before use.
