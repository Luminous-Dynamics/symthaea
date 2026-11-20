# VOICE-FDX-001A — Full-Duplex Turn Controller

Status: source candidate; not qualified.

## Purpose

Provide a modality-neutral runtime controller for spoken turn-taking, interruption, backchannels, slow-down, and explicit stop.

This controller is general communication infrastructure. Adult/intimate dialogue may consume it later, but the controller itself contains no sexual/intimate semantics and creates no physical authority.

## Orthogonal state

The controller tracks user-floor and assistant-output state separately:

```text
UserFloor:
  Silent
  Speaking(utterance_id, start)

AssistantOutput:
  Silent
  Speaking(turn_id, start)
  YieldRequested(turn_id, request_time, reason)
```

This allows real overlap and interruption to be represented rather than forcing a single alternating-turn enum.

## Interruption

If user speech begins while assistant output is active:

```text
Assistant Speaking
+ User SpeechStart
→ Assistant YieldRequested(UserInterruption)
```

The renderer/output layer must separately acknowledge that output actually stopped. The resulting receipt records request time, stop-ack time, and latency.

Interrupted turns do not auto-resume. A future response is a fresh higher-layer decision and a fresh assistant turn identity.

## Backchannels

A classified backchannel records whether assistant output was active but does not steal the user floor or request assistant yield.

Classification itself is outside this module.

## Explicit stop

`user_stop` latches a stop state.

If output is active, the controller requests yield with `ExplicitStop`; if output is already silent, the stop receipt has zero output-stop latency.

Assistant speech cannot begin while stop is latched. Returning to speech requires an explicit fresh session epoch through `reset_session`.

Repeated identical stop signals are idempotent.

## Slow-down / de-escalation

`user_slow_down` latches a de-escalation requirement until a higher layer explicitly acknowledges a compliant plan reference.

Any assistant-speech permit exposes whether de-escalation is currently required.

Repeated identical slow-down signals are idempotent. Stop dominates slow-down.

## Timing semantics

Accepted events use a monotonic nanosecond domain. Rejected events are transactional: they do not advance the accepted-event clock or partially mutate valid state.

The controller exposes evidence sufficient to measure:

- user interruption -> output-stop acknowledgement latency;
- explicit stop -> output-stop acknowledgement latency;
- user-floor release -> assistant speech-start latency;
- slow-down -> compliant-plan acknowledgement latency;
- whether a backchannel occurred while assistant output was active.

## Identity / replay

User event/utterance identities and assistant turn identities are replay-resistant within one session epoch. A fresh higher session epoch resets those namespaces explicitly.

## Core invariants

```text
backchannel != interruption
interruption request != output actually stopped
stop request != automatic future resume
slow-down request != automatically satisfied
voice output != physical authority
rejected input != accepted state transition
```

## Tests

The integration suite covers:

- interruption-triggered yielding and measured acknowledgement latency;
- no auto-resume after interruption;
- backchannel non-interruption;
- latching explicit stop and stop latency;
- zero-latency stop while already silent;
- slow-down latching and explicit de-escalation acknowledgement;
- user-floor exclusion of assistant speech;
- rejected event leaving clock/state unchanged;
- response latency after user-floor release;
- natural end vs yield acknowledgement separation;
- explicit fresh session reset;
- non-monotonic timestamp rejection.

## Nonclaims

VOICE-FDX-001A does not implement ASR, end-of-turn classification, TTS, acoustic echo cancellation, voice likeness authorization, semantic response generation, physical action, or claim human-level conversational naturalness.
