# Symthaea Presentation Outcome Receipts v0.1

**Date:** 2026-11-27  
**Status:** Rust state machine and fixtures added; runtime integration and qualification pending

## Purpose

A presentation permit records an authorization decision. It does not prove that a renderer accepted a command, that pixels/audio/haptics reached an output device, or that output subsequently stopped.

The receipt layer gives those outcomes separate states and timestamps. It is a renderer-neutral protocol intended to make failed, delayed, cancelled and unobserved outputs visible rather than silently counting them as success.

## Lifecycle

```text
Permitted
   | \
   |  +--> NotSubmitted (terminal)
   v
Authorized
   | \
   |  +--> Failed / NotSubmitted
   v
Submitted
   | \
   |  +--> Failed (terminal)
   +-----> OutcomeUnknown (unresolved)
   |            | \
   v            |  +--> Failed (terminal)
Observed <------+
   |
   +-----------> Failed (terminal)
   |
   v
Stopped (terminal)
```

Allowed transitions:

- initial → Permitted
- Permitted → Authorized or NotSubmitted
- Authorized → Submitted, NotSubmitted or Failed
- Submitted → Observed, OutcomeUnknown or Failed
- OutcomeUnknown → Observed or Failed
- Observed → Stopped or Failed

Stopped, Failed and NotSubmitted are terminal. A second terminal transition is rejected. OutcomeUnknown is deliberately unresolved, not a success state; a delayed observer may later resolve it to Observed or Failed. If no observer ever reports, the ledger retains the unresolved state.

## Rust implementation

`crates/domains/symthaea-soma/src/embodiment_receipt.rs` exports:

- `PresentationReceiptLedger`: bounded in-memory append-only transition ledger, limited to 65,536 attempts per instance.
- `record_permitted`: stores the exact policy decision timestamp and opaque permit binding.
- `record_authorized`: rechecks the live `PresentationPolicy`, appends Authorized, and returns the exact refreshed opaque permit.
- `record_submitted`: appends Submitted only after the adapter acknowledges acceptance of that exact authorized permit. This is still not proof that output was observed.
- `record_observed`: requires a named observer reporting Started.
- `record_stopped`: requires a named observer reporting Stopped; a stop command alone is insufficient.
- `record_outcome_unknown`, `record_not_submitted`, and `record_failed`: preserve unresolved and failure outcomes.
- `verify_chain`: checks every BLAKE3 digest and previous-digest link in the local ledger.

Observation sources distinguish adapter callbacks, compositor reports, audio-device loopback, haptic-driver acknowledgement, and external observers. The source is disclosed, not treated as proof of independence. The observer type must match the authorized output channel.

Every receipt preserves session, clock domain, event and grant identity, the complete canonical-cue digest, event sequence, semantic event kind, event expiry, grant expiry, effective permit expiry, source/build provenance, epistemic status, privacy class, form, output channel/destination, intensity, motion mode, adapter build and observer metadata. The initial decision timestamp and every transition timestamp retain their clock-domain identifiers; an Authorized record uses the policy clock, while Submitted/Observed/Stopped use the adapter/output-observer clocks. No raw audio/video is included.

## Integrity and trust limits

The chain uses an unkeyed, domain-separated BLAKE3 digest over the serialized receipt body, including the prior digest. This detects accidental or partial modification of the current in-memory chain.

It does **not**:
- authenticate the adapter, observer, user, or consent grant;
- prevent an attacker from rewriting the entire chain and replacing its final head;
- provide trusted time, hardware attestation, secure persistence, or anti-fork guarantees;
- prove that a reported observer is physically independent;
- prove user perception, comprehension, or consent;
- authorize sensor capture, remote export, or physical actuation.

Persisting the chain safely needs an independently trusted checkpoint/anchor, key custody, authenticated observer identity, and a separate storage threat model. Do not claim a cryptographic audit trail from this in-memory digest chain alone.

## Clock treatment

Source decision time and renderer/output observation time may use different monotonic clocks. The ledger preserves both IDs and never compares raw nanosecond values across different domains. Regression is checked only within the same domain. A cross-device latency measurement needs a separately qualified clock synchronization or mapping protocol; receipt timestamps alone do not invent one.

## Machine-readable contracts

- `SYMTHAEA_PRESENTATION_RECEIPT_SCHEMA_V1_2026-11-27.json` defines a serialized transition record.
- `SYMTHAEA_PRESENTATION_RECEIPT_FIXTURES_V1_2026-11-27.json` contains five accepted lifecycle sequences and five negative sequence/timing controls.

The fixture file is a research artifact; it is not yet consumed automatically by the Rust tests. JSON files have been parsed and structurally inspected, but an independent JSON Schema 2020-12 validator has not been run.

## Tests and qualification

Seventeen Rust tests are authored for:
- permit versus submission distinction;
- required final policy recheck;
- rejection of submission acknowledgements carrying a stale pre-authorization permit;
- separation between Authorized and adapter-acknowledged Submitted;
- observed output and observed stop;
- delayed observation resolving an unknown outcome;
- failure resolution;
- rejection of a stop command without stopped observation;
- terminal-state immutability;
- not-submitted terminal state;
- duplicate attempt IDs;
- clock regression within one domain and no ordering across domains;
- denial leaving the state unchanged;
- digest-chain verification;
- separate source and renderer clock domains;
- canonical cue digest preservation in the receipt;
- observer/source compatibility with the output channel.

These tests have **not** been compiled or executed. The branch's GitHub workflows have returned skipped, not passed. No actual renderer or hardware output observer is integrated yet.

## Next integration step

The first Living Constellation renderer should hold the policy synchronization lock across recording Authorized (which performs the final permit recheck), renderer submission, and recording the adapter's Submitted acknowledgement. A compositor/frame callback then reports Observed; a separate stop callback or observer reports Stopped. A missing callback is OutcomeUnknown, never success.
