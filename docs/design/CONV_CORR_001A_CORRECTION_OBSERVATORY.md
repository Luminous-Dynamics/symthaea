# CONV-CORR-001A — explicit correction observatory

Status: source candidate only. No format/compile/test/Clippy PASS is established until exact-head qualification executes.

## Purpose

Measure how quickly and faithfully Symthaea adapts after an explicit correction without conflating initial prediction quality, correction quality, durability, or cross-context generalization.

The observatory is content-blind. It consumes pre-labeled semantic correction/observation events and emits deterministic evidence receipts. It does not decide whether arbitrary natural-language content is semantically correct and does not mutate memory or preferences itself.

## Core distinctions

```text
initial prediction quality != correction quality
first compliant turn != no later recurrence
correction in context A != rewrite context B
temporary override != durable preference
no observations != successful correction
```

## Correction identity

An `ExplicitCorrectionV1` binds:

- correction ID;
- subject reference;
- semantic key;
- exact context ID;
- durability (`TurnOnly`, `Session`, `DurableExplicit`);
- payload (`Replace`, `Retract`, `NarrowScope`);
- logical correction turn;
- optional timestamp;
- domain-separated BLAKE3 commitment.

Opaque references are used for corrected values/scope. Raw private conversation text is not required.

## Observation semantics

Post-correction observations are one of:

- `ApplicableCompliant`;
- `ApplicableRecurrence`;
- `ApplicableAmbiguous`;
- `OutOfScopeUnchanged`;
- `OutOfScopeChangedConsistentWithCorrection`.

Applicable observations must match the correction's exact semantic key and context. Out-of-scope observations must not masquerade as exact-scope observations.

Observation IDs are unique. Input order is canonicalized by logical turn and observation ID before the observation-set commitment is computed.

## Receipt

`CorrectionObservationReceiptV1` reports separately:

- adaptation status;
- recurrence status;
- spillover status;
- first compliant logical-turn delta;
- optional first compliant wall-clock latency;
- recurrence before adaptation;
- recurrence after adaptation;
- ambiguous applicable observations;
- applicable observation count;
- out-of-scope observation count;
- spillover count;
- timestamp coverage;
- deterministic observation-set and receipt commitments.

A one-turn adaptation followed by later recurrence therefore remains visible as both a fast adaptation and a regression.

## Evidence discipline

```text
zero post-correction observations
→ adaptation NotEstablished

applicable recurrence only
→ NoCompliantObservation

first compliant observation
→ Adapted

later recurrence
→ RecurrenceObserved remains explicit

out-of-scope behavior changed consistently with correction
→ SpilloverObserved
```

The observatory emits no single universal conversation score.

## Tests

The integration suite covers:

- immediate adaptation;
- delayed adaptation;
- recurrence before and after adaptation;
- no-evidence behavior;
- no-compliant-observation behavior;
- out-of-scope spillover and non-spillover;
- temporary/session/durable identity separation;
- exact context/key enforcement;
- duplicate observation rejection;
- canonical ordering commitments;
- logical-turn/timestamp freshness;
- correction commitment tampering.

## Nonclaims

This tranche does not establish natural-language correction classification, automatic memory rewriting, preference correctness, psychological diagnosis, universal conversational quality, or any physical/user authority.
