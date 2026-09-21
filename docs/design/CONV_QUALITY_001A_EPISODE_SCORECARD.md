# CONV-QUALITY-001A — Multi-turn conversation-quality episode scorecard

## Purpose

Provide a source-only integration contract for #5274 that binds independent full-duplex, correction, and persona-continuity evidence without force-merging their unqualified candidate branches or inventing a universal conversation-quality score.

## Core theorem

```text
evidence completeness != conversation quality

complete evidence + Stop violation != good conversation
fast correction + later recurrence != corrected forever
stable persona + ignored correction != good conversation
missing evidence != success
```

## Manifest identity

`ConversationQualityEpisodeManifestV1` binds:

- episode ID/version;
- fixture reference;
- evaluator-set reference;
- exact full-duplex component source head + receipt schema + receipt commitment + evidence state;
- exact correction component source head + receipt schema + receipt commitment + evidence state;
- exact persona-continuity component source head + receipt schema + receipt commitment + evidence state.

All three component kinds are required exactly once. Component ordering is canonicalized. Source heads are exact lowercase 40-hex Git identities. Component receipt commitments are domain-labelled BLAKE3 commitments.

Manifest validation rechecks canonical IDs/keys/refs and every component reference rather than assuming construction occurred through the smart constructors.

## Independent scorecards

### Voice

Carries independently:

- interruption semantics and latency;
- Stop semantics and latency;
- SlowDown semantics and acknowledgement latency;
- response latency;
- backchannel semantics;
- incomplete-trace count;
- runtime-rejection count.

Latency budget evidence distinguishes `NotEvaluated`, `NotEstablished`, `WithinBudget`, and `Exceeded`. `WithinBudget`/`Exceeded` require an actual latency measurement. Receipt construction revalidates nested latency values so deserialized/publicly-constructed invalid states fail closed.

### Correction

Carries:

- first compliant turn delta;
- optional first-compliant wall-clock latency;
- recurrence before adaptation;
- recurrence after adaptation;
- unrelated-context spillover;
- ambiguous applicable observations;
- applicable observation count.

Fast first adaptation never erases later recurrence.

### Persona

Carries:

- critical/noncritical violations;
- global/context-band violations;
- evaluator-disagreement groups;
- unobserved required facets;
- facet/profile/context coverage.

Persona continuity cannot compensate for failed Stop/correction behavior.

## Evidence coverage

The episode derives only an evidence-coverage state:

```text
NotEstablished
Partial
Complete
Uncertain
Invalid
```

This is not a quality verdict.

- any invalid component evidence -> `Invalid`;
- otherwise any uncertain component evidence -> `Uncertain`;
- all components not established -> `NotEstablished`;
- all three established -> `Complete`;
- otherwise -> `Partial`.

Thus a fully measured episode containing a Stop violation remains evidence-complete.

## Receipt integrity

The final domain-separated BLAKE3 commitment binds:

- receipt schema;
- episode and manifest identities;
- exact component receipt commitments;
- evidence coverage;
- every field in all three scorecards.

Validation performs two independent checks:

1. reconstruct derived fields from the frozen manifest + scorecards and require exact equality;
2. recompute the commitment over the stored receipt object itself.

This prevents an old valid commitment from being replayed after mutating derived fields such as evidence coverage, stored component receipt references, schema, or latency evidence.

## Tests

Initial tests cover:

- complete evidence with Stop violation remains evidence-complete;
- one-turn correction with later recurrence preserves both facts;
- partial/all-missing/uncertain/invalid evidence states;
- component ordering invariance and subject substitution;
- duplicate/missing component kinds;
- missing latency cannot claim `WithinBudget`;
- manually/deserialized invalid nested latency is rejected;
- scorecard evidence state must match the frozen component reference;
- manifest and receipt commitment tampering;
- derived-field tampering with an old commitment;
- schema tampering;
- correction/persona internal coverage consistency.

## Nonclaims

This tranche does not execute or qualify component candidates, classify arbitrary natural-language corrections, measure subjective naturalness, establish human-level conversation/personhood, or create user/physical authority. Component qualification states remain independently authoritative for claims about those source subjects.
