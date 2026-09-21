# CONV-QUALITY-001A — Multi-turn conversation-quality episode scorecard

## Purpose

Provide a source-only integration contract for #5274 that can bind independent full-duplex, correction, and persona-continuity evidence without merging their candidate branches or inventing a universal conversation-quality score.

This tranche is intentionally component-agnostic: it records exact subject and receipt identities plus independent scorecard slices. Later adapters may lower qualified component receipts into this contract.

## Core theorem

```text
evidence completeness != conversation quality

complete evidence + stop violation != good conversation
fast correction + later recurrence != corrected forever
stable persona + ignored correction != good conversation
missing evidence != success
```

## Manifest identity

`ConversationQualityEpisodeManifestV1` binds:

- episode ID/version;
- exact fixture reference;
- evaluator-set reference;
- exact full-duplex component source head + receipt schema + receipt commitment + evidence state;
- exact correction component source head + receipt schema + receipt commitment + evidence state;
- exact persona-continuity component source head + receipt schema + receipt commitment + evidence state.

All three component kinds are required exactly once. Component ordering is canonicalized through a `BTreeMap`; duplicate or missing component kinds fail closed.

The source head must be an exact lowercase 40-hex Git commit identity. Component receipts must use a domain-labelled BLAKE3 commitment.

## Independent scorecard slices

### Voice

Carries, independently:

- interruption semantics and latency evidence;
- Stop semantics and latency evidence;
- SlowDown semantics and acknowledgement latency;
- response latency;
- backchannel semantics;
- incomplete-trace count;
- runtime-rejection count.

Latency evidence distinguishes:

- `NotEvaluated`;
- `NotEstablished`;
- `WithinBudget`;
- `Exceeded`.

A `WithinBudget` or `Exceeded` finding requires an actual latency measurement. Missing requested evidence cannot be represented as within budget.

### Correction

Carries:

- first compliant turn delta;
- optional first-compliant wall-clock latency;
- recurrence before adaptation;
- recurrence after adaptation;
- unrelated-context spillover;
- ambiguous applicable observations;
- applicable observation count.

The scorecard deliberately permits `first_compliant_turn_delta = 1` and `recurrence_after_adaptation > 0`; fast adaptation never erases later regression.

### Persona

Carries:

- critical/noncritical violations;
- global/context-band violations;
- evaluator-disagreement groups;
- unobserved required facets;
- facet/profile/context coverage.

Persona continuity remains an independent evidence family and cannot compensate for failed Stop/correction behavior.

## Evidence coverage

The episode derives only an **evidence coverage** state:

```text
NotEstablished
Partial
Complete
Uncertain
Invalid
```

This is not a quality verdict.

Rules:

- any invalid component evidence -> `Invalid`;
- otherwise any uncertain component evidence -> `Uncertain`;
- all components not established -> `NotEstablished`;
- all three established -> `Complete`;
- otherwise -> `Partial`.

Therefore a fully measured episode containing a Stop violation remains `Complete` evidence.

## Commitments

The manifest and final receipt use independent domain-separated BLAKE3 commitments. The final receipt commitment covers:

- manifest commitment;
- exact component receipt commitments;
- derived evidence coverage;
- every field in the three independent scorecards.

Changing a component source/receipt, recurrence count, latency finding, persona violation, or evidence state changes receipt identity.

## Initial tests

- complete evidence with Stop violation remains evidence-complete;
- one-turn correction with later recurrence preserves both facts;
- one missing component evidence lane -> partial;
- all missing -> not established;
- uncertainty cannot be averaged away;
- invalid evidence remains explicit;
- component input ordering does not change manifest identity;
- component subject substitution changes manifest identity;
- duplicate/missing component kinds fail closed;
- missing latency cannot claim `WithinBudget`;
- scorecard evidence state must match frozen component reference;
- manifest/receipt commitment tampering fails;
- correction/persona coverage inconsistencies fail.

## Nonclaims

This tranche does not:

- execute the component candidates;
- prove component qualification;
- classify natural-language corrections;
- measure subjective naturalness;
- establish human-level conversation;
- establish consciousness/personhood;
- create user or physical authority;
- produce a scalar conversation-quality score.

The component candidate qualification states remain independently visible and authoritative for claims about those component subjects.
