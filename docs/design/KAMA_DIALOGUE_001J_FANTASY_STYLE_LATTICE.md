# KAMA-DIALOGUE-001J — Fantasy Style Lattice and Adaptive Pacing

Status: source candidate; not qualified.

## Purpose

Represent adult fantasy/romantic dialogue style as a multidimensional, non-authoritative proposal rather than a single intensity scalar.

This tranche composes three already-defined boundaries:

- `AdultFantasySessionV1` — active adult-only conversational eligibility;
- `FantasyPreferenceModelV1` — preference evidence plus hard topic boundaries;
- `FantasyWorldStateV1` — fiction-only world/scene continuity.

It does not create a session, verify age, authorize likeness, unblock a topic, authorize physical contact, or create motor authority.

## Style dimensions

The V1 lattice materializes all existing preference dimensions:

- romance;
- playfulness;
- directness;
- verbal intensity;
- tenderness;
- initiative;
- suspense;
- humor;
- narrative density;
- callback density.

A separate optional `descriptive_balance` represents descriptive-vs-conversational rendering style.

## Evidence precedence

Per-turn explicit overrides apply only to the current proposal. They do not rewrite durable preference evidence.

Otherwise the planner consumes the existing preference model's effective estimate. Missing evidence remains `None`/`Unknown` rather than receiving a population or demographic default.

For directness, verbal intensity, and initiative, unknown state causes `ClarifyBeforeIncreasing` rather than guessed escalation.

## Pacing

Pacing is explicit:

- `Continue` — maintain when sensitive dimensions are known; otherwise clarify before increasing;
- `Hold` — no increase in verbal intensity, initiative, or suspense above the previous proposal in the same session;
- `SlowDown` — the same hard ceiling plus an explicit downstream obligation to decrease intensity; no universal numeric decay rate is invented;
- `Stop` — latches the `AdultFantasySessionV1` into stopped state and emits no style proposal.

The planner only uses a previous proposal as a ceiling when the session ID and session epoch are exactly the same. Old-session state cannot constrain or authorize a new session.

## Binding

Every proposal binds:

- session ID and session epoch;
- fantasy world ID;
- scene ID and scene epoch;
- hard-boundary epoch;
- topic ID;
- proposal epoch.

This makes stale style proposals identifiable when session, scene, or boundary state changes.

## Core invariants

```text
style preference != permission
per-turn override != durable preference
fantasy world != real-world fact
Hold/SlowDown != permission to increase elsewhere
Stop != zero-intensity continuation
```

And the broader authority rule remains:

```text
Preference may suggest.
Session state may admit conversation.
Boundaries may constrain.
Nothing in this module grants physical authority.
```

## Tests

The integration suite covers:

- explicit current-turn override beating stored preference without mutating it;
- unknown sensitive axes producing clarification-before-increase;
- hard topic boundary dominance;
- exact session/world binding;
- Hold ceilings on intensity-bearing axes;
- SlowDown no-increase + must-decrease semantics;
- Stop latching the fantasy session and emitting no proposal;
- malformed/duplicate overrides failing closed;
- proposal binding to session/scene/boundary epochs.

## Nonclaims

This tranche does not establish generation quality, sexual/romantic suitability, psychological inference quality, long-term preference correctness, voice/prosody quality, memory truth, consent, physical safety, or product superiority.
