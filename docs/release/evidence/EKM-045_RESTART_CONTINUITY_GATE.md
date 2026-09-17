# EKM-045 — Restart Receipt Continuity Gate

## Purpose

EKM-045 separates **validity** from **freshness/continuity** for restart candidates.

A candidate may pass every EKM-035/041/043 semantic check and still be an old state replayed after a newer checkpoint. This tranche compares an EKM-044 validation receipt with a caller-supplied trusted anchor and classifies the relationship without constructing or activating restart state.

## Types

`TrustedRestartValidationAnchorV1`

- capture cycle;
- EKM-044 receipt digest;
- constructible from a validated EKM-044 receipt;
- does not provide durable/tamper-resistant storage itself.

`RestartContinuityDispositionV1`

- `IdempotentReplay`
- `ForwardProgress`
- `Rollback`
- `SameCycleEquivocation`

`RestartContinuityDecisionV1`

- candidate and anchor cycles;
- candidate and anchor receipt digests;
- disposition;
- `further_review_eligible`;
- `quarantine_construction_authorized = false`;
- `activation_authorized = false`.

## Rules

Given trusted anchor A and validated candidate C:

1. `digest(C) == digest(A)` → `IdempotentReplay`.
2. different digest and `cycle(C) < cycle(A)` → `Rollback`.
3. different digest and `cycle(C) == cycle(A)` → `SameCycleEquivocation`.
4. different digest and `cycle(C) > cycle(A)` → `ForwardProgress`.

Only exact replay and forward progress are eligible for a later review layer. Neither case authorizes quarantine construction or activation.

## Trust boundary

The gate is only as rollback-resistant as the caller's anchor storage.

If an attacker can roll back or replace the trusted anchor itself, EKM-045 cannot infer a newer historical checkpoint from the candidate alone. Production deployment therefore needs an independently protected monotonic anchor mechanism appropriate to its threat model (for example an operator-controlled durable store, append-only log, signed checkpoint, hardware monotonic counter, or external transparency service).

EKM-045 deliberately does not choose or implement that deployment mechanism.

## Negative controls

Tests cover:

- exact receipt → idempotent replay;
- older valid receipt → rollback;
- different valid receipt at the same capture cycle → equivocation;
- newer different valid receipt → forward progress;
- all outcomes retain `quarantine_construction_authorized = false` and `activation_authorized = false`.

## Non-claims

EKM-045 does not:

- prove an anchor was stored securely;
- provide anti-rollback persistence;
- authenticate receipt origin;
- sign checkpoints;
- construct restart capsules or quarantine images;
- hydrate or activate state;
- mutate evidence, beliefs, causal/world-model/action state;
- perform file/database/network I/O;
- qualify EKM-044 or any parent while CI has not executed.

## Qualification boundary

CI remains authoritative. At authoring time no Actions run had yet attached to the EKM-044 exact head, while parent EKM-043 CI #7065 remained queued. No format, compile, Clippy, test, runtime, or restore qualification is inferred.
