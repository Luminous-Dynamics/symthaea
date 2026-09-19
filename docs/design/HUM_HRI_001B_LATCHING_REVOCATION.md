# HUM-HRI-001B — Latching Human-Contact Revocation

Status: source-design candidate
Issue: #4681
Authority: consent lifecycle only; **no motor authority**

## Purpose

Bind the scoped semantic consent introduced by HUM-HRI-001A to a monotonic runtime lifecycle where revocation destroys the currently active consent epoch and recovery cannot silently resume it.

## Core theorem

For an active consent epoch `N`:

`Active(N) -> revoke -> WithdrawRequired(N) -> Idle`

There is deliberately no transition from `WithdrawRequired(N)` or `Idle` back to `Active(N)`.

Re-entry requires a separately supplied live scope with `consent_epoch > N`.

## State machine

- `Idle`
- `Active { consent_epoch }`
- `WithdrawRequired { revoked_consent_epoch, reason }`

`acknowledge_withdrawal_complete()` only records that protective withdrawal/disengagement has completed. It does not restore authority.

## Revocation semantics

Revocation is latching and idempotent while withdrawal is already required. Repeated stop signals do not weaken or replace the first latched stop reason.

The session records the highest consent epoch that has reached a terminal/revoked state. Any later attempt to admit that epoch or an older one fails closed as stale.

## Freshness

A scope can enter `Active` only when all are true:

- the lifecycle is `Idle`;
- participant identity matches the session;
- session identity matches the session;
- the semantic scope is live at the admission time;
- the scope's consent epoch is strictly greater than every terminal/revoked consent epoch previously seen by this runtime session.

A newer scope cannot replace an active scope in place. The existing epoch must first leave the active state.

## Separation of concerns

This lifecycle:

- does not authenticate that a person granted the semantic scope;
- does not grant motor authority;
- does not select a disengagement trajectory;
- does not define force, pressure, temperature, velocity, or energy limits;
- does not make preference, affect, physiology, or operator evidence sufficient for consent;
- does not make a reboot/configuration change safe by itself.

Follow-on work must bind authentic consent verification, exact `HumanContact` IR, bounded protective disengagement, continuity breaks/re-arming, tactile evidence, close-contact physical envelopes, final safety projection, and HAL interlocks.

## Safety invariant

Stopping is monotonic. Recovery may clear the mechanics of withdrawal, but only fresh explicit authority may create a new active epoch.

## Nonclaims

This source contract establishes no HIL/hardware qualification, human-contact safety, human-trial authorization, legal compliance, medical claim, or product-safety certification.
