# Symthaea Inference Fabric — IF-7 Conservative Resource Guard v1

Status: local quota/circuit-authority child of IF-6.

## Purpose

IF-7 separates provider-observed resource hints from locally authorized inference
spend. Provider headers and receipt evidence may tighten availability and trigger
cooldowns, but they never replenish request/token authority within a quota epoch.

The guard is intentionally local and conservative. It is a prerequisite for a
future deterministic router; it is not itself a provider selector.

## Authority model

`InferenceResourceAuthority` defines the locally authorized envelope for one
provider/account scope:

- quota epoch;
- credential epoch;
- optional request budget;
- optional token budget.

A `None` budget means the local deployment deliberately imposed no numeric cap. It
does **not** mean an unknown provider quota has been proven unlimited.

Only `advance_quota_epoch()` may replenish request/token authority. Provider reset
headers, Retry-After, token usage, request limits, or optimistic remaining values
cannot mint quota.

## Pre-dispatch reservations

Every provider attempt first transfers an `InferenceResourceReservation` out of the
guard. Reservation immediately charges:

- one request, when request authority is bounded;
- the caller's conservative maximum token exposure, when token authority is bounded;
- any tighter provider-observed remaining ceilings.

This prevents concurrent callers from both spending the same last request/token
capacity.

Reservations are non-Clone and non-Serde. They bind:

- a caller-supplied non-zero guard-instance identity;
- reservation generation;
- quota epoch;
- credential epoch;
- reserved token exposure.

A reservation from one guard cannot settle another guard even when both have the
same generation number.

## Abandoned attempts

Dropping a reservation does not refund authority. When the runtime knows an attempt
will never reach settlement it may call `abandon()`. Abandon retires the active
capability but leaves the charged request/token budget consumed.

This avoids two unsafe behaviors at once:

1. automatic refunds based on unproven non-consumption;
2. abandoned active capabilities permanently blocking an explicit epoch transition.

## Provider observations only tighten

IF-6 wire evidence may reduce the provider-observed request/token ceilings. More
optimistic observations later in the same quota epoch cannot increase them.

Provider-declared token usage likewise cannot refund the conservative token
reservation. A later refund design requires a stronger accounting theorem.

Provider reset durations are retained only as informational hints. They do not
advance the quota epoch or replenish availability.

## Rate limits and circuit breaking

A 429 settlement may open a Retry-After cooldown, but provider-requested durations
are capped by local `max_backoff_millis` policy.

Repeated transport/provider failures open a bounded exponential local circuit after
the configured failure threshold.

Cooldown expiry only removes the time gate. It does not refill exhausted quota.

## Credential failures

Authentication/authorization rejection blocks the current credential epoch. A
strictly newer local credential epoch is required to clear that block.

Quota and credential epoch transitions are rejected while active reservations
exist. Old-epoch attempts must first settle or be explicitly abandoned.

Credential rotation does not replenish quota.

## Intended router integration

A future router should use this guard as a hard eligibility boundary:

`policy/capability admission -> resource guard eligibility/reservation -> deterministic ranking -> permit/execution`

Quality, latency, locality, or preference scores must never bypass privacy,
credential, quota, or cooldown rejection.

## Explicit non-claims

- no durable/distributed quota ledger;
- no cross-process reservation coordination;
- no provider reset becoming quota authority;
- no automatic token/request refunds;
- no trusted provider billing reconciliation;
- no provider registry;
- no routing/scoring policy;
- no current `LLMBackend` factory change;
- no claim that provider-reported remaining/usage values are truthful.
