# MON-053 — Temporal freshness and one-use admission

This standard-library-only Go package is an in-memory reference model for temporal appraisal and race-safe one-use admission.

## Temporal rule

Given capture interval `[t_min, t_max]` and trusted effective revocation time `R`:
- `t_max < R` → qualified.
- `R <= t_min` → blocked.
- otherwise → indeterminate.

A point capture at `R` is blocked. An interval whose upper bound equals `R` but begins earlier is indeterminate. Unknown revocation time is indeterminate.

The claim lease is the minimum expiry of all mandatory dependencies. Empty, duplicate, unknown or zero-expiry entries fail closed.

## Admission model

A permit binds claim ID/digest, target, operation, trust epoch, target fencing token, nonce and deadline. The in-memory store serializes permit issue, revocation, trust-epoch changes, fencing, nonce consumption and the mutation callback under one mutex. The nonce is consumed before the callback; failure or unknown outcome requires reconciliation and never makes it reusable.

A global trust-epoch change invalidates outstanding permits and marks previous claim appraisals stale. Unaffected claims must be explicitly re-appraised before new permits can be issued. Revoked claims cannot be resurrected; a new versioned claim is required.

The mutex is this model's linearization boundary. The callback must be bounded and must not re-enter the store. A production implementation must use a durable transactional/linearizable system of record, authenticate token contents under the trust root, persist event ordering and prove fencing at the actual mutation boundary. This package does none of those things and must not be used to authorize production changes.

## Run

```sh
./CHECKS.sh
```

The seven temporal boundary vectors are SHA-256 pinned. The concurrency tests include 256 simultaneous consumers racing for one nonce and 10,000 paired concurrent admission/revocation schedules under the race detector. These are process-local model tests, not distributed consensus or real system-of-record transactions.
