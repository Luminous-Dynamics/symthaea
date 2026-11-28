# MON-053 temporal admission protocol v0.1

## Temporal claim appraisal

Inputs are a bounded capture interval, an authenticated effective revocation timestamp and a versioned trust checkpoint. The reference function returns:

- `qualified` only when `capture_latest < revocation_effective_at`.
- `blocked` when `revocation_effective_at <= capture_earliest`.
- `indeterminate` when the interval crosses the revocation boundary.
- `indeterminate` when the revocation time or clock ordering cannot be established.

Equality is intentional: a point capture exactly at revocation is blocked; an interval that reaches the revocation instant from earlier is indeterminate. This avoids converting uncertainty into a positive historical proof.

## Dependency leases

A claim's lease is the minimum expiry of its mandatory dependency set. Empty sets, duplicate IDs, unknown expiries and zero timestamps are errors. Optional context is not included in the lease, matching the required/context distinction in MON-050.

## Admission state machine

Permit scope comprises claim ID, claim digest, target, operation, trust epoch, per-target fencing generation, nonce and expiry. The reference store serializes:
1. permit issue,
2. trust epoch changes,
3. claim revocation/re-appraisal,
4. fencing generation changes,
5. one-use nonce consumption,
6. mutation callback.

The nonce is consumed and target fence advanced before the callback. If the callback returns an error, the caller must reconcile external state before creating another mutation request. The original nonce cannot be replayed.

A trust epoch change makes all outstanding permits stale and all prior claim appraisals stale. An unaffected claim must be explicitly re-appraised under the new epoch before a new permit is issued. Revoked claims cannot be reactivated using the re-appraisal method; a new versioned claim is required.

## Linearization contract and limitations

The mutex is the only linearization mechanism in this reference implementation. It guarantees one serialized order among admission callbacks, revocation and fencing operations within one process. It does not prove:
- multi-process or multi-region consensus,
- durability across process crash,
- atomic commit with an external database or infrastructure API,
- resistance to a compromised process or clock,
- cryptographic authenticity of permit structures,
- recovery of an unknown external mutation outcome.

A production realization must place the nonce ledger and mutation decision in the system of record's transactional or consensus boundary, use signed/MACed permits bound to the authenticated trust root, persist event sequence/checkpoint evidence, apply fencing at the actual target, and provide an idempotent reconciliation protocol for unknown outcomes.

## Frozen test evidence

The seven-case temporal corpus is pinned by raw SHA-256 in `vectors_test.go`. The Go package uses only the standard library. Local validation includes unit tests, race detection, 50 repeated test runs, 10,000 concurrent admission/revocation schedules under the race detector, `go vet`, and `go build`.

These are executable reference-model tests, not evidence of production readiness, distributed atomicity or reliability.
