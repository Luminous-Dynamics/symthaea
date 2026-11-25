# MON-011 — Causal Evidence Join Integrity (CEJI) preflight

MON-009 qualifies sink-side commits and independent readback. MON-010 qualifies source/claim lineage and decisions. Those pieces can each be locally valid while still being incorrectly joined: a decision can cite the wrong source digest, an intent can change the approved principal or operation, a sink event can bind a different request, or a readback can refer to a different revision.

MON-011 qualifies the joins *between* those layers.

## Chain

`source assertion -> authorized decision -> action intent -> sink commit -> independent readback`

The package checks:

- source evidence ID, evidence digest and exact statement digest binding to the decision;
- explicit decision authority state and exact approval/policy digest;
- decision-to-intent binding for principal, tenant, resource, mutation path, operation ID, request digest and approval digest;
- intent-to-sink binding plus exact policy digest;
- separately authenticated commit-time authority evidence bound to principal, tenant/resource/path and policy digest, with validity interval, revocation state/time, revocation-check freshness and a trust domain independent from the sink when required;
- sink-to-observation binding for sink ID/digest, operation, tenant/resource, state revision and state digest;
- independent sink/observer trust domains, causal timestamp order and bounded readback freshness;
- explicit `qualified`, `indeterminate`, `contradicted`, `stale` and `unauthorized` outcomes.

A missing observation is indeterminate, never success. A trace ID alone is not a causal evidence join. Measurement evidence cannot satisfy the decision's authority state. A boolean `authority_at_commit` is not accepted in place of a scoped authority evidence record.

## Run

```sh
sh reproduce.sh
```

The Go implementation uses only the standard library. The sources are published as gzip snapshots; the runner extracts them to a temporary directory before executing tests and vet.

## Local preflight

Go 1.23.2 / linux/amd64:
- 33 named test functions;
- `go test -race -count=5 ./...` passed;
- `go vet ./...` passed;
- 92.5% statement coverage.

These results cover a synthetic normalized-evidence preflight only.

## Trust boundary

Every `Authenticated` field is an assertion supplied by an adapter after separate verification. This package does not authenticate publisher signatures, decision signatures, authority credentials, COSE/SCITT receipts, trusted timestamps or destination transactions. It does not prove that a source statement is true or independently establish that a trust-domain label is honest. Those checks must be implemented and tested by adapters and deployment controls.

The package does not compute a universal/cross-language canonical payload digest. It requires adapters to supply already-verified SHA-256 digests and exact references. A `qualified` result means only that the supplied normalized records satisfy these join invariants and configured policy checks.

## Test corpus

The suite mutates source and decision linkage, approval bytes, decision authority, operation identity, intent/sink references, authority scope, expiry, revocation, revocation freshness, observer independence, readback revision/digest, timestamps, observation freshness, and missing/unauthenticated evidence. Tests do not call a real database, COSE verifier, transparency service or live destination.

## Next gates

1. Bind official RFC 9942/9943 and Figure Statements/Decision Provenance vectors through maintained adapters.
2. Add independently authored mutation/property tests and fuzzing for identifier/digest substitution.
3. Connect this preflight to MON-009 v0.2.3 reconciler output and MON-010 claimgraph/CPDC output using exact serialized evidence bytes.
4. Compare matched end-to-end fault traces against PostgreSQL outbox/logical decoding, DBOS, Temporal and SCITT/Rekor; report W0 throughput separately from W6 effect-qualified throughput.
5. Execute PostgreSQL race/rollback/revocation integration and a physically observed effect trial before making production claims.
