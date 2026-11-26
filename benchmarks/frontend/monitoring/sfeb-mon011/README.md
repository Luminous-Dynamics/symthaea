# MON-011 — Causal Evidence Join Integrity (CEJI) preflight

MON-009 qualifies sink-side commits and independent readback. MON-010 qualifies source/claim lineage and decisions. Those pieces can each be locally valid while still being incorrectly joined: a decision can cite the wrong source digest, an intent can change the approved principal or operation, a sink event can bind a different request, or a readback can refer to a different revision.

MON-011 qualifies the joins *between* those layers. It now includes a byte-exact composed trace across the source/claim/decision/action/effect chain, plus explicit interoperability mappings for OAR and SCITT transparency receipts.

## Chain

`source assertion -> authorized decision -> action intent -> sink commit -> independent readback`

The package checks:

- source evidence ID, evidence digest and exact statement digest binding to the decision;
- explicit decision authority state and exact approval/policy digest binding;
- decision-to-intent binding for principal, tenant, resource, mutation path, operation ID, request digest and approval digest;
- intent-to-sink binding plus exact policy digest;
- separately authenticated commit-time authority evidence bound to principal, tenant/resource/path and policy digest, with validity interval, revocation status/time, revocation-check freshness and a trust domain independent from the sink when required;
- sink-to-observation binding for sink ID/digest, operation, tenant/resource, state revision and state digest;
- independent sink/observer trust domains, causal timestamp order and bounded readback freshness;
- explicit `qualified`, `indeterminate`, `contradicted`, `stale` and `unauthorized` outcomes.

A missing observation is indeterminate, never success. A trace ID alone is not a causal evidence join. Measurement evidence cannot satisfy the decision's authority state. A boolean `authority_at_commit` is not accepted in place of a scoped authority evidence record.

## Run

```sh
sh reproduce.sh
```

The Go implementation uses only the standard library. The sources are published as gzip snapshots; the runner expands them into a temporary module before executing tests and vet.

## Local preflight

Go 1.23.2 / linux/amd64:
- 34 named test functions;
- `go test -race -count=5 ./...` passed;
- `go vet` passed;
- 93.4% statement coverage.

These results cover a synthetic normalized-evidence preflight only.

## Composed byte-exact trace

See [composed-e2e/README.md](composed-e2e/README.md). This second preflight joins the MON-010 source assertion, typed claim, transformation and independent replay to the exact decision, approval/display commitment, request bytes, MON-009 commit-time authority check, sink commit and independent readback. The current local run has 30 tests, five race-enabled runs passed, `go vet` passed and 67.9% statement coverage.

It explicitly separates correction valid-time from when the verifier learned about it. A retroactive correction can challenge the decision basis while preserving the fact that the destination effect was committed and independently read back. A later-effective correction raises a separate current-risk review. A correction not yet recorded by the as-of cutoff does not leak into that view.

## Interoperability

- [OAR-INTEROP-v1.md](OAR-INTEROP-v1.md) maps MON to OAR decision, approval, authority re-check, execution commitments and outcome classes. MON does not define a competing action receipt.
- [TRANSPARENCY-PROFILE-QUALIFICATION-v1.md](../sfeb-mon009/TRANSPARENCY-PROFILE-QUALIFICATION-v1.md) distinguishes RFC9162_SHA256 from the SCITT MMR profile and defines receipt-position/consistency tests.

## Matched industry benchmark

See [MATCHED-BENCHMARK-PROTOCOL-v1.md](MATCHED-BENCHMARK-PROTOCOL-v1.md). It defines W0-W6, CPDC C0-C7, fair comparisons against PostgreSQL outbox/CDC, DBOS, Temporal and SCITT/Rekor, and explicit target gates. It does not claim matched industry measurements have already been run.

## Trust boundary

Every `Authenticated` field is an assertion supplied by an adapter after separate verification. This package does not authenticate publisher signatures, decision signatures, authority credentials, COSE/SCITT receipts, trusted timestamps or destination transactions. It does not prove that a source statement is true or independently establish that a trust-domain label is honest. Those checks must be implemented and tested by adapters and deployment controls.

The package does not compute a universal/cross-language canonical payload digest. It requires adapters to supply already-verified SHA-256 digests and exact references. A `qualified` result means only that the supplied normalized records satisfy these cross-layer join invariants and configured policy checks.

## Test corpus

The suite mutates source and decision linkage, approval bytes, decision authority, operation identity, intent/sink references, authority scope, expiry, revocation, revocation freshness, authority-evidence/sink independence, observer independence, readback revision/digest, timestamps, observation freshness and missing/unauthenticated evidence. Tests do not call a real database, COSE verifier, transparency service or live destination.

## Next gates

1. Bind official RFC 9942/9943 and Figure Statements/Decision Provenance vectors through maintained adapters.
2. Add independently authored mutation/property tests and fuzzing for identifier/digest substitution.
3. Connect this preflight to MON-009 v0.2.3 reconciler output and MON-010 claimgraph/CPDC output using exact serialized evidence bytes.
4. Compare matched end-to-end fault traces against PostgreSQL outbox/logical decoding, DBOS, Temporal and SCITT/Rekor; report W0 throughput separately from W6 effect-qualified throughput.
5. Execute PostgreSQL race/rollback/revocation integration and a physically observed effect trial before making production claims.


The OAR outcome projection is executable in [`oar-outcome/`](oar-outcome/README.md), with three named tests, race ×5, vet and 100% statement coverage.
