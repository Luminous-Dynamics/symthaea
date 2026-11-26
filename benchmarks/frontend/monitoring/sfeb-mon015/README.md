# SFEB MON-015 — independent assurance mutation and holdout harness

**Status:** local semantic model, 18 deterministic mutation cases, five fuzz targets, and microbenchmarks pass in the recorded Go environment. This is not a cryptographic verifier, PostgreSQL integration result, upstream standards-conformance result, or production benchmark.

**Parent:** MON-014 cross-profile assurance (research/sfeb-mon014-cross-profile-assurance-v0).

## Why MON-015 exists

MON-014 established a semantic join across normalized producer, sink, and observer evidence. MON-015 turns the join into an executable challenge harness. It asks whether an implementation can be tricked into promoting an action when an operation identifier, tenant, request digest, effect status, readback principal, revision, authority outcome, or causal reference is changed.

The research shift is from "does a verifier accept valid evidence?" to **"which specific false claims can an adversary make it accept, and can an independent holdout reproduce that failure?"** The mutation corpus defines false claims before the code is exercised.

## Included

- AAC -06 five-state effect projection: planned, dispatched, confirmed, failed, reverted.
- The planned/dispatched request-digest MAY/SHOULD semantics are preserved; absent request digest is not silently upgraded into a hard failure.
- Confirmed effect requires a verified response digest; a claimed confirmed mode without that binding derives at most dispatched-unconfirmed and fails qualification.
- Effect-attestation grade: gate-executed above runtime-claimed and host-served-observed; unknown values remain informational and never grade upward.
- Verdict/effect orthogonality and independent assurance dimensions: attestation mode, effect mode, ledger mode, and optional cross-party rung.
- OAR 0.3 lifecycle projection: root decision, optional approval after a held decision, execution, and exact execution-bound confirmation.
- Digital-effect qualification requiring verified producer and sink evidence, exact scope and request digest, destination commit, separate observer principal, exact revision/state digest, control-root independence, and valid authority at commit.
- Explicit outcomes for refused, failed, executed-unconfirmed, not-confirmed, indeterminate, contradicted, shared-control-root and authority-uncheckable evidence.
- Eighteen deterministic, independently specified single-field mutations and five bounded fuzz targets.
- Go microbenchmarks for digest equality, digest incomparability, full digital-effect join, and OAR lifecycle projection.

## Run

From this directory:

    cd semantic-adapter
    go test -race -count=5 ./...
    go vet ./...
    go test -count=1 -cover ./...
    go test -run '^$' -bench . -benchmem

The semantic-adapter/README.md lists each fuzz target and its 30-second command.

## Important boundaries

- Inputs are normalized results from maintained OAR, AAC, COSE and SCITT verifiers. MON-015 does not validate signatures, JCS, COSE protected headers, SCITT receipts, timestamp authorities, or MMR proofs.
- The OAR lifecycle projection does not replace the upstream OAR verifier. It checks cross-record semantic relationships after upstream verification has passed.
- qualified_digital_effect is strictly a digital destination-state result. It is not evidence of a physical-world outcome.
- A separate principal is not necessarily an independent control domain. Control-root independence must be supplied by deployment evidence and assessed separately.
- The harness does not prove complete capture of all external actions. Completeness is always relative to a declared, independently monitored observation boundary.
- Local fuzz smoke and microbenchmarks are not population-wide security guarantees or industry comparisons.

## Next qualification gates

1. Feed actual OAR 0.3 bundles through pinned upstream TypeScript and Python verifiers, then pass their normalized reports into this adapter.
2. Feed maintained AAC -06 and COSE/SCITT verifier outputs; preserve verifier versions and trust-anchor digests.
3. Run blinded holdout mutations authored independently from this implementation.
4. Execute MON-013 SQL on disposable PostgreSQL instances, including simultaneous sessions, revocation races, rollback, lost acknowledgement, role bypass and WAL/restore controls.
5. Run matched DBOS and Temporal workloads with equal transaction and durability semantics, then publish the assurance-cost frontier.
6. Extend W6 to real actuators, independent sensors, calibration records, and tamper-evident physical observation.
