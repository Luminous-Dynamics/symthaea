# SFEB MON-014 — cross-profile evidence reconciliation

**Status:** standard-library Go semantic reference and adapter protocol. No OAR, COSE or SCITT cryptographic verification is reimplemented here.
**Parent:** MON-013 PostgreSQL effect assurance.

## Distinct contribution

MON-014 prevents false evidence joins caused by treating digest strings as self-describing. Algorithm, profile, version, payload type, canonicalization, encoding, purpose and subject must all be compatible. If digest contexts differ or are missing, the result is incomparable or indeterminate—not equal.

It preserves the independent assurance dimensions introduced by Agent Action Capsule -06:
- effect mode: not applicable, dispatched-unconfirmed, confirmed;
- effect-attestation grade: gate-executed is stronger; runtime-claimed and host-served-observed are equal; unknown values never grade above runtime-claimed;
- cross-party rung: unilateral fallback, acknowledged receipt, full bilateral;
- ledger custody: standalone, chained, anchored.

These dimensions do not collapse into one score. Bilateral linkage does not prove independent administration; anchored transparency does not prove destination state; a producer-confirmed response does not prove a sink mutation.

## Maintained verifier policy

Do not build a replacement OAR verifier. The public OAR repository is pinned to commit `dd1dc0393385048c26b2b47ac543859a1ba6cdd2`; its TypeScript core, CLI and Python verifier are version 0.6.0, and its public corpus contains 18 main plus 6 compatibility cases. OAR 0.3 is the public implementation format. OAR -01's new constructions are planned, not implemented in the current repository.

## Local semantic preflight

Go 1.23.2 / linux-amd64:
- `gofmt -w *.go` — passed.
- `go test -race -count=5 ./...` — passed.
- `go vet ./...` — passed.
- `go test -count=1 -cover ./...` — passed, 77.1% statement coverage.

See [digest-join/RESULTS-v1.md](digest-join/RESULTS-v1.md). The semantic checker consumes normalized verification results; it does not authenticate signatures, timestamps, or log receipts.

## Run the semantic suite

    cd digest-join
    go test -race -count=5 ./...
    go vet ./...
    go test -count=1 -cover ./...

## Run the pinned upstream OAR verifier suite

See [UPSTREAM-VERIFIER-PREFLIGHT-v1.md](UPSTREAM-VERIFIER-PREFLIGHT-v1.md). The runner requires the exact public OAR source commit and a Node/pnpm environment. It invokes upstream frozen-lockfile install, build, typecheck and tests; it does not download moving latest packages.

## Important limits

- A qualified result here means only that verified normalized inputs satisfy the declared cross-profile join rules.
- No test here authenticates upstream records; maintained upstream verifiers must run first.
- A valid OAR confirmation or Agent Action Capsule confirmed response remains an input claim. MON still requires its own sink commit and separately authorized exact-revision readback.
- This semantic preflight is not an industry benchmark or production qualification.
