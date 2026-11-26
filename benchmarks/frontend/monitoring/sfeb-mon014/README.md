# SFEB MON-014 — cross-profile evidence reconciliation

**Status:** standard-library Go semantic reference and adapter protocol. No OAR, COSE or SCITT cryptographic verification is reimplemented here.
**Parent:** MON-013 PostgreSQL effect assurance.

## Distinct contribution

MON-014 addresses a subtle source of false joins: equal-looking digest strings do not necessarily identify equal evidence. Algorithm, profile, version, payload type, canonicalization, encoding, purpose and subject must all be compatible. If digest contexts differ or are missing, the result is incomparable or indeterminate—not equal.

It also preserves the independent assurance dimensions introduced by Agent Action Capsule -06:
- effect mode: not applicable, dispatched-unconfirmed, confirmed;
- effect-attestation grade: gate-executed is stronger; runtime-claimed and host-served-observed are equal; unknown values never grade above runtime-claimed;
- cross-party rung: unilateral fallback, acknowledged receipt, full bilateral;
- ledger custody: standalone, chained, anchored.

These dimensions do not collapse into a single score. In particular, full bilateral references do not prove independent administration; anchored transparency does not prove destination state; a producer-confirmed response does not prove an external sink mutation.

## Maintained verifier policy

Do not build a replacement OAR verifier. The public OAR repository pins a TypeScript core and CLI at 0.6.0, a Python verifier at 0.6.0, and 18 main plus 6 compatibility test vectors. Its main branch at this snapshot is pinned to commit dd1dc0393385048c26b2b47ac543859a1ba6cdd2. The upstream project states that all three verifiers run the vector corpus in CI.

OAR 0.3 is the current public implementation format. The OAR -01 Internet-Draft says its new constructions are planned, not yet implemented by the current repository. MON must report current OAR verifier conformance separately from an -01 design mapping.

## Local preflight

Go 1.23.2 / linux-amd64.
Run the semantic suite using the commands below and preserve the command transcript in the PR evidence. The Go checker consumes normalized verification states; it does not authenticate signatures, timestamps or log receipts.

## Run the local semantic suite

    cd digest-join
    go test -race -count=5 ./...
    go vet ./...
    go test -count=1 -cover ./...

## Run the pinned upstream OAR verifier suite

See UPSTREAM-VERIFIER-PREFLIGHT-v1.md. The runner requires the exact public OAR source commit and a Node/pnpm environment. It invokes upstream frozen-lockfile build, typecheck and tests; it does not download or trust arbitrary moving latest packages.

## Important limits

- A qualified result here means only that verified normalized inputs satisfy the declared cross-profile join rules.
- No test here authenticates upstream records; upstream verifiers must run first.
- A valid OAR confirmation or Agent Action Capsule confirmed response remains an input claim. MON still requires its own sink commit and independently authorized exact-revision readback.
- This semantic preflight is not an industry benchmark or production qualification.
