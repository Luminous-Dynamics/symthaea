# OAR outcome-class compatibility preflight

This standard-library Go package projects the Open Action Record lifecycle vocabulary onto OAR's three relying-party outcome classes:

- `confirmed` → `committed`
- `denied`, `rejected`, `expired`, `refused` → `refused`
- every other, unknown, non-terminal or unconfirmed outcome → `unresolved`

An `executed` receipt means the producer recorded execution, not that a read-only system-of-record confirmation exists. A producer/API success claim cannot be promoted to `committed` without the OAR confirmation outcome.

## Run

```sh
go test -race -count=5 ./...
go vet ./...
go test -count=1 -cover ./...
```

Go 1.23.2 / linux/amd64: three named tests, race suite ×5 passed, `go vet` passed, 100% statement coverage.

This is a semantic projection check based on the OAR outcome table. It is **not** an OAR bundle parser, Ed25519/JCS/sealer/RFC 3161 verifier, nor an execution of the official 24-vector main/compatibility corpus. MON must use maintained OAR verifiers for those checks; this package validates only the mapping boundary.

References: [OAR public repository](https://github.com/openactionrecord/oar), [OAR SPEC](https://github.com/openactionrecord/oar/blob/main/SPEC.md), [OAR test vectors](https://github.com/openactionrecord/oar/tree/main/vectors).
