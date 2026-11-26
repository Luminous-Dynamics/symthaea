# MON-014 digest-join local preflight result v1

Environment: Go 1.23.2 / linux/amd64.

- `gofmt -w *.go` — passed.
- `go test -race -count=5 ./...` — passed.
- `go vet ./...` — passed.
- `go test -count=1 -cover ./...` — passed, 77.1% statement coverage.
- 10 named tests cover digest-context equivalence/incomparability, missing/invalid digests, AAC-06 effect-mode and attestation grades, producer/sink identity joins, exact readback revision, distinct observer identity, control-root independence and commit-time authority outcomes.

This is a local standard-library semantic preflight over normalized adapter results. It does not verify OAR records, COSE signatures, SCITT receipts, timestamps or external-world state. It does not establish standards conformance, database integration or industry superiority.
