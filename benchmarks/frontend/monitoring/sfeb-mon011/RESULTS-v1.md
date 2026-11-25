# MON-011 local preflight result v1

Environment: Go 1.23.2, linux/amd64.

- 33 named test functions.
- `go test -race -count=5 ./...` — pass.
- `go vet ./...` — pass.
- `go test -count=1 -cover ./...` — 92.5% statement coverage.

The test suite includes causal reference/digest binding, exact identity and request scope, separate authority evidence at commit, authority expiry and revocation status, independent authority/sink and observer/sink trust domains, exact readback revision/digest, causal ordering, freshness and fail-closed missing evidence.

This is a synthetic semantic preflight over normalized inputs. It is not a COSE/SCITT or signature conformance result, database integration, external product benchmark, production qualification or SOTA claim.
