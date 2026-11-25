# MON-011 local preflight result v1

Environment: Go 1.23.2, linux/amd64.

- 34 named test functions.
- `go test -race -count=5 ./...` — pass.
- `go vet ./...` — pass.
- `go test -count=1 -cover ./...` — 93.4% statement coverage.

The tests cover source/decision/intent/sink/observation reference binding, identity and request scope, authority evidence at commit, scope/policy match, validity interval, revocation status and freshness, authority/sink trust-domain separation, observer independence, exact readback revision/digest, causal order, freshness and fail-closed missing evidence.

This is a synthetic semantic preflight over normalized inputs. It is not a COSE/SCITT or signature conformance result, database integration, external product benchmark, production qualification or SOTA claim.
