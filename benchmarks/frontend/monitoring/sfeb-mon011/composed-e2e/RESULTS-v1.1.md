# MON-011 composed evidence trace preflight result v1.1

Environment: Go 1.23.2 / linux/amd64.

- 30 named adversarial tests.
- `go test -race -count=5 ./...` — pass.
- `go vet ./...` — pass.
- `go test -count=1 -cover ./...` — 67.9% statement coverage.
- ZIP integrity check — pass.

The trace binds source assertion, claim semantics, transformation inputs/outputs, independent replay, decision, approval/display commitments, exact request bytes, commit-time authority evidence, sink event and independent readback.

Temporal tests separate source correction valid-time from recorded-time. A retroactive correction can mark the decision basis stale while preserving the independently confirmed committed effect. A later-effective correction triggers a separate current-risk review. Corrections not yet recorded by the audit cutoff do not leak into earlier as-of views.

This is a semantic preflight over normalized inputs. It does not implement OAR/SCITT signatures, RFC 8785, COSE, a live PostgreSQL sink or physical instrumentation. No external SOTA result is claimed.
