# MON-009 reconciler v0.2.3 source overlay

This directory contains the newer v0.2.3 source and SQL overlay that was locally tested. It is deliberately versioned separately from the initial public `reconciler/` snapshot so reviewers can compare the two without silently replacing or losing the original.

The Go sources and SQL are gzip-compressed to preserve exact bytes. From the repository root, run:

```sh
set -eu
src=benchmarks/frontend/monitoring/sfeb-mon009/reconciler
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp "$src/go.mod" "$tmp/go.mod"
gzip -dc "$src/v0.2.3/coverage.go.gz" > "$tmp/coverage.go"
gzip -dc "$src/v0.2.3/coverage_test.go.gz" > "$tmp/coverage_test.go"
(
  cd "$tmp"
  go test -race -count=5 ./...
  go vet ./...
  go test -count=1 -cover ./...
)
gzip -t "$src/v0.2.3/schema.sql.gz"
gzip -t "$src/v0.2.3/integration.sql.gz"
```

Local preflight on Go 1.23.2 / linux/amd64: five race-enabled test-suite runs passed, `go vet` passed, 84.7% statement coverage, 36 named test functions. These results apply to the extracted Go reference package only.

## Exact source hashes

SHA-256 of the compressed artifacts:

- `coverage.go.gz`: `83b1734b28a9e374561b4e7b75a255f5cf74c7ce081a61b525c3ce262ba3b766`
- `coverage_test.go.gz`: `22af7d0dc8bf5dcbb51dae12f46def909d1659449639fcf58444fe7a5152fbbf`
- `schema.sql.gz`: `18070155db579eb36722f8a67b814a7dc8b3f9f3b5be16ac0dbf73f999608007`
- `integration.sql.gz`: `c781377c41a657e594586127fe59ea7290eef8e3e74fb8e7d336544e921d3d5c`

## Security and execution limits

- Authority evidence is a normalized, pre-authenticated snapshot. This package does not verify signatures, certificates, COSE, SCITT, revocation proof or policy-authority credentials.
- PostgreSQL integration SQL remains an unexecuted fixture in this environment. No PostgreSQL server/client, Docker or Podman was available. Do not interpret the Go preflight as database verification.
- SQL receipts are unsigned JSON derived from committed ledger state, not RFC 9942 receipts.
- The SQL digest profile and trust assumptions are described in the parent reconciler and PostgreSQL documentation. A connection `session_user` is not a distinct end-user identity when a shared service pool is used.
- This overlay is a review snapshot, not a production qualification.

