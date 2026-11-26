#!/usr/bin/env sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp "$root/go.mod" "$tmp/go.mod"
gzip -dc "$root/e2e.go.gz" > "$tmp/e2e.go"
gzip -dc "$root/e2e_test.go.gz" > "$tmp/e2e_test.go"
(
  cd "$tmp"
  go test -race -count=5 ./...
  go vet ./...
  go test -count=1 -cover ./...
)
