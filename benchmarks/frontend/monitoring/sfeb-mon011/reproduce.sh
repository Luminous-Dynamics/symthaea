#!/usr/bin/env sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp "$root/go.mod" "$tmp/go.mod"
gzip -dc "$root/join.go.gz" > "$tmp/join.go"
gzip -dc "$root/join_test.go.gz" > "$tmp/join_test.go"
(
  cd "$tmp"
  go test -race -count=5 ./...
  go vet ./...
  go test -count=1 -cover ./...
)
