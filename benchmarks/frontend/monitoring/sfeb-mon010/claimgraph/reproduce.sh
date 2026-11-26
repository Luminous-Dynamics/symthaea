#!/usr/bin/env sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM
cp "$root/go.mod" "$tmpdir/go.mod"
gzip -dc "$root/claimgraph.go.gz" > "$tmpdir/claimgraph.go"
gzip -dc "$root/claimgraph_test.go.gz" > "$tmpdir/claimgraph_test.go"
cd "$tmpdir"
go test -race -count=5 ./...
go vet ./...
go test -count=1 -cover ./...
