#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unformatted="$(gofmt -l join/*.go cmd/mace-join/*.go)"
if [[ -n "$unformatted" ]]; then
  printf 'gofmt required:\n%s\n' "$unformatted" >&2
  exit 1
fi
go test -count=1 ./...
go test -count=1 -race ./...
go vet ./...
tmp_binary="$(mktemp)"
trap 'rm -f "$tmp_binary"' EXIT
go build -o "$tmp_binary" ./cmd/mace-join
