#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

unformatted="$(gofmt -l matrix/*.go)"
if [[ -n "$unformatted" ]]; then
  printf 'gofmt required:\n%s\n' "$unformatted" >&2
  exit 1
fi

go test -count=1 ./...
go test -count=1 -race ./...
go vet ./...
