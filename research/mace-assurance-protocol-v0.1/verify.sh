#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

unformatted="$(gofmt -l matrix/*.go cmd/mace-profilec/*.go)"
if [[ -n "$unformatted" ]]; then
  printf 'gofmt required:\n%s\n' "$unformatted" >&2
  exit 1
fi

go test -count=1 ./...
go test -count=1 -race ./...
go vet ./...

profile_output="$(mktemp)"
trap 'rm -f "$profile_output"' EXIT
go run ./cmd/mace-profilec --input fixtures/profile-compile-request.json > "$profile_output"
grep -q '"inventory_sha256"' "$profile_output"
printf 'profile compilation fixture: PASS\n'
