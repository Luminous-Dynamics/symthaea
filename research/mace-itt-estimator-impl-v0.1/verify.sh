#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
gofmt -w ./itt ./cmd
go test -count=1 ./...
go test -count=1 -race ./...
go vet ./...
go test -cover ./itt
go run ./cmd/mace-itt -bootstrap 2000 -seed 1 -roster fixtures/assignment-roster.json < fixtures/pairs.jsonl > /tmp/mace-itt-report.json
if printf '%s\n' '{"schema_version":"mace-itt-pair-v999","pair":{}}' | go run ./cmd/mace-itt >/dev/null 2>&1; then
  echo "unsupported schema unexpectedly accepted" >&2
  exit 1
fi
printf 'MACE ITT verification: PASS\n'
