#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
command -v go >/dev/null 2>&1 || { echo "go required" >&2; exit 69; }
command -v jq >/dev/null 2>&1 || { echo "jq required" >&2; exit 69; }
jq -e '.profile=="mon037-unified-recovery-scenarios-v1" and (.scenarios|length)>=20 and ([.scenarios[].id]|unique|length)==(.scenarios|length)' "$ROOT/SCENARIOS-v1.json" >/dev/null
jq -e '.profile=="mon037-unified-recovery-campaign-input-v1" and .components.mon034_timeline_result.profile=="mon034-postgres-timeline-ancestry-result-v1"' "$ROOT/INPUT-example.json" >/dev/null
cd "$ROOT"
gofmt -w ./*.go ./cmd/qualify/*.go
go vet ./...
go test ./...
go test -race ./...
BIN="$(mktemp)"
OUT="$(mktemp)"
trap 'rm -f "$BIN" "$OUT"' EXIT
go build -o "$BIN" ./cmd/qualify
"$BIN" --input INPUT-example.json > "$OUT"
jq -e '.component_verdict=="qualified" and .campaign_verdict=="indeterminate" and .traffic_activation_permitted==false' "$OUT" >/dev/null
go test -run '^$' -bench BenchmarkEvaluate1000Events100Transactions -benchmem -count=5
echo "MON-037 format, vet, unit, race, CLI and benchmark checks passed."
