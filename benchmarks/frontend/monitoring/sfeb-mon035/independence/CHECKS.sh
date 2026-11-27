#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
command -v go >/dev/null 2>&1 || { echo "go is required" >&2; exit 69; }
command -v jq >/dev/null 2>&1 || { echo "jq is required" >&2; exit 69; }
jq -e '.profile == "mon035-observer-independence-input-v1" and (.observers|length)==2 and (.nodes|length)>0' "$ROOT/INPUT-example.json" >/dev/null
jq -e '.properties.observers.maxItems == 2 and (.properties.nodes.maxItems <= 50000)' "$ROOT/INPUT-v1.schema.json" >/dev/null
cd "$ROOT"
gofmt -w ./*.go ./cmd/qualify/*.go
go vet ./...
go test ./...
go test -race ./...
go test -run '^$' -bench BenchmarkEvaluateTwoObserverClosures1000 -benchmem -count=5
echo "MON-035 JSON, vet, unit, race and benchmark checks passed. No external infrastructure was queried."
