#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
command -v go >/dev/null 2>&1 || { echo "go is required" >&2; exit 69; }
command -v jq >/dev/null 2>&1 || { echo "jq is required" >&2; exit 69; }

jq -e '.profile == "mon035-observer-independence-input-v1" and (.observers|length)==2 and (.nodes|length)>0' "$ROOT/INPUT-example.json" >/dev/null
jq -e '.properties.observers.maxItems == 2 and (.properties.nodes.maxItems <= 50000)' "$ROOT/INPUT-v1.schema.json" >/dev/null

cd "$ROOT"
if [ -n "$(gofmt -l ./*.go ./cmd/qualify/*.go)" ]; then
  echo "Go files need gofmt" >&2
  exit 1
fi

go vet ./...
go test ./...
go test -race ./...

BIN="$(mktemp)"
SHARED_INPUT="$(mktemp)"
SHARED_OUTPUT="$(mktemp)"
trap 'rm -f "$BIN" "$SHARED_INPUT" "$SHARED_OUTPUT"' EXIT
go build -o "$BIN" ./cmd/qualify
"$BIN" --input INPUT-example.json | jq -e '.result.verdict == "qualified" and .result.independent == true and (.result.shared_domains|length)==0' >/dev/null
jq '(.nodes[] | select(.id == "b-network").failure_domain_id) = "a:network"' INPUT-example.json > "$SHARED_INPUT"
set +e
"$BIN" --input "$SHARED_INPUT" > "$SHARED_OUTPUT"
STATUS=$?
set -e
[ "$STATUS" -eq 2 ] || { echo "shared-domain CLI fixture did not exit blocked (2): $STATUS" >&2; exit 1; }
jq -e '.result.verdict == "blocked" and (.result.shared_domains | any(.class == "network" and .domain_id == "a:network"))' "$SHARED_OUTPUT" >/dev/null

go test -run '^$' -bench BenchmarkEvaluateTwoObserverClosures1000 -benchmem -count=5
echo "MON-035 JSON, vet, unit, race, CLI and benchmark checks passed. No external infrastructure was queried."
