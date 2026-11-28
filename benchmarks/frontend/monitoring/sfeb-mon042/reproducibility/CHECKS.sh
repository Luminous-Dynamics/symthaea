#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
for cmd in go jq; do command -v "$cmd" >/dev/null 2>&1 || { echo "$cmd required" >&2; exit 69; }; done
cd "$ROOT"
gofmt -w ./mon042.go ./mon042_test.go ./cmd/qualify/main.go
go vet ./...
go test ./...
go test -race ./...
BIN="$(mktemp)"; OUT="$(mktemp)"; BAD="$(mktemp)"
trap 'rm -f "$BIN" "$OUT" "$BAD"' EXIT
go build -o "$BIN" ./cmd/qualify
set +e
"$BIN" --input INPUT-example.json >"$OUT"; STATUS=$?
set -e
[ "$STATUS" -eq 3 ] || { echo "incomplete example must be indeterminate (3), got $STATUS" >&2; exit 1; }
jq -e '.design_verdict=="indeterminate" and .campaign_verdict=="indeterminate" and .production_use_permitted==false' "$OUT" >/dev/null
jq '.unexpected=true' INPUT-example.json >"$BAD"
set +e
"$BIN" --input "$BAD" >/dev/null 2>&1; UNKNOWN=$?
set -e
[ "$UNKNOWN" -eq 65 ] || { echo "unknown field was not rejected" >&2; exit 1; }
cat INPUT-example.json INPUT-example.json >"$BAD"
set +e
"$BIN" --input "$BAD" >/dev/null 2>&1; TRAILING=$?
set -e
[ "$TRAILING" -eq 65 ] || { echo "trailing JSON was not rejected" >&2; exit 1; }
if command -v check-jsonschema >/dev/null 2>&1; then
 check-jsonschema --schemafile INPUT-v1.schema.json INPUT-example.json
elif command -v python3 >/dev/null 2>&1; then
 python3 -c 'import json, jsonschema; s=json.load(open("INPUT-v1.schema.json")); jsonschema.Draft202012Validator.check_schema(s); jsonschema.validate(json.load(open("INPUT-example.json")),s)'
else
 echo "schema validator unavailable; schema validation skipped"
fi
go test -run '^$' -bench BenchmarkEvaluate29IndependentClusters -benchmem -count=5
echo "MON-042 formatting, vet, unit, race, strict CLI, schema and benchmark checks passed."
