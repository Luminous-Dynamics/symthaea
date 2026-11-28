#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
for cmd in go jq; do command -v "$cmd" >/dev/null 2>&1 || { echo "$cmd is required" >&2; exit 69; }; done
cd "$ROOT"
gofmt -w ./capability.go ./capability_test.go ./cmd/qualify/main.go
go vet ./...
go test ./...
go test -race ./...
BIN="$(mktemp)"; OUT="$(mktemp)"; BAD="$(mktemp)"
trap 'rm -f "$BIN" "$OUT" "$BAD"' EXIT
go build -o "$BIN" ./cmd/qualify

set +e
"$BIN" --input INPUT-example.json >"$OUT"; INCOMPLETE_STATUS=$?
set -e
[ "$INCOMPLETE_STATUS" -eq 3 ] || { echo "missing required capability must be indeterminate (3), got $INCOMPLETE_STATUS" >&2; exit 1; }
jq -e '.coverage_verdict=="indeterminate" and .campaign_verdict=="indeterminate" and .production_action_permitted==false and (.missing_required|length)==1' "$OUT" >/dev/null

set +e
"$BIN" --input fixtures/INPUT-valid-synthetic.json >"$OUT"; VALID_STATUS=$?
set -e
[ "$VALID_STATUS" -eq 0 ] || { echo "synthetic supported-scope fixture must qualify offline coverage (0), got $VALID_STATUS" >&2; exit 1; }
jq -e '.capability_verdict=="qualified" and .coverage_verdict=="qualified" and .campaign_verdict=="indeterminate" and .production_action_permitted==false and .covered_required_count==1' "$OUT" >/dev/null

jq '.unexpected_field=true' INPUT-example.json >"$BAD"
set +e
"$BIN" --input "$BAD" >/dev/null 2>&1; UNKNOWN_STATUS=$?
set -e
[ "$UNKNOWN_STATUS" -eq 65 ] || { echo "unknown JSON field was not rejected" >&2; exit 1; }
cat INPUT-example.json INPUT-example.json >"$BAD"
set +e
"$BIN" --input "$BAD" >/dev/null 2>&1; TRAILING_STATUS=$?
set -e
[ "$TRAILING_STATUS" -eq 65 ] || { echo "trailing JSON document was not rejected" >&2; exit 1; }

if command -v check-jsonschema >/dev/null 2>&1; then
 check-jsonschema --schemafile INPUT-v1.schema.json INPUT-example.json fixtures/INPUT-valid-synthetic.json
elif command -v python3 >/dev/null 2>&1; then
 python3 -c 'import json, jsonschema; s=json.load(open("INPUT-v1.schema.json")); jsonschema.Draft202012Validator.check_schema(s); [jsonschema.validate(json.load(open(p)),s) for p in ["INPUT-example.json","fixtures/INPUT-valid-synthetic.json"]]'
else
 echo "JSON Schema validation skipped: no validator installed"
fi

go test -run '^$' -bench BenchmarkEvaluateTwoRequiredItems -benchmem -count=5
echo "MON-044 gofmt, vet, unit, race, CLI, schema-if-available and benchmark checks passed."
