#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
for cmd in go jq; do command -v "$cmd" >/dev/null 2>&1 || { echo "$cmd is required" >&2; exit 69; }; done
cd "$ROOT"
gofmt -w ./invalidation.go ./invalidation_test.go ./cmd/qualify/main.go
go vet ./...
go test ./...
go test -race ./...
BIN="$(mktemp)"; OUT="$(mktemp)"; BAD="$(mktemp)"
trap 'rm -f "$BIN" "$OUT" "$BAD"' EXIT
go build -o "$BIN" ./cmd/qualify

set +e
"$BIN" --input INPUT-example.json >"$OUT"; HEALTHY_STATUS=$?
set -e
[ "$HEALTHY_STATUS" -eq 0 ] || { echo "healthy offline graph must exit 0, got $HEALTHY_STATUS" >&2; exit 1; }
jq -e '.graph_verdict=="qualified" and .campaign_verdict=="indeterminate" and .production_action_permitted==false and (.impacted_claim_ids|length)==0 and .unaffected_claim_ids==["recovery-complete"]' "$OUT" >/dev/null

set +e
"$BIN" --input INPUT-revoked-example.json >"$OUT"; REVOKED_STATUS=$?
set -e
[ "$REVOKED_STATUS" -eq 2 ] || { echo "revoked required evidence must exit blocked (2), got $REVOKED_STATUS" >&2; exit 1; }
jq -e '.graph_verdict=="blocked" and .campaign_verdict=="indeterminate" and .production_action_permitted==false and .impacted_claim_ids==["recovery-complete"] and (.invalidated_node_ids|index("postgres-source-truth")!=null)' "$OUT" >/dev/null

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
 check-jsonschema --schemafile INPUT-v1.schema.json INPUT-example.json INPUT-revoked-example.json
elif command -v python3 >/dev/null 2>&1; then
 python3 -c 'import json, jsonschema; s=json.load(open("INPUT-v1.schema.json")); jsonschema.Draft202012Validator.check_schema(s); [jsonschema.validate(json.load(open(p)),s) for p in ["INPUT-example.json","INPUT-revoked-example.json"]]'
else
 echo "JSON Schema validation skipped: no validator installed"
fi

go test -run '^$' -bench BenchmarkEvaluate1000NodeEvidenceGraph -benchmem -count=5
echo "MON-045 format, vet, unit, race, CLI, schema-if-available and benchmark checks passed."
