#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
for cmd in go jq; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "$cmd is required" >&2; exit 69; }
done

jq -e '
  .profile == "mon039-counterfactual-fault-scenarios-v1"
  and (.scenarios | length) >= 10
  and ([.scenarios[].id] | unique | length) == (.scenarios | length)
' "$ROOT/SCENARIOS-v1.json" >/dev/null

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/cmd"
cp "$ROOT"/go.mod "$ROOT"/*.go "$TMP"/
cp "$ROOT"/cmd/minimize/main.go "$TMP/cmd/main.go"
cd "$TMP"

gofmt -w ./*.go ./cmd/*.go
go vet ./...
go test ./...
go test -race ./...
go build -o "$TMP/mon039" ./cmd

if command -v check-jsonschema >/dev/null 2>&1; then
  check-jsonschema --schemafile "$ROOT/INPUT-v1.schema.json" "$ROOT/INPUT-example.json"
fi

set +e
"$TMP/mon039" --input "$ROOT/INPUT-example.json" > "$TMP/result.json"
STATUS=$?
set -e
[ "$STATUS" -eq 3 ] || { echo "offline fixture must exit indeterminate (3), got $STATUS" >&2; exit 1; }
jq -e '
  .full_fault_set_verdict == "blocked"
  and .minimal_fault_ids == ["fault-a","fault-c"]
  and .minimality_kind == "global_candidate_minimum"
  and .evidence_complete == true
  and .campaign_verdict == "indeterminate"
  and .production_action_permitted == false
  and .subset_evaluations == 7
' "$TMP/result.json" >/dev/null

jq '.observations |= map(select(.fault_ids | length > 0))' "$ROOT/INPUT-example.json" > "$TMP/missing.json"
set +e
"$TMP/mon039" --input "$TMP/missing.json" > "$TMP/missing-result.json"
MISSING_STATUS=$?
set -e
[ "$MISSING_STATUS" -eq 3 ] || { echo "missing subset must remain indeterminate" >&2; exit 1; }
jq -e '.minimality_kind == "not_established" and .evidence_complete == false' "$TMP/missing-result.json" >/dev/null

jq '.code_revision = "ffffffffffffffffffffffffffffffffffffffff"' "$ROOT/INPUT-example.json" > "$TMP/stale.json"
set +e
"$TMP/mon039" --input "$TMP/stale.json" >/dev/null 2>&1
STALE_STATUS=$?
set -e
[ "$STALE_STATUS" -eq 2 ] || { echo "stale code revision must block" >&2; exit 1; }

jq '.unexpected_field = true' "$ROOT/INPUT-example.json" > "$TMP/unknown.json"
set +e
"$TMP/mon039" --input "$TMP/unknown.json" >/dev/null 2>&1
UNKNOWN_STATUS=$?
set -e
[ "$UNKNOWN_STATUS" -eq 65 ] || { echo "unknown JSON field was not rejected" >&2; exit 1; }

go test -run '^$' -bench 'Benchmark(ExhaustiveTwelveFaults|OneMinimalSixtyFourFaults)$' -benchmem -count=5
echo "MON-039 format, vet, unit, race, CLI, signature, completeness and benchmark checks passed."
