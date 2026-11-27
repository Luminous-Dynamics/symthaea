#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

for cmd in go jq sha256sum; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "$cmd is required" >&2; exit 69; }
done

jq -e '
  .profile == "mon038-kafka-partition-capture-scenarios-v1"
  and (.scenarios | length) >= 20
  and ([.scenarios[].id] | unique | length) == (.scenarios | length)
' "$ROOT/SCENARIOS-v1.json" >/dev/null

jq -e '
  .profile == "mon038-kafka-partition-capture-input-v1"
  and .recovery_boundary.claim_id == "claim-cdc"
  and .topic_inventory_complete == true
  and (.topic_inventory | length) == 2
  and (.partition_captures | length) == 3
  and ([.partition_captures[].isolation_level] | all(. == "read_committed"))
  and ([.partition_captures[].cleanup_policy] | all(. == "delete"))
' "$ROOT/INPUT-example.json" >/dev/null

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/cmd"
cp "$ROOT"/go.mod "$ROOT"/*.go "$TMP"/
cp "$ROOT"/cmd/qualify/main.go "$TMP/cmd/main.go"

cd "$TMP"
gofmt -w ./*.go ./cmd/*.go
go vet ./...
go test ./...
go test -race ./...
go build -o "$TMP/mon038" ./cmd

"$TMP/mon038" --input "$ROOT/INPUT-example.json" --derive-fixture-proof-digests > "$TMP/bound-input.json"
if command -v check-jsonschema >/dev/null 2>&1; then
  check-jsonschema --schemafile "$ROOT/INPUT-v1.schema.json" "$TMP/bound-input.json"
fi

set +e
"$TMP/mon038" --input "$TMP/bound-input.json" > "$TMP/result.json"
STATUS=$?
set -e
[ "$STATUS" -eq 0 ] || { echo "component-qualified offline fixture should exit 0, got $STATUS" >&2; exit 1; }
jq -e '
  .capture_verdict == "qualified"
  and .campaign_verdict == "indeterminate"
  and .production_use_permitted == false
  and .global_order_inferred == false
  and .partition_count == 3
  and .business_event_count == 4
  and .offset_gaps_accepted == 2
' "$TMP/result.json" >/dev/null

jq '.unexpected_field = true' "$TMP/bound-input.json" > "$TMP/unknown.json"
set +e
"$TMP/mon038" --input "$TMP/unknown.json" >/dev/null 2>&1
UNKNOWN_STATUS=$?
set -e
[ "$UNKNOWN_STATUS" -eq 65 ] || { echo "unknown JSON field was not rejected" >&2; exit 1; }

jq '.mode = "runtime_candidate"' "$TMP/bound-input.json" > "$TMP/runtime.json"
set +e
"$TMP/mon038" --input "$TMP/runtime.json" --derive-fixture-proof-digests >/dev/null 2>&1
RUNTIME_DERIVE_STATUS=$?
set -e
[ "$RUNTIME_DERIVE_STATUS" -eq 65 ] || { echo "runtime fixture digest derivation was not rejected" >&2; exit 1; }

jq '.partition_captures[0].isolation_level = "read_uncommitted"' "$TMP/bound-input.json" > "$TMP/unsafe-unbound.json"
"$TMP/mon038" --input "$TMP/unsafe-unbound.json" --derive-fixture-proof-digests > "$TMP/unsafe.json"
set +e
"$TMP/mon038" --input "$TMP/unsafe.json" > "$TMP/unsafe-result.json"
UNSAFE_STATUS=$?
set -e
[ "$UNSAFE_STATUS" -eq 2 ] || { echo "read_uncommitted mutation must exit blocked (2), got $UNSAFE_STATUS" >&2; exit 1; }
jq -e '.capture_verdict == "blocked" and .production_use_permitted == false' "$TMP/unsafe-result.json" >/dev/null

go test -run '^$' -bench BenchmarkEvaluate10000Records256Partitions -benchmem -count=5
echo "MON-038 gofmt, vet, tests, race, CLI, schema, fail-closed and benchmark checks passed."
