#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
for cmd in go jq sha256sum; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "$cmd required" >&2; exit 69; }
done

jq -e '.profile=="mon037-unified-recovery-scenarios-v1" and (.scenarios|length)>=20 and ([.scenarios[].id]|unique|length)==(.scenarios|length)' "$ROOT/SCENARIOS-v1.json" >/dev/null
jq -e '
 .profile=="mon037-unified-recovery-campaign-input-v1"
 and .source_truth_complete==true
 and .observed_capture_complete==true
 and .transaction_report_complete==true
 and .surface_inventory_complete==true
 and (.source_events|length)==(.observed_events|length)
 and (.source_events[0].data_collection|length)>0
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
go build -o "$TMP/mon037" ./cmd

SOURCE_MANIFEST_SHA="$(sha256sum "$ROOT/fixtures/manifests/source-truth.manifest" | awk '{print $1}')"
KAFKA_MANIFEST_SHA="$(sha256sum "$ROOT/fixtures/manifests/observed-kafka-capture.manifest" | awk '{print $1}')"
TX_MANIFEST_SHA="$(sha256sum "$ROOT/fixtures/manifests/transaction-report.manifest" | awk '{print $1}')"
SURFACE_MANIFEST_SHA="$(sha256sum "$ROOT/fixtures/manifests/required-surfaces.manifest" | awk '{print $1}')"

jq \
  --arg source "$SOURCE_MANIFEST_SHA" \
  --arg kafka "$KAFKA_MANIFEST_SHA" \
  --arg tx "$TX_MANIFEST_SHA" \
  --arg surface "$SURFACE_MANIFEST_SHA" \
  '.source_truth_manifest_sha256=$source
   | .observed_capture_manifest_sha256=$kafka
   | .transaction_report_sha256=$tx
   | .surface_inventory_sha256=$surface' \
  "$ROOT/INPUT-example.json" > "$TMP/unbound-input.json"

"$TMP/mon037" --input "$TMP/unbound-input.json" --derive-fixture-record-digests > "$TMP/input.json"

if command -v check-jsonschema >/dev/null 2>&1; then
  check-jsonschema --schemafile "$ROOT/INPUT-v1.schema.json" "$TMP/input.json"
fi

set +e
"$TMP/mon037" --input "$TMP/input.json" > "$TMP/result.json"
STATUS=$?
set -e
[ "$STATUS" -eq 3 ] || { echo "offline fixture must exit indeterminate (3), got $STATUS" >&2; exit 1; }
jq -e '
 .component_verdict=="qualified"
 and .campaign_verdict=="indeterminate"
 and .traffic_activation_permitted==false
 and .reconciliation.duplicate_replays==0
' "$TMP/result.json" >/dev/null

jq '.unexpected=true' "$TMP/input.json" > "$TMP/unknown.json"
set +e
"$TMP/mon037" --input "$TMP/unknown.json" >/dev/null 2>&1
UNKNOWN_STATUS=$?
set -e
[ "$UNKNOWN_STATUS" -eq 65 ] || { echo "unknown JSON field was not rejected" >&2; exit 1; }

jq '.mode="runtime_candidate"' "$TMP/input.json" > "$TMP/runtime.json"
set +e
"$TMP/mon037" --input "$TMP/runtime.json" --derive-fixture-record-digests >/dev/null 2>&1
RUNTIME_DERIVE_STATUS=$?
set -e
[ "$RUNTIME_DERIVE_STATUS" -eq 65 ] || { echo "runtime-candidate digest derivation was not rejected" >&2; exit 1; }

go test -run '^$' -bench BenchmarkEvaluate1000Events100Transactions -benchmem -count=5
echo "MON-037 format, vet, unit, race, CLI, record-set digest and benchmark checks passed."
