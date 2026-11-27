#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
command -v go >/dev/null 2>&1 || { echo "go required" >&2; exit 69; }
command -v jq >/dev/null 2>&1 || { echo "jq required" >&2; exit 69; }
command -v sha256sum >/dev/null 2>&1 || { echo "sha256sum required" >&2; exit 69; }

jq -e '.profile=="mon037-unified-recovery-scenarios-v1" and (.scenarios|length)>=20 and ([.scenarios[].id]|unique|length)==(.scenarios|length)' "$ROOT/SCENARIOS-v1.json" >/dev/null

SOURCE_MANIFEST_SHA="$(sha256sum "$ROOT/fixtures/manifests/source-truth.manifest" | awk '{print $1}')"
KAFKA_MANIFEST_SHA="$(sha256sum "$ROOT/fixtures/manifests/observed-kafka-capture.manifest" | awk '{print $1}')"
TX_MANIFEST_SHA="$(sha256sum "$ROOT/fixtures/manifests/transaction-report.manifest" | awk '{print $1}')"
SURFACE_MANIFEST_SHA="$(sha256sum "$ROOT/fixtures/manifests/required-surfaces.manifest" | awk '{print $1}')"
DEDUPE_MANIFEST_SHA="$(sha256sum "$ROOT/fixtures/manifests/replay-deduplication-proof.manifest" | awk '{print $1}')"
EXCLUSION_MANIFEST_SHA="$(sha256sum "$ROOT/fixtures/manifests/row-stream-exclusion.manifest" | awk '{print $1}')"

INPUT="$(mktemp)"
BIN="$(mktemp)"
OUT="$(mktemp)"
trap 'rm -f "$INPUT" "$BIN" "$OUT"' EXIT

jq \
  --arg source "$SOURCE_MANIFEST_SHA" \
  --arg kafka "$KAFKA_MANIFEST_SHA" \
  --arg tx "$TX_MANIFEST_SHA" \
  --arg surface "$SURFACE_MANIFEST_SHA" \
  '.source_truth_manifest_sha256=$source
   | .observed_capture_manifest_sha256=$kafka
   | .transaction_report_sha256=$tx
   | .surface_inventory_sha256=$surface' \
  "$ROOT/INPUT-example.json" > "$INPUT"

jq -e '
  .profile=="mon037-unified-recovery-campaign-input-v1"
  and .source_truth_complete==true
  and .observed_capture_complete==true
  and .transaction_report_complete==true
  and .surface_inventory_complete==true
  and (.source_events|length)==(.observed_events|length)
  and (.source_events[0].data_collection|length)>0
  and (.transactions[0].data_collections|has("public.ledger"))
' "$INPUT" >/dev/null

cd "$ROOT"
gofmt -w ./*.go ./cmd/qualify/*.go
go vet ./...
go test ./...
go test -race ./...
go build -o "$BIN" ./cmd/qualify
"$BIN" --input "$INPUT" > "$OUT"
jq -e '
  .component_verdict=="qualified"
  and .campaign_verdict=="indeterminate"
  and .traffic_activation_permitted==false
  and (.reconciliation.duplicate_replays==0)
' "$OUT" >/dev/null

# CLI parser is fail-closed on unknown fields and malformed JSON.
printf '{"unexpected":true}\n' > "$INPUT"
set +e
"$BIN" --input "$INPUT" >/dev/null 2>&1
UNKNOWN_STATUS=$?
set -e
[ "$UNKNOWN_STATUS" -eq 65 ] || { echo "unknown JSON field was not rejected" >&2; exit 1; }

go test -run '^$' -bench BenchmarkEvaluate1000Events100Transactions -benchmem -count=5
echo "MON-037 format, vet, unit, race, CLI, manifest and benchmark checks passed."
