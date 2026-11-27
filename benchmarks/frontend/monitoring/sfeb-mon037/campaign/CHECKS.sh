#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
command -v go >/dev/null 2>&1 || { echo "go required" >&2; exit 69; }
command -v jq >/dev/null 2>&1 || { echo "jq required" >&2; exit 69; }
command -v sha256sum >/dev/null 2>&1 || { echo "sha256sum required" >&2; exit 69; }
jq -e '.profile=="mon037-unified-recovery-scenarios-v1" and (.scenarios|length)>=20 and ([.scenarios[].id]|unique|length)==(.scenarios|length)' "$ROOT/SCENARIOS-v1.json" >/dev/null
jq -e '.profile=="mon037-unified-recovery-campaign-input-v1" and .components.mon034_timeline_result.profile=="mon034-postgres-timeline-ancestry-result-v1" and .source_truth_complete==true and .observed_capture_complete==true and .transaction_report_complete==true and .surface_inventory_complete==true and (.source_events[0].data_collection|length)>0' "$ROOT/INPUT-example.json" >/dev/null
verify_manifest() {
  expected="$1"
  file="$2"
  actual="$(sha256sum "$ROOT/$file" | awk '{print $1}')"
  [ "$expected" = "$actual" ] || { echo "manifest digest mismatch: $file" >&2; exit 1; }
}
verify_manifest "$(jq -r '.source_truth_manifest_sha256' "$ROOT/INPUT-example.json")" "fixtures/manifests/source-truth.manifest"
verify_manifest "$(jq -r '.observed_capture_manifest_sha256' "$ROOT/INPUT-example.json")" "fixtures/manifests/observed-kafka-capture.manifest"
verify_manifest "$(jq -r '.transaction_report_sha256' "$ROOT/INPUT-example.json")" "fixtures/manifests/transaction-report.manifest"
verify_manifest "$(jq -r '.surface_inventory_sha256' "$ROOT/INPUT-example.json")" "fixtures/manifests/required-surfaces.manifest"
verify_manifest "da4afcc43078abcea80e62e0c0ca374fde9d9df429b1a8c252f1d5e8a6e1d8b8" "fixtures/manifests/replay-deduplication-proof.manifest"
verify_manifest "9fbb127d2f5020d96a88704e5ae6dd72b6d91b1b8b39a581654142201efedf92" "fixtures/manifests/row-stream-exclusion.manifest"
if command -v check-jsonschema >/dev/null 2>&1; then
  check-jsonschema --schemafile "$ROOT/INPUT-v1.schema.json" "$ROOT/INPUT-example.json"
fi
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
