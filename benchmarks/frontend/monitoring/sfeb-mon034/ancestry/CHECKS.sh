#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
command -v go >/dev/null 2>&1 || { echo "go is required" >&2; exit 69; }
command -v jq >/dev/null 2>&1 || { echo "jq is required" >&2; exit 69; }

jq -e '
  .profile == "mon034-postgres-timeline-ancestry-input-v1"
  and .recovered_timeline_id == .history_timeline_id
  and .recovered_timeline_id > 1
  and (.history_sha256 | test("^[0-9a-fA-F]{64}$"))
  and (.history_text | length > 0)
' "$ROOT/INPUT-example.json" >/dev/null

jq -e '
  .profile == "mon034-postgres-timeline-ancestry-input-v1"
  and .recovered_timeline_id == 1
  and .history_timeline_id == 1
  and .history_sha256 == ""
  and .history_text == ""
' "$ROOT/INPUT-example-tli1.json" >/dev/null

jq -e '
  .properties.profile.const == "mon034-postgres-timeline-ancestry-input-v1"
  and (.allOf | length == 1)
' "$ROOT/INPUT-v1.schema.json" >/dev/null

cd "$ROOT"
gofmt -w ./*.go ./cmd/qualify/*.go
go vet ./...
go test ./...
go test -race ./...
go test -run '^$' -bench 'Benchmark(ParseTimelineHistory1000|EvaluateTimelineChain1000)' -benchmem -count=5
echo "MON-034 JSON, vet, unit, race and benchmark checks passed. No external database service was started."
