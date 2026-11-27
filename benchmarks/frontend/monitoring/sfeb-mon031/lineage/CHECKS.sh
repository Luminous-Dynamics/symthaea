#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
command -v go >/dev/null 2>&1 || { echo "go is required" >&2; exit 69; }
command -v jq >/dev/null 2>&1 || { echo "jq is required" >&2; exit 69; }
jq -e '.profile == "mon031-upstream-recovery-regression-matrix-v1" and (.scenarios|length==8) and ([.scenarios[].id]|unique|length==8) and all(.scenarios[]; .upstream_url|startswith("https://"))' "$ROOT/REGRESSION-MATRIX-v1.json" >/dev/null
jq -e '.profile == "mon031-backup-lineage-input-v1" and (.backups|length>=1)' "$ROOT/INPUT-example.json" >/dev/null
jq -e '.properties.profile.const == "mon031-backup-lineage-input-v1"' "$ROOT/INPUT-v1.schema.json" >/dev/null
jq -e '.properties.profile.const == "mon031-upstream-recovery-regression-matrix-v1"' "$ROOT/REGRESSION-MATRIX-v1.schema.json" >/dev/null
cd "$ROOT"
gofmt -w ./*.go ./cmd/qualify/*.go
go vet ./...
go test ./...
go test -race ./...
echo "MON-031 JSON, Go vet, unit and race checks passed. No external recovery service was started."
