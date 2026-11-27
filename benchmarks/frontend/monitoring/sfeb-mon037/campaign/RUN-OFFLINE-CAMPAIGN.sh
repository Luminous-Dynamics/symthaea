#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
HEAD_SHA="$(git -C "$REPO" rev-parse HEAD)"
[ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "Refusing dirty worktree" >&2; exit 73; }
for cmd in go jq sha256sum; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "$cmd required" >&2; exit 69; }
done
RUN_ID="$MON037_RUN_ID"
if [ -z "$RUN_ID" ]; then RUN_ID="offline-$(date -u +%Y%m%dT%H%M%SZ)"; fi
case "$RUN_ID" in *[!A-Za-z0-9_-]*|'') echo "invalid run ID" >&2; exit 64 ;; esac
OUT_ROOT="$MON037_OUT_DIR"
if [ -z "$OUT_ROOT" ]; then OUT_ROOT="$HERE/out"; fi
mkdir -p "$OUT_ROOT"
OUT="$OUT_ROOT/$RUN_ID"
mkdir "$OUT"
trap 'echo "Campaign logs: '"$OUT"'"' EXIT
: > "$OUT/steps.jsonl"

step() {
  local name="$1"
  shift
  local rc=0
  "$@" >"$OUT/$name.log" 2>&1 || rc=$?
  local digest
  digest="$(sha256sum "$OUT/$name.log" | awk '{print $1}')"
  jq -n --arg name "$name" --arg status "$([ "$rc" -eq 0 ] && echo passed || echo failed)" \
    --arg log "$name.log" --arg digest "$digest" --argjson rc "$rc" \
    '{name:$name,status:$status,log:$log,log_sha256:$digest,exit_code:$rc}' >> "$OUT/steps.jsonl"
  [ "$rc" -eq 0 ] || echo "FAILED: $name (exit $rc)" >&2
}

step mon025-static-preflight bash "$REPO/benchmarks/frontend/monitoring/sfeb-mon025/scripts/static-preflight.sh"
step mon024-vet go -C "$REPO/benchmarks/frontend/monitoring/sfeb-mon024/proof-graph" vet ./...
step mon024-tests go -C "$REPO/benchmarks/frontend/monitoring/sfeb-mon024/proof-graph" test ./...
step mon024-race go -C "$REPO/benchmarks/frontend/monitoring/sfeb-mon024/proof-graph" test -race ./...
step mon034-checks bash "$REPO/benchmarks/frontend/monitoring/sfeb-mon034/ancestry/CHECKS.sh"
step mon035-checks bash "$REPO/benchmarks/frontend/monitoring/sfeb-mon035/independence/CHECKS.sh"
step mon036-checks bash "$REPO/benchmarks/frontend/monitoring/sfeb-mon036/continuity/CHECKS.sh"
step mon037-checks bash "$HERE/CHECKS.sh"

jq -s --arg run "$RUN_ID" --arg head "$HEAD_SHA" --arg date "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{profile:"mon037-offline-preflight-receipt-v1",run_id:$run,repository_head:$head,captured_at_utc:$date,worktree_clean:true,mode:"offline_only",live_services_started:false,live_faults_injected:false,steps:.,all_passed:(all(.[];.status=="passed"))}' \
  "$OUT/steps.jsonl" > "$OUT/run-receipt.json"
sha256sum "$OUT/run-receipt.json" > "$OUT/SHA256SUMS"
cat "$OUT/run-receipt.json"
jq -e '.all_passed==true' "$OUT/run-receipt.json" >/dev/null
