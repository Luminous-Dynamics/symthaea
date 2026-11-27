#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
HEAD_SHA="$(git -C "$REPO" rev-parse HEAD)"
[ -z "$(git -C "$REPO" status --porcelain)" ] || {
  echo "Refusing offline campaign on a dirty worktree" >&2
  exit 73
}

for cmd in go jq sha256sum; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "$cmd is required" >&2; exit 69; }
done

RUN_ID="$(printenv MON039_RUN_ID || true)"
if [ -z "$RUN_ID" ]; then RUN_ID="offline-$(date -u +%Y%m%dT%H%M%SZ)"; fi
case "$RUN_ID" in *[!A-Za-z0-9_-]*|'') echo "invalid run ID" >&2; exit 64 ;; esac
OUT_ROOT="$(printenv MON039_OUT_DIR || true)"
if [ -z "$OUT_ROOT" ]; then OUT_ROOT="$HERE/out"; fi
mkdir -p "$OUT_ROOT"
OUT="$OUT_ROOT/$RUN_ID"
mkdir "$OUT"
trap 'printf "Campaign logs: %s\n" "$OUT"' EXIT
: > "$OUT/steps.jsonl"

step() {
  local name="$1"
  shift
  local rc=0
  "$@" >"$OUT/$name.log" 2>&1 || rc=$?
  local digest
  digest="$(sha256sum "$OUT/$name.log" | awk '{print $1}')"
  jq -n --arg name "$name" \
    --arg status "$([ "$rc" -eq 0 ] && echo passed || echo failed)" \
    --arg log "$name.log" --arg digest "$digest" --argjson rc "$rc" \
    '{name:$name,status:$status,log:$log,log_sha256:$digest,exit_code:$rc}' >> "$OUT/steps.jsonl"
  [ "$rc" -eq 0 ] || echo "FAILED: $name (exit $rc)" >&2
}

step mon037-integration-checks bash "$REPO/benchmarks/frontend/monitoring/sfeb-mon037/campaign/CHECKS.sh"
step mon038-partition-vector-checks bash "$REPO/benchmarks/frontend/monitoring/sfeb-mon038/capture/CHECKS.sh"
step mon039-fault-reduction-checks bash "$HERE/CHECKS.sh"

jq -s --arg run "$RUN_ID" --arg head "$HEAD_SHA" --arg date "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{profile:"mon039-offline-campaign-receipt-v1",run_id:$run,repository_head:$head,captured_at_utc:$date,worktree_clean:true,mode:"offline_only",live_services_started:false,live_faults_injected:false,steps:.,all_passed:(all(.[];.status=="passed"))}' \
  "$OUT/steps.jsonl" > "$OUT/run-receipt.json"
sha256sum "$OUT/run-receipt.json" > "$OUT/SHA256SUMS"
cat "$OUT/run-receipt.json"
jq -e '.all_passed == true' "$OUT/run-receipt.json" >/dev/null
