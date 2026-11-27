#!/usr/bin/env bash
set -euo pipefail
CONNECTOR="$1"
TARGET="$2"
TIMEOUT_SECONDS="$3"
INTERVAL_SECONDS="$4"
OUT_DIR="$5"
CONNECT_URL="$(printenv MON025_CONNECT_URL || true)"
[ -n "$CONNECT_URL" ] || CONNECT_URL="http://127.0.0.1:8083"
case "$CONNECTOR" in *[!A-Za-z0-9_-]*|'') echo "invalid connector name" >&2; exit 64 ;; esac
case "$TARGET" in lsn_events_processed_1|transactional_message) ;; *) echo "invalid target" >&2; exit 64 ;; esac
case "$TIMEOUT_SECONDS" in *[!0-9]*|'') echo "timeout must be a positive integer" >&2; exit 64 ;; esac
case "$INTERVAL_SECONDS" in *[!0-9]*|'') echo "interval must be a positive integer" >&2; exit 64 ;; esac
[ "$TIMEOUT_SECONDS" -gt 0 ] && [ "$INTERVAL_SECONDS" -gt 0 ] || { echo "timeout and interval must be positive" >&2; exit 64; }
[ ! -e "$OUT_DIR" ] || { echo "output directory already exists" >&2; exit 73; }
mkdir -p "$(dirname "$OUT_DIR")"
mkdir "$OUT_DIR"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
elapsed=0
while [ "$elapsed" -lt "$TIMEOUT_SECONDS" ]; do
  if curl -fsS "$CONNECT_URL/connectors/$CONNECTOR/status" >"$tmp/status.json" 2>/dev/null &&
     jq -e '.connector.state == "RUNNING" and ([.tasks[]?.state] | all(. == "RUNNING"))' "$tmp/status.json" >/dev/null &&
     curl -fsS "$CONNECT_URL/connectors/$CONNECTOR/offsets" >"$tmp/offsets.json" 2>/dev/null; then
    rm -f "$tmp/proof.json"
    if bash "$(dirname "$0")/verify-offset-target.sh" "$tmp/offsets.json" "$TARGET" "$tmp/proof.json" >/dev/null 2>&1; then
      cp "$tmp/status.json" "$OUT_DIR/status.json"
      cp "$tmp/offsets.json" "$OUT_DIR/offsets.json"
      cp "$tmp/proof.json" "$OUT_DIR/proof.json"
      captured_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      jq -n --arg connector "$CONNECTOR" --arg target "$TARGET" --arg captured "$captured_at" \
        '{profile:"mon033-offset-target-observation-v1",version:"1.0.0",connector:$connector,
          target:$target,target_reached:true,captured_at:$captured,offsets_mutated:false}' >"$OUT_DIR/observation.json"
      sha256sum "$OUT_DIR"/*.json >"$OUT_DIR/SHA256SUMS"
      echo "Target reached after approximately $elapsed seconds; evidence: $OUT_DIR"
      exit 0
    fi
  fi
  sleep "$INTERVAL_SECONDS"
  elapsed=$((elapsed + INTERVAL_SECONDS))
done
captured_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
jq -n --arg connector "$CONNECTOR" --arg target "$TARGET" --arg captured "$captured_at" \
  '{profile:"mon033-offset-target-observation-v1",version:"1.0.0",connector:$connector,
    target:$target,target_reached:false,verdict:"indeterminate",
    reason:"target_offset_not_observed_before_timeout",captured_at:$captured,offsets_mutated:false}' >"$OUT_DIR/observation.json"
echo "Target was not observed within the bounded window; this is indeterminate, not a pass." >&2
exit 3
