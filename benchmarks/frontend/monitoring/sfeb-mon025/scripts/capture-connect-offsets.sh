#!/usr/bin/env bash
set -euo pipefail
RUN_ID="$1"
OUT_DIR="$2"
CONNECTOR="mon025-pg-source"
CONNECT_URL="$(printenv MON025_CONNECT_URL || true)"
[ -n "$CONNECT_URL" ] || CONNECT_URL="http://127.0.0.1:8083"
case "$RUN_ID" in *[!A-Za-z0-9_-]*|'') echo "invalid run ID" >&2; exit 64 ;; esac
[ -n "$OUT_DIR" ] || { echo "output directory required" >&2; exit 64; }
[ ! -e "$OUT_DIR" ] || { echo "output path already exists" >&2; exit 73; }
mkdir -p "$(dirname "$OUT_DIR")"
mkdir "$OUT_DIR"
curl -fsS "$CONNECT_URL/connectors/$CONNECTOR/status" >"$OUT_DIR/status.json"
curl -fsS "$CONNECT_URL/connectors/$CONNECTOR/offsets" >"$OUT_DIR/offsets.json"
jq -e 'type == "object" and (.offsets | type == "array") and (.offsets | length == 1)' "$OUT_DIR/offsets.json" >/dev/null || {
  echo "expected exactly one source partition in Kafka Connect offsets response" >&2; exit 65;
}
captured_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
digest="$(sha256sum "$OUT_DIR/offsets.json" | awk '{print $1}')"
jq -n --arg run "$RUN_ID" --arg captured "$captured_at" --arg digest "$digest" \
  --slurpfile status "$OUT_DIR/status.json" --slurpfile offsets "$OUT_DIR/offsets.json" \
  '{profile:"mon033-connect-offset-capture-v1",version:"1.0.0",run_id:$run,
    captured_at:$captured,read_only:true,offsets_sha256:$digest,
    connector_status:$status[0],source_partition_count:($offsets[0].offsets|length)}' >"$OUT_DIR/capture.json"
sha256sum "$OUT_DIR/status.json" "$OUT_DIR/offsets.json" "$OUT_DIR/capture.json" >"$OUT_DIR/SHA256SUMS"
jq -e '.connector.state == "RUNNING" and ([.tasks[]?.state] | all(. == "RUNNING"))' "$OUT_DIR/status.json" >/dev/null || {
  echo "offsets captured, but connector/task was not RUNNING; not a qualified observation" >&2
  exit 3
}
echo "Read-only offset capture written to $OUT_DIR"
