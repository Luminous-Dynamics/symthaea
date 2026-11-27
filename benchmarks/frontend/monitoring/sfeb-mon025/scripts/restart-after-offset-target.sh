#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$1"
ALLOW="$(printenv MON025_ALLOW_OFFSET_RESTART || true)"
[ "$ALLOW" = "YES" ] || { echo "Set MON025_ALLOW_OFFSET_RESTART=YES for a disposable MON-025 lab only" >&2; exit 64; }
[ -s "$RUN_DIR/proof.json" ] || { echo "verified proof.json is required" >&2; exit 66; }
jq -e '.profile == "mon033-postgres-offset-target-v1" and .target_reached == true and .offset.lsn != null' "$RUN_DIR/proof.json" >/dev/null || {
  echo "target proof invalid; refusing restart" >&2; exit 65;
}
[ -f "$ROOT/.env" ] || { echo ".env is required" >&2; exit 66; }
grep -Eq '^COMPOSE_PROJECT_NAME=.+$' "$ROOT/.env" || {
  echo "a unique COMPOSE_PROJECT_NAME is required for the disposable lab" >&2; exit 64;
}
docker compose --env-file "$ROOT/.env" -f "$ROOT/compose/compose.yaml" --project-directory "$ROOT/compose" restart connect
CONNECT_URL="$(printenv MON025_CONNECT_URL || true)"
[ -n "$CONNECT_URL" ] || CONNECT_URL="http://127.0.0.1:8083"
attempt=0
while [ "$attempt" -lt 60 ]; do
  attempt=$((attempt + 1))
  if curl -fsS "$CONNECT_URL/connectors/mon025-pg-source/status" >"$RUN_DIR/restart-status.json" 2>/dev/null &&
     jq -e '.connector.state == "RUNNING" and ([.tasks[]?.state] | all(. == "RUNNING"))' "$RUN_DIR/restart-status.json" >/dev/null; then
    break
  fi
  sleep 1
done
[ "$attempt" -lt 60 ] || { echo "connector did not return to RUNNING within 60 seconds" >&2; exit 3; }
curl -fsS "$CONNECT_URL/connectors/mon025-pg-source/offsets" >"$RUN_DIR/after-restart-offsets.json"
jq -n --slurpfile before "$RUN_DIR/proof.json" --slurpfile after "$RUN_DIR/after-restart-offsets.json" \
  --arg restarted_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{profile:"mon033-offset-restart-observation-v1",version:"1.0.0",
    before_target:$before[0],after_restart_offsets:$after[0],
    restarted_at:$restarted_at,connect_service_restarted:true,
    independent_cdc_reconciliation_pending:true}' >"$RUN_DIR/restart-observation.json"
sha256sum "$RUN_DIR"/restart-*.json "$RUN_DIR/after-restart-offsets.json" >"$RUN_DIR/restart-SHA256SUMS"
echo "Connect restarted and connector recovered. This is not a recovery pass until SQL/CDC reconciliation completes."
