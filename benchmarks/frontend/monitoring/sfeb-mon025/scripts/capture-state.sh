#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/compose/compose.yaml"
ENV_FILE="$ROOT/.env"
DC=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" --project-directory "$ROOT/compose")

MAJOR="${1:-18}"
SERVICE="${2:-postgres$MAJOR}"
OUTPUT="${3:-$ROOT/out/$SERVICE-state.json}"
case "$MAJOR" in 18|19) ;; *) echo "usage: $0 18|19 [service] [output.json]" >&2; exit 64 ;; esac
[ -f "$ENV_FILE" ] || { echo ".env is required" >&2; exit 66; }
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a
: "${MON025_DB_PASSWORD:?Set MON025_DB_PASSWORD in .env}"
mkdir -p "$(dirname "$OUTPUT")"
case "$SERVICE" in
  restore18) export COMPOSE_PROFILES="pg18,restore18" ;;
  restore19) export COMPOSE_PROFILES="pg19,restore19" ;;
  *) export COMPOSE_PROFILES="pg$MAJOR" ;;
esac

capture="$("${DC[@]}" exec -T \
  -e PGPASSWORD="$MON025_DB_PASSWORD" \
  "$SERVICE" \
  psql -X -q -A -t -h 127.0.0.1 -U "${MON025_DB_USER:-mon025}" -d "${MON025_DB_NAME:-mon025}" \
  -v ON_ERROR_STOP=1 \
  -c "BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY" \
  -c "SELECT jsonb_build_object('database', current_database(), 'captured_at', transaction_timestamp()::text, 'capture_lsn', CASE WHEN pg_is_in_recovery() THEN pg_last_wal_replay_lsn()::text ELSE pg_current_wal_lsn()::text END, 'system_identifier', (SELECT system_identifier::text FROM pg_control_system()), 'timeline_id', (SELECT timeline_id::text FROM pg_control_checkpoint()), 'is_in_recovery', pg_is_in_recovery())::text" \
  -f /lab/postgres/state-manifest.sql \
  -c "COMMIT")"

metadata="$(printf '%s\n' "$capture" | sed -n '1p')"
state="$(printf '%s\n' "$capture" | sed -n '2p')"
[ -n "$metadata" ] && [ -n "$state" ] || {
  echo "capture did not produce the expected metadata and state lines" >&2
  printf '%s\n' "$capture" >&2
  exit 65
}
jq -n --argjson metadata "$metadata" --argjson state "$state" '{profile:"mon025-state-capture-v1",metadata:$metadata,state:$state}' >"$OUTPUT"
jq -e '.metadata.capture_lsn != null and .metadata.timeline_id != null and (.state | has("ledger_digest"))' "$OUTPUT" >/dev/null
sha256sum "$OUTPUT"
echo "Quiesce all fixture writers before capture. Use metadata.capture_lsn and metadata.timeline_id as the exact PITR boundary."
echo "State manifest written: $OUTPUT"
