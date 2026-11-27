#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/compose/compose.yaml"
ENV_FILE="$ROOT/.env"
DC=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" --project-directory "$ROOT/compose")

MAJOR="${1:-18}"
SERVICE="${2:-restore$MAJOR}"
TARGET_LSN="${3:-}"
TARGET_TIMELINE="${4:-}"
TIMEOUT_SECONDS="${5:-180}"
case "$MAJOR" in 18|19) ;; *) echo "usage: $0 18|19 restore18|restore19 HEX/HEX TIMELINE_ID [timeout_seconds]" >&2; exit 64 ;; esac
[[ "$SERVICE" == "restore18" || "$SERVICE" == "restore19" ]] || { echo "service must be a restore target" >&2; exit 64; }
[[ "$TARGET_LSN" =~ ^[0-9A-Fa-f]{1,8}/[0-9A-Fa-f]{1,8}$ ]] || { echo "invalid target LSN" >&2; exit 64; }
[[ "$TARGET_TIMELINE" =~ ^[1-9][0-9]*$ ]] || { echo "invalid target timeline" >&2; exit 64; }
target_timeline_hex="$(printf "%08X" "$TARGET_TIMELINE")"
[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "timeout must be a positive integer" >&2; exit 64; }
[ -f "$ENV_FILE" ] || { echo ".env is required" >&2; exit 66; }
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a
: "${MON025_DB_PASSWORD:?Set MON025_DB_PASSWORD in .env}"
if [ "$SERVICE" = "restore18" ]; then export COMPOSE_PROFILES="pg18,restore18"; else export COMPOSE_PROFILES="pg19,restore19"; fi

for ((attempt=1; attempt<=TIMEOUT_SECONDS; attempt++)); do
  if "${DC[@]}" exec -T "$SERVICE" pg_isready -U "${MON025_DB_USER:-mon025}" -d "${MON025_DB_NAME:-mon025}" >/dev/null 2>&1; then
    ready="$("${DC[@]}" exec -T -e PGPASSWORD="$MON025_DB_PASSWORD" "$SERVICE" \
      psql -X -q -A -t -h 127.0.0.1 -U "${MON025_DB_USER:-mon025}" -d "${MON025_DB_NAME:-mon025}" \
      -v ON_ERROR_STOP=1 -c "SELECT (pg_is_in_recovery() AND pg_last_wal_replay_lsn() IS NOT NULL AND pg_last_wal_replay_lsn() >= '$TARGET_LSN'::pg_lsn AND substring(pg_walfile_name(pg_last_wal_replay_lsn()) from 1 for 8) = '$target_timeline_hex')::text")" || true
    if [ "$ready" = "true" ]; then
      observed="$("${DC[@]}" exec -T -e PGPASSWORD="$MON025_DB_PASSWORD" "$SERVICE" \
        psql -X -q -A -t -h 127.0.0.1 -U "${MON025_DB_USER:-mon025}" -d "${MON025_DB_NAME:-mon025}" \
        -v ON_ERROR_STOP=1 -c "SELECT jsonb_build_object('in_recovery',pg_is_in_recovery(),'replay_lsn',pg_last_wal_replay_lsn()::text,'replay_time',pg_last_xact_replay_timestamp()::text)::text")"
      printf '%s\n' "$observed"
      echo "Restore target reached or passed the requested LSN on its pinned timeline and remains in recovery pause."
      exit 0
    fi
  fi
  sleep 1
done
echo "Restore did not reach the requested LSN within $TIMEOUT_SECONDS seconds. Do not enable traffic; inspect PostgreSQL logs and archive continuity." >&2
exit 1
