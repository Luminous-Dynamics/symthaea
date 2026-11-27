#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/compose/compose.yaml"
ENV_FILE="$ROOT/.env"
DC=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" --project-directory "$ROOT/compose")

MAJOR="${1:-18}"
case "$MAJOR" in 18|19) ;; *) echo "usage: $0 18|19 [output.json]" >&2; exit 64 ;; esac
OUTPUT="${2:-$ROOT/out/wal-archive-pg$MAJOR.json}"
[ -f "$ENV_FILE" ] || { echo ".env is required" >&2; exit 66; }
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a
: "${MON025_DB_PASSWORD:?Set MON025_DB_PASSWORD in .env}"
export COMPOSE_PROFILES="pg$MAJOR"
mkdir -p "$(dirname "$OUTPUT")"

segment="$("${DC[@]}" exec -T -e PGPASSWORD="$MON025_DB_PASSWORD" "postgres$MAJOR" \
  psql -X -q -A -t -h 127.0.0.1 -U "${MON025_DB_USER:-mon025}" -d "${MON025_DB_NAME:-mon025}" \
  -v ON_ERROR_STOP=1 -c "SELECT pg_walfile_name(pg_current_wal_lsn())")"
switch_lsn="$("${DC[@]}" exec -T -e PGPASSWORD="$MON025_DB_PASSWORD" "postgres$MAJOR" \
  psql -X -q -A -t -h 127.0.0.1 -U "${MON025_DB_USER:-mon025}" -d "${MON025_DB_NAME:-mon025}" \
  -v ON_ERROR_STOP=1 -c "SELECT pg_switch_wal()::text")"
archive_dir="/var/lib/postgresql/$MAJOR/docker/mon025-archive"
archived=false
for attempt in $(seq 1 60); do
  if "${DC[@]}" exec -T "postgres$MAJOR" sh -c "test -s '$archive_dir/$segment'"; then
    archived=true
    break
  fi
  sleep 1
done
[ "$archived" = true ] || { echo "WAL segment did not reach archive: $segment" >&2; exit 1; }

stats="$("${DC[@]}" exec -T -e PGPASSWORD="$MON025_DB_PASSWORD" "postgres$MAJOR" \
  psql -X -q -A -t -h 127.0.0.1 -U "${MON025_DB_USER:-mon025}" -d "${MON025_DB_NAME:-mon025}" \
  -v ON_ERROR_STOP=1 -c "SELECT jsonb_build_object('last_archived_wal', last_archived_wal, 'last_archived_time', last_archived_time, 'failed_count', failed_count, 'last_failed_wal', last_failed_wal, 'last_failed_time', last_failed_time)::text FROM pg_stat_archiver")"
jq -n --arg segment "$segment" --arg switch_lsn "$switch_lsn" --argjson stats "$stats" \
  '{profile:"mon025-wal-archive-seal-v1",segment_sealed:$segment,switch_lsn:$switch_lsn,archiver:$stats}' >"$OUTPUT"
sha256sum "$OUTPUT"
echo "Archived WAL segment observed: $segment"
echo "This seals one current segment only; it does not prove the complete backup-to-target WAL interval. The restore must consume the requested target and fail on any missing segment."
