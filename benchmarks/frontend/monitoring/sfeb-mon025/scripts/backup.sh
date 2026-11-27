#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/compose/compose.yaml"
ENV_FILE="$ROOT/.env"
DC=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" --project-directory "$ROOT/compose")

MAJOR="${1:-18}"
BACKUP_ID="${2:-}"
case "$MAJOR" in 18|19) ;; *) echo "usage: $0 18|19 BACKUP_ID" >&2; exit 64 ;; esac
case "$BACKUP_ID" in *[!A-Za-z0-9_-]*|'') echo "backup ID must use only A-Z a-z 0-9 _ -" >&2; exit 64 ;; esac
[ -f "$ENV_FILE" ] || { echo ".env is required" >&2; exit 66; }
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a
: "${MON025_DB_PASSWORD:?Set MON025_DB_PASSWORD in .env}"
export COMPOSE_PROFILES="pg$MAJOR"

"${DC[@]}" exec -T --user postgres \
  -e PGPASSWORD="$MON025_DB_PASSWORD" \
  "postgres$MAJOR" \
  pg_basebackup -h 127.0.0.1 -p 5432 -U "${MON025_DB_USER:-mon025}" \
    -D "/backups/$BACKUP_ID" -Fp -Xs -P -c fast

echo "Physical base backup created at /backups/$BACKUP_ID in the named backup volume."
echo "Capture the requested target timestamp and SQL truth manifest independently; archive coverage must be checked before restore."
