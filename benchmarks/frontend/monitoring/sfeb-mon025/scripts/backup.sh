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
mkdir -p "$ROOT/out"

"${DC[@]}" exec -T --user postgres \
  -e PGPASSWORD="$MON025_DB_PASSWORD" \
  "postgres$MAJOR" \
  pg_basebackup -h 127.0.0.1 -p 5432 -U "${MON025_DB_USER:-mon025}" \
    -D "/backups/$BACKUP_ID" -Fp -Xs -P -c fast

# Verify the backup manifest and WAL included in the base backup.
# This does not verify future archived WAL required for a later PITR target.
"${DC[@]}" exec -T --user postgres "postgres$MAJOR" pg_verifybackup "/backups/$BACKUP_ID"
"${DC[@]}" exec -T --user postgres "postgres$MAJOR" \
  sha256sum "/backups/$BACKUP_ID/backup_manifest" >"$ROOT/out/$BACKUP_ID.backup-manifest.sha256"

echo "Physical base backup and its embedded WAL verified: /backups/$BACKUP_ID"
echo "Future archived WAL coverage must still be sealed and verified before a later PITR target."
