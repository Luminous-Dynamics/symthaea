#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/compose/compose.yaml"
ENV_FILE="$ROOT/.env"
DC=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" --project-directory "$ROOT/compose")

MAJOR="${1:-18}"
BATCH_ID="${2:-}"
COUNT="${3:-100}"
case "$MAJOR" in 18|19) ;; *) echo "usage: $0 18|19 BATCH_ID COUNT" >&2; exit 64 ;; esac
case "$BATCH_ID" in *[!A-Za-z0-9_-]*|'') echo "batch ID must use only A-Z a-z 0-9 _ -" >&2; exit 64 ;; esac
case "$COUNT" in *[!0-9]*|'') echo "count must be a positive integer" >&2; exit 64 ;; esac
[ "$COUNT" -gt 0 ] && [ "$COUNT" -le 100000 ] || { echo "count must be in 1..100000" >&2; exit 64; }
[ -f "$ENV_FILE" ] || { echo ".env is required" >&2; exit 66; }
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a
: "${MON025_DB_PASSWORD:?Set MON025_DB_PASSWORD in .env}"
export COMPOSE_PROFILES="pg$MAJOR"

"${DC[@]}" exec -T \
  -e PGPASSWORD="$MON025_DB_PASSWORD" \
  "postgres$MAJOR" \
  psql -X -h 127.0.0.1 -U "${MON025_DB_USER:-mon025}" -d "${MON025_DB_NAME:-mon025}" \
  -v ON_ERROR_STOP=1 -v batch_id="$BATCH_ID" -v batch_size="$COUNT" \
  -f /lab/postgres/seed-batch.sql

echo "Seeded batch=$BATCH_ID expected_count=$COUNT. Capture this source-generated expectation before fault injection."
