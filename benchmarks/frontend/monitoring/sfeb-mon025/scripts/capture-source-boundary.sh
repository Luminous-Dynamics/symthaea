#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/compose/compose.yaml"
ENV_FILE="$ROOT/.env"
DC=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" --project-directory "$ROOT/compose")

MAJOR="${1:-18}"
OUTPUT="${2:-$ROOT/out/source-boundary-pg$MAJOR.json}"
case "$MAJOR" in 18|19) ;; *) echo "usage: $0 18|19 [output.json]" >&2; exit 64 ;; esac
[ -f "$ENV_FILE" ] || { echo ".env is required" >&2; exit 66; }
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a
: "${MON025_DB_PASSWORD:?Set MON025_DB_PASSWORD in .env}"
export COMPOSE_PROFILES="pg$MAJOR"
mkdir -p "$(dirname "$OUTPUT")"

"${DC[@]}" exec -T -e PGPASSWORD="$MON025_DB_PASSWORD" "postgres$MAJOR" \
  psql -X -h 127.0.0.1 -U "${MON025_DB_USER:-mon025}" -d "${MON025_DB_NAME:-mon025}" \
  -v ON_ERROR_STOP=1 -At -f /lab/postgres/source-boundary.sql >"$OUTPUT"

jq -e 'has("system_identifier") and has("timeline_id") and has("slots") and has("publications")' "$OUTPUT" >/dev/null
sha256sum "$OUTPUT"
