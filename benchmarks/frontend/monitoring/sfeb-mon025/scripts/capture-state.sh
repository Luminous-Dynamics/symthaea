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

"${DC[@]}" exec -T \
  -e PGPASSWORD="$MON025_DB_PASSWORD" \
  "$SERVICE" \
  psql -X -h 127.0.0.1 -U "${MON025_DB_USER:-mon025}" -d "${MON025_DB_NAME:-mon025}" \
  -v ON_ERROR_STOP=1 -At -f /lab/postgres/state-manifest.sql >"$OUTPUT"

jq -e 'type == "object" and has("ledger_digest") and has("schema_digest")' "$OUTPUT" >/dev/null
sha256sum "$OUTPUT"
echo "State manifest written: $OUTPUT"
