#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/compose/compose.yaml"
ENV_FILE="$ROOT/.env"
DC=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" --project-directory "$ROOT/compose")

MAJOR="${1:-18}"
case "$MAJOR" in 18|19) ;; *) echo "usage: $0 18|19" >&2; exit 64 ;; esac
[ -f "$ENV_FILE" ] || { echo ".env is required" >&2; exit 66; }
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a
: "${MON025_CONNECTOR_DB_USER:?Set MON025_CONNECTOR_DB_USER in .env}"
: "${MON025_CONNECTOR_DB_PASSWORD:?Set MON025_CONNECTOR_DB_PASSWORD in .env}"
CONNECT_URL="${MON025_CONNECT_URL:-http://127.0.0.1:8083}"
PG_SERVICE="postgres$MAJOR"
CONNECTOR_NAME="mon025-pg-source"
SLOT_NAME="mon025_slot_pg$MAJOR"
TOPIC_PREFIX="mon025_pg$MAJOR"

curl -fsS "$CONNECT_URL/" >/dev/null
config="$(jq -n \
  --arg host "$PG_SERVICE" \
  --arg user "$MON025_CONNECTOR_DB_USER" \
  --arg password "$MON025_CONNECTOR_DB_PASSWORD" \
  --arg database "${MON025_DB_NAME:-mon025}" \
  --arg slot "$SLOT_NAME" \
  --arg topic "$TOPIC_PREFIX" \
  '{
    "connector.class":"io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname":$host,
    "database.port":"5432",
    "database.user":$user,
    "database.password":$password,
    "database.dbname":$database,
    "topic.prefix":$topic,
    "plugin.name":"pgoutput",
    "slot.name":$slot,
    "publication.name":"mon025_pub",
    "publication.autocreate.mode":"disabled",
    "table.include.list":"public.ledger",
    "snapshot.mode":"initial",
    "tombstones.on.delete":"false",
    "provide.transaction.metadata":"true",
    "heartbeat.interval.ms":"1000",
    "decimal.handling.mode":"string",
    "include.schema.changes":"true"
  }')"

response="$(mktemp)"
trap 'rm -f "$response"' EXIT
curl -fsS -X PUT \
  -H 'Content-Type: application/json' \
  --data "$config" \
  "$CONNECT_URL/connectors/$CONNECTOR_NAME/config" >"$response"
jq -e '.["connector.class"] == "io.debezium.connector.postgresql.PostgresConnector" and .["slot.name"] == "'"$SLOT_NAME"'"' "$response" >/dev/null
echo "Registered $CONNECTOR_NAME against $PG_SERVICE. Connector task state still requires independent inspection."
curl -fsS "$CONNECT_URL/connectors/$CONNECTOR_NAME/status" | jq .
