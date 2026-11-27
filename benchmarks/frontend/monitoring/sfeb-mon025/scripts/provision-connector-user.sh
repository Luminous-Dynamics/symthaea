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
: "${MON025_DB_PASSWORD:?Set MON025_DB_PASSWORD in .env}"
: "${MON025_CONNECTOR_DB_USER:?Set MON025_CONNECTOR_DB_USER in .env}"
: "${MON025_CONNECTOR_DB_PASSWORD:?Set MON025_CONNECTOR_DB_PASSWORD in .env}"
export COMPOSE_PROFILES="pg$MAJOR"

"${DC[@]}" exec -T -e PGPASSWORD="$MON025_DB_PASSWORD" -e MON025_CONNECTOR_DB_PASSWORD="$MON025_CONNECTOR_DB_PASSWORD" "postgres$MAJOR" \
  psql -X -q -v ON_ERROR_STOP=1 -h 127.0.0.1 -U "${MON025_DB_USER:-mon025}" -d "${MON025_DB_NAME:-mon025}" \
  -v connector_user="$MON025_CONNECTOR_DB_USER" -v database_name="${MON025_DB_NAME:-mon025}" <<'SQL'
\getenv connector_password MON025_CONNECTOR_DB_PASSWORD
SELECT format('CREATE ROLE %I', :'connector_user')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'connector_user')
\gexec
ALTER ROLE :"connector_user" WITH LOGIN REPLICATION PASSWORD :'connector_password';
GRANT CONNECT ON DATABASE :"database_name" TO :"connector_user";
GRANT USAGE ON SCHEMA public TO :"connector_user";
GRANT SELECT ON public.ledger TO :"connector_user";
SQL

echo "Dedicated least-privilege connector role provisioned for PostgreSQL $MAJOR."
echo "The role has REPLICATION, CONNECT, schema USAGE and SELECT on the published ledger table; it is not a superuser."
