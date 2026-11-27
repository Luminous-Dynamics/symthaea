#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/compose/compose.yaml"
ENV_FILE="$ROOT/.env"
DC=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" --project-directory "$ROOT/compose")

usage() {
  cat <<'USAGE'
Usage:
  scripts/labctl.sh preflight
  scripts/labctl.sh up 18|19
  scripts/labctl.sh status 18|19
  scripts/labctl.sh stop 18|19
  scripts/labctl.sh down
  scripts/labctl.sh restore 18|19 BACKUP_ID HEX/HEX TIMELINE_ID
  scripts/labctl.sh clean

clean removes all MON-025 named volumes and requires MON025_ALLOW_VOLUME_DELETION=YES.
USAGE
}

require_major() {
  case "${1:-}" in 18|19) ;; *) echo "PostgreSQL major must be 18 or 19" >&2; exit 64 ;; esac
}

preflight() {
  for cmd in docker jq curl date; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "required command missing: $cmd" >&2; exit 69; }
  done
  [ -f "$ENV_FILE" ] || { echo "copy .env.example to .env and set MON025_DB_PASSWORD" >&2; exit 66; }
  # shellcheck disable=SC1090
  set -a
  . "$ENV_FILE"
  set +a
  [ -n "${MON025_DB_PASSWORD:-}" ] || { echo "MON025_DB_PASSWORD must be set" >&2; exit 64; }
  docker compose version
  docker info >/dev/null
  "${DC[@]}" config --quiet
}

case "${1:-}" in
  preflight)
    preflight
    echo "MON-025 local preflight passed. No containers were started."
    ;;
  up)
    require_major "${2:-}"
    preflight
    export COMPOSE_PROFILES="pg${2}"
    "${DC[@]}" up -d "postgres${2}" kafka connect
    "${DC[@]}" ps
    ;;
  status)
    require_major "${2:-}"
    export COMPOSE_PROFILES="pg${2}"
    "${DC[@]}" ps
    ;;
  stop)
    require_major "${2:-}"
    export COMPOSE_PROFILES="pg${2},restore${2}"
    "${DC[@]}" stop "postgres${2}" "restore${2}" connect kafka
    ;;
  restore)
    require_major "${2:-}"
    [ "$#" -eq 5 ] || { usage; exit 64; }
    backup_id="$3"
    target_lsn="$4"
    target_timeline="$5"
    case "$backup_id" in *[!A-Za-z0-9_-]*|'') echo "invalid backup ID" >&2; exit 64 ;; esac
    [[ "$target_lsn" =~ ^[0-9A-Fa-f]{1,8}/[0-9A-Fa-f]{1,8}$ ]] || { echo "invalid PostgreSQL LSN" >&2; exit 64; }
    [[ "$target_timeline" =~ ^[1-9][0-9]*$ ]] || { echo "invalid recovery timeline" >&2; exit 64; }
    preflight
    export COMPOSE_PROFILES="pg${2},restore${2}"
    export MON025_BACKUP_ID="$backup_id"
    export MON025_RECOVERY_TARGET_LSN="$target_lsn"
    export MON025_RECOVERY_TARGET_TIMELINE="$target_timeline"
    "${DC[@]}" up -d "postgres${2}" "restore${2}"
    "${DC[@]}" ps
    ;;
  down)
    [ -f "$ENV_FILE" ] || { echo ".env is required" >&2; exit 66; }
    "${DC[@]}" down
    ;;
  clean)
    [ "${MON025_ALLOW_VOLUME_DELETION:-}" = "YES" ] || {
      echo "Refusing volume deletion. Set MON025_ALLOW_VOLUME_DELETION=YES explicitly." >&2
      exit 77
    }
    [ -f "$ENV_FILE" ] || { echo ".env is required" >&2; exit 66; }
    "${DC[@]}" down --volumes
    ;;
  *)
    usage
    exit 64
    ;;
esac
