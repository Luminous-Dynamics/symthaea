#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/compose/compose.yaml"
ENV_FILE="$ROOT/.env"
PROJECT="${COMPOSE_PROJECT_NAME:-mon025}"
DC=(docker compose --project-name "$PROJECT" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" --project-directory "$ROOT/compose")
MAJOR="${1:-18}"
ACTION="${2:-}"
case "$MAJOR" in 18|19) ;; *) echo "usage: $0 18|19 connect-kill|broker-kill|postgres-kill|connect-restart|broker-restart|postgres-restart" >&2; exit 64 ;; esac
[ "${MON025_ALLOW_FAULT_INJECTION:-}" = "YES" ] || { echo "Refusing fault injection. Set MON025_ALLOW_FAULT_INJECTION=YES for this disposable lab." >&2; exit 77; }
[ -f "$ENV_FILE" ] || { echo ".env is required" >&2; exit 66; }
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a
PROJECT="${COMPOSE_PROJECT_NAME:-mon025}"
DC=(docker compose --project-name "$PROJECT" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" --project-directory "$ROOT/compose")

case "$ACTION" in
  connect-kill) service=connect; operation=kill ;;
  broker-kill) service=kafka; operation=kill ;;
  postgres-kill) service="postgres$MAJOR"; operation=kill ;;
  connect-restart) service=connect; operation=restart ;;
  broker-restart) service=kafka; operation=restart ;;
  postgres-restart) service="postgres$MAJOR"; operation=restart ;;
  *) echo "unsupported fault action: $ACTION" >&2; exit 64 ;;
esac

container_id="$("${DC[@]}" ps -q "$service")"
[ -n "$container_id" ] || { echo "target service is not running: $service" >&2; exit 69; }
labels="$(docker inspect "$container_id" | jq -r '.[0].Config.Labels')"
project_label="$(printf '%s' "$labels" | jq -r '.["com.docker.compose.project"] // empty')"
config_label="$(printf '%s' "$labels" | jq -r '.["com.docker.compose.project.config_files"] // empty')"
[ "$project_label" = "$PROJECT" ] || { echo "container is not in isolated project $PROJECT" >&2; exit 77; }
case "$config_label" in *"$COMPOSE_FILE"*) ;; *) echo "container config label does not identify this MON-025 Compose file" >&2; exit 77 ;; esac

fault_at="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
before_state="$(docker inspect "$container_id" | jq -r '.[0].State.Status')"
case "$operation" in
  kill) "${DC[@]}" kill --signal SIGKILL "$service" ;;
  restart) "${DC[@]}" restart --timeout 0 "$service" ;;
esac
after_state="$(docker inspect "$container_id" | jq -r '.[0].State.Status')"
exit_code="$(docker inspect "$container_id" | jq -r '.[0].State.ExitCode')"
mkdir -p "$ROOT/out"
jq -n --arg action "$ACTION" --arg service "$service" --arg container "$container_id" \
  --arg injected_at_utc "$fault_at" --arg before_state "$before_state" --arg after_state "$after_state" \
  --argjson exit_code "$exit_code" \
  '{profile:"mon025-fault-injection-v1",action:$action,service:$service,container_id:$container,injected_at_utc:$injected_at_utc,before_state:$before_state,observed_state_after_command:$after_state,observed_exit_code:$exit_code}' \
  >>"$ROOT/out/fault-injections.jsonl"
echo "Recorded $ACTION against isolated Compose service $service. Inspect service state and reconcile the side effect before any further mutation."
