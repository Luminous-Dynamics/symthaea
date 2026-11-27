#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/compose/compose.yaml"
ENV_FILE="$ROOT/.env"
DC=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" --project-directory "$ROOT/compose")

MAJOR="${1:-18}"
BATCH_ID="${2:-}"
EXPECTED_COUNT="${3:-}"
case "$MAJOR" in 18|19) ;; *) echo "usage: $0 18|19 BATCH_ID EXPECTED_COUNT" >&2; exit 64 ;; esac
case "$BATCH_ID" in *[!A-Za-z0-9_-]*|'') echo "invalid batch ID" >&2; exit 64 ;; esac
case "$EXPECTED_COUNT" in *[!0-9]*|'') echo "expected count must be a positive integer" >&2; exit 64 ;; esac
[ "$EXPECTED_COUNT" -gt 0 ] || { echo "expected count must be positive" >&2; exit 64; }
[ -f "$ENV_FILE" ] || { echo ".env is required" >&2; exit 66; }
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a
CONNECT_URL="${MON025_CONNECT_URL:-http://127.0.0.1:8083}"
TOPIC_PREFIX="mon025_pg$MAJOR"
DATA_TOPIC="$TOPIC_PREFIX.public.ledger"
TX_TOPIC="$TOPIC_PREFIX.transaction"

curl -fsS "$CONNECT_URL/connectors/mon025-pg-source/status" |
  jq -e '.connector.state == "RUNNING" and ([.tasks[]?.state] | all(. == "RUNNING"))' >/dev/null

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
rows="$tmp/rows.jsonl"
tx="$tmp/transactions.jsonl"
report="$tmp/transaction-report.json"

consume_topic() {
  local topic="$1" output="$2"
  set +e
  "${DC[@]}" exec -T kafka /opt/kafka/bin/kafka-console-consumer.sh \
    --bootstrap-server kafka:9092 --topic "$topic" --from-beginning --timeout-ms 15000 >"$output"
  local rc=$?
  set -e
  if [ "$rc" -gt 1 ]; then
    echo "Kafka consumer failed for $topic with status $rc" >&2
    return "$rc"
  fi
}
consume_topic "$DATA_TOPIC" "$rows"
consume_topic "$TX_TOPIC" "$tx"

jq -n --arg batch "$BATCH_ID" --argjson expected "$EXPECTED_COUNT" --slurpfile rows "$rows" --slurpfile tx "$tx" \
  -f "$ROOT/scripts/transaction-metadata-report.jq" >"$report"

jq . "$report"
sha256sum "$report"
jq -e '.complete == true' "$report" >/dev/null || {
  echo "Transaction closure is incomplete or unproven. Do not qualify the source interval." >&2
  exit 1
}
echo "Transaction BEGIN/END boundaries, END event_count and per-event order match the captured batch."
