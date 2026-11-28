#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/compose/compose.yaml"
ENV_FILE="$ROOT/.env"
DC=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" --project-directory "$ROOT/compose")
MAJOR="${1:-18}"
BATCH_ID="${2:-}"
EXPECTED_COUNT="${3:-}"
case "$MAJOR" in 18|19) ;; *) echo "usage: $0 18|19 BATCH_ID EXPECTED_ROW_COUNT" >&2; exit 64 ;; esac
case "$BATCH_ID" in *[!A-Za-z0-9_-]*|'') echo "invalid batch ID" >&2; exit 64 ;; esac
case "$EXPECTED_COUNT" in *[!0-9]*|'') echo "expected row count must be a positive integer" >&2; exit 64 ;; esac
[ "$EXPECTED_COUNT" -gt 0 ] || { echo "expected row count must be positive" >&2; exit 64; }
[ -f "$ENV_FILE" ] || { echo ".env is required" >&2; exit 66; }
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a
CONNECT_URL="${MON025_CONNECT_URL:-http://127.0.0.1:8083}"
TOPIC_PREFIX="mon025_pg$MAJOR"
DATA_TOPIC="$TOPIC_PREFIX.public.ledger"
MESSAGE_TOPIC="$TOPIC_PREFIX.message"
TX_TOPIC="$TOPIC_PREFIX.transaction"
MESSAGE_PREFIX="mon025_transaction"
curl -fsS "$CONNECT_URL/connectors/mon025-pg-source/status" |
  jq -e '.connector.state == "RUNNING" and ([.tasks[]?.state] | all(. == "RUNNING"))' >/dev/null

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
rows="$tmp/rows.jsonl"
messages="$tmp/messages.jsonl"
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
assert_single_partition() {
  local topic="$1" description count
  if ! description="$("${DC[@]}" exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:9092 --describe --topic "$topic")"; then
    echo "cannot describe required topic $topic" >&2
    return 65
  fi
  count="$(printf '%s\n' "$description" | grep -oE 'PartitionCount: [0-9]+' | head -n1 | awk '{print $2}' || true)"
  [ "$count" = "1" ] || {
    echo "topic $topic must have exactly one partition for this profile's order adjudication (observed: ${count:-unknown})" >&2
    return 65
  }
}
for topic in "$DATA_TOPIC" "$MESSAGE_TOPIC" "$TX_TOPIC"; do
  assert_single_partition "$topic"
done
consume_topic "$DATA_TOPIC" "$rows"
consume_topic "$MESSAGE_TOPIC" "$messages"
consume_topic "$TX_TOPIC" "$tx"
jq -n --arg batch "$BATCH_ID" --arg message_prefix "$MESSAGE_PREFIX" --argjson expected "$EXPECTED_COUNT" \
  --slurpfile rows "$rows" --slurpfile messages "$messages" --slurpfile tx "$tx" \
  -f "$ROOT/scripts/transaction-metadata-report.jq" >"$report"
jq . "$report"
sha256sum "$report"
jq -e '.complete == true' "$report" >/dev/null || {
  echo "Multi-topic transaction closure is incomplete or unproven. Do not qualify the source interval." >&2
  exit 1
}
echo "Row and transactional logical-message events reconcile to BEGIN/END boundaries, event_count, data_collections, total_order and per-collection order."
