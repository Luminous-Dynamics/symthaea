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

jq -n --arg batch "$BATCH_ID" --argjson expected "$EXPECTED_COUNT" --slurpfile rows "$rows" --slurpfile tx "$tx" '
  [ $rows[0][]
    | select((.payload.after? // null) != null and .payload.after.batch_id == $batch)
    | {
        event_id: .payload.after.event_id,
        account_id: .payload.after.account_id,
        amount: .payload.after.amount,
        transaction_id: (.payload.transaction.id // null),
        total_order: ((.payload.transaction.total_order // null) | if type == "string" then tonumber else . end),
        data_collection_order: ((.payload.transaction.data_collection_order // null) | if type == "string" then tonumber else . end)
      }
  ] as $raw_events
  | ($raw_events | group_by(.event_id)) as $event_groups
  | [ $event_groups[] | select((map(tojson) | unique | length) > 1) | .[0].event_id ] as $conflicts
  | [ $event_groups[] | .[0] ] as $events
  | [ $tx[0][] | .payload | select(.status? == "BEGIN") ] as $begins
  | [ $tx[0][] | .payload | select(.status? == "END") ] as $ends
  | ($events | group_by(.transaction_id)) as $groups
  | [ $groups[]
      | . as $group
      | ($group[0].transaction_id) as $tid
      | ($tid | if type == "string" then split(":")[0] else "" end) as $xid
      | ([$ends[] | select(.id == $tid)]) as $end
      | ([$begins[] | select((.id | split(":")[0]) == $xid)]) as $begin
      | ($group | map(.total_order) | sort) as $orders
      | ($group | map(.data_collection_order) | sort) as $collection_orders
      | {
          transaction_id: $tid,
          row_event_count: ($group | length),
          begin_records: ($begin | length),
          end_records: ($end | length),
          announced_event_count: (if ($end | length) == 1 then ($end[0].event_count | if type == "string" then tonumber else . end) else null end),
          total_order: $orders,
          data_collection_order: $collection_orders,
          boundary_pair_complete: (($begin | length) == 1 and ($end | length) == 1),
          event_count_matches: (($end | length) == 1 and (($end[0].event_count | if type == "string" then tonumber else . end) == ($group | length))),
          total_order_contiguous: ($orders == ([range(1; ($group | length) + 1)])),
          collection_order_contiguous: ($collection_orders == ([range(1; ($group | length) + 1)]))
        }
    ] as $checks
  | {
      profile:"mon025-transaction-closure-v1",
      batch_id:$batch,
      expected_row_count:$expected,
      observed_raw_messages:($raw_events | length),
      observed_unique_event_ids:($events | length),
      exact_duplicate_replays:(($raw_events | length) - ($events | length)),
      conflicting_event_ids:$conflicts,
      events_without_transaction_id:([$events[] | select(.transaction_id == null)] | length),
      transaction_groups:$checks,
      complete:(
        (($events | length) == $expected)
        and (($conflicts | length) == 0)
        and (all($events[]; .transaction_id != null))
        and (all($checks[]; .boundary_pair_complete and .event_count_matches and .total_order_contiguous and .collection_order_contiguous))
      )
    }' >"$report"

jq . "$report"
sha256sum "$report"
jq -e '.complete == true' "$report" >/dev/null || {
  echo "Transaction closure is incomplete or unproven. Do not qualify the source interval." >&2
  exit 1
}
echo "Transaction BEGIN/END boundaries, END event_count and per-event order match the captured batch."
