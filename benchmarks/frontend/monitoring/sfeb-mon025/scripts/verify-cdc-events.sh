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
: "${MON025_DB_PASSWORD:?Set MON025_DB_PASSWORD in .env}"
CONNECT_URL="${MON025_CONNECT_URL:-http://127.0.0.1:8083}"
TOPIC="mon025_pg$MAJOR.public.ledger"
export COMPOSE_PROFILES="pg$MAJOR"

curl -fsS "$CONNECT_URL/connectors/mon025-pg-source/status" |
  jq -e '.connector.state == "RUNNING" and ([.tasks[]?.state] | all(. == "RUNNING"))' >/dev/null

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
expected="$tmp/expected.json"
raw="$tmp/topic.jsonl"
result="$tmp/result.json"

expected_json="$("${DC[@]}" exec -T -e PGPASSWORD="$MON025_DB_PASSWORD" "postgres$MAJOR" \
  psql -X -q -A -t -h 127.0.0.1 -U "${MON025_DB_USER:-mon025}" -d "${MON025_DB_NAME:-mon025}" \
  -v ON_ERROR_STOP=1 -v batch_id="$BATCH_ID" -f /lab/postgres/source-expected-batch.sql)"
printf '%s\n' "$expected_json" >"$expected"
jq -e 'type == "array" and length == '"$EXPECTED_COUNT" "$expected" >/dev/null || {
  echo "source oracle does not contain the requested expected number of batch rows" >&2
  exit 65
}

set +e
"${DC[@]}" exec -T kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server kafka:9092 \
  --topic "$TOPIC" --from-beginning --timeout-ms 15000 >"$raw"
consumer_status=$?
set -e
# The console consumer may return non-zero when its bounded timeout expires.
# The independent event-set and payload checks below determine whether the batch arrived.
if [ "$consumer_status" -gt 1 ]; then
  echo "Kafka console consumer failed with status $consumer_status" >&2
  exit "$consumer_status"
fi

jq -s --arg batch "$BATCH_ID" --slurpfile expected "$expected" '
  [ .[]
    | select((.payload.after? // null) != null and .payload.after.batch_id == $batch)
    | .payload.after
    | {event_id, batch_id, account_id, amount}
  ] as $rows
  | ($rows | group_by(.event_id)) as $groups
  | [ $groups[] | select((map(tojson) | unique | length) > 1) | .[0].event_id ] as $conflicts
  | [ $groups[] | .[0] ] as $unique
  | ($expected[0] // []) as $want
  | ($want | map({key:.event_id, value:.}) | from_entries) as $expected_map
  | ($unique | map({key:.event_id, value:.}) | from_entries) as $actual_map
  | [ $want[] | select(($actual_map[.event_id] // null) == null) | .event_id ] as $missing
  | [ $want[] | select(($actual_map[.event_id] // null) != .) | .event_id ] as $mismatched
  | [ $unique[] | select(($expected_map[.event_id] // null) == null) | .event_id ] as $extra
  | {
      expected_count: ($want | length),
      observed_messages: ($rows | length),
      unique_observed_count: ($unique | length),
      duplicate_count: (([$groups[] | length] | add // 0) - ($unique | length)),
      missing_event_ids: $missing,
      extra_event_ids: $extra,
      conflicting_event_ids: $conflicts,
      mismatched_payload_event_ids: $mismatched
    }
' "$raw" >"$result"

jq . "$result"
jq -e '(.missing_event_ids | length) == 0 and (.extra_event_ids | length) == 0 and (.conflicting_event_ids | length) == 0 and (.mismatched_payload_event_ids | length) == 0 and .unique_observed_count == .expected_count' "$result" >/dev/null

echo "CDC event IDs and source-derived payload fields match. Exact duplicate replays are reported separately and are not confused with conflicting identity reuse."
echo "Generated-column fields are intentionally not treated as covered by this row event comparison; MON-022 accounts for that surface separately."
