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
  psql -X -h 127.0.0.1 -U "${MON025_DB_USER:-mon025}" -d "${MON025_DB_NAME:-mon025}" \
  -v ON_ERROR_STOP=1 -v batch_id="$BATCH_ID" -At \
  -c "SELECT coalesce(json_agg(event_id ORDER BY event_id), '[]'::json) FROM public.ledger WHERE batch_id = :'batch_id'")"
printf '%s\n' "$expected_json" >"$expected"
jq -e 'type == "array" and length == '"$EXPECTED_COUNT" "$expected" >/dev/null || {
  echo "source oracle does not contain the requested expected number of batch IDs" >&2
  exit 65
}

set +e
"${DC[@]}" exec -T kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server kafka:9092 \
  --topic "$TOPIC" --from-beginning --timeout-ms 15000 >"$raw"
consumer_status=$?
set -e
# The console consumer may return non-zero when its bounded timeout expires.
# The independent JSON/event-set checks below determine whether the batch arrived.
if [ "$consumer_status" -gt 1 ]; then
  echo "Kafka console consumer failed with status $consumer_status" >&2
  exit "$consumer_status"
fi

jq -s --arg batch "$BATCH_ID" --slurpfile expected "$expected" '
  [ .[] | select((.payload.after? // null) != null and .payload.after.batch_id == $batch) | .payload.after ] as $rows
  | ($rows | group_by(.event_id)) as $groups
  | [ $groups[] | select((map(tojson) | unique | length) > 1) | .[0].event_id ] as $conflicts
  | [ $groups[] | .[0].event_id ] as $actual
  | ($expected[0] // []) as $want
  | {
      expected_count: ($want | length),
      observed_messages: ($rows | length),
      unique_observed_count: ($actual | length),
      duplicate_count: (([$groups[] | length] | add // 0) - ($actual | length)),
      missing_event_ids: [$want[] | select(. as $id | ($actual | index($id)) == null)],
      extra_event_ids: [$actual[] | select(. as $id | ($want | index($id)) == null)],
      conflicting_event_ids: $conflicts
    }
' "$raw" >"$result"

cat "$result" | jq .
jq -e '(.missing_event_ids | length) == 0 and (.extra_event_ids | length) == 0 and (.conflicting_event_ids | length) == 0 and .unique_observed_count == .expected_count' "$result" >/dev/null

echo "CDC event-set check passed. Exact duplicate replays are reported separately; downstream idempotency still requires its own test."
