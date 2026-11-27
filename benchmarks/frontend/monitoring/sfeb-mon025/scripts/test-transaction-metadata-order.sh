#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FILTER="$ROOT/scripts/transaction-metadata-report.jq"
command -v jq >/dev/null 2>&1 || { echo "jq is required" >&2; exit 69; }
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
BATCH="order-fixture"
TX="transaction-fixture"

write_row() {
  local destination="$1" event="$2" order="$3" amount="$4"
  jq -nc --arg batch "$BATCH" --arg event "$event" --arg tx "$TX" \
    --arg account "account-01" --arg amount "$amount" --argjson order "$order" \
    '{payload:{after:{batch_id:$batch,event_id:$event,account_id:$account,amount:$amount},transaction:{id:$tx,total_order:$order,data_collection_order:$order}}}' \
    >>"$destination"
}
write_boundaries() {
  local destination="$1"
  jq -nc --arg tx "$TX" '{payload:{id:$tx,status:"BEGIN"}}' >>"$destination"
  jq -nc --arg tx "$TX" '{payload:{id:$tx,status:"END",event_count:2}}' >>"$destination"
}
run_report() {
  local rows="$1" tx="$2" output="$3"
  jq -n --arg batch "$BATCH" --argjson expected 2 --slurpfile rows "$rows" --slurpfile tx "$tx" -f "$FILTER" >"$output"
}

# Ordered positive control.
: >"$TMP/ordered-rows.jsonl"
write_row "$TMP/ordered-rows.jsonl" event-1 1 10.00
write_row "$TMP/ordered-rows.jsonl" event-2 2 20.00
write_boundaries "$TMP/ordered-tx.jsonl"
run_report "$TMP/ordered-rows.jsonl" "$TMP/ordered-tx.jsonl" "$TMP/ordered.json"
jq -e '.complete and .transaction_groups[0].total_order == [1,2]' "$TMP/ordered.json" >/dev/null

# Out-of-order negative control: sorting the set would hide this fault.
: >"$TMP/reversed-rows.jsonl"
write_row "$TMP/reversed-rows.jsonl" event-2 2 20.00
write_row "$TMP/reversed-rows.jsonl" event-1 1 10.00
run_report "$TMP/reversed-rows.jsonl" "$TMP/ordered-tx.jsonl" "$TMP/reversed.json"
jq -e '(.complete == false) and (.transaction_groups[0].total_order == [2,1]) and (.transaction_groups[0].total_order_contiguous == false)' "$TMP/reversed.json" >/dev/null

# Exact duplicate replay remains acceptable and counted.
: >"$TMP/replay-rows.jsonl"
write_row "$TMP/replay-rows.jsonl" event-1 1 10.00
write_row "$TMP/replay-rows.jsonl" event-2 2 20.00
write_row "$TMP/replay-rows.jsonl" event-1 1 10.00
run_report "$TMP/replay-rows.jsonl" "$TMP/ordered-tx.jsonl" "$TMP/replay.json"
jq -e '.complete and .exact_duplicate_replays == 1 and .transaction_groups[0].total_order == [1,2]' "$TMP/replay.json" >/dev/null

# Conflicting duplicate identity is rejected.
: >"$TMP/conflict-rows.jsonl"
write_row "$TMP/conflict-rows.jsonl" event-1 1 10.00
write_row "$TMP/conflict-rows.jsonl" event-2 2 20.00
write_row "$TMP/conflict-rows.jsonl" event-1 1 999.00
run_report "$TMP/conflict-rows.jsonl" "$TMP/ordered-tx.jsonl" "$TMP/conflict.json"
jq -e '(.complete == false) and (.conflicting_event_ids | length == 1)' "$TMP/conflict.json" >/dev/null


# Multiple transactions in one capture: order must be checked within each transaction.
cat >"$TMP/multirows.jsonl" <<'JSON'
{"payload":{"after":{"batch_id":"order-fixture","event_id":"a1","account_id":"a","amount":1},"transaction":{"id":"tx-a","total_order":1,"data_collection_order":1}}}
{"payload":{"after":{"batch_id":"order-fixture","event_id":"b1","account_id":"b","amount":1},"transaction":{"id":"tx-b","total_order":1,"data_collection_order":1}}}
{"payload":{"after":{"batch_id":"order-fixture","event_id":"a2","account_id":"a","amount":2},"transaction":{"id":"tx-a","total_order":2,"data_collection_order":2}}}
JSON
cat >"$TMP/multitx.jsonl" <<'JSON'
{"payload":{"id":"tx-a","status":"BEGIN"}}
{"payload":{"id":"tx-a","status":"END","event_count":2}}
{"payload":{"id":"tx-b","status":"BEGIN"}}
{"payload":{"id":"tx-b","status":"END","event_count":1}}
JSON
jq -n --arg batch "$BATCH" --argjson expected 3 --slurpfile rows "$TMP/multirows.jsonl" --slurpfile tx "$TMP/multitx.jsonl" -f "$FILTER" >"$TMP/multi.json"
jq -e '.complete and (.transaction_groups|length == 2)' "$TMP/multi.json" >/dev/null

# Missing event identity must not qualify, even for a one-event expected denominator.
jq -nc --arg batch "$BATCH" --arg tx "$TX" '{payload:{after:{batch_id:$batch,account_id:"a",amount:1},transaction:{id:$tx,total_order:1,data_collection_order:1}}}' >"$TMP/no-id-rows.jsonl"
jq -nc --arg tx "$TX" '{payload:{id:$tx,status:"BEGIN"}}' >"$TMP/no-id-tx.jsonl"
jq -nc --arg tx "$TX" '{payload:{id:$tx,status:"END",event_count:1}}' >>"$TMP/no-id-tx.jsonl"
jq -n --arg batch "$BATCH" --argjson expected 1 --slurpfile rows "$TMP/no-id-rows.jsonl" --slurpfile tx "$TMP/no-id-tx.jsonl" -f "$FILTER" >"$TMP/no-id.json"
jq -e '(.complete|not) and .invalid_event_ids == 1' "$TMP/no-id.json" >/dev/null

# Exact duplicate BEGIN/END envelopes are counted, not treated as missing boundaries.
cp "$TMP/ord.jsonl" "$TMP/boundary-replay-rows.jsonl"
cat "$TMP/tx.jsonl" "$TMP/tx.jsonl" >"$TMP/boundary-replay-tx.jsonl"
jq -n --arg batch "$BATCH" --argjson expected 2 --slurpfile rows "$TMP/boundary-replay-rows.jsonl" --slurpfile tx "$TMP/boundary-replay-tx.jsonl" -f "$FILTER" >"$TMP/boundary-replay.json"
jq -e '.complete and .exact_duplicate_begin_replays == 1 and .exact_duplicate_end_replays == 1' "$TMP/boundary-replay.json" >/dev/null

# Conflicting BEGIN under an existing transaction identity is a hard conflict.
jq -nc --arg tx "$TX" '{payload:{id:$tx,status:"BEGIN",ts_ms:1}}' >"$TMP/boundary-conflict-tx.jsonl"
jq -nc --arg tx "$TX" '{payload:{id:$tx,status:"BEGIN",ts_ms:2}}' >>"$TMP/boundary-conflict-tx.jsonl"
jq -nc --arg tx "$TX" '{payload:{id:$tx,status:"END",event_count:2}}' >>"$TMP/boundary-conflict-tx.jsonl"
jq -n --arg batch "$BATCH" --argjson expected 2 --slurpfile rows "$TMP/ord.jsonl" --slurpfile tx "$TMP/boundary-conflict-tx.jsonl" -f "$FILTER" >"$TMP/boundary-conflict.json"
jq -e '(.complete|not) and (.conflicting_begin_ids|length == 1)' "$TMP/boundary-conflict.json" >/dev/null

echo "MON-030 transaction observed-order fixtures passed (9 fixture classes)."
