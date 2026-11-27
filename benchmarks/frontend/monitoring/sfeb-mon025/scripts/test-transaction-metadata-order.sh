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

echo "MON-030 transaction observed-order fixtures passed."
