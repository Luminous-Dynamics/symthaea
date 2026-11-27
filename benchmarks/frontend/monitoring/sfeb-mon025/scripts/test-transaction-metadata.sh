#!/usr/bin/env bash
set -euo pipefail
FILTER="$(dirname "$0")/transaction-metadata-report.jq"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
BATCH="batch-032"
PREFIX="mon025_transaction"
write_row() {
  local out="$1" event="$2" opid="$3" total="$4" collection="$5" amount="$6"
  jq -nc --arg batch "$BATCH" --arg event "$event" --arg opid "$opid" --arg amount "$amount" --argjson total "$total" --argjson collection "$collection" \
    '{payload:{source:{schema:"public",table:"ledger",lsn:"0/10"},after:{batch_id:$batch,event_id:$event,account_id:"account-01",amount:$amount},transaction:{id:$opid,total_order:$total,data_collection_order:$collection}}}' >>"$out"
}
write_message() {
  local out="$1" opid="$2" total="$3" collection="$4" content="$5"
  jq -nc --arg prefix "$PREFIX" --arg opid "$opid" --arg content "$content" --argjson total "$total" --argjson collection "$collection" \
    '{payload:{source:{lsn:"0/11"},op:"m",message:{prefix:$prefix,content:$content},transaction:{id:$opid,total_order:$total,data_collection_order:$collection}}}' >>"$out"
}
write_boundaries() {
  local out="$1" begin="$2" end="$3" event_count="$4" row_count="$5" message_count="$6"
  jq -nc --arg id "$begin" '{payload:{id:$id,status:"BEGIN"}}' >>"$out"
  jq -nc --arg id "$end" --argjson event_count "$event_count" --argjson row_count "$row_count" --argjson message_count "$message_count" \
    '{payload:{id:$id,status:"END",event_count:$event_count,data_collections:[{data_collection:"LOGICAL_DECODING_MESSAGE",event_count:$message_count},{data_collection:"public.ledger",event_count:$row_count}]}}' >>"$out"
}
run() {
  local rows="$1" messages="$2" tx="$3" expected="$4" output="$5"
  jq -n --arg batch "$BATCH" --arg message_prefix "$PREFIX" --argjson expected "$expected" --slurpfile rows "$rows" --slurpfile messages "$messages" --slurpfile tx "$tx" -f "$FILTER" >"$output"
}
: >"$TMP/rows.jsonl"; : >"$TMP/messages.jsonl"; : >"$TMP/tx.jsonl"
write_row "$TMP/rows.jsonl" evt-1 42:0/10 1 1 10.00
write_row "$TMP/rows.jsonl" evt-2 42:0/12 3 2 20.00
write_message "$TMP/messages.jsonl" 42:0/11 2 1 encoded-content
write_boundaries "$TMP/tx.jsonl" 42:0/9 42:0/13 3 2 1
run "$TMP/rows.jsonl" "$TMP/messages.jsonl" "$TMP/tx.jsonl" 2 "$TMP/positive.json"
jq -e '.complete and .transaction_groups[0].transaction_event_count == 3 and .transaction_groups[0].total_order == [1,2,3]' "$TMP/positive.json" >/dev/null
: >"$TMP/empty-messages.jsonl"
run "$TMP/rows.jsonl" "$TMP/empty-messages.jsonl" "$TMP/tx.jsonl" 2 "$TMP/missing-message.json"
jq -e '(.complete|not) and (.transaction_groups[0].event_count_matches|not) and (.transaction_groups[0].total_order_contiguous|not)' "$TMP/missing-message.json" >/dev/null
: >"$TMP/wrong-tx.jsonl"
write_boundaries "$TMP/wrong-tx.jsonl" 99:0/9 99:0/13 3 2 1
run "$TMP/rows.jsonl" "$TMP/messages.jsonl" "$TMP/wrong-tx.jsonl" 2 "$TMP/missing-boundary.json"
jq -e '(.complete|not) and (.transaction_groups[0].boundary_pair_complete|not)' "$TMP/missing-boundary.json" >/dev/null
: >"$TMP/reversed-rows.jsonl"
write_row "$TMP/reversed-rows.jsonl" evt-2 42:0/12 3 2 20.00
write_row "$TMP/reversed-rows.jsonl" evt-1 42:0/10 1 1 10.00
run "$TMP/reversed-rows.jsonl" "$TMP/messages.jsonl" "$TMP/tx.jsonl" 2 "$TMP/reordered.json"
jq -e '(.complete|not) and (.transaction_groups[0].row_collection_order_contiguous|not)' "$TMP/reordered.json" >/dev/null
cat "$TMP/rows.jsonl" "$TMP/rows.jsonl" >"$TMP/replayed-rows.jsonl"
run "$TMP/replayed-rows.jsonl" "$TMP/messages.jsonl" "$TMP/tx.jsonl" 2 "$TMP/row-replay.json"
jq -e '.complete and .exact_duplicate_row_replays == 2' "$TMP/row-replay.json" >/dev/null
cat "$TMP/messages.jsonl" "$TMP/messages.jsonl" >"$TMP/replayed-messages.jsonl"
run "$TMP/rows.jsonl" "$TMP/replayed-messages.jsonl" "$TMP/tx.jsonl" 2 "$TMP/message-replay.json"
jq -e '.complete and .exact_duplicate_logical_message_replays == 1' "$TMP/message-replay.json" >/dev/null
: >"$TMP/conflicting-messages.jsonl"
write_message "$TMP/conflicting-messages.jsonl" 42:0/11 2 1 encoded-content
write_message "$TMP/conflicting-messages.jsonl" 42:0/11 2 1 altered-content
run "$TMP/rows.jsonl" "$TMP/conflicting-messages.jsonl" "$TMP/tx.jsonl" 2 "$TMP/message-conflict.json"
jq -e '(.complete|not) and (.conflicting_logical_message_event_ids|length == 1)' "$TMP/message-conflict.json" >/dev/null
: >"$TMP/colliding-messages.jsonl"
write_message "$TMP/colliding-messages.jsonl" 42:0/11 1 1 colliding-order
run "$TMP/rows.jsonl" "$TMP/colliding-messages.jsonl" "$TMP/tx.jsonl" 2 "$TMP/order-collision.json"
jq -e '(.complete|not) and (.transaction_groups[0].duplicate_total_order_groups > 0)' "$TMP/order-collision.json" >/dev/null
: >"$TMP/two-message-rows.jsonl"; : >"$TMP/two-messages.jsonl"; : >"$TMP/two-message-tx.jsonl"
write_row "$TMP/two-message-rows.jsonl" evt-a 42:0/20 1 1 10.00
write_row "$TMP/two-message-rows.jsonl" evt-b 42:0/23 4 2 20.00
write_message "$TMP/two-messages.jsonl" 42:0/21 2 1 first-message
write_message "$TMP/two-messages.jsonl" 42:0/22 3 1 second-message
jq -nc --arg id "42:0/19" '{payload:{id:$id,status:"BEGIN"}}' >>"$TMP/two-message-tx.jsonl"
jq -nc --arg id "42:0/24" '{payload:{id:$id,status:"END",event_count:4,data_collections:[{data_collection:"public.ledger",event_count:2},{data_collection:"LOGICAL_DECODING_MESSAGE@one",event_count:1},{data_collection:"LOGICAL_DECODING_MESSAGE@two",event_count:1}]}}' >>"$TMP/two-message-tx.jsonl"
run "$TMP/two-message-rows.jsonl" "$TMP/two-messages.jsonl" "$TMP/two-message-tx.jsonl" 2 "$TMP/two-messages.json"
jq -e '.complete and .transaction_groups[0].transaction_event_count == 4 and .transaction_groups[0].message_data_collection_order == [1,1]' "$TMP/two-messages.json" >/dev/null
: >"$TMP/wrong-collections.jsonl"
jq -nc --arg id "42:0/9" '{payload:{id:$id,status:"BEGIN"}}' >>"$TMP/wrong-collections.jsonl"
jq -nc --arg id "42:0/13" '{payload:{id:$id,status:"END",event_count:3,data_collections:[{data_collection:"public.ledger",event_count:3}]}}' >>"$TMP/wrong-collections.jsonl"
run "$TMP/rows.jsonl" "$TMP/messages.jsonl" "$TMP/wrong-collections.jsonl" 2 "$TMP/collection-mismatch.json"
jq -e '(.complete|not) and (.transaction_groups[0].data_collections_match|not)' "$TMP/collection-mismatch.json" >/dev/null
echo "MON-032 composite-ID and multi-topic transaction fixtures passed (10 fixture classes)."
