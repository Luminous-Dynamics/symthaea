def num:
  if type == "number" then .
  elif type == "string" then (try tonumber catch null)
  else null end;
def tx_key($value):
  if ($value | type) == "string" and ($value | test("^[0-9]+:[^:]+$"))
  then ($value | split(":")[0]) else null end;
def contiguous($values):
  $values == [range(1; ($values | length) + 1)];
def normalize_collections:
  ( . // [] ) | map({data_collection: .data_collection, event_count: (.event_count | num)}) | sort_by(.data_collection);
[
  $rows | to_entries[] | . as $entry | .value
  | select((.payload.after? // null) != null and .payload.after.batch_id == $batch)
  | {observed_index:$entry.key,event_class:"row",event_id:.payload.after.event_id,
     operation_id:(.payload.transaction.id // null),
     transaction_id:tx_key(.payload.transaction.id // null),
     total_order:(.payload.transaction.total_order | num),
     collection_order:(.payload.transaction.data_collection_order | num),
     collection:((.payload.source.schema // "") + "." + (.payload.source.table // "")),
     account_id:.payload.after.account_id,amount:.payload.after.amount}
] as $raw_rows
| ($raw_rows | group_by(.event_id)) as $row_groups
| [$row_groups[] | select((map(del(.observed_index) | tojson) | unique | length) > 1) | .[0].event_id] as $conflicting_row_event_ids
| [$row_groups[] | sort_by(.observed_index) | .[0]] as $row_events
| ($row_events | map(.transaction_id) | map(select(. != null)) | unique) as $batch_transaction_ids
| [
    $messages | to_entries[] | . as $entry | .value
    | select((.payload.transaction.id? // null) != null)
    | . as $record
    | tx_key($record.payload.transaction.id) as $tid
    | select($tid != null and ($batch_transaction_ids | index($tid)) != null)
    | select(($record.payload.message.prefix // "") == $message_prefix)
    | {observed_index:$entry.key,event_class:"logical_message",
       event_id:("\($tid)|\(($record.payload.transaction.total_order | num) // "invalid")"),
       operation_id:$record.payload.transaction.id,transaction_id:$tid,
       total_order:($record.payload.transaction.total_order | num),
       collection_order:($record.payload.transaction.data_collection_order | num),
       collection:"LOGICAL_DECODING_MESSAGE",message_prefix:$record.payload.message.prefix,
       message_content:$record.payload.message.content,source_lsn:$record.payload.source.lsn}
] as $raw_messages
| ($raw_messages | group_by(.event_id)) as $message_groups
| [$message_groups[] | select((map(del(.observed_index) | tojson) | unique | length) > 1) | .[0].event_id] as $conflicting_message_event_ids
| [$message_groups[] | sort_by(.observed_index) | .[0]] as $message_events
| [
    $tx | to_entries[] | . as $entry | .value.payload
    | select(.status == "BEGIN" or .status == "END")
    | . as $boundary
    | tx_key($boundary.id) as $tid
    | select($tid != null and ($batch_transaction_ids | index($tid)) != null)
    | {observed_index:$entry.key,transaction_id:$tid,id:$boundary.id,status:$boundary.status,
       event_count:($boundary.event_count | num),data_collections:($boundary.data_collections // [])}
] as $raw_boundaries
| [$raw_boundaries[] | select(.status == "BEGIN")] as $raw_begins
| [$raw_boundaries[] | select(.status == "END")] as $raw_ends
| ($raw_begins | group_by(.transaction_id)) as $begin_groups
| ($raw_ends | group_by(.transaction_id)) as $end_groups
| [$begin_groups[] | select((map(del(.observed_index) | tojson) | unique | length) > 1) | .[0].transaction_id] as $conflicting_begin_ids
| [$end_groups[] | select((map(del(.observed_index) | tojson) | unique | length) > 1) | .[0].transaction_id] as $conflicting_end_ids
| [$begin_groups[] | sort_by(.observed_index) | .[0]] as $begins
| [$end_groups[] | sort_by(.observed_index) | .[0]] as $ends
| [
    $batch_transaction_ids[] as $tid
    | ([$row_events[] | select(.transaction_id == $tid)]) as $tx_rows
    | ([$message_events[] | select(.transaction_id == $tid)]) as $tx_messages
    | ($tx_rows + $tx_messages) as $events
    | ([$begins[] | select(.transaction_id == $tid)]) as $begin
    | ([$ends[] | select(.transaction_id == $tid)]) as $end
    | ($events | map(.total_order) | sort) as $total_orders
    | ($tx_rows | sort_by(.observed_index) | map(.collection_order)) as $row_collection_orders
    | ($tx_messages | sort_by(.observed_index) | map(.collection_order)) as $message_collection_orders
    | (if ($end | length) == 1 then $end[0].event_count else null end) as $announced_count
    | (if ($end | length) == 1 then ($end[0].data_collections | normalize_collections) else [] end) as $declared_collections
    | [$declared_collections[] | select(.data_collection == "public.ledger")] as $declared_row_collections
    | [$declared_collections[] | select(.data_collection != "public.ledger")] as $declared_message_collections
    | ($declared_message_collections | map(.event_count) | add // 0) as $declared_message_event_count
    | ($events | map(.total_order) | group_by(.) | map(select(length > 1)) | length) as $duplicate_total_order_groups
    | {transaction_id:$tid,row_event_count:($tx_rows | length),
       logical_message_count:($tx_messages | length),transaction_event_count:($events | length),
       begin_records:($begin | length),end_records:($end | length),
       begin_operation_id:(if ($begin | length) == 1 then $begin[0].id else null end),
       end_operation_id:(if ($end | length) == 1 then $end[0].id else null end),
       announced_event_count:$announced_count,total_order:$total_orders,
       row_data_collection_order:$row_collection_orders,
       message_data_collection_order:$message_collection_orders,
       declared_data_collections:$declared_collections,
       declared_row_collection_count:($declared_row_collections | length),
       declared_message_collection_entries:$declared_message_collections,
       observed_row_collection_count:($tx_rows | length),
       observed_logical_message_count:($tx_messages | length),
       boundary_pair_complete:(($begin | length) == 1 and ($end | length) == 1),
       event_count_matches:(($end | length) == 1 and $announced_count == ($events | length)),
       data_collections_match:(($end | length) == 1
         and ($declared_row_collections | length) == 1
         and $declared_row_collections[0].event_count == ($tx_rows | length)
         and $declared_message_event_count == ($tx_messages | length)
         and (($tx_messages | length) == 0 or ($declared_message_collections | length) >= 1)
         and (($tx_messages | length) > 0 or ($declared_message_collections | length) == 0)),
       total_order_contiguous:contiguous($total_orders),
       row_collection_order_contiguous:contiguous($row_collection_orders),
       message_collection_order_positive:all($message_collection_orders[]; . != null and . > 0),
       duplicate_total_order_groups:$duplicate_total_order_groups}
  ] as $transaction_groups
| {profile:"mon025-multitopic-transaction-closure-v3",batch_id:$batch,message_prefix:$message_prefix,
   expected_row_count:$expected,
   observed_raw_row_messages:($raw_rows | length),observed_unique_row_events:($row_events | length),
   exact_duplicate_row_replays:(($raw_rows | length) - ($row_events | length)),
   conflicting_row_event_ids:$conflicting_row_event_ids,
   observed_raw_logical_messages:($raw_messages | length),observed_unique_logical_messages:($message_events | length),
   exact_duplicate_logical_message_replays:(($raw_messages | length) - ($message_events | length)),
   conflicting_logical_message_event_ids:$conflicting_message_event_ids,
   exact_duplicate_begin_replays:(($raw_begins | length) - ($begins | length)),
   exact_duplicate_end_replays:(($raw_ends | length) - ($ends | length)),
   conflicting_begin_transaction_ids:$conflicting_begin_ids,
   conflicting_end_transaction_ids:$conflicting_end_ids,
   row_events_without_valid_transaction_id:([$row_events[] | select(.transaction_id == null)] | length),
   row_events_without_valid_event_id:([$row_events[] | select((.event_id | type) != "string" or .event_id == "")] | length),
   invalid_row_order_or_collection_order:([$row_events[] | select(.total_order == null or .total_order <= 0 or .collection_order == null or .collection_order <= 0)] | length),
   invalid_message_order_or_collection_order:([$message_events[] | select(.total_order == null or .total_order <= 0 or .collection_order == null or .collection_order <= 0)] | length),
   transaction_groups:$transaction_groups,
   complete:(($row_events | length) == $expected
     and ($conflicting_row_event_ids | length) == 0
     and ($conflicting_message_event_ids | length) == 0
     and ($conflicting_begin_ids | length) == 0
     and ($conflicting_end_ids | length) == 0
     and all($row_events[]; .transaction_id != null and (.event_id | type) == "string" and .event_id != "" and .total_order != null and .total_order > 0 and .collection_order != null and .collection_order > 0)
     and all($message_events[]; .total_order != null and .total_order > 0 and .collection_order != null and .collection_order > 0)
     and all($transaction_groups[]; .boundary_pair_complete and .event_count_matches and .data_collections_match and .total_order_contiguous and .row_collection_order_contiguous and .message_collection_order_positive and .duplicate_total_order_groups == 0))}
