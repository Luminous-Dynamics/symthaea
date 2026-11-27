[
  $rows | to_entries[] | . as $entry | $entry.value
  | select((.payload.after? // null) != null and .payload.after.batch_id == $batch)
  | {
      observed_index: $entry.key,
      event_id: .payload.after.event_id,
      account_id: .payload.after.account_id,
      amount: .payload.after.amount,
      transaction_id: (.payload.transaction.id // null),
      total_order: ((.payload.transaction.total_order // null) | if type == "string" then tonumber else . end),
      data_collection_order: ((.payload.transaction.data_collection_order // null) | if type == "string" then tonumber else . end)
    }
] as $raw_events
| ($raw_events | group_by(.event_id)) as $event_groups
| [ $event_groups[] | select((map(del(.observed_index) | tojson) | unique | length) > 1) | .[0].event_id ] as $conflicts
| [ $event_groups[] | sort_by(.observed_index) | .[0] ] as $events
| [ $tx[] | .payload | select(.status? == "BEGIN") ] as $raw_begins
| ($raw_begins | group_by(.id)) as $begin_groups
| [ $begin_groups[] | select((map(tojson) | unique | length) > 1) | .[0].id ] as $begin_conflicts
| [ $begin_groups[] | .[0] ] as $begins
| [ $tx[] | .payload | select(.status? == "END") ] as $raw_ends
| ($raw_ends | group_by(.id)) as $end_groups
| [ $end_groups[] | select((map(tojson) | unique | length) > 1) | .[0].id ] as $end_conflicts
| [ $end_groups[] | .[0] ] as $ends
| ($events | group_by(.transaction_id)) as $groups
| [ $groups[]
    | . as $group
    | ($group[0].transaction_id) as $tid
    | ([$ends[] | select(.id == $tid)]) as $end
    | ([$begins[] | select(.id == $tid)]) as $begin
    | ($group | sort_by(.observed_index) | map(.total_order)) as $orders
    | ($group | sort_by(.observed_index) | map(.data_collection_order)) as $collection_orders
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
    profile: "mon025-transaction-closure-v2",
    batch_id: $batch,
    expected_row_count: $expected,
    observed_raw_messages: ($raw_events | length),
    observed_unique_event_ids: ($events | length),
    exact_duplicate_replays: (($raw_events | length) - ($events | length)),
    exact_duplicate_begin_replays: (($raw_begins | length) - ($begins | length)),
    exact_duplicate_end_replays: (($raw_ends | length) - ($ends | length)),
    conflicting_event_ids: $conflicts,
    conflicting_begin_ids: $begin_conflicts,
    conflicting_end_ids: $end_conflicts,
    events_without_transaction_id: ([$events[] | select(.transaction_id == null)] | length),
    transaction_groups: $checks,
    complete: (
      (($events | length) == $expected)
      and (($conflicts | length) == 0)
      and (($begin_conflicts | length) == 0)
      and (($end_conflicts | length) == 0)
      and (all($events[]; .transaction_id != null))
      and (all($checks[]; .boundary_pair_complete and .event_count_matches and .total_order_contiguous and .collection_order_contiguous))
    )
  }
