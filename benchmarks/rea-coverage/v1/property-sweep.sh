#!/bin/sh
# Exhaustive small-domain interval-complement property test.
# This test is independent of the public fixed-case oracle and invokes no REA provider.
set -eu
command -v jq >/dev/null 2>&1 || { echo 'jq required' >&2; exit 1; }

fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

# Same normalized-report acceptance contract as ENG-REA-003, kept here as a
# tiny independent executable so the interval property sweep can run alone.
assess() {
    jq -ner --argjson c "$1" '
      def complement($length; $chunks):
        ([$chunks[] | {start: .range.offset, end: (.range.offset + .range.length)}] | sort_by(.start, .end))
        | reduce .[] as $r ({cursor: 0, gaps: []};
            .gaps += (if .cursor < $r.start then [{offset: .cursor, length: ($r.start - .cursor)}] else [] end)
            | .cursor = (if $r.end > .cursor then $r.end else .cursor end)
          )
        | if .cursor < $length then .gaps + [{offset: .cursor, length: ($length - .cursor)}] else .gaps end;
      if ($c.selection_length | type) != "number" or $c.selection_length <= 0 or
         ($c.selection_length | floor) != $c.selection_length or
         ($c.selection_offset | type) != "number" or $c.selection_offset < 0 or
         ($c.selection_offset | floor) != $c.selection_offset then
        "invalid-input"
      elif any($c.chunks[]?;
          (.range.offset | type) != "number" or
          (.range.length | type) != "number" or
          (.range.offset | floor) != .range.offset or
          (.range.length | floor) != .range.length or
          .range.offset < 0 or .range.length <= 0 or
          (.range.offset + .range.length) > $c.selection_length) then
        "invalid-report"
      else
        (complement($c.selection_length; $c.chunks)) as $derived |
        ([$c.extraction_gaps[]? | select(.kind == "unreported-input-region") |
           {offset: .range.offset, length: .range.length,
            root_offset: .root_file_range.offset, root_length: .root_file_range.length}
         ] | sort_by(.offset, .length)) as $actual |
        ([$derived[] | {offset: .offset, length: .length,
           root_offset: ($c.selection_offset + .offset), root_length: .length}
         ] | sort_by(.offset, .length)) as $expected |
        (if ($derived | length) > 0 or ($c.partial_signals | length) > 0 or
            any($c.chunks[]?; .handler == null or .encrypted == true) or
            ($c.extraction_gaps | length) > 0
         then "partial" else "complete" end) as $coverage |
        if $actual == $expected and $c.coverage == $coverage then "accept" else "reject" end
      end
    '
}

count=0
for length in 1 2 3 4 5 6 7 8; do
    limit=$((1 << length))
    mask=0
    while [ "$mask" -lt "$limit" ]; do
        bits='['
        i=0
        while [ "$i" -lt "$length" ]; do
            bit=$(((mask >> i) & 1))
            if [ "$i" -gt 0 ]; then bits="$bits,"; fi
            bits="$bits$bit"
            i=$((i + 1))
        done
        bits="$bits]"
        # Vary source base offset so correct root-file range mapping is checked
        # over many offsets as well as many coverage patterns.
        base=$(((mask * 17 + length * 13) % 997))
        case_json=$(jq -cn --argjson len "$length" --argjson bits "$bits" --argjson base "$base" --arg id "SW-$length-$mask" '
          def gapruns($bits):
            reduce range(0; ($bits | length)) as $i
              ({rows: [], start: null};
                if $bits[$i] == 0 then
                  if .start == null then .start = $i else . end
                elif .start != null then
                  .rows += [{offset: .start, length: ($i - .start)}] | .start = null
                else . end
              )
            | if .start != null then .rows += [{offset: .start, length: (($bits | length) - .start)}] else . end
            | .rows;
          $bits as $b |
          [range(0; $len) as $i | select($b[$i] == 1) |
            {range: {offset: $i, length: 1}, handler: "known", encrypted: false}] as $chunks |
          gapruns($b) as $g |
          {id: $id, selection_offset: $base, selection_length: $len, chunks: $chunks,
           coverage: (if ($g | length) > 0 then "partial" else "complete" end), partial_signals: [],
           extraction_gaps: [$g[] | {kind: "unreported-input-region", range: .,
             root_file_range: {offset: ($base + .offset), length: .length}}]}
        ')
        actual=$(assess "$case_json") || fail "jq assessment failed for length=$length mask=$mask"
        [ "$actual" = accept ] || fail "length=$length mask=$mask expected bytewise reference to be accepted, got $actual"
        count=$((count + 1))
        mask=$((mask + 1))
    done
done

[ "$count" -eq 510 ] || fail "expected 510 patterns, generated $count"
printf 'PASS: %s exhaustive coverage masks for lengths 1..8\n' "$count"
printf 'PASS: independent bytewise-gap oracle agrees with interval-union implementation\n'
printf 'PASS: selected-to-root offset mapping across %s deterministic source offsets\n' "$count"
printf 'NOTE: synthetic property test only; no REA provider or hardware was exercised.\n'
