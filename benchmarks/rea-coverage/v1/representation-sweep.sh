#!/bin/sh
# Exhaustive small-domain interval-representation invariance sweep.
# Checks the interval-union complement against an independent bytewise oracle.
set -eu
command -v jq >/dev/null 2>&1 || { echo 'jq required' >&2; exit 1; }

jq -ner '
  def p2($n): reduce range(0; $n) as $i (1; . * 2);
  def ranges($n):
    [range(0; $n) as $s | range(($s + 1); ($n + 1)) as $e |
      {offset: $s, length: ($e - $s)}];
  def union_gaps($n; $chunks):
    ([$chunks[] | {start: .offset, end: (.offset + .length)}] | sort_by(.start, .end))
    | reduce .[] as $r ({cursor: 0, gaps: []};
        .gaps += (if .cursor < $r.start then [{offset: .cursor, length: ($r.start - .cursor)}] else [] end)
        | .cursor = (if $r.end > .cursor then $r.end else .cursor end))
    | if .cursor < $n then .gaps + [{offset: .cursor, length: ($n - .cursor)}] else .gaps end;
  def bytewise_gaps($n; $chunks):
    [range(0; $n) as $b |
      select(all($chunks[]?; .offset > $b or (.offset + .length) <= $b)) | $b]
    | reduce .[] as $b ({gaps: [], start: null, prev: null};
        if .start == null then .start = $b | .prev = $b
        elif $b == (.prev + 1) then .prev = $b
        else .gaps += [{offset: .start, length: (.prev - .start + 1)}]
             | .start = $b | .prev = $b end)
    | if .start != null then .gaps + [{offset: .start, length: (.prev - .start + 1)}] else .gaps end;
  def coverage_bitmap($n; $chunks):
    [range(0; $n) as $b |
      any($chunks[]?; .offset <= $b and (.offset + .length) > $b)];
  reduce range(1; 6) as $n
    ({patterns: 0, representations: 0, mismatches: 0};
      (ranges($n)) as $all |
      (p2($all | length)) as $limit |
      reduce range(0; $limit) as $mask (.; 
        ([$all | to_entries[] | select(((($mask / p2(.key)) | floor) % 2) == 1) | .value]) as $selected |
        ([$selected, ($selected | reverse), ($selected + $selected), ($selected | sort_by(.length, .offset)), ($selected | sort_by(.length, .offset) | reverse)]) as $variants |
        .patterns += 1 |
        reduce $variants[] as $chunks (.; 
          (union_gaps($n; $chunks)) as $u |
          (bytewise_gaps($n; $chunks)) as $b |
          .representations += 1 |
          if $u == $b then . else .mismatches += 1 end
        )
      )
    )
  | if .patterns != 33866 or .representations != (33866 * 5) or .mismatches != 0 then
      error("unexpected exhaustive sweep result: \(.patterns) sets, \(.representations) representations, \(.mismatches) mismatches")
    else
      "PASS: \(.patterns) exhaustive interval subsets for lengths 1..5",
      "PASS: \(.representations) representations (original, reversed, duplicated, ascending-length, descending-length)",
      "PASS: interval-union gaps exactly match independent bytewise gap runs",
      "NOTE: synthetic contract property test only; REA/Unblob and hardware not exercised."
    end
' </dev/null
