#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 SOURCE_MANIFEST.json RESTORE_MANIFEST.json [REPORT.json]" >&2
  exit 64
fi
source_file="$1"
restore_file="$2"
REPORT="${3:-$ROOT/out/state-comparison.json}"
[ -s "$source_file" ] && [ -s "$restore_file" ] || { echo "both manifests must exist and be non-empty" >&2; exit 66; }
mkdir -p "$(dirname "$REPORT")"

jq -n --slurpfile source "$source_file" --slurpfile restored "$restore_file" \
  '($source[0] | if has("state") then .state else . end) as $s
   | ($restored[0] | if has("state") then .state else . end) as $r
   | ($source[0].metadata // {}) as $sm
   | ($restored[0].metadata // {}) as $rm
   | (($s | keys) + ($r | keys) | unique) as $keys
   | [ $keys[] as $k | {
        surface: $k,
        source_present: ($s | has($k)),
        restored_present: ($r | has($k)),
        equal: (($s | has($k)) and ($r | has($k)) and ($s[$k] == $r[$k])),
        source_value: (if ($s | has($k)) then $s[$k] else null end),
        restored_value: (if ($r | has($k)) then $r[$k] else null end)
      } ] as $rows
   | {
       profile:"mon025-surface-comparison-v1",
       metadata_checks:{
         database_equal: (($sm.database != null) and ($sm.database == $rm.database)),
         system_identifier_equal: (($sm.system_identifier != null) and ($sm.system_identifier == $rm.system_identifier)),
         timeline_equal: (($sm.timeline_id != null) and ($sm.timeline_id == $rm.timeline_id)),
         restore_still_in_recovery: ($rm.is_in_recovery == true)
       },
       surface_results:$rows,
       equal:(
         all($rows[]; .equal)
         and ($sm.database != null and $sm.database == $rm.database)
         and ($sm.system_identifier != null and $sm.system_identifier == $rm.system_identifier)
         and ($sm.timeline_id != null and $sm.timeline_id == $rm.timeline_id)
         and ($rm.is_in_recovery == true)
       )
     }' >"$REPORT"

jq . "$REPORT"
sha256sum "$REPORT"
jq -e '.equal == true' "$REPORT" >/dev/null || {
  echo "State surfaces diverge. Use the per-surface report to scope coverage; this blocks an unqualified full-state claim." >&2
  exit 1
}
echo "All captured state surfaces match."
