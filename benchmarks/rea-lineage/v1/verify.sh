#!/bin/sh
# ENG-REA-004 normalized extraction-lineage contract verifier.
# Does not run REA, execute extracted code, or access devices.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CASES="$ROOT/cases.json"
ORACLE="$ROOT/oracle.json"
fail(){ printf 'FAIL: %s\n' "$*" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || fail "jq required"
[ -f "$CASES" ] && [ -f "$ORACLE" ] || fail "manifest/oracle missing"
jq -e '.schema_version=="symthaea.rea-lineage-manifest.v1" and .benchmark_id=="ENG-REA-004" and (.cases|length)==18' "$CASES" >/dev/null || fail "manifest schema/count mismatch"
jq -e '.schema_version=="symthaea.rea-lineage-oracle.v1" and .benchmark_id=="ENG-REA-004" and (.expected_dispositions|length)==18' "$ORACLE" >/dev/null || fail "oracle schema/count mismatch"
case_ids=$(jq -er '[.cases[].id] | sort | join(",")' "$CASES")
oracle_ids=$(jq -er '.expected_dispositions|keys|sort|join(",")' "$ORACLE")
[ "$case_ids" = "$oracle_ids" ] || fail "case IDs differ or are duplicated"

assess(){
  jq -ner --argjson c "$1" '
    def uint($n): ($n|type)=="number" and $n>=0 and ($n|floor)==$n;
    def validrange($r;$limit):
      $r!=null and uint($r.offset) and uint($r.length) and $r.length>0 and ($r.offset+$r.length)<=$limit;
    $c.tasks as $tasks |
    ($tasks|map(.path)) as $paths |
    ($tasks|map(select(.path=="$input"))|length) as $root_count |
    if ($c.selection_offset|type)!="number" or ($c.selection_length|type)!="number" or
       ($c.selection_offset|floor)!=$c.selection_offset or $c.selection_offset<0 or
       ($c.selection_length|floor)!=$c.selection_length or $c.selection_length<=0 then
      "reject"
    elif ($paths|unique|length)!=($paths|length) or $root_count!=1 then
      "reject"
    elif any($tasks[] as $t; ($t.path|type)!="string" or ($t.blob_id|type)!="string" or ($t.blob_id|length)==0 or ((uint($t.size))|not)) then
      "reject"
    elif ([$tasks[]|select(.path=="$input")|.size][0])!=$c.selection_length then
      "reject"
    elif ($c.derivations|length)!=(($tasks|length)-1) then
      "reject"
    elif any($c.derivations[] as $e;
       ([$tasks[]|select(.path==$e.parent_path)]|length)!=1 or
       ([$tasks[]|select(.path==$e.child_path)]|length)!=1 or
       $e.parent_path==$e.child_path or $e.child_path=="$input" or
       ($e.provider_blob_id|type)!="string" or ($e.provider_blob_id|length)==0 or
       ([$tasks[]|select(.path==$e.child_path)|.blob_id][0])!=$e.provider_blob_id or
       (if $e.stage_kind=="multi-file" then $e.parent_file_range!=null
        elif $e.parent_file_range==null then false
        else
          ([$tasks[]|select(.path==$e.parent_path)|.size][0]) as $parent_size |
          (validrange($e.parent_file_range;$parent_size)|not) or
          (any($c.chunks[]? as $pc;
             $pc.input_path==$e.parent_path and
             $pc.provider_id==$e.provider_blob_id and
             $pc.range==$e.parent_file_range)|not)
        end)
      ) then "reject"
    elif any($paths[] as $p;
       if $p=="$input" then any($c.derivations[]?; .child_path==$p)
       else ([$c.derivations[]?|select(.child_path==$p)]|length)!=1 end) then
      "reject"
    elif any($c.chunks[] as $ch;
       ([$tasks[]|select(.path==$ch.input_path)]|length)!=1 or
       (
         ([$tasks[]|select(.path==$ch.input_path)|.size][0]) as $task_size |
         (validrange($ch.range;$task_size)|not) or
         (if $ch.input_path=="$input" then
            $ch.root_file_range==null or
            $ch.root_file_range.offset!=($c.selection_offset+$ch.range.offset) or
            $ch.root_file_range.length!=$ch.range.length or
            (validrange($ch.root_file_range;($c.selection_offset+$c.selection_length))|not)
          else $ch.root_file_range!=null end)
       )
      ) then "reject"
    else "accept" end
  '
}

accepted=0
rejected=0
while IFS= read -r case_json; do
 id=$(printf '%s' "$case_json"|jq -er '.id')
 expected=$(jq -er --arg id "$id" '.expected_dispositions[$id]' "$ORACLE")
 actual=$(assess "$case_json") || fail "$id verifier error"
 [ "$actual" = "$expected" ] || fail "$id expected $expected, got $actual"
 permuted=$(printf '%s' "$case_json"|jq -c '.tasks |= reverse | .derivations |= reverse | .chunks |= reverse')
 permuted_result=$(assess "$permuted") || fail "$id permutation verifier error"
 [ "$permuted_result" = "$actual" ] || fail "$id disposition changed under record permutation"
 case "$actual" in
  accept) accepted=$((accepted+1));;
  reject) rejected=$((rejected+1));;
  *) fail "$id unexpected disposition $actual";;
 esac
 printf 'PASS %s -> %s (record permutation invariant)\n' "$id" "$actual"
done <<EOF_CASES
$(jq -c '.cases[]' "$CASES")
EOF_CASES
printf '\nPASS ENG-REA-004: 18/18 dispositions matched; task/edge/chunk order invariance checked.\n'
printf 'Accepted valid lineage: %s\nRejected invalid or misleading lineage: %s\n' "$accepted" "$rejected"
printf 'NOTE: synthetic contract only; no real REA provider or hardware was executed.\n'
