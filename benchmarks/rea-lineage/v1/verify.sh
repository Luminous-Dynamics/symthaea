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
jq -e '.schema_version=="symthaea.rea-lineage-manifest.v1" and .benchmark_id=="ENG-REA-004" and (.cases|length)==20' "$CASES" >/dev/null || fail "manifest schema/count mismatch"
jq -e '.schema_version=="symthaea.rea-lineage-oracle.v1" and .benchmark_id=="ENG-REA-004" and (.expected_dispositions|length)==20' "$ORACLE" >/dev/null || fail "oracle schema/count mismatch"
case_ids=$(jq -er '[.cases[].id] | sort | join(",")' "$CASES")
oracle_ids=$(jq -er '.expected_dispositions|keys|sort|join(",")' "$ORACLE")
[ "$case_ids" = "$oracle_ids" ] || fail "case IDs differ or are duplicated"

assess(){
  jq -ner --argjson c "$1" '
    def uint($n): ($n|type)=="number" and $n>=0 and ($n|floor)==$n;
    def validrange($r;$limit):
      $r!=null and uint($r.offset) and uint($r.length) and $r.length>0 and ($r.offset+$r.length)<=$limit;
    def badtask($t):
      ($t.path|type)!="string" or ($t.path|length)==0 or ($t.blob_id|type)!="string" or
      ($t.blob_id|length)==0 or ((uint($t.size))|not);
    def badedge($e;$c;$tasks):
      ($tasks|map(select(.path==$e.parent_path))) as $parents |
      ($tasks|map(select(.path==$e.child_path))) as $children |
      if ($parents|length)!=1 or ($children|length)!=1 or
         $e.parent_path==$e.child_path or $e.child_path=="$input" or
         ($e.provider_blob_id|type)!="string" or ($e.provider_blob_id|length)==0 or
         ($e.stage_kind|type)!="string" or ($e.stage_kind|length)==0 then
        true
      elif $children[0].blob_id!=$e.provider_blob_id then true
      elif $e.stage_kind=="multi-file" then $e.parent_file_range!=null
      elif $e.parent_file_range==null then false
      else
        ($parents[0].size) as $parent_size |
        (validrange($e.parent_file_range;$parent_size)|not) or
        (any($c.chunks[]?;
          .input_path==$e.parent_path and
          .provider_id==$e.provider_blob_id and
          .range==$e.parent_file_range)|not)
      end;
    def baddegree($p;$c):
      if $p=="$input" then any($c.derivations[]?; .child_path==$p)
      else ([$c.derivations[]?|select(.child_path==$p)]|length)!=1 end;
    def badchunk($ch;$c;$tasks):
      ($tasks|map(select(.path==$ch.input_path))) as $matches |
      if ($matches|length)!=1 then true
      elif ($ch.provider_id|type)!="string" or ($ch.provider_id|length)==0 then true
      elif (validrange($ch.range;$matches[0].size)|not) then true
      elif $ch.input_path=="$input" then
        if $ch.root_file_range==null then true
        elif (validrange($ch.root_file_range;($c.selection_offset+$c.selection_length))|not) then true
        elif $ch.root_file_range.offset!=($c.selection_offset+$ch.range.offset) then true
        elif $ch.root_file_range.length!=$ch.range.length then true
        else false end
      else $ch.root_file_range!=null end;
    $c.tasks as $tasks |
    ($tasks|map(.path)) as $paths |
    ($tasks|map(select(.path=="$input"))|length) as $root_count |
    if ($c.selection_offset|type)!="number" or ($c.selection_length|type)!="number" or
       ($c.selection_offset|floor)!=$c.selection_offset or $c.selection_offset<0 or
       ($c.selection_length|floor)!=$c.selection_length or $c.selection_length<=0 then
      "reject"
    elif ($paths|unique|length)!=($paths|length) or $root_count!=1 then
      "reject"
    elif any($tasks[]; badtask(.)) then
      "reject"
    elif ([$tasks[]|select(.path=="$input")|.size][0])!=$c.selection_length then
      "reject"
    elif ($c.derivations|length)!=(($tasks|length)-1) then
      "reject"
    elif any($c.derivations[]; badedge(.;$c;$tasks)) then
      "reject"
    elif any($paths[]; baddegree(.;$c)) then
      "reject"
    elif any($c.chunks[]; badchunk(.;$c;$tasks)) then
      "reject"
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
printf '\nPASS ENG-REA-004: 20/20 dispositions matched; task/edge/chunk order invariance checked.\n'
printf 'Accepted valid lineage: %s\nRejected invalid or misleading lineage: %s\n' "$accepted" "$rejected"
printf 'NOTE: synthetic contract only; no real REA provider or hardware was executed.\n'
