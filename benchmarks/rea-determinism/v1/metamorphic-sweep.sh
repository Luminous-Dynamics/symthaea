#!/bin/sh
# ENG-REA-006 deterministic metamorphic sweep for semantic report normalization.
# Generates new derived reports only; never mutates the checked-in cases.json.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CASES="$ROOT/cases.json"
NORMALIZER="$ROOT/normalize.jq"
fail(){ printf 'FAIL: %s\n' "$*" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || fail "jq required"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum required"
[ -f "$CASES" ] && [ -f "$NORMALIZER" ] || fail "fixture or normalizer missing"
SOURCE_SHA=$(git rev-parse HEAD 2>/dev/null || printf 'unknown')
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

# A known producer ID must change consistently at both its report definition and
# every Task.blob_id reference. Only Unblob-generated blob/chunk IDs are varied.
variant_jq='
def new_id($seed; $id): if $id=="" then "" else ("pid-" + ($seed|tostring) + "-" + $id) end;
def rebase($old; $new; $p):
  if ($p|type)!="string" then error("non-string path")
  elif $p==$old then $new
  elif ($p|startswith($old + "/")) then $new + $p[($old|length):]
  else $p end;
map(
  .task |= (.blob_id=new_id($seed;.blob_id))
  | .task.path=rebase($oldroot;$newroot;.task.path)
  | .subtasks |= map(.blob_id=new_id($seed;.blob_id) | .path=rebase($oldroot;$newroot;.path))
  | .reports |= map(
      walk(if type=="object" then
        if has("blob_id") then .blob_id=new_id($seed;.blob_id)
        elif ((.__typename__ // "") as $t | ["ChunkReport","UnknownChunkReport","MultiFileReport"]|index($t))!=null
          then .id=new_id($seed;.id)
        else . end
      else . end)
      | walk(if type=="object" then with_entries(
          if (["path","carve_dir","extraction_path","link_path"]|index(.key))!=null and (.value|type)=="string"
            then .value=rebase($oldroot;$newroot;.value)
          elif .key=="paths" and (.value|type)=="array" then .value |= map(rebase($oldroot;$newroot;.))
          else . end)
        else . end)
    )
)
| map(.reports |= sort_by((tojson|explode|reduce .[] as $code ($seed; ((.*33+$code)%2147483647))))
       | .subtasks |= sort_by((tojson|explode|reduce .[] as $code ($seed; ((.*33+$code)%2147483647)))))
| sort_by((tojson|explode|reduce .[] as $code ($seed; ((.*33+$code)%2147483647))))
';

project(){
  fixture=$1
  root=$2
  input=$3
  jq -c --arg f "$fixture" '.fixtures[$f].report' "$CASES" > "$input"
  jq --arg root "$root" -cS -f "$NORMALIZER" "$input" | sha256sum | awk '{print $1}'
}

passed=0
for fixture in baseline multi-file-baseline; do
  original_root=$(jq -er --arg fixture "$fixture" '.fixtures[$fixture].root_path' "$CASES")
  case "$fixture" in
    baseline) new_name=firmware;;
    multi-file-baseline) new_name=multi-file;;
  esac
  jq -c --arg f "$fixture" '.fixtures[$f].report' "$CASES" > "$TMP/base.json"
  base_digest=$(jq --arg root "$original_root" -cS -f "$NORMALIZER" "$TMP/base.json" | sha256sum | awk '{print $1}') \
    || fail "$fixture baseline did not normalize"
  [ -n "$base_digest" ] || fail "$fixture baseline has empty digest"
  seed=1
  while [ "$seed" -le 256 ]; do
    new_root="/tmp/rea-meta-$seed/$new_name"
    input="$TMP/$fixture-$seed.json"
    transformed="$TMP/$fixture-$seed.transformed.json"
    jq --arg oldroot "$original_root" --arg newroot "$new_root" --argjson seed "$seed" "$variant_jq" "$TMP/base.json" > "$transformed" \
      || fail "$fixture seed $seed could not transform report"
    digest=$(jq --arg root "$new_root" -cS -f "$NORMALIZER" "$transformed" | sha256sum | awk '{print $1}') \
      || fail "$fixture seed $seed did not normalize"
    [ "$digest" = "$base_digest" ] || fail "$fixture seed $seed changed semantic digest: expected $base_digest got $digest"
    passed=$((passed+1))
    if [ "$((seed % 64))" -eq 0 ]; then
      printf 'PASS %s: %s/256 variants preserve semantic digest %s\n' "$fixture" "$seed" "$base_digest"
    fi
    seed=$((seed+1))
  done
  printf 'PASS %s: 256/256 equivalent metamorphic variants; digest=%s\n' "$fixture" "$base_digest"
done

printf '\nPASS ENG-REA-006: %s/512 equivalent variants preserve semantics.\n' "$passed"
printf 'Seed range: 1..256 per fixture; source checkout HEAD: %s\n' "$SOURCE_SHA"
printf 'Transformations: scratch-root rebase, generated producer IDs, report/task/subtask ordering.\n'
printf 'Negative controls are enforced by the adjacent ENG-REA-005 fixed corpus and mutation controls.\n'
printf 'NOTE: metamorphic fixture sweep only; no extraction provider, firmware execution, emulator, or hardware was run.\n'
