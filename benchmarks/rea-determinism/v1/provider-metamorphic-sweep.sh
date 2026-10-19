#!/bin/sh
# ENG-REA-006 metamorphic stress over an actual Unblob-produced report.
# Arguments: raw report, source scratch root, expected semantic SHA-256, output receipt.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
NORMALIZER="$ROOT/normalize.jq"
fail(){ printf 'FAIL: %s\n' "$*" >&2; exit 1; }
[ "$#" -eq 4 ] || fail "usage: $0 RAW_REPORT SOURCE_ROOT EXPECTED_SEMANTIC_SHA RECEIPT_JSON"
RAW_REPORT=$1
SOURCE_ROOT=$2
EXPECTED_SEMANTIC_SHA=$3
RECEIPT=$4
command -v jq >/dev/null 2>&1 || fail "jq required"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum required"
[ -s "$RAW_REPORT" ] || fail "actual provider raw report missing"
[ -f "$NORMALIZER" ] || fail "semantic normalizer missing"

raw_sha_before=$(sha256sum "$RAW_REPORT" | awk '{print $1}')
normalizer_sha=$(sha256sum "$NORMALIZER" | awk '{print $1}')
jq --arg root "$SOURCE_ROOT" -cS -f "$NORMALIZER" "$RAW_REPORT" > "$RECEIPT.baseline.json" \
  || fail "actual raw report does not satisfy semantic normalizer"
baseline_sha=$(sha256sum "$RECEIPT.baseline.json" | awk '{print $1}')
[ "$baseline_sha" = "$EXPECTED_SEMANTIC_SHA" ] || fail "caller semantic digest does not match independently normalized raw report"

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
          . as $entry |
          if (["path","carve_dir","extraction_path","link_path"]|index($entry.key))!=null and ($entry.value|type)=="string"
            then .value=rebase($oldroot;$newroot;$entry.value)
          elif $entry.key=="paths" and ($entry.value|type)=="array" then .value |= map(rebase($oldroot;$newroot;.))
          else . end)
        else . end)
    )
)
| map(.reports |= sort_by((tojson|explode|reduce .[] as $code ($seed; ((.*33+$code)%2147483647))))
       | .subtasks |= sort_by((tojson|explode|reduce .[] as $code ($seed; ((.*33+$code)%2147483647)))))
| sort_by((tojson|explode|reduce .[] as $code ($seed; ((.*33+$code)%2147483647))))
';

TMP=$(mktemp -d)
trap 'rm -rf "$TMP" "$RECEIPT.baseline.json"' EXIT HUP INT TERM
passed=0
seed=1
while [ "$seed" -le 64 ]; do
  newroot="/tmp/rea-provider-meta-$seed"
  transformed="$TMP/provider-$seed.json"
  projection="$TMP/provider-$seed.projection.json"
  jq --arg oldroot "$SOURCE_ROOT" --arg newroot "$newroot" --argjson seed "$seed" "$variant_jq" "$RAW_REPORT" > "$transformed" \
    || fail "provider report seed $seed could not be transformed"
  jq --arg root "$newroot" -cS -f "$NORMALIZER" "$transformed" > "$projection" \
    || fail "provider report seed $seed did not normalize"
  digest=$(sha256sum "$projection" | awk '{print $1}')
  [ "$digest" = "$baseline_sha" ] || fail "actual provider semantic digest changed for equivalent seed $seed: expected $baseline_sha got $digest"
  passed=$((passed+1))
  if [ "$((seed % 16))" -eq 0 ]; then
    printf 'PASS actual-provider metamorphic sweep: %s/64 equivalent report variants preserve %s\n' "$seed" "$baseline_sha"
  fi
  seed=$((seed+1))
done
raw_sha_after=$(sha256sum "$RAW_REPORT" | awk '{print $1}')
[ "$raw_sha_before" = "$raw_sha_after" ] || fail "metamorphic sweep modified the exact raw provider report"
jq -n \
  --arg raw_report_sha256 "$raw_sha_before" \
  --arg normalizer_sha256 "$normalizer_sha" \
  --arg semantic_projection_sha256 "$baseline_sha" \
  --arg source_root "$SOURCE_ROOT" \
  --argjson seed_start 1 \
  --argjson seed_end 64 \
  --argjson equivalent_variants "$passed" \
  '{schema_version:"symthaea.rea-provider-metamorphic-receipt.v1",raw_report_sha256:$raw_report_sha256,normalizer_sha256:$normalizer_sha256,semantic_projection_sha256:$semantic_projection_sha256,source_root_for_audit:$source_root,seed_start:$seed_start,seed_end:$seed_end,equivalent_variants:$equivalent_variants,all_variants_preserved_semantic_projection:($equivalent_variants==64),raw_report_unchanged:true,transformations:["scratch-root rebase","coherent generated producer-ID remap","TaskResult ordering","report ordering","subtask ordering"],claims_ceiling:"64 metamorphic variants derived from one actual Unblob 26.6.4 report; not 64 provider executions, and not a firmware behavior or device-qualification result."}' > "$RECEIPT"
jq -e --arg raw "$raw_sha_before" --arg norm "$normalizer_sha" --arg sem "$baseline_sha" \
  '.schema_version=="symthaea.rea-provider-metamorphic-receipt.v1" and .raw_report_sha256==$raw and .normalizer_sha256==$norm and .semantic_projection_sha256==$sem and .equivalent_variants==64 and .all_variants_preserved_semantic_projection==true and .raw_report_unchanged==true' \
  "$RECEIPT" >/dev/null || fail "provider metamorphic receipt failed consistency validation"
printf 'PASS ENG-REA-006 real-provider metamorphic sweep: %s/64 equivalent variants; raw report unchanged; semantic digest=%s\n' "$passed" "$baseline_sha"
