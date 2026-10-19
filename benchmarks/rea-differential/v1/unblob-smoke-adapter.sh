#!/usr/bin/env bash
# Narrow first adapter for pinned Unblob gzip -> tar -> payload.txt smoke fixtures.
# Emits observations; the Rust scorer decides pass/partial/fail. Never executes extracted content.
set -euo pipefail
fail(){ printf 'FAIL: %s\n' "$*" >&2; exit 1; }
[[ $# -eq 6 ]] || fail "usage: $0 RAW_REPORT INPUT_GZIP EXTRACTED_ROOT ORACLE.json PROVIDER_VERSION OBSERVATION.json"
RAW_REPORT=$1
INPUT_GZIP=$2
EXTRACTED_ROOT=$3
ORACLE=$4
PROVIDER_VERSION=$5
OUT=$6
for file in "$RAW_REPORT" "$INPUT_GZIP" "$ORACLE"; do [[ -f "$file" ]] || fail "required input missing: $file"; done
[[ -d "$EXTRACTED_ROOT" ]] || fail "extraction root missing: $EXTRACTED_ROOT"
command -v jq >/dev/null 2>&1 || fail "jq required"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum required"
command -v wc >/dev/null 2>&1 || fail "wc required"
command -v find >/dev/null 2>&1 || fail "find required"
command -v awk >/dev/null 2>&1 || fail "awk required"
command -v grep >/dev/null 2>&1 || fail "grep required"
if find "$EXTRACTED_ROOT" -type l -print -quit | grep -q .; then
  fail "provider extraction contains symlinks outside this smoke adapter's scope"
fi
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

ROOT_TASK=$(jq -ce '[.[] | select(.task.depth==0)] | if length==1 then .[0] else error("expected exactly one depth-zero source task") end' "$RAW_REPORT") \
  || fail "cannot identify the Unblob source task"
ROOT_PATH=$(printf '%s\n' "$ROOT_TASK" | jq -er '.task.path')
[[ "$ROOT_PATH" == "$INPUT_GZIP" ]] || fail "raw report root task path does not identify the supplied input"
CHILD_TASK=$(jq -ce '[.[] | select(.task.depth==1 and any(.reports[]?; .__typename__=="ChunkReport" and .handler_name=="tar"))] | if length==1 then .[0] else error("expected exactly one depth-one task with a tar ChunkReport") end' "$RAW_REPORT") \
  || fail "cannot identify the Unblob tar-stream coordinate space"
CHILD_PATH=$(printf '%s\n' "$CHILD_TASK" | jq -er '.task.path')
case "$CHILD_PATH" in "$EXTRACTED_ROOT"/*) ;; *) fail "tar-stream coordinate path escapes the extraction root: $CHILD_PATH";; esac
[[ -f "$CHILD_PATH" ]] || fail "Unblob tar-stream bytes are not present at the reported path"
[[ "$CHILD_PATH" != "$INPUT_GZIP" ]] || fail "root and child coordinate spaces unexpectedly alias"

ROOT_SHA=$(sha256sum "$INPUT_GZIP" | awk '{print $1}')
ROOT_LEN=$(wc -c < "$INPUT_GZIP" | tr -d ' ')
CHILD_SHA=$(sha256sum "$CHILD_PATH" | awk '{print $1}')
CHILD_LEN=$(wc -c < "$CHILD_PATH" | tr -d ' ')

ROOT_GZIP_COUNT=$(printf '%s\n' "$ROOT_TASK" | jq '[.reports[]? | select(.__typename__=="ChunkReport" and .handler_name=="gzip")] | length')
ROOT_GZIP_START=$(printf '%s\n' "$ROOT_TASK" | jq -r '[.reports[]? | select(.__typename__=="ChunkReport" and .handler_name=="gzip")][0].start_offset // 0')
ROOT_GZIP_END=$(printf '%s\n' "$ROOT_TASK" | jq -r '[.reports[]? | select(.__typename__=="ChunkReport" and .handler_name=="gzip")][0].end_offset // 0')
CHILD_TAR_COUNT=$(printf '%s\n' "$CHILD_TASK" | jq '[.reports[]? | select(.__typename__=="ChunkReport" and .handler_name=="tar")] | length')
CHILD_TAR_START=$(printf '%s\n' "$CHILD_TASK" | jq -r '[.reports[]? | select(.__typename__=="ChunkReport" and .handler_name=="tar")][0].start_offset // 0')
CHILD_TAR_END=$(printf '%s\n' "$CHILD_TASK" | jq -r '[.reports[]? | select(.__typename__=="ChunkReport" and .handler_name=="tar")][0].end_offset // 0')

# Every emitted leaf remains visible to the scorer; canonicalize paths relative
# to the deepest provider-reported extraction directory, preserving subdirectories.
find "$EXTRACTED_ROOT" -type f ! -path "$CHILD_PATH" -print | LC_ALL=C sort > "$TMP/leaf-paths.txt"
: > "$TMP/files.ndjson"
while IFS= read -r LEAF; do
  [[ -n "$LEAF" ]] || continue
  WRAPPER=$(jq -r --arg leaf "$LEAF" '
    [.[] |
      select(any(.reports[]?; .__typename__=="StatReport" and .is_dir==true)) |
      .task.path as $dir |
      select($leaf|startswith($dir + "/")) |
      $dir
    ] | sort_by(length) | last // empty
  ' "$RAW_REPORT")
  if [[ -z "$WRAPPER" ]]; then WRAPPER=$EXTRACTED_ROOT; fi
  CANONICAL_PATH=$(printf '%s\n' "$LEAF" | awk -v prefix="$WRAPPER/" 'index($0,prefix)==1 {print substr($0,length(prefix)+1)}')
  [[ -n "$CANONICAL_PATH" ]] || fail "cannot map provider output to a normalized relative path: $LEAF"
  case "$CANONICAL_PATH" in /*|../*|*/../*|*\\*) fail "unsafe normalized output path: $CANONICAL_PATH";; esac
  LEAF_SHA=$(sha256sum "$LEAF" | awk '{print $1}')
  LEAF_LEN=$(wc -c < "$LEAF" | tr -d ' ')
  LOGICAL_ID=$(jq -r --arg path "$CANONICAL_PATH" '[.expected_files[] | select(.path==$path) | .logical_id][0] // ("unmatched:" + $path)' "$ORACLE")
  jq -cn --arg path "$CANONICAL_PATH" --arg sha "$LEAF_SHA" --argjson length "$LEAF_LEN" --arg logical_id "$LOGICAL_ID" \
    '{path:$path,sha256:$sha,byte_length:$length,logical_id:$logical_id}' >> "$TMP/files.ndjson"
done < "$TMP/leaf-paths.txt"
jq -s '.' "$TMP/files.ndjson" > "$TMP/files.json"

jq -n \
  --slurpfile raw "$RAW_REPORT" \
  --slurpfile oracle "$ORACLE" \
  --slurpfile files "$TMP/files.json" \
  --arg root_path "$ROOT_PATH" --arg child_path "$CHILD_PATH" \
  --arg root_sha "$ROOT_SHA" --argjson root_len "$ROOT_LEN" \
  --arg child_sha "$CHILD_SHA" --argjson child_len "$CHILD_LEN" \
  --arg provider_version "$PROVIDER_VERSION" \
  --argjson root_gzip_count "$ROOT_GZIP_COUNT" --argjson root_start "$ROOT_GZIP_START" --argjson root_end "$ROOT_GZIP_END" \
  --argjson child_tar_count "$CHILD_TAR_COUNT" --argjson child_start "$CHILD_TAR_START" --argjson child_end "$CHILD_TAR_END" \
  '
  def chunk_spans($task_path; $space):
    [$raw[0][] | select(.task.path==$task_path) | .reports[]? |
      select(.__typename__=="ChunkReport" or .__typename__=="UnknownChunkReport") |
      select((.start_offset|type)=="number" and (.end_offset|type)=="number") |
      {
        coordinate_space_id:$space,
        start:.start_offset,
        end:.end_offset,
        disposition:(if .__typename__=="ChunkReport" then "identified" else "unknown" end),
        kind:(if .__typename__=="ChunkReport" then ((.handler_name // "unknown")+"-container") else "unknown-chunk" end)
      }],
  def ledger($task_path; $space; $length):
    (chunk_spans($task_path; $space) | sort_by(.start,.end)) as $spans
    | (reduce $spans[] as $span ({cursor:0,spans:[]};
        .spans +=
          (if $span.start>.cursor
           then [{coordinate_space_id:$space,start:.cursor,end:$span.start,disposition:"unknown"}]
           else [] end)
          + [{coordinate_space_id:$space,start:$span.start,end:$span.end,disposition:$span.disposition}]
        | .cursor=(if $span.end>.cursor then $span.end else .cursor end)
      )) as $state
    | (if ($spans|length)==0
       then [{coordinate_space_id:$space,start:0,end:$length,disposition:"unknown"}]
       else $state.spans + (if $state.cursor<$length then [{coordinate_space_id:$space,start:$state.cursor,end:$length,disposition:"unknown"}] else [] end)
       end),
  def region_rows($task_path; $space):
    [chunk_spans($task_path; $space)[] | {coordinate_space_id:.coordinate_space_id,kind:.kind,start:.start,end:.end}],
  ($oracle[0]) as $o
  | {
    schema_version:"symthaea.rea-differential-observation.v1",
    case_id:$o.case_id,
    provider_id:"unblob",
    provider_version:$provider_version,
    capability:"supported",
    source:{sha256:$root_sha,byte_length:$root_len},
    coordinate_spaces:[
      {id:"root",sha256:$root_sha,byte_length:$root_len,parent_coordinate_space_id:null,parent_start:null,parent_end:null,transform_kind:"source"},
      {id:"tar-stream",sha256:$child_sha,byte_length:$child_len,parent_coordinate_space_id:"root",parent_start:$root_start,parent_end:$root_end,transform_kind:"decompress:gzip"}
    ],
    accounting:(ledger($root_path;"root";$root_len) + ledger($child_path;"tar-stream";$child_len)),
    regions:(region_rows($root_path;"root") + region_rows($child_path;"tar-stream")),
    files:$files[0],
    complete_extraction_claim:(
      $root_gzip_count==1 and $root_start==0 and $root_end==$root_len and
      $child_tar_count==1 and $child_start==0 and $child_end==$child_len and
      ((ledger($root_path;"root";$root_len)+ledger($child_path;"tar-stream";$child_len)) |
       all(.[];.disposition=="identified" or .disposition=="padding"))
    )
  }' > "$OUT" || fail "could not convert Unblob raw report to differential observation"
printf 'PASS Unblob observation normalized: case=%s root_bytes=%s tar_bytes=%s outputs=%s\n' \
  "$(jq -er '.case_id' "$OUT")" "$ROOT_LEN" "$CHILD_LEN" "$(jq -er '.files|length' "$OUT")"
