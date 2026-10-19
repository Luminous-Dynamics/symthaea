#!/usr/bin/env bash
# Build an oracle strictly from known fixture-construction bytes, never a provider report.
set -euo pipefail
fail(){ printf 'FAIL: %s\n' "$*" >&2; exit 1; }
[[ $# -eq 6 ]] || fail "usage: $0 INPUT_GZIP EXPECTED_TAR EXPECTED_PAYLOAD CASE_ID LOGICAL_ID ORACLE_OUT"
INPUT_GZIP=$1
EXPECTED_TAR=$2
EXPECTED_PAYLOAD=$3
CASE_ID=$4
LOGICAL_ID=$5
OUT=$6
for file in "$INPUT_GZIP" "$EXPECTED_TAR" "$EXPECTED_PAYLOAD"; do
  [[ -f "$file" ]] || fail "known-answer fixture file missing: $file"
done
command -v jq >/dev/null 2>&1 || fail "jq required"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum required"
command -v wc >/dev/null 2>&1 || fail "wc required"
command -v tar >/dev/null 2>&1 || fail "tar required"
command -v sed >/dev/null 2>&1 || fail "sed required"
TAR_MEMBERS=$(tar -tf "$EXPECTED_TAR" | sed '/^$/d')
TAR_MEMBER_COUNT=$(printf '%s\n' "$TAR_MEMBERS" | wc -l | tr -d ' ')
[[ "$TAR_MEMBER_COUNT" -eq 1 && "$TAR_MEMBERS" == "payload.txt" ]] || fail "smoke oracle builder is restricted to a one-member payload.txt tar"
INPUT_SHA=$(sha256sum "$INPUT_GZIP" | awk '{print $1}')
INPUT_LEN=$(wc -c < "$INPUT_GZIP" | tr -d ' ')
TAR_SHA=$(sha256sum "$EXPECTED_TAR" | awk '{print $1}')
TAR_LEN=$(wc -c < "$EXPECTED_TAR" | tr -d ' ')
PAYLOAD_SHA=$(sha256sum "$EXPECTED_PAYLOAD" | awk '{print $1}')
PAYLOAD_LEN=$(wc -c < "$EXPECTED_PAYLOAD" | tr -d ' ')
jq -n \
  --arg case_id "$CASE_ID" \
  --arg logical_id "$LOGICAL_ID" \
  --arg input_sha "$INPUT_SHA" --argjson input_len "$INPUT_LEN" \
  --arg tar_sha "$TAR_SHA" --argjson tar_len "$TAR_LEN" \
  --arg payload_sha "$PAYLOAD_SHA" --argjson payload_len "$PAYLOAD_LEN" \
  '{
    schema_version:"symthaea.rea-differential-oracle.v1",
    corpus_id:"rea-differential-v1",
    case_id:$case_id,
    source:{sha256:$input_sha,byte_length:$input_len},
    coordinate_spaces:[
      {id:"root",sha256:$input_sha,byte_length:$input_len,parent_coordinate_space_id:null,parent_start:null,parent_end:null,transform_kind:"source"},
      {id:"tar-stream",sha256:$tar_sha,byte_length:$tar_len,parent_coordinate_space_id:"root",parent_start:0,parent_end:$input_len,transform_kind:"decompress:gzip"}
    ],
    expected_regions:[
      {coordinate_space_id:"root",kind:"gzip-container",start:0,end:$input_len},
      {coordinate_space_id:"tar-stream",kind:"tar-container",start:0,end:$tar_len}
    ],
    expected_files:[
      {path:"payload.txt",sha256:$payload_sha,byte_length:$payload_len,logical_id:$logical_id}
    ]
  }' > "$OUT"
printf 'PASS oracle built from known fixture bytes: case=%s input=%s tar=%s payload=%s\n' "$CASE_ID" "$INPUT_SHA" "$TAR_SHA" "$PAYLOAD_SHA"
