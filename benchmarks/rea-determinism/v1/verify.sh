#!/bin/sh
# ENG-REA-005 deterministic semantic projection contract.
# Does not execute firmware or start an external extraction provider.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CASES="$ROOT/cases.json"
ORACLE="$ROOT/oracle.json"
NORMALIZER="$ROOT/normalize.jq"
fail(){ printf 'FAIL: %s\n' "$*" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || fail "jq required"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum required"
[ -f "$CASES" ] && [ -f "$ORACLE" ] && [ -f "$NORMALIZER" ] || fail "manifest/oracle/normalizer missing"
jq -e '.schema_version=="symthaea.rea-determinism-cases.v1" and .benchmark_id=="ENG-REA-005" and (.fixtures|length)>=4 and (.cases|length)==24' "$CASES" >/dev/null || fail "manifest schema/count mismatch"
jq -e '.schema_version=="symthaea.rea-determinism-oracle.v1" and .benchmark_id=="ENG-REA-005" and (.expected_relations|length)==24' "$ORACLE" >/dev/null || fail "oracle schema/count mismatch"
case_ids=$(jq -er '[.cases[].id] | sort | join(",")' "$CASES")
oracle_ids=$(jq -er '.expected_relations|keys|sort|join(",")' "$ORACLE")
[ "$case_ids" = "$oracle_ids" ] || fail "case IDs differ or are duplicated"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

apply_mutation(){
  mutation=$1
  input=$2
  output=$3
  case "$mutation" in
    identity) cp "$input" "$output" ;;
    child-hash) jq 'map(if .task.depth==1 then .reports |= map(if .__typename__=="HashReport" then .sha256="3333333333333333333333333333333333333333333333333333333333333333" else . end) else . end)' "$input" > "$output" ;;
    root-hash) jq 'map(if .task.depth==0 then .reports |= map(if .__typename__=="HashReport" then .sha256="7777777777777777777777777777777777777777777777777777777777777777" else . end) else . end)' "$input" > "$output" ;;
    chunk-range) jq 'map(if .task.depth==0 then .reports |= map(if .__typename__=="ChunkReport" then (.start_offset=1 | .size=11) else . end) else . end)' "$input" > "$output" ;;
    handler) jq 'map(if .task.depth==0 then .reports |= map(if .__typename__=="ChunkReport" then .handler_name="lzma" else . end) else . end)' "$input" > "$output" ;;
    child-path) jq 'walk(if type=="string" then gsub("rootfs[.]squashfs";"rootfs2.squashfs") else . end)' "$input" > "$output" ;;
    dangling-subtask) jq 'map(if .task.depth==0 then .subtasks[0].chunk_id="missing:1" else . end)' "$input" > "$output" ;;
    duplicate-producer-id) jq 'map(if .task.depth==1 then .reports |= map(if .__typename__=="ChunkReport" then .id="231:7" else . end) else . end)' "$input" > "$output" ;;
    unknown-report) jq 'map(if .task.depth==0 then .reports += [{"payload":"future schema"}] else . end)' "$input" > "$output" ;;
    outside-root-path) jq 'map(if .task.depth==1 then .reports |= map(if .__typename__=="StatReport" then .path="/etc/passwd" else . end) else . end)' "$input" > "$output" ;;
    mismatched-child-chunk) jq 'map(if .task.depth==1 then .task.chunk_id="231:8" else . end)' "$input" > "$output" ;;
    duplicate-task-path) jq 'map(if .task.depth==1 then .task.path="/tmp/rea-run-a/firmware.bin" else . end)' "$input" > "$output" ;;
    missing-root) jq 'map(select(.task.depth!=0))' "$input" > "$output" ;;
    malformed-range) jq 'map(if .task.depth==0 then .reports |= map(if .__typename__=="ChunkReport" then .size=99 else . end) else . end)' "$input" > "$output" ;;
    wrong-depth) jq 'map(if .task.depth==1 then .task.depth=2 else . end)' "$input" > "$output" ;;
    duplicate-edge) jq 'map(if .task.depth==0 then .subtasks += [.subtasks[0]] else . end)' "$input" > "$output" ;;
    reverse-order) jq 'map(.reports |= reverse | .subtasks |= reverse) | reverse' "$input" > "$output" ;;
    remove-child-hash) jq 'map(if .task.depth==1 then .reports |= map(select(.__typename__!="HashReport")) else . end)' "$input" > "$output" ;;
    multifile-dangling) jq 'map(if .task.depth==0 then .subtasks[0].chunk_id="not-present" else . end)' "$input" > "$output" ;;
    unknown-chunk-id) jq 'map(if .task.depth==0 then .reports |= map(if .__typename__=="UnknownChunkReport" then .id="new-pid:900" else . end) else . end)' "$input" > "$output" ;;
    parent-traversal-path) jq 'map(if .task.depth==1 then .task.path="../escape.bin" else . end)' "$input" > "$output" ;;
    missing-typename) jq 'map(if .task.depth==1 then .reports |= map(if .__typename__=="HashReport" then del(.__typename__) else . end) else . end)' "$input" > "$output" ;;
    *) fail "unimplemented mutation: $mutation" ;;
  esac
}

canonicalize(){
  fixture=$1; mutation=$2; stem=$3
  root_path=$(jq -er --arg f "$fixture" '.fixtures[$f].root_path' "$CASES") || return 2
  jq -c --arg f "$fixture" '.fixtures[$f].report' "$CASES" > "$TMP/$stem.raw.json" || return 2
  apply_mutation "$mutation" "$TMP/$stem.raw.json" "$TMP/$stem.input.json" || return 2
  if jq --arg root "$root_path" -cS -f "$NORMALIZER" "$TMP/$stem.input.json" > "$TMP/$stem.normalized.json" 2> "$TMP/$stem.err"; then
    sha256sum "$TMP/$stem.normalized.json" | awk '{print $1}'
  else
    return 3
  fi
}

passed=0
equivalent=0
changed=0
rejected=0
while IFS= read -r case_json; do
  id=$(printf '%s' "$case_json" | jq -er '.id')
  left_fixture=$(printf '%s' "$case_json" | jq -er '.left_fixture')
  right_fixture=$(printf '%s' "$case_json" | jq -er '.right_fixture')
  mutation=$(printf '%s' "$case_json" | jq -er '.mutation')
  expected=$(jq -er --arg id "$id" '.expected_relations[$id]' "$ORACLE")
  left_hash=$(canonicalize "$left_fixture" identity "$id-left") || fail "$id left fixture failed to normalize"
  if right_hash=$(canonicalize "$right_fixture" "$mutation" "$id-right"); then
    if [ "$left_hash" = "$right_hash" ]; then actual=equivalent; else actual=changed; fi
  else
    actual=reject
  fi
  [ "$actual" = "$expected" ] || fail "$id expected $expected, got $actual"
  case "$actual" in
    equivalent) equivalent=$((equivalent+1));;
    changed) changed=$((changed+1));;
    reject) rejected=$((rejected+1));;
  esac
  printf 'PASS %s -> %s (%s)\n' "$id" "$actual" "$mutation"
  passed=$((passed+1))
done <<EOF_CASES
$(jq -c '.cases[]' "$CASES")
EOF_CASES
printf '\nPASS ENG-REA-005: %s/24 relations matched independent oracle.\n' "$passed"
printf 'Equivalent (volatile IDs / temp paths / ordering ignored): %s\nSemantic differences preserved: %s\nMalformed or ambiguous reports rejected: %s\n' "$equivalent" "$changed" "$rejected"
printf 'NOTE: fixture-based contract only; no external provider, firmware execution, emulator, or hardware was run.\n'
