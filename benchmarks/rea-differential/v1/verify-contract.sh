#!/bin/sh
# Offline checks for the provider-neutral extraction-fidelity contract.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$ROOT/../../../" && pwd)
CASES="$ROOT/cases.json"
ORACLE="$ROOT/example-oracle.json"
OBSERVATION="$ROOT/example-observation.json"
fail(){ printf 'FAIL: %s\n' "$*" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || fail "jq required"
command -v cargo >/dev/null 2>&1 || fail "cargo required"
[ -f "$CASES" ] && [ -f "$ORACLE" ] && [ -f "$OBSERVATION" ] || fail "catalog or scorecard examples missing"

jq -e '
  .schema_version=="symthaea.rea-differential-cases.v1" and
  .corpus_id=="rea-differential-v1" and
  (.cases|length)==13 and
  ([.cases[].case_id]|unique|length)==13 and
  all(.cases[]; (.mutations|length)>0 and (.required_region_kinds|unique|length)==(.required_region_kinds|length))
' "$CASES" >/dev/null || fail "case catalog schema or uniqueness invariant failed"

jq -e '
  .schema_version=="symthaea.rea-differential-oracle.v1" and
  .source.byte_length==10 and (.source.sha256|test("^[0-9a-f]{64}$")) and
  (.expected_files|length)==1
' "$ORACLE" >/dev/null || fail "known-answer oracle fixture is malformed"

jq -e '
  .schema_version=="symthaea.rea-differential-observation.v1" and
  .capability=="supported" and .complete_extraction_claim==true and
  (.source.sha256|test("^[0-9a-f]{64}$"))
' "$OBSERVATION" >/dev/null || fail "known-answer provider observation fixture is malformed"

cd "$REPO_ROOT"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT HUP INT TERM
run_score(){
  observation=$1
  output=$2
  cargo run --quiet --package symthaea-rea-evidence-verifier -- \
    score-case --oracle "$ORACLE" --observation "$observation" > "$output" 2> "$output.stderr"
}
expect_nonpass(){
  label=$1
  filter=$2
  expected_status=$3
  jq "$filter" "$OBSERVATION" > "$TMP/observation-$label.json" || fail "could not construct mutation $label"
  if run_score "$TMP/observation-$label.json" "$TMP/score-$label.json"; then
    fail "mutation unexpectedly exited zero: $label"
  fi
  jq -e --arg status "$expected_status" '.status==$status and (.complete_extraction_claim_accepted|not)' "$TMP/score-$label.json" >/dev/null \
    || { cat "$TMP/score-$label.json" >&2; fail "mutation $label did not produce status $expected_status"; }
  printf 'PASS differential mutation rejected: %s -> %s\n' "$label" "$expected_status"
}

run_score "$OBSERVATION" "$TMP/score-pass.json" || { cat "$TMP/score-pass.json.stderr" >&2; fail "valid known-answer fixture was rejected"; }
jq -e '.status=="pass" and .complete_extraction_claim_accepted==true and .metrics.known_file_recall_percent==100 and .metrics.unexplained_bytes==0' "$TMP/score-pass.json" >/dev/null \
  || fail "valid known-answer fixture did not earn a full pass"
printf 'PASS exact known-answer fixture earns a full pass.\n'

expect_nonpass wrong-source-hash '.source.sha256=("c"*64)' fail
expect_nonpass wrong-output-content '.files[0].sha256=("f"*64)' fail
expect_nonpass duplicate-path '.files += [.files[0]]' fail
expect_nonpass overlapping-accounting '.accounting=[{start:0,end:7,disposition:"identified"},{start:5,end:10,disposition:"padding"}]' fail
expect_nonpass false-region '.regions[1].end=11' fail
expect_nonpass false-completeness '.accounting=[{start:0,end:7,disposition:"identified"},{start:7,end:10,disposition:"unknown"}]' fail
expect_nonpass unsafe-path '.files[0].path="../escape.bin"' fail
expect_nonpass unresolved-partial '.accounting=[{start:0,end:7,disposition:"identified"},{start:7,end:10,disposition:"unknown"}] | .complete_extraction_claim=false' partial
expect_nonpass unsupported-emits-output '.capability="unsupported"' fail
expect_nonpass honest-abstention '.capability="unsupported" | .files=[] | .regions=[] | .accounting=[] | .complete_extraction_claim=false' abstain

printf '\nPASS ENG-REA-009 scorecard contract: 1 valid case passes and 10 negative/abstention controls behave as specified.\n'
