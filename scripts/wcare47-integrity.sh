#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 4
cd "$ROOT"

check_blob() {
  local path="$1" expected="$2" actual
  actual="$(git hash-object "$path" 2>/dev/null || true)"
  [[ "$actual" == "$expected" ]] || {
    printf '%s\n' "{\"authority\":\"MeasurementOnly\",\"classification\":\"INVALID_PROTOCOL\",\"detail\":\"blob_mismatch:$path:$actual\",\"runtime_authority_granted\":false}"
    exit 4
  }
}

check_blob "docs/release/evidence/WCARE47_STANDALONE_LOCK_ADMISSION_PROTOCOL_V1.md" "331d911cb60a0e9e899a4f9ca359381702090b66"
check_blob "docs/release/evidence/WCARE47_LOCK_ADMISSION_RESULT_SCHEMA_V1.json" "e16c822c4b41bf2cede04745ad4f0cefdab973d9"
check_blob "scripts/wcare47_lock_admit.py" "0456dd638b11d17efdc64ce3d64cbcebab060308"
check_blob "scripts/wcare47_selftest.py" "1fe7dbbee0ec9d3336410d12ab521a77efd7211c"

python3 -m py_compile scripts/wcare47_lock_admit.py scripts/wcare47_selftest.py || {
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"python_compile_failed","runtime_authority_granted":false}'
  exit 4
}

NEGATIVE_OUTPUT="$(python3 scripts/wcare47_selftest.py 2>&1)"
NEGATIVE_STATUS=$?

[[ $NEGATIVE_STATUS -eq 0 ]] || {
  printf '%s\n' "$NEGATIVE_OUTPUT"
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"wcare47_negative_fixture_failed","runtime_authority_granted":false}'
  exit 4
}

for marker in \
  'PASS_WCARE47_PRELOCK_FAIL_CLOSED' \
  'PASS_WCARE47_NEGATIVE_FIXTURE' \
  '"classification":"INFRASTRUCTURE_INDETERMINATE"' \
  '"detail":"candidate_lock_missing"' \
  '"exact_source_subject_bound":true' \
  '"rust_toolchain_subject_bound":true' \
  '"lock_admitted":false' \
  '"wcare42_executable_qualification_established":false' \
  '"runtime_authority_granted":false'; do
  grep -Fq "$marker" <<<"$NEGATIVE_OUTPUT" || {
    printf '%s\n' "$NEGATIVE_OUTPUT"
    printf '%s\n' "{\"authority\":\"MeasurementOnly\",\"classification\":\"INVALID_PROTOCOL\",\"detail\":\"missing_negative_fixture_marker:$marker\",\"runtime_authority_granted\":false}"
    exit 4
  }
done

ACTUAL_OUTPUT="$(python3 scripts/wcare47_lock_admit.py 2>&1)"
ACTUAL_STATUS=$?

PARSED="$(
  python3 -c 'import json,sys; r=json.loads(sys.stdin.read()); print("\t".join([str(r.get("classification","")), str(r.get("detail","")), "1" if r.get("candidate_lock_present") else "0", "1" if r.get("lock_admitted") else "0"]))' <<<"$ACTUAL_OUTPUT"
)" || {
  printf '%s\n' "$ACTUAL_OUTPUT"
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"actual_result_not_json","runtime_authority_granted":false}'
  exit 4
}

IFS=$'\t' read -r CLASSIFICATION DETAIL CANDIDATE_PRESENT LOCK_ADMITTED <<<"$PARSED"

if [[ $ACTUAL_STATUS -eq 3 && "$CLASSIFICATION" == "INFRASTRUCTURE_INDETERMINATE" && "$DETAIL" == "candidate_lock_missing" && "$CANDIDATE_PRESENT" == "0" && "$LOCK_ADMITTED" == "0" ]]; then
  CURRENT_STATE="PRELOCK_MISSING"
elif [[ $ACTUAL_STATUS -eq 0 && "$CLASSIFICATION" == "LOCK_ADMITTED" && "$CANDIDATE_PRESENT" == "1" && "$LOCK_ADMITTED" == "1" ]]; then
  CURRENT_STATE="LOCK_ADMITTED"
else
  printf '%s\n' "$NEGATIVE_OUTPUT"
  printf '%s\n' "$ACTUAL_OUTPUT"
  printf '%s\n' "{\"authority\":\"MeasurementOnly\",\"classification\":\"INVALID_PROTOCOL\",\"detail\":\"current_checkout_not_admissible:$CLASSIFICATION:$DETAIL\",\"runtime_authority_granted\":false}"
  if [[ $ACTUAL_STATUS -eq 1 || $ACTUAL_STATUS -eq 3 || $ACTUAL_STATUS -eq 4 ]]; then
    exit "$ACTUAL_STATUS"
  fi
  exit 4
fi

printf '%s\n' "$NEGATIVE_OUTPUT"
printf '%s\n' "$ACTUAL_OUTPUT"
printf '%s\n' 'PASS_PROTOCOL_INTEGRITY'
printf '%s\n' "PASS_WCARE47_CURRENT_STATE:$CURRENT_STATE"
