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
check_blob "scripts/wcare47_selftest.py" "bea208b796d20b01c69f9b12c1073775c7cc9293"

python3 -m py_compile scripts/wcare47_lock_admit.py scripts/wcare47_selftest.py || {
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"python_compile_failed","runtime_authority_granted":false}'
  exit 4
}

set +e
OUTPUT="$(python3 scripts/wcare47_selftest.py 2>&1)"
STATUS=$?
set -e

[[ $STATUS -eq 0 ]] || {
  printf '%s\n' "$OUTPUT"
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"wcare47_selftest_failed","runtime_authority_granted":false}'
  exit 4
}

for marker in \
  'PASS_WCARE47_PRELOCK_FAIL_CLOSED' \
  '"classification":"INFRASTRUCTURE_INDETERMINATE"' \
  '"detail":"candidate_lock_missing"' \
  '"exact_source_subject_bound":true' \
  '"rust_toolchain_subject_bound":true' \
  '"lock_admitted":false' \
  '"wcare42_executable_qualification_established":false' \
  '"runtime_authority_granted":false'; do
  grep -Fq "$marker" <<<"$OUTPUT" || {
    printf '%s\n' "$OUTPUT"
    printf '%s\n' "{\"authority\":\"MeasurementOnly\",\"classification\":\"INVALID_PROTOCOL\",\"detail\":\"missing_marker:$marker\",\"runtime_authority_granted\":false}"
    exit 4
  }
done

printf '%s\n' "$OUTPUT"
printf '%s\n' 'PASS_PROTOCOL_INTEGRITY'
printf '%s\n' '{"authority":"MeasurementOnly","classification":"INFRASTRUCTURE_INDETERMINATE","detail":"candidate_lock_missing","lock_admitted":false,"wcare42_executable_qualification_established":false,"runtime_authority_granted":false}'