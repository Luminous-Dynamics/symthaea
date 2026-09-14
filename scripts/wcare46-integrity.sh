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

check_blob "docs/release/evidence/WCARE46_EXACT_CHILD_TREE_INTEGRATION_PROTOCOL_V1.md" "40e3f34b41a5d9ccb8e813b60837423aa28ae253"
check_blob "docs/release/evidence/WCARE46_PREFLIGHT_RESULT_SCHEMA_V1.json" "1b263b8aad3bbddfc973371599ee6e0ca39358e3"
check_blob "scripts/wcare46-preflight.py" "fd921dcd66ea85e872fd5e1bf30b1609a67ea432"
check_blob "scripts/wcare46_selftest.py" "7a6c38e974fd0be3f8762df6e5b31976272b506d"

python3 -m py_compile scripts/wcare46-preflight.py scripts/wcare46_selftest.py || {
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"python_compile_failed","runtime_authority_granted":false}'
  exit 4
}

set +e
OUTPUT="$(python3 scripts/wcare46_selftest.py 2>&1)"
STATUS=$?
set -e

[[ $STATUS -eq 0 ]] || {
  printf '%s\n' "$OUTPUT"
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"wcare46_selftest_failed","runtime_authority_granted":false}'
  exit 4
}

for marker in \
  'PASS_WCARE46_EXACT_TREE_INTEGRATION' \
  '"classification":"CHILD_EXECUTION_INDETERMINATE"' \
  '"lineage_convergence_verified":true' \
  '"full_child_tree_census_verified":true' \
  '"standalone_lock_present":false' \
  '"child_execution_established":false' \
  '"runtime_authority_granted":false'; do
  grep -Fq "$marker" <<<"$OUTPUT" || {
    printf '%s\n' "$OUTPUT"
    printf '%s\n' "{\"authority\":\"MeasurementOnly\",\"classification\":\"INVALID_PROTOCOL\",\"detail\":\"missing_marker:$marker\",\"runtime_authority_granted\":false}"
    exit 4
  }
done

printf '%s\n' "$OUTPUT"
printf '%s\n' 'PASS_PROTOCOL_INTEGRITY'
printf '%s\n' '{"authority":"MeasurementOnly","classification":"CHILD_EXECUTION_INDETERMINATE","exact_child_tree_integration_verified":true,"standalone_lock_present":false,"child_execution_established":false,"runtime_authority_granted":false}'