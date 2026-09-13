#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"not_in_git_worktree","runtime_authority_granted":false}'
  exit 4
}
cd "$ROOT"

BASE_WCARE37="f3aaac0eefe01436c1e287b7de79ec0cbe5e4132"

emit() {
  local classification="$1" detail="$2"
  printf '{"authority":"MeasurementOnly","classification":"%s","detail":"%s","head":"%s","wcare37_base":"%s","runtime_authority_granted":false}\n' \
    "$classification" "$detail" "$(git rev-parse HEAD)" "$BASE_WCARE37"
}

if ! git merge-base --is-ancestor "$BASE_WCARE37" HEAD; then
  emit "INVALID_PROTOCOL" "exact_wcare37_base_not_ancestor"
  exit 4
fi

check_blob() {
  local path="$1" expected="$2" actual
  actual="$(git hash-object "$path" 2>/dev/null || true)"
  if [[ "$actual" != "$expected" ]]; then
    emit "INVALID_PROTOCOL" "blob_mismatch:$path:$actual"
    exit 4
  fi
}

check_blob "docs/release/evidence/WCARE38_AUTHENTICATED_PANEL_PROTOCOL_V1.md" "ebffc2a6bf8f5114a6e852791ef881c4b0219bcf"
check_blob "docs/release/evidence/WCARE38_AUTHENTICATED_PANEL_PLAN_SCHEMA_V1.json" "2b2650bf1b5b7269209e36415e5a74fcd77ed080"
check_blob "docs/release/evidence/WCARE38_ATTESTATION_PACKAGE_MANIFEST_SCHEMA_V1.json" "a87ecc428c4974672a01f322367fba62735c0bd2"
check_blob "docs/release/evidence/WCARE38_AUTHENTICATED_PANEL_RESULT_SCHEMA_V1.json" "f374cd71a864eb179d68c815abf4e9b0420de9ca"
check_blob "scripts/wcare38-qualify.py" "885b952a8a75fba59a75544429b0338fd0ff2519"
check_blob "scripts/wcare38_qualify_authenticated_panel.py" "cf198d78e76eab05a64c3489bdd51f538f026a1d"
check_blob "scripts/wcare38_monotonicity_selftest.py" "e9e82d13cfdef768cb5873359d77f98de465495a"
check_blob "scripts/wcare38_adversarial_selftest.py" "b522f8e81114ee9d41383e5409f6bf6c465c89df"

if ! python3 scripts/wcare38_monotonicity_selftest.py >/tmp/wcare38-monotonicity-selftest.json 2>/tmp/wcare38-monotonicity-selftest.stderr; then
  emit "INVALID_PROTOCOL" "monotonicity_selftest_failed"
  exit 4
fi
if ! grep -q '"classification":"PASS_MONOTONICITY_SELFTEST"' /tmp/wcare38-monotonicity-selftest.json; then
  emit "INVALID_PROTOCOL" "monotonicity_selftest_did_not_report_pass"
  exit 4
fi

if ! python3 scripts/wcare38_adversarial_selftest.py >/tmp/wcare38-adversarial-selftest.json 2>/tmp/wcare38-adversarial-selftest.stderr; then
  emit "INVALID_PROTOCOL" "adversarial_source_selftest_failed"
  exit 4
fi
if ! grep -q '"classification":"PASS_ADVERSARIAL_SOURCE_SELFTEST"' /tmp/wcare38-adversarial-selftest.json; then
  emit "INVALID_PROTOCOL" "adversarial_source_selftest_did_not_report_pass"
  exit 4
fi

emit "PASS_PROTOCOL_INTEGRITY" "exact_wcare38_bytes_and_adversarial_tests_match"
exit 0
