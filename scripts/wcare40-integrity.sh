#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"not_in_git_worktree","runtime_authority_granted":false}'
  exit 4
}
cd "$ROOT"

BASE_WCARE39="42423b1e3d26d4203e40378df060ea025a3c6a65"

emit() {
  local classification="$1" detail="$2"
  printf '{"authority":"MeasurementOnly","classification":"%s","detail":"%s","head":"%s","wcare39_base":"%s","runtime_authority_granted":false}\n' \
    "$classification" "$detail" "$(git rev-parse HEAD)" "$BASE_WCARE39"
}

if ! git merge-base --is-ancestor "$BASE_WCARE39" HEAD; then
  emit "INVALID_PROTOCOL" "exact_wcare39_base_not_ancestor"
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

check_blob "docs/release/evidence/WCARE40_EXECUTION_REPLICATION_PROTOCOL_V1.md" "da09bf7adef07065a22bc5a2dd6e471d08aa139e"
check_blob "docs/release/evidence/WCARE40_REPLICATION_PLAN_SCHEMA_V1.json" "244f8b8fa063238d22c408365f5655f365cc70b6"
check_blob "docs/release/evidence/WCARE40_BUILDER_PROVENANCE_SCHEMA_V1.json" "d9dcceb5da983cc411e44cb1527bca2052bd7cd2"
check_blob "docs/release/evidence/WCARE40_BUILDER_RELATION_SCHEMA_V1.json" "8b00caac8bf934636789b74d048cabcce63b29ee"
check_blob "docs/release/evidence/WCARE40_REPLICATION_RESULT_SCHEMA_V1.json" "7c2f752527504c453a25bd4b3f04c586e3f69d3a"
check_blob "scripts/wcare40_verify_replication.py" "a368dc46b33272d6f424212a29ac26d5170b2e5a"
check_blob "scripts/wcare40-qualify.py" "13675951a5a8bd21c341cf6779341aa9e497b4d1"
check_blob "scripts/wcare40_selftest.py" "f782b10b17ebf41ce42647c4ee84eb938b4d0fb5"
check_blob "scripts/wcare40_frontdoor_selftest.py" "b95fc828b71a8f980bba0b3b5d5c29c25a6870e9"

if ! python3 - <<'PY'
from pathlib import Path
import json
for path in (
    Path('docs/release/evidence/WCARE40_REPLICATION_PLAN_SCHEMA_V1.json'),
    Path('docs/release/evidence/WCARE40_BUILDER_PROVENANCE_SCHEMA_V1.json'),
    Path('docs/release/evidence/WCARE40_BUILDER_RELATION_SCHEMA_V1.json'),
    Path('docs/release/evidence/WCARE40_REPLICATION_RESULT_SCHEMA_V1.json'),
):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AssertionError(f'non-object schema: {path}')
for path in (
    Path('scripts/wcare40_verify_replication.py'),
    Path('scripts/wcare40-qualify.py'),
    Path('scripts/wcare40_selftest.py'),
    Path('scripts/wcare40_frontdoor_selftest.py'),
):
    compile(path.read_text(), str(path), 'exec')
PY
then
  emit "INVALID_PROTOCOL" "schema_parse_or_python_compile_failed"
  exit 4
fi

if ! python3 scripts/wcare40_selftest.py >/tmp/wcare40-selftest.json 2>/tmp/wcare40-selftest.stderr; then
  emit "INVALID_PROTOCOL" "core_replication_selftest_failed"
  exit 4
fi
if ! grep -q '"classification":"PASS_WCARE40_SELFTEST"' /tmp/wcare40-selftest.json; then
  emit "INVALID_PROTOCOL" "core_replication_selftest_did_not_report_pass"
  exit 4
fi

if ! python3 scripts/wcare40_frontdoor_selftest.py >/tmp/wcare40-frontdoor-selftest.json 2>/tmp/wcare40-frontdoor-selftest.stderr; then
  emit "INVALID_PROTOCOL" "frontdoor_replication_selftest_failed"
  exit 4
fi
if ! grep -q '"classification":"PASS_WCARE40_FRONTDOOR_SELFTEST"' /tmp/wcare40-frontdoor-selftest.json; then
  emit "INVALID_PROTOCOL" "frontdoor_replication_selftest_did_not_report_pass"
  exit 4
fi
if ! grep -q '"forged_final_outcome_rejected":true' /tmp/wcare40-frontdoor-selftest.json; then
  emit "INVALID_PROTOCOL" "frontdoor_forged_outcome_guard_not_verified"
  exit 4
fi

emit "PASS_PROTOCOL_INTEGRITY" "exact_wcare40_bytes_and_adversarial_campaigns_match"
exit 0
