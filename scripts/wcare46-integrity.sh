#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"not_in_git_worktree","runtime_authority_granted":false}'
  exit 4
}
cd "$ROOT"

check_blob() {
  local path="$1" expected="$2" actual
  actual="$(git hash-object "$path" 2>/dev/null || true)"
  if [[ "$actual" != "$expected" ]]; then
    printf '%s\n' "{\"authority\":\"MeasurementOnly\",\"classification\":\"INVALID_PROTOCOL\",\"detail\":\"blob_mismatch:$path:$actual\",\"runtime_authority_granted\":false}"
    exit 4
  fi
}

check_blob "docs/release/evidence/WCARE46_SELECTIVE_CHILD_TREE_INTEGRATION_PROTOCOL_V1.md" "782cd50330cde06afeeb0b3df692b4208c148284"
check_blob "docs/release/evidence/WCARE46_CHILD_TREE_MANIFEST_V1.json" "d3eb490f9dda631d0cd35a6addf19157882e934c"
check_blob "scripts/wcare46_child_tree_verify.py" "ee9d2fe28c0fe5778791b8b0f12b416fae02f020"
check_blob "scripts/wcare46_selftest.py" "556ec8fe73c9ca3803fa2e267593c1a31b845f51"

if ! python3 - <<'PY'
from pathlib import Path
import json
manifest = json.loads(Path('docs/release/evidence/WCARE46_CHILD_TREE_MANIFEST_V1.json').read_text())
assert manifest['protocol_version'] == 'wcare46-selective-child-tree-integration-v1'
assert manifest['authority'] == 'MeasurementOnly'
assert manifest['wcare42_file_count'] == 9
assert manifest['wcare43_file_count'] == 15
assert manifest['total_file_count'] == 24
assert len(manifest['entries']) == 24
assert len({e['path'] for e in manifest['entries']}) == 24
assert all(e['mode'] == '100644' for e in manifest['entries'])
assert manifest['wcare42_standalone_lock_expected_absent'] is True
for field in (
    'child_reexecution_established',
    'child_verifier_lineage_established',
    'builder_authentication_established',
    'preregistration_temporal_precedence_established',
    'authenticated_preregistered_replication_established',
    'runtime_authority_granted',
):
    assert manifest[field] is False
for path in ('scripts/wcare46_child_tree_verify.py', 'scripts/wcare46_selftest.py'):
    compile(Path(path).read_text(), path, 'exec')
PY
then
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"manifest_or_python_integrity_failed","runtime_authority_granted":false}'
  exit 4
fi

if ! python3 scripts/wcare46_selftest.py >/tmp/wcare46-selftest.json 2>/tmp/wcare46-selftest.stderr; then
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"wcare46_selftest_failed","runtime_authority_granted":false}'
  exit 4
fi
for marker in \
  '"classification":"PASS_WCARE46_SELFTEST"' \
  '"exact_24_file_integration_verified":true' \
  '"exact_blob_and_mode_identity_verified":true' \
  '"child_tree_absence_blockers_removed":true' \
  '"wcare42_lock_blocker_preserved":true' \
  '"child_reexecution_blocker_preserved":true' \
  '"promotion_remains_blocked":true' \
  '"runtime_authority_granted":false'
do
  if ! grep -q "$marker" /tmp/wcare46-selftest.json; then
    printf '%s\n' "{\"authority\":\"MeasurementOnly\",\"classification\":\"INVALID_PROTOCOL\",\"detail\":\"selftest_missing_marker:$marker\",\"runtime_authority_granted\":false}"
    exit 4
  fi
done

printf '%s\n' '{"authority":"MeasurementOnly","classification":"PASS_PROTOCOL_INTEGRITY","detail":"exact_wcare46_child_tree_bytes_and_fail_closed_transition_match","exact_child_tree_integration_established":true,"child_execution_indeterminate":true,"authenticated_preregistered_replication_established":false,"runtime_authority_granted":false}'
exit 0
