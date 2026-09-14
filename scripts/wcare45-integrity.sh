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

check_blob "docs/release/evidence/WCARE45_CHILD_VERIFIER_PROMOTION_PROTOCOL_V1.md" "85db0ee96aa0e3673f8e8f586a7bdff93f569b4a"
check_blob "docs/release/evidence/WCARE45_PREFLIGHT_RESULT_SCHEMA_V1.json" "b0640098c16b59ec126031dd4f7eeae7698dc1b5"
check_blob "scripts/wcare45-preflight.py" "bd76fe8cd51c6d45599030d4f89a60fdffe9f575"
check_blob "scripts/wcare45_selftest.py" "ef02f19b46f4728326af908212d1a66a979c01de"

python3 - <<'PY' || exit 4
from pathlib import Path
import json
schema = json.loads(Path('docs/release/evidence/WCARE45_PREFLIGHT_RESULT_SCHEMA_V1.json').read_text())
assert schema['additionalProperties'] is False
for field in (
    'child_verifier_lineage_established',
    'wcare42_executable_qualification_established',
    'wcare43_external_execution_lineage_established',
    'builder_authentication_established',
    'preregistration_temporal_precedence_established',
    'authenticated_preregistered_replication_established',
    'runtime_authority_granted',
):
    assert schema['properties'][field]['const'] is False
for path in ('scripts/wcare45-preflight.py','scripts/wcare45_selftest.py'):
    compile(Path(path).read_text(), path, 'exec')
PY

if ! python3 scripts/wcare45_selftest.py >/tmp/wcare45-selftest.json 2>/tmp/wcare45-selftest.stderr; then
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"preflight_selftest_failed","runtime_authority_granted":false}'
  exit 4
fi
for marker in \
  '"classification":"PASS_WCARE45_PREFLIGHT_SELFTEST"' \
  '"exact_three_parent_ancestry_verified":true' \
  '"ancestry_does_not_imply_child_tree_integration":true' \
  '"missing_child_implementations_remain_visible":true' \
  '"missing_wcare42_lock_remains_visible":true' \
  '"child_execution_indeterminate_not_promoted":true' \
  '"runtime_authority_granted":false'
do
  grep -q "$marker" /tmp/wcare45-selftest.json || exit 4
done

printf '%s\n' '{"authority":"MeasurementOnly","classification":"PASS_PROTOCOL_INTEGRITY","detail":"exact_wcare45_preflight_bytes_and_fail_closed_campaign_match","child_verifier_lineage_established":false,"authenticated_preregistered_replication_established":false,"runtime_authority_granted":false}'
exit 0
