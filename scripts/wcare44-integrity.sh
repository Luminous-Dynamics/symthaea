#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"not_in_git_worktree","runtime_authority_granted":false}'
  exit 4
}
cd "$ROOT"

BASE_WCARE41="89243a3072a5ff510c8d6d7ea4958644be40f879"

emit() {
  local classification="$1" detail="$2"
  printf '{"authority":"MeasurementOnly","classification":"%s","detail":"%s","head":"%s","wcare41_base":"%s","child_verifier_lineage_established":false,"builder_authentication_established":false,"preregistration_temporal_precedence_established":false,"authenticated_preregistered_replication_established":false,"runtime_authority_granted":false}\n' \
    "$classification" "$detail" "$(git rev-parse HEAD)" "$BASE_WCARE41"
}

if ! git merge-base --is-ancestor "$BASE_WCARE41" HEAD; then
  emit "INVALID_PROTOCOL" "exact_wcare41_base_not_ancestor"
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

check_blob "docs/release/evidence/WCARE44_AUTHENTICATED_REPLICATION_AGGREGATION_PROTOCOL_V1.md" "2c9a65f0cae32a4276f200e440cd643c43537888"
check_blob "docs/release/evidence/WCARE44_BUILDER_AUTH_OBSERVATION_SCHEMA_V1.json" "f129ae7fef8d20856277a97f6f549b9313e98e8c"
check_blob "docs/release/evidence/WCARE44_TEMPORAL_OBSERVATION_SCHEMA_V1.json" "2fabbb71170d0efe34e01a2166d197a44e056b3b"
check_blob "docs/release/evidence/WCARE44_CANDIDATE_RESULT_SCHEMA_V1.json" "2db071910816c1354bc8e14a46e8118ea8052a97"
check_blob "scripts/wcare44_candidate_kernel.py" "c5e0f52e34c4753028a00cf6891c8725d04e6b9a"
check_blob "scripts/wcare44_candidate_qualify.py" "e8f1e9d9eb9dc90710d37acefc19504bb6401e66"
check_blob "scripts/wcare44_candidate_selftest.py" "1449466d0f237871360904c0228f12e8f7af2231"
check_blob "scripts/wcare44_frontdoor_selftest.py" "4ecf05516dfb96d40450ca3e90e81679499bd812"

if ! python3 - <<'PY'
from pathlib import Path
import json
for path in (
    Path('docs/release/evidence/WCARE44_BUILDER_AUTH_OBSERVATION_SCHEMA_V1.json'),
    Path('docs/release/evidence/WCARE44_TEMPORAL_OBSERVATION_SCHEMA_V1.json'),
    Path('docs/release/evidence/WCARE44_CANDIDATE_RESULT_SCHEMA_V1.json'),
):
    value = json.loads(path.read_text())
    assert isinstance(value, dict) and value.get('additionalProperties') is False
result_schema = json.loads(Path('docs/release/evidence/WCARE44_CANDIDATE_RESULT_SCHEMA_V1.json').read_text())
for field in (
    'child_verifier_lineage_established',
    'wcare42_executable_qualification_established',
    'wcare43_external_execution_lineage_established',
    'builder_authentication_established',
    'preregistration_temporal_precedence_established',
    'authenticated_preregistered_replication_established',
    'runtime_authority_granted',
):
    assert result_schema['properties'][field]['const'] is False, field
for path in (
    Path('scripts/wcare44_candidate_kernel.py'),
    Path('scripts/wcare44_candidate_qualify.py'),
    Path('scripts/wcare44_candidate_selftest.py'),
    Path('scripts/wcare44_frontdoor_selftest.py'),
):
    compile(path.read_text(), str(path), 'exec')
PY
then
  emit "INVALID_PROTOCOL" "schema_parse_or_python_compile_failed"
  exit 4
fi

if ! python3 scripts/wcare44_candidate_selftest.py >/tmp/wcare44-candidate-selftest.json 2>/tmp/wcare44-candidate-selftest.stderr; then
  emit "INVALID_PROTOCOL" "candidate_adversarial_campaign_failed"
  exit 4
fi
for marker in \
  '"classification":"PASS_WCARE44_CANDIDATE_SELFTEST"' \
  '"full_candidate_conjunction_without_final_promotion_verified":true' \
  '"missing_relation_collapses_one_separation":true' \
  '"missing_endpoint_collapses_multiple_separations":true' \
  '"baseline_edge_cannot_be_removed":true' \
  '"temporal_only_cannot_authenticate_builders":true' \
  '"builder_only_cannot_establish_temporal_precedence":true' \
  '"synthetic_temporal_cannot_establish_candidate_precedence":true' \
  '"indeterminate_builder_state_preserved":true' \
  '"duplicate_observation_rejected":true' \
  '"forged_wcare40_component_count_rejected":true' \
  '"invalid_temporal_state_rejected":true' \
  '"runtime_authority_granted":false'
do
  if ! grep -q "$marker" /tmp/wcare44-candidate-selftest.json; then
    emit "INVALID_PROTOCOL" "candidate_selftest_missing_marker:$marker"
    exit 4
  fi
done

if ! python3 scripts/wcare44_frontdoor_selftest.py >/tmp/wcare44-frontdoor-selftest.json 2>/tmp/wcare44-frontdoor-selftest.stderr; then
  emit "INVALID_PROTOCOL" "frontdoor_adversarial_campaign_failed"
  exit 4
fi
for marker in \
  '"classification":"PASS_WCARE44_FRONTDOOR_SELFTEST"' \
  '"valid_candidate_path_preserved":true' \
  '"missing_builder_field_rejected":true' \
  '"unknown_builder_field_rejected":true' \
  '"wrong_builder_verifier_identity_rejected":true' \
  '"wrong_temporal_verifier_identity_rejected":true' \
  '"malformed_builder_boolean_rejected":true' \
  '"malformed_auth_plan_time_rejected":true' \
  '"malformed_backend_contract_rejected":true' \
  '"missing_temporal_field_rejected":true' \
  '"final_promotion_remains_blocked":true' \
  '"runtime_authority_granted":false'
do
  if ! grep -q "$marker" /tmp/wcare44-frontdoor-selftest.json; then
    emit "INVALID_PROTOCOL" "frontdoor_selftest_missing_marker:$marker"
    exit 4
  fi
done

emit "PASS_PROTOCOL_INTEGRITY" "exact_wcare44_candidate_frontdoor_bytes_and_adversarial_campaigns_match"
exit 0
