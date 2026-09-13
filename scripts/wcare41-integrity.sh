#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"not_in_git_worktree","runtime_authority_granted":false}'
  exit 4
}
cd "$ROOT"

BASE_WCARE40="4748c5b8b0f22781c5af4ca37a0ccd0f7e80584d"

emit() {
  local classification="$1" detail="$2"
  printf '{"authority":"MeasurementOnly","classification":"%s","detail":"%s","head":"%s","wcare40_base":"%s","runtime_authority_granted":false}\n' \
    "$classification" "$detail" "$(git rev-parse HEAD)" "$BASE_WCARE40"
}

if ! git merge-base --is-ancestor "$BASE_WCARE40" HEAD; then
  emit "INVALID_PROTOCOL" "exact_wcare40_base_not_ancestor"
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

check_blob "docs/release/evidence/WCARE41_AUTHENTICATED_PREREGISTRATION_PROTOCOL_V1.md" "8df5bbe387f4c221d55aa2ac16203b57ed602779"
check_blob "docs/release/evidence/WCARE41_AUTHENTICATION_PLAN_SCHEMA_V1.json" "32683abe68e44a1b1bfce2b9c08e2bcd7041877f"
check_blob "docs/release/evidence/WCARE41_BUILDER_ATTESTATION_ENVELOPE_SCHEMA_V1.json" "e2ffc9b452378120ed0cb40ae459bab8b9a76cde"
check_blob "docs/release/evidence/WCARE41_TEMPORAL_PROOF_PACKAGE_SCHEMA_V1.json" "2771b0af9b319a697716ff5d195724f42c222897"
check_blob "docs/release/evidence/WCARE41_BUILDER_ISSUER_TRUST_POLICY_SCHEMA_V1.json" "27ca0805dc041b3573b10f4accdee8db40a4e6de"
check_blob "docs/release/evidence/WCARE41_TEMPORAL_VERIFIER_POLICY_SCHEMA_V1.json" "4d2acc1ac5fb4332bdcb934e35284d21f6957e41"
check_blob "docs/release/evidence/WCARE41_AUTHENTICATION_RESULT_SCHEMA_V1.json" "17cecb49abea363e7c93a4ce8b3f3ed22910eb1f"
check_blob "scripts/wcare41_contract_selftest.py" "d428422aa5a5c65e5c108a513034f67a4d8a9bfa"

if ! python3 - <<'PY'
from pathlib import Path
import json
for path in (
    Path('docs/release/evidence/WCARE41_AUTHENTICATION_PLAN_SCHEMA_V1.json'),
    Path('docs/release/evidence/WCARE41_BUILDER_ATTESTATION_ENVELOPE_SCHEMA_V1.json'),
    Path('docs/release/evidence/WCARE41_TEMPORAL_PROOF_PACKAGE_SCHEMA_V1.json'),
    Path('docs/release/evidence/WCARE41_BUILDER_ISSUER_TRUST_POLICY_SCHEMA_V1.json'),
    Path('docs/release/evidence/WCARE41_TEMPORAL_VERIFIER_POLICY_SCHEMA_V1.json'),
    Path('docs/release/evidence/WCARE41_AUTHENTICATION_RESULT_SCHEMA_V1.json'),
):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AssertionError(f'non-object schema: {path}')
compile(Path('scripts/wcare41_contract_selftest.py').read_text(), 'scripts/wcare41_contract_selftest.py', 'exec')
PY
then
  emit "INVALID_PROTOCOL" "schema_parse_or_python_compile_failed"
  exit 4
fi

if ! python3 scripts/wcare41_contract_selftest.py >/tmp/wcare41-contract-selftest.json 2>/tmp/wcare41-contract-selftest.stderr; then
  emit "INVALID_PROTOCOL" "contract_selftest_failed"
  exit 4
fi
for marker in \
  '"classification":"PASS_WCARE41_CONTRACT_SELFTEST"' \
  '"cryptographic_authentication_executed":false' \
  '"external_temporal_verification_executed":false' \
  '"late_commitment_rejected_as_preregistration":true' \
  '"authentication_monotonicity_verified":true'
do
  if ! grep -q "$marker" /tmp/wcare41-contract-selftest.json; then
    emit "INVALID_PROTOCOL" "contract_selftest_missing_marker:$marker"
    exit 4
  fi
done

emit "PASS_PROTOCOL_INTEGRITY" "exact_wcare41_contract_bytes_and_claim_algebra_match"
exit 0
