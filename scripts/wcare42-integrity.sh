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
  printf '{"authority":"MeasurementOnly","classification":"%s","detail":"%s","head":"%s","wcare41_base":"%s","cryptographic_execution_qualified":false,"builder_authentication_established":false,"preregistration_temporal_precedence_established":false,"runtime_authority_granted":false}\n' \
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

check_blob "tools/wcare42_builder_attestation_verifier/Cargo.toml" "5410040e5616241dd4ba581af8f297675d083830"
check_blob "tools/wcare42_builder_attestation_verifier/src/main.rs" "1c300a455f054d118e55556aac81b623824629bc"
check_blob "tools/wcare42_builder_attestation_verifier/tests/golden.rs" "666242f74fb302f9be2b62fa4b3050f3ee9ffefd"
check_blob "docs/release/evidence/WCARE42_BUILDER_ATTESTATION_VERIFIER_PROTOCOL_V1.md" "1cd6e3f5f2f4edcf4fccf45b35c4c68bdb9c12a5"
check_blob "docs/release/evidence/WCARE42_BUILDER_ATTESTATION_RESULT_SCHEMA_V1.json" "b2d6b4b61af46c8924cdb95b2f958df3d3d7ab96"
check_blob "docs/release/evidence/WCARE42_BUILDER_ATTESTATION_GOLDEN_VECTOR_V1.json" "dc42f404537795f37a9bf178d3f30fd52c74a355"
check_blob "scripts/wcare42-qualify.sh" "8a80d1a74e409503a72b4607425cc61d8072ca2e"

if ! bash -n scripts/wcare42-qualify.sh; then
  emit "INVALID_PROTOCOL" "qualifier_shell_syntax_failed"
  exit 4
fi

if ! python3 - <<'PY'
import json
from pathlib import Path
v = json.loads(Path('docs/release/evidence/WCARE42_BUILDER_ATTESTATION_GOLDEN_VECTOR_V1.json').read_text())
assert v['producer'] == 'python-cryptography-ed25519'
assert v['canonical_message_sha256'] == '2054958f9c99f594131c7335f89e8ce5dc68b46ba92eb91a3201eafadc4f523f'
assert v['signature_ed25519_hex'] == 'd9bc6dae63c3fc60c50b2c1f443a6a9d714006659c31380609b35d85a0efe4cdf15b5cb1f862ce618c540ae5617862f51b661c06acb1d83b174acd296ea56c06'
for path in (
    'docs/release/evidence/WCARE42_BUILDER_ATTESTATION_RESULT_SCHEMA_V1.json',
):
    value = json.loads(Path(path).read_text())
    assert isinstance(value, dict) and value.get('additionalProperties') is False
PY
then
  emit "INVALID_PROTOCOL" "golden_vector_or_schema_integrity_failed"
  exit 4
fi

if [[ -f tools/wcare42_builder_attestation_verifier/Cargo.lock ]]; then
  emit "INVALID_PROTOCOL" "prelock_review_unit_unexpectedly_contains_cargo_lock"
  exit 4
fi

set +e
BLOCKER_OUTPUT="$(scripts/wcare42-qualify.sh missing-envelope.json missing-policy.json missing-plan.json missing-result.json missing-subject.json 2026-09-13T12:00:00Z 2>/tmp/wcare42-blocker.stderr)"
BLOCKER_STATUS=$?
set -e
if [[ "$BLOCKER_STATUS" -ne 3 ]]; then
  emit "INVALID_PROTOCOL" "qualifier_did_not_fail_closed_without_lock:$BLOCKER_STATUS"
  exit 4
fi
if [[ "$BLOCKER_OUTPUT" != *'"classification":"INFRASTRUCTURE_INDETERMINATE"'* ]] || [[ "$BLOCKER_OUTPUT" != *'"detail":"standalone_cargo_lock_missing"'* ]]; then
  emit "INVALID_PROTOCOL" "qualifier_lock_blocker_receipt_mismatch"
  exit 4
fi

emit "PASS_SOURCE_INTEGRITY" "exact_wcare42_prelock_bytes_and_fail_closed_blocker_match"
exit 0
