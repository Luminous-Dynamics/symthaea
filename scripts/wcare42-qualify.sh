#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"not_in_git_worktree","runtime_authority_granted":false}'
  exit 4
}
cd "$ROOT"

BASE_WCARE41="89243a3072a5ff510c8d6d7ea4958644be40f879"
MANIFEST="tools/wcare42_builder_attestation_verifier/Cargo.toml"
LOCK="tools/wcare42_builder_attestation_verifier/Cargo.lock"

emit() {
  local classification="$1" detail="$2"
  printf '{"authority":"MeasurementOnly","classification":"%s","detail":"%s","head":"%s","wcare41_base":"%s","builder_authentication_established":false,"preregistration_temporal_precedence_established":false,"runtime_authority_granted":false}\n' \
    "$classification" "$detail" "$(git rev-parse HEAD)" "$BASE_WCARE41"
}

if [[ "$#" -ne 6 ]]; then
  emit "INVALID_PROTOCOL" "usage: wcare42-qualify.sh ENVELOPE POLICY WCARE40_PLAN WCARE40_RESULT SUBJECT_RECEIPT EVALUATION_UTC"
  exit 4
fi

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
check_blob "tools/wcare42_builder_attestation_verifier/tests/golden.rs" "b00fb90e0bd8863e71fb697151483cc395a63bb0"
check_blob "docs/release/evidence/WCARE42_BUILDER_ATTESTATION_VERIFIER_PROTOCOL_V1.md" "1cd6e3f5f2f4edcf4fccf45b35c4c68bdb9c12a5"
check_blob "docs/release/evidence/WCARE42_BUILDER_ATTESTATION_RESULT_SCHEMA_V1.json" "b2d6b4b61af46c8924cdb95b2f958df3d3d7ab96"
check_blob "docs/release/evidence/WCARE42_BUILDER_ATTESTATION_GOLDEN_VECTOR_V1.json" "dc42f404537795f37a9bf178d3f30fd52c74a355"

if [[ ! -f "$LOCK" ]]; then
  emit "INFRASTRUCTURE_INDETERMINATE" "standalone_cargo_lock_missing"
  exit 3
fi

LOCK_HASH="$(git hash-object "$LOCK" 2>/dev/null || true)"
if [[ -z "$LOCK_HASH" ]]; then
  emit "INVALID_PROTOCOL" "standalone_cargo_lock_unhashable"
  exit 4
fi

if ! cargo test --manifest-path "$MANIFEST" --locked; then
  emit "FAIL_VERIFIER" "locked_cross_implementation_tests_failed"
  exit 1
fi

RESULT="$(mktemp)"
trap 'rm -f "$RESULT"' EXIT
set +e
cargo run --quiet --manifest-path "$MANIFEST" --locked -- "$1" "$2" "$3" "$4" "$5" "$6" >"$RESULT"
VERIFY_STATUS=$?
set -e
cat "$RESULT"

python3 - "$RESULT" "$VERIFY_STATUS" "$LOCK_HASH" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
status = int(sys.argv[2])
lock_blob = sys.argv[3]
try:
    value = json.loads(path.read_text())
except Exception:
    raise SystemExit(4)

if value.get("authority") != "MeasurementOnly":
    raise SystemExit(4)
if value.get("verifier_protocol_version") != "wcare42-builder-attestation-verifier-v1":
    raise SystemExit(4)
if value.get("builder_authentication_established") is not False:
    raise SystemExit(4)
if value.get("preregistration_temporal_precedence_established") is not False:
    raise SystemExit(4)
if value.get("runtime_authority_granted") is not False:
    raise SystemExit(4)

expected = {
    "ATTESTATION_ACCEPTED": 0,
    "SIGNATURE_VALID_ISSUER_UNTRUSTED": 2,
    "ATTESTATION_REJECTED": 4,
    "INFRASTRUCTURE_INDETERMINATE": 3,
}.get(value.get("disposition"), 4)
if status != expected:
    raise SystemExit(4)
raise SystemExit(expected)
PY
