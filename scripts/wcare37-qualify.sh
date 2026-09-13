#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  printf '%s\n' '{"classification":"INFRASTRUCTURE_INDETERMINATE","detail":"not_in_git_worktree"}'
  exit 3
}
cd "$ROOT"

BASE_WCARE36="1ac9092cd5eadb6d33a496b5429b38209e35285b"
MANIFEST="tools/wcare37_attestation_verifier/Cargo.toml"
LOCKFILE="tools/wcare37_attestation_verifier/Cargo.lock"
TARGET_DIR="target/wcare37"
mkdir -p "$TARGET_DIR"

emit() {
  local classification="$1" detail="$2"
  printf '{"authority":"MeasurementOnly","classification":"%s","detail":"%s","head":"%s","wcare36_base":"%s","dependency_lock_present":%s,"runtime_authority_granted":false}\n' \
    "$classification" "$detail" "$(git rev-parse HEAD)" "$BASE_WCARE36" "$([[ -f "$LOCKFILE" ]] && echo true || echo false)"
}

if ! git merge-base --is-ancestor "$BASE_WCARE36" HEAD; then
  emit "INVALID_PROTOCOL" "exact_wcare36_base_not_ancestor"
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

check_blob "docs/release/evidence/WCARE37_ATTESTATION_PROTOCOL_V1.md" "cbde37860e957856943ffbd3d27ad7fdff39fe2a"
check_blob "docs/release/evidence/WCARE37_ATTESTATION_ENVELOPE_SCHEMA_V1.json" "92f3113ad231b4af2b221acd8f6ebaa259e4df14"
check_blob "docs/release/evidence/WCARE37_ISSUER_TRUST_POLICY_SCHEMA_V1.json" "ac8ee43b2c93c47bcb388c5bbf65dcea81de4ccd"
check_blob "docs/release/evidence/WCARE37_ATTESTATION_RESULT_SCHEMA_V1.json" "f10bea5dc98fd27b01d543f19c542cb77b288f5c"
check_blob "$MANIFEST" "309d44a1047548f3d9cec5a2de21126a905c2de1"
check_blob "tools/wcare37_attestation_verifier/src/main.rs" "72748afe656be7762a1a7c4a5ef69c73d7783624"

# The exact standalone dependency graph is part of the theorem. Do not allow an
# unlocked Cargo resolution to mint PASS_VERIFIER/PASS_ATTESTATION.
if [[ ! -f "$LOCKFILE" ]]; then
  emit "INFRASTRUCTURE_INDETERMINATE" "standalone_cargo_lock_missing"
  exit 3
fi

TEST_LOG="$TARGET_DIR/cargo-test.log"
if ! cargo test --locked --manifest-path "$MANIFEST" >"$TEST_LOG" 2>&1; then
  if grep -Eqi 'could not resolve host|failed to download|network.*unreachable|timed out|no space left|resource temporarily unavailable' "$TEST_LOG"; then
    emit "INFRASTRUCTURE_INDETERMINATE" "verifier_test_infrastructure_failure"
    exit 3
  fi
  emit "FAIL_VERIFIER" "standalone_verifier_tests_failed"
  exit 1
fi

if [[ "${1:-}" == "--self-test" ]]; then
  emit "PASS_VERIFIER" "exact_blob_and_locked_verifier_tests_passed"
  exit 0
fi

if [[ "$#" -ne 6 ]]; then
  emit "INFRASTRUCTURE_INDETERMINATE" "usage_requires_six_attestation_arguments_or_self_test"
  exit 3
fi

RESULT_FILE="$TARGET_DIR/attestation-result.json"
set +e
cargo run --quiet --locked --manifest-path "$MANIFEST" -- "$@" >"$RESULT_FILE" 2>"$TARGET_DIR/cargo-run.stderr"
STATUS=$?
set -e
cat "$RESULT_FILE"

case "$STATUS" in
  0) exit 0 ;;
  1) exit 1 ;;
  2) exit 2 ;;
  3) exit 3 ;;
  *)
    emit "INFRASTRUCTURE_INDETERMINATE" "unexpected_verifier_exit:$STATUS"
    exit 3
    ;;
esac
