#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Verify the isolated Q1 synthetic-ranking contract and emit an execution receipt.
set -u
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
MANIFEST="$SCRIPT_DIR/Cargo.toml"
CATALOG="$SCRIPT_DIR/hostile-cases.v0.json"
SOURCE="$SCRIPT_DIR/src/lib.rs"
DEFAULT_RECEIPT="$REPO_ROOT/target/organ-design-q1-receipt.json"
RECEIPT_PATH="${1:-$DEFAULT_RECEIPT}"
SOURCE_COMMIT="${2:-${Q1_SOURCE_COMMIT:-${GITHUB_SHA:-}}}"

if [[ -z "$SOURCE_COMMIT" ]]; then
  SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
fi
if [[ ! "$SOURCE_COMMIT" =~ ^[[:xdigit:]]{40,64}$ ]]; then
  echo "q1-verifier: source commit must be an explicit 40-64 character Git object ID" >&2
  exit 2
fi
SOURCE_COMMIT="$(printf '%s' "$SOURCE_COMMIT" | tr 'A-F' 'a-f')"
if ! command -v sha256sum >/dev/null 2>&1; then
  echo "q1-verifier: sha256sum is required to bind the exact source and fixture bytes" >&2
  exit 2
fi

CATALOG_SHA="$(sha256sum "$CATALOG" | awk '{print $1}')"
SOURCE_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
MANIFEST_SHA="$(sha256sum "$MANIFEST" | awk '{print $1}')"
TOOLCHAIN="unavailable"
if command -v rustc >/dev/null 2>&1; then
  TOOLCHAIN="$(rustc --version)"
fi

FMT_RESULT="not_run"
TEST_RESULT="not_run"
FMT_EXIT=125
TEST_EXIT=125
STATUS="NOT_RUN"
NOTES="Rust/Cargo were unavailable; no Q1 test command was executed."

if command -v cargo >/dev/null 2>&1; then
  if cargo fmt --manifest-path "$MANIFEST" -- --check; then
    FMT_RESULT="pass"; FMT_EXIT=0
  else
    FMT_EXIT=$?; FMT_RESULT="fail"
  fi
  if cargo test --manifest-path "$MANIFEST"; then
    TEST_RESULT="pass"; TEST_EXIT=0
  else
    TEST_EXIT=$?; TEST_RESULT="fail"
  fi

  if [[ "$FMT_RESULT" == "fail" || "$TEST_RESULT" == "fail" ]]; then
    STATUS="FAIL"
    NOTES="At least one required Q1 command failed; inspect the preceding output."
  elif [[ "$FMT_RESULT" == "pass" && "$TEST_RESULT" == "pass" ]]; then
    STATUS="PASS"
    NOTES="PASS is limited to software-contract tests for synthetic candidate ranking; it is not biological or physical validation."
  else
    STATUS="BLOCKED"
    NOTES="The complete Q1 software gate did not execute because a required command was unavailable."
  fi
fi

mkdir -p "$(dirname -- "$RECEIPT_PATH")"
cat > "$RECEIPT_PATH" <<JSON
{
  "schema_version": "0.1.0",
  "benchmark_id": "ORGAN-DESIGN-Q1",
  "context_of_use": "synthetic-only candidate ranking contract",
  "status": "$STATUS",
  "evidence_tier": "synthetic_simulation",
  "code_commit": "$SOURCE_COMMIT",
  "toolchain": "$TOOLCHAIN",
  "cargo_manifest_sha256": "$MANIFEST_SHA",
  "ranking_source_sha256": "$SOURCE_SHA",
  "hostile_case_catalog_sha256": "$CATALOG_SHA",
  "hostile_case_count": 18,
  "format_check": "$FMT_RESULT",
  "test_command": "$TEST_RESULT",
  "format_exit_code": $FMT_EXIT,
  "test_exit_code": $TEST_EXIT,
  "claim_ceiling": "synthetic simulation only; no physical, tissue, organ-function, transplant, or clinical claim",
  "notes": "$NOTES"
}
JSON

echo "Q1 status: $STATUS"
echo "Q1 receipt: $RECEIPT_PATH"
echo "Q1 source SHA-256: $SOURCE_SHA"
echo "Q1 hostile-corpus SHA-256: $CATALOG_SHA"

case "$STATUS" in
  PASS) exit 0 ;;
  FAIL) exit 1 ;;
  BLOCKED) exit 3 ;;
  *) exit 4 ;;
esac
