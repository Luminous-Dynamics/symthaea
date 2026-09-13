#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  printf '%s\n' '{"authority":"MeasurementOnly","classification":"INVALID_PROTOCOL","detail":"not_in_git_worktree","runtime_authority_granted":false}'
  exit 4
}
cd "$ROOT"

BASE_WCARE38="d35527069fdb67937c2000c181172cfd9a634274"

emit() {
  local classification="$1" detail="$2"
  printf '{"authority":"MeasurementOnly","classification":"%s","detail":"%s","head":"%s","wcare38_base":"%s","runtime_authority_granted":false}\n' \
    "$classification" "$detail" "$(git rev-parse HEAD)" "$BASE_WCARE38"
}

if ! git merge-base --is-ancestor "$BASE_WCARE38" HEAD; then
  emit "INVALID_PROTOCOL" "exact_wcare38_base_not_ancestor"
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

check_blob "docs/release/evidence/WCARE39_EXECUTION_CAPSULE_PROTOCOL_V1.md" "c196824cba4afd25ec6e59d6215a33e24a97cbb3"
check_blob "docs/release/evidence/WCARE39_EXECUTION_CAPSULE_SCHEMA_V1.json" "0d3936654d947062664999bbb59063e9f24f1b24"
check_blob "docs/release/evidence/WCARE39_COMMAND_PLAN_SCHEMA_V1.json" "e2b0a027daaf3518a21e36b5f6c97952879ad470"
check_blob "docs/release/evidence/WCARE39_EXECUTION_STATUS_SCHEMA_V1.json" "35afbd3c4dbd7e5bd765509ffb1839b543c228af"
check_blob "scripts/wcare39_execution_capsule.py" "17c43109a59047052737b39a5d12465b2d824eb1"
check_blob "scripts/wcare39_selftest.py" "42f0bc87ffafe3c359af7774f50545b80a17c8fe"

if ! python3 - <<'PY'
from pathlib import Path
for path in (Path('scripts/wcare39_execution_capsule.py'), Path('scripts/wcare39_selftest.py')):
    compile(path.read_text(), str(path), 'exec')
PY
then
  emit "INVALID_PROTOCOL" "python_source_compile_failed"
  exit 4
fi

if ! python3 scripts/wcare39_selftest.py >/tmp/wcare39-selftest.json 2>/tmp/wcare39-selftest.stderr; then
  emit "INVALID_PROTOCOL" "synthetic_execution_capsule_campaign_failed"
  exit 4
fi
if ! grep -q '"classification":"PASS_WCARE39_SELFTEST"' /tmp/wcare39-selftest.json; then
  emit "INVALID_PROTOCOL" "synthetic_campaign_did_not_report_pass"
  exit 4
fi

emit "PASS_PROTOCOL_INTEGRITY" "exact_wcare39_bytes_and_synthetic_campaign_match"
exit 0
