#!/usr/bin/env bash
set -u -o pipefail

if [[ "$#" -ne 2 ]]; then
  printf '%s\n' '{"authority":"MeasurementOnly","protocol_version":"wcare39-execution-capsule-v1","classification":"INVALID_CAPSULE","detail":"usage: wcare39-qualify.sh PLAN EVIDENCE_DIR","runtime_authority_granted":false}'
  exit 4
fi

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  printf '%s\n' '{"authority":"MeasurementOnly","protocol_version":"wcare39-execution-capsule-v1","classification":"INFRASTRUCTURE_INDETERMINATE","detail":"not_in_git_worktree","runtime_authority_granted":false}'
  exit 3
}
cd "$ROOT"

RESULT="$(mktemp)"
trap 'rm -f "$RESULT"' EXIT

set +e
python3 scripts/wcare39_execution_capsule.py run "$1" "$2" >"$RESULT"
RUNNER_STATUS=$?
set -e
cat "$RESULT"

python3 - "$RESULT" "$RUNNER_STATUS" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
runner_status = int(sys.argv[2])
try:
    value = json.loads(path.read_text())
except Exception:
    raise SystemExit(3)

classification = value.get("classification")
outcome = value.get("subject_outcome")
if classification == "QUALIFIED_EXECUTION":
    if outcome == "PASS":
        raise SystemExit(0)
    if outcome in {"FAIL", "INVALID"}:
        raise SystemExit(1)
    raise SystemExit(3)
if classification == "ENVIRONMENT_DRIFT":
    raise SystemExit(2)
if classification == "INFRASTRUCTURE_INDETERMINATE":
    raise SystemExit(3)
if classification == "INVALID_CAPSULE":
    raise SystemExit(4)
raise SystemExit(runner_status if runner_status in {1, 2, 3, 4} else 3)
PY
