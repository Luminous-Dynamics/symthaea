#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

: "${SYMTHAEA_SUBJECT_REVISION:?set SYMTHAEA_SUBJECT_REVISION to the exact checked-out commit SHA}"

subject="${SYMTHAEA_SUBJECT_REVISION,,}"
if [[ ! "$subject" =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]]; then
  echo "error: SYMTHAEA_SUBJECT_REVISION must be a full 40- or 64-hex revision" >&2
  exit 2
fi

actual_head="$(git rev-parse HEAD | tr '[:upper:]' '[:lower:]')"
if [[ "$actual_head" != "$subject" ]]; then
  echo "error: checked-out HEAD $actual_head != declared subject $subject" >&2
  exit 3
fi

before_tree="$(git rev-parse 'HEAD^{tree}')"
before_status="$(git status --porcelain=v1 --untracked-files=all)"
if [[ -n "$before_status" ]]; then
  echo "error: qualification checkout is not clean before execution" >&2
  printf '%s\n' "$before_status" >&2
  exit 4
fi

receipt_path="${SYMTHAEA_ADAPTIVE_RECEIPT_PATH:-$repo_root/target/rq-006z/adaptive-qualification-receipt.json}"
export SYMTHAEA_ADAPTIVE_RECEIPT_PATH="$receipt_path"
rm -f "$receipt_path"

cargo run --locked --quiet --bin adaptive_qualification_subject

python3 - "$receipt_path" "$subject" <<'PY'
import json
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
subject = sys.argv[2]
if not path.is_file():
    raise SystemExit(f"receipt missing: {path}")

receipt = json.loads(path.read_text())
if receipt.get("subject_revision", "").lower() != subject:
    raise SystemExit("receipt subject revision does not match checked-out subject")
if receipt.get("control_version") != "rq-006z-adaptive-control-v2":
    raise SystemExit("unexpected adaptive qualification control version")
if receipt.get("action_space_version") != "adaptive-action-space-v1":
    raise SystemExit("unexpected adaptive action-space version")
if receipt.get("learner_state_before") != receipt.get("learner_state_after"):
    raise SystemExit("frozen qualifier mutated learner state")
if len(receipt.get("actions", [])) != receipt.get("max_steps"):
    raise SystemExit("receipt does not contain exactly max_steps action records")
if receipt.get("forced_actions_applied") != len(receipt.get("forced_schedule", [])):
    raise SystemExit("forced schedule was not applied exactly")
if not receipt.get("action_space"):
    raise SystemExit("receipt action-space manifest is empty")
for field in (
    "action_space_commitment",
    "forced_schedule_commitment",
    "question_digest",
    "learner_state_before",
    "learner_state_after",
    "final_state_digest",
    "receipt_commitment",
):
    value = receipt.get(field, "")
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SystemExit(f"{field} is not a 64-hex digest")
print("outer_receipt_structure=PASS")
PY

after_head="$(git rev-parse HEAD | tr '[:upper:]' '[:lower:]')"
after_tree="$(git rev-parse 'HEAD^{tree}')"
after_status="$(git status --porcelain=v1 --untracked-files=all)"

if [[ "$after_head" != "$actual_head" ]]; then
  echo "error: HEAD changed during qualification" >&2
  exit 5
fi
if [[ "$after_tree" != "$before_tree" ]]; then
  echo "error: committed tree changed during qualification" >&2
  exit 6
fi
if [[ "$after_status" != "$before_status" ]]; then
  echo "error: checkout worktree/index changed during qualification" >&2
  printf 'before:\n%s\nafter:\n%s\n' "$before_status" "$after_status" >&2
  exit 7
fi

receipt_sha256="$(sha256sum "$receipt_path" | awk '{print $1}')"
printf 'subject_conformance=PASS\n'
printf 'checkout_immutable=PASS\n'
printf 'receipt_sha256=%s\n' "$receipt_sha256"
printf 'receipt_path=%s\n' "$receipt_path"
