#!/usr/bin/env bash
set -euo pipefail

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "required environment variable $name is missing" >&2
    exit 2
  fi
}

for name in \
  SYMTHAEA_SUBJECT_REVISION \
  SYMTHAEA_ARC_DATASET_ROOT \
  SYMTHAEA_ARC_DATASET_REPOSITORY \
  SYMTHAEA_ARC_DATASET_REVISION \
  SYMTHAEA_ARC_DATASET_TREE \
  SYMTHAEA_ARC_DATASET_MANIFEST_PATH \
  SYMTHAEA_ARC_POLICY_RESULTS_PATH \
  SYMTHAEA_ARC_SMOKE_RECEIPT_PATH \
  SYMTHAEA_ARC_SMOKE_TASK_FILES \
  SYMTHAEA_ARC_CANDIDATE_BUDGET \
  SYMTHAEA_ARC_POLICY_SEED; do
  require_env "$name"
done

if [[ ! "$SYMTHAEA_SUBJECT_REVISION" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "SYMTHAEA_SUBJECT_REVISION must be an exact 40-hex revision" >&2
  exit 2
fi
if [[ ! "$SYMTHAEA_ARC_DATASET_REVISION" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "SYMTHAEA_ARC_DATASET_REVISION must be an exact 40-hex revision" >&2
  exit 2
fi
if [[ ! "$SYMTHAEA_ARC_DATASET_TREE" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "SYMTHAEA_ARC_DATASET_TREE must be an exact 40-hex tree" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
subject_head_before="$(git rev-parse HEAD)"
subject_tree_before="$(git rev-parse 'HEAD^{tree}')"
subject_status_before="$(git status --porcelain=v1 --untracked-files=all)"
dataset_head_before="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" rev-parse HEAD)"
dataset_tree_before="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" rev-parse 'HEAD^{tree}')"
dataset_status_before="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" status --porcelain=v1 --untracked-files=all)"

if [[ "$subject_head_before" != "$SYMTHAEA_SUBJECT_REVISION" ]]; then
  echo "Symthaea HEAD does not match declared smoke subject" >&2
  exit 2
fi
if [[ -n "$subject_status_before" ]]; then
  echo "Symthaea checkout must be clean before smoke measurement" >&2
  exit 2
fi
if [[ "$dataset_head_before" != "$SYMTHAEA_ARC_DATASET_REVISION" ]]; then
  echo "dataset HEAD does not match declared smoke subject" >&2
  exit 2
fi
if [[ "$dataset_tree_before" != "$SYMTHAEA_ARC_DATASET_TREE" ]]; then
  echo "dataset tree does not match declared smoke tree" >&2
  exit 2
fi
if [[ -n "$dataset_status_before" ]]; then
  echo "dataset checkout must be clean before smoke measurement" >&2
  exit 2
fi
if [[ ! -s "$SYMTHAEA_ARC_DATASET_MANIFEST_PATH" ]]; then
  echo "qualified dataset manifest is missing" >&2
  exit 2
fi

policy_bin="${SYMTHAEA_ARC_POLICY_BIN:-$repo_root/target/debug/arc_budgeted_policy}"
receipt_bin="${SYMTHAEA_ARC_SMOKE_RECEIPT_BIN:-$repo_root/target/debug/arc_smoke_receipt}"
if [[ ! -x "$policy_bin" ]]; then
  echo "prebuilt ARC policy binary not found: $policy_bin" >&2
  exit 2
fi
if [[ ! -x "$receipt_bin" ]]; then
  echo "prebuilt ARC smoke receipt binary not found: $receipt_bin" >&2
  exit 2
fi

mkdir -p "$(dirname "$SYMTHAEA_ARC_POLICY_RESULTS_PATH")" "$(dirname "$SYMTHAEA_ARC_SMOKE_RECEIPT_PATH")"

# Critical information-flow boundary: manifest paths/commitments and evaluator-only provenance are
# absent from the policy process environment. The frozen policy gets only the exact fields required
# to choose/evaluate native ARC actions. Hidden expected grids remain evaluator-side within the
# frozen v2 binary's post-seal phase; this harness adds no new path from manifest provenance to policy.
env -i \
  PATH="$PATH" \
  SYMTHAEA_SUBJECT_REVISION="$SYMTHAEA_SUBJECT_REVISION" \
  SYMTHAEA_ARC_DATASET_VERSION="$SYMTHAEA_ARC_DATASET_REVISION" \
  SYMTHAEA_ARC_SPLIT="training" \
  SYMTHAEA_ARC_DATA_DIR="$SYMTHAEA_ARC_DATASET_ROOT/data" \
  SYMTHAEA_ARC_POLICY_RESULTS_PATH="$SYMTHAEA_ARC_POLICY_RESULTS_PATH" \
  SYMTHAEA_ARC_MAX_TASKS="$SYMTHAEA_ARC_SMOKE_TASK_FILES" \
  SYMTHAEA_ARC_CANDIDATE_BUDGET="$SYMTHAEA_ARC_CANDIDATE_BUDGET" \
  SYMTHAEA_ARC_POLICY_SEED="$SYMTHAEA_ARC_POLICY_SEED" \
  "$policy_bin"

if [[ ! -s "$SYMTHAEA_ARC_POLICY_RESULTS_PATH" ]]; then
  echo "ARC policy smoke report was not produced" >&2
  exit 2
fi

"$receipt_bin"

if [[ ! -s "$SYMTHAEA_ARC_SMOKE_RECEIPT_PATH" ]]; then
  echo "ARC smoke receipt was not produced" >&2
  exit 2
fi

python3 - "$SYMTHAEA_ARC_DATASET_MANIFEST_PATH" "$SYMTHAEA_ARC_POLICY_RESULTS_PATH" "$SYMTHAEA_ARC_SMOKE_RECEIPT_PATH" <<'PY'
import hashlib
import json
import os
import sys

manifest_path, report_path, receipt_path = sys.argv[1:]
with open(manifest_path, "rb") as handle:
    manifest_bytes = handle.read()
with open(report_path, "rb") as handle:
    report_bytes = handle.read()
with open(receipt_path, "r", encoding="utf-8") as handle:
    receipt = json.load(handle)

if receipt["schema_version"] != 1:
    raise SystemExit("unexpected smoke receipt schema")
if receipt["receipt_domain"] != "symthaea/reasoning/arc-public-training-smoke/v1":
    raise SystemExit("unexpected smoke receipt domain")
if receipt["authority"] != "DevelopmentProbe" or receipt["contamination_status"] != "Exposed":
    raise SystemExit("smoke receipt has invalid authority/contamination semantics")
if receipt["subject_revision"] != os.environ["SYMTHAEA_SUBJECT_REVISION"]:
    raise SystemExit("smoke receipt subject mismatch")
if receipt["dataset_revision"] != os.environ["SYMTHAEA_ARC_DATASET_REVISION"]:
    raise SystemExit("smoke receipt dataset revision mismatch")
if receipt["dataset_tree"] != os.environ["SYMTHAEA_ARC_DATASET_TREE"]:
    raise SystemExit("smoke receipt dataset tree mismatch")
if receipt["task_limit"] != int(os.environ["SYMTHAEA_ARC_SMOKE_TASK_FILES"]):
    raise SystemExit("smoke receipt task limit mismatch")
if receipt["candidate_budget"] != int(os.environ["SYMTHAEA_ARC_CANDIDATE_BUDGET"]):
    raise SystemExit("smoke receipt budget mismatch")
if receipt["random_seed_root"] != int(os.environ["SYMTHAEA_ARC_POLICY_SEED"]):
    raise SystemExit("smoke receipt seed mismatch")
if hashlib.sha256(manifest_bytes).hexdigest() != receipt["dataset_manifest_file_sha256"]:
    raise SystemExit("independent manifest-file SHA-256 mismatch")
if hashlib.sha256(report_bytes).hexdigest() != receipt["policy_report_sha256"]:
    raise SystemExit("independent policy-report SHA-256 mismatch")
if len(receipt["selected_task_paths"]) != int(os.environ["SYMTHAEA_ARC_SMOKE_TASK_FILES"]):
    raise SystemExit("smoke receipt selected-task list has wrong size")
PY

subject_head_after="$(git rev-parse HEAD)"
subject_tree_after="$(git rev-parse 'HEAD^{tree}')"
subject_status_after="$(git status --porcelain=v1 --untracked-files=all)"
dataset_head_after="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" rev-parse HEAD)"
dataset_tree_after="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" rev-parse 'HEAD^{tree}')"
dataset_status_after="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" status --porcelain=v1 --untracked-files=all)"

if [[ "$subject_head_after" != "$subject_head_before" || "$subject_tree_after" != "$subject_tree_before" ]]; then
  echo "Symthaea checkout changed during smoke measurement" >&2
  exit 2
fi
if [[ -n "$subject_status_after" ]]; then
  echo "Symthaea checkout became dirty during smoke measurement" >&2
  exit 2
fi
if [[ "$dataset_head_after" != "$dataset_head_before" || "$dataset_tree_after" != "$dataset_tree_before" ]]; then
  echo "ARC dataset checkout changed during smoke measurement" >&2
  exit 2
fi
if [[ -n "$dataset_status_after" ]]; then
  echo "ARC dataset checkout became dirty during smoke measurement" >&2
  exit 2
fi

printf 'qualified ARC public-training smoke subject %s dataset %s\n' \
  "$subject_head_after" "$dataset_head_after"
