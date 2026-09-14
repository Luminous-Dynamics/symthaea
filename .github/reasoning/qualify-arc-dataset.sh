#!/usr/bin/env bash
set -euo pipefail

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "required environment variable $name is missing" >&2
    exit 2
  fi
}

require_full_sha() {
  local label="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "$label must be an exact 40-hex Git object id" >&2
    exit 2
  fi
}

require_env SYMTHAEA_SUBJECT_REVISION
require_env SYMTHAEA_ARC_DATASET_ROOT
require_env SYMTHAEA_ARC_DATASET_REVISION
require_env SYMTHAEA_ARC_DATASET_TREE
require_env SYMTHAEA_ARC_SPLIT
require_env SYMTHAEA_ARC_DATASET_MANIFEST_PATH
require_env SYMTHAEA_ARC_EXPECTED_TASKS

require_full_sha "Symthaea subject revision" "$SYMTHAEA_SUBJECT_REVISION"
require_full_sha "ARC dataset revision" "$SYMTHAEA_ARC_DATASET_REVISION"
require_full_sha "ARC dataset tree" "$SYMTHAEA_ARC_DATASET_TREE"

if [[ "$SYMTHAEA_ARC_SPLIT" != "training" && "$SYMTHAEA_ARC_SPLIT" != "evaluation" ]]; then
  echo "SYMTHAEA_ARC_SPLIT must be training or evaluation" >&2
  exit 2
fi
if [[ ! "$SYMTHAEA_ARC_EXPECTED_TASKS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SYMTHAEA_ARC_EXPECTED_TASKS must be a positive integer" >&2
  exit 2
fi

export SYMTHAEA_ARC_DATASET_REPOSITORY="${SYMTHAEA_ARC_DATASET_REPOSITORY:-arcprize/ARC-AGI-2}"
expected_remote="https://github.com/${SYMTHAEA_ARC_DATASET_REPOSITORY}.git"

repo_root="$(git rev-parse --show-toplevel)"
subject_head_before="$(git rev-parse HEAD)"
subject_tree_before="$(git rev-parse 'HEAD^{tree}')"
subject_status_before="$(git status --porcelain=v1 --untracked-files=all)"

if [[ "$subject_head_before" != "$SYMTHAEA_SUBJECT_REVISION" ]]; then
  echo "Symthaea HEAD $subject_head_before does not equal declared subject $SYMTHAEA_SUBJECT_REVISION" >&2
  exit 2
fi
if [[ -n "$subject_status_before" ]]; then
  echo "Symthaea checkout must be clean before dataset qualification" >&2
  printf '%s\n' "$subject_status_before" >&2
  exit 2
fi

if [[ -L "$SYMTHAEA_ARC_DATASET_ROOT" ]]; then
  echo "ARC dataset root must not be a symlink: $SYMTHAEA_ARC_DATASET_ROOT" >&2
  exit 2
fi
if [[ ! -d "$SYMTHAEA_ARC_DATASET_ROOT/.git" ]]; then
  echo "ARC dataset root is not a Git checkout: $SYMTHAEA_ARC_DATASET_ROOT" >&2
  exit 2
fi

origin_url="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" remote get-url origin)"
if [[ "$origin_url" != "$expected_remote" ]]; then
  echo "ARC dataset origin $origin_url does not equal qualified repository $expected_remote" >&2
  exit 2
fi

python3 - "$SYMTHAEA_ARC_DATASET_ROOT" "$SYMTHAEA_ARC_DATASET_MANIFEST_PATH" <<'PY'
import os
import sys

root = os.path.realpath(sys.argv[1])
output = os.path.realpath(os.path.abspath(sys.argv[2]))
if os.path.commonpath([root, output]) == root:
    raise SystemExit("dataset manifest output must be outside the dataset checkout")
PY

dataset_head_before="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" rev-parse HEAD)"
dataset_tree_before="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" rev-parse 'HEAD^{tree}')"
dataset_status_before="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" status --porcelain=v1 --untracked-files=all)"

if [[ "$dataset_head_before" != "$SYMTHAEA_ARC_DATASET_REVISION" ]]; then
  echo "ARC dataset HEAD $dataset_head_before does not equal declared subject $SYMTHAEA_ARC_DATASET_REVISION" >&2
  exit 2
fi
if [[ "$dataset_tree_before" != "$SYMTHAEA_ARC_DATASET_TREE" ]]; then
  echo "ARC dataset tree $dataset_tree_before does not equal declared tree $SYMTHAEA_ARC_DATASET_TREE" >&2
  exit 2
fi
if [[ -n "$dataset_status_before" ]]; then
  echo "ARC dataset checkout must be clean before qualification" >&2
  printf '%s\n' "$dataset_status_before" >&2
  exit 2
fi

manifest_bin="${SYMTHAEA_ARC_DATASET_MANIFEST_BIN:-$repo_root/target/debug/arc_dataset_manifest}"
if [[ ! -x "$manifest_bin" ]]; then
  echo "prebuilt ARC dataset manifest binary not found or executable: $manifest_bin" >&2
  exit 2
fi

"$manifest_bin"

if [[ ! -s "$SYMTHAEA_ARC_DATASET_MANIFEST_PATH" ]]; then
  echo "dataset manifest was not produced: $SYMTHAEA_ARC_DATASET_MANIFEST_PATH" >&2
  exit 2
fi

python3 - "$SYMTHAEA_ARC_DATASET_MANIFEST_PATH" <<'PY'
import hashlib
import json
import os
import re
import sys

path = sys.argv[1]
root = os.path.realpath(os.environ["SYMTHAEA_ARC_DATASET_ROOT"])
with open(path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)

required = {
    "schema_version",
    "manifest_domain",
    "canonical_encoding",
    "repository",
    "revision",
    "tree",
    "split",
    "task_count",
    "files",
    "canonical_byte_length",
    "manifest_blake3",
    "manifest_sha256",
}
missing = sorted(required.difference(manifest))
if missing:
    raise SystemExit(f"manifest is missing required fields: {missing}")
if manifest["schema_version"] != 1:
    raise SystemExit("unexpected dataset manifest schema")
if manifest["manifest_domain"] != "symthaea/reasoning/arc-dataset-manifest/v1":
    raise SystemExit("unexpected dataset manifest domain")
if manifest["canonical_encoding"] != "length-prefixed-le64-v1":
    raise SystemExit("unexpected canonical encoding")
if manifest["repository"] != os.environ["SYMTHAEA_ARC_DATASET_REPOSITORY"]:
    raise SystemExit("manifest repository identity does not match qualified repository")
if manifest["revision"] != os.environ["SYMTHAEA_ARC_DATASET_REVISION"]:
    raise SystemExit("manifest revision does not match qualified revision")
if manifest["tree"] != os.environ["SYMTHAEA_ARC_DATASET_TREE"]:
    raise SystemExit("manifest tree does not match qualified tree")
if manifest["split"] != os.environ["SYMTHAEA_ARC_SPLIT"]:
    raise SystemExit("manifest split does not match qualified split")
expected_tasks = int(os.environ["SYMTHAEA_ARC_EXPECTED_TASKS"])
if manifest["task_count"] != expected_tasks:
    raise SystemExit(
        f"manifest task count {manifest['task_count']} does not equal expected {expected_tasks}"
    )
if manifest["task_count"] != len(manifest["files"]):
    raise SystemExit("task_count does not match file list")
if manifest["canonical_byte_length"] <= 0:
    raise SystemExit("canonical manifest encoding is empty")

hex64 = re.compile(r"^[0-9a-f]{64}$")
if not hex64.fullmatch(manifest["manifest_blake3"]):
    raise SystemExit("invalid manifest BLAKE3 digest")
if not hex64.fullmatch(manifest["manifest_sha256"]):
    raise SystemExit("invalid manifest SHA-256 digest")

paths = [entry["relative_path"] for entry in manifest["files"]]
if paths != sorted(paths):
    raise SystemExit("manifest paths are not sorted")
if len(paths) != len(set(paths)):
    raise SystemExit("manifest contains duplicate paths")

expected_prefix = f"data/{manifest['split']}/"
for entry in manifest["files"]:
    relative = entry["relative_path"]
    if not relative.startswith(expected_prefix):
        raise SystemExit(f"task path escaped selected split: {relative}")
    if entry["byte_length"] <= 0:
        raise SystemExit(f"empty ARC task: {relative}")
    if not hex64.fullmatch(entry["blake3"]):
        raise SystemExit(f"invalid BLAKE3 for {relative}")
    if not hex64.fullmatch(entry["sha256"]):
        raise SystemExit(f"invalid SHA-256 for {relative}")

    task_path = os.path.realpath(os.path.join(root, relative))
    if os.path.commonpath([root, task_path]) != root:
        raise SystemExit(f"task path escapes dataset checkout: {relative}")
    with open(task_path, "rb") as handle:
        payload = handle.read()
    if len(payload) != entry["byte_length"]:
        raise SystemExit(f"byte length mismatch for {relative}")
    if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise SystemExit(f"independent SHA-256 mismatch for {relative}")
PY

subject_head_after="$(git rev-parse HEAD)"
subject_tree_after="$(git rev-parse 'HEAD^{tree}')"
subject_status_after="$(git status --porcelain=v1 --untracked-files=all)"
dataset_head_after="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" rev-parse HEAD)"
dataset_tree_after="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" rev-parse 'HEAD^{tree}')"
dataset_status_after="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" status --porcelain=v1 --untracked-files=all)"

if [[ "$subject_head_after" != "$subject_head_before" || "$subject_tree_after" != "$subject_tree_before" ]]; then
  echo "Symthaea subject changed during dataset qualification" >&2
  exit 2
fi
if [[ -n "$subject_status_after" ]]; then
  echo "Symthaea checkout became dirty during dataset qualification" >&2
  printf '%s\n' "$subject_status_after" >&2
  exit 2
fi
if [[ "$dataset_head_after" != "$dataset_head_before" || "$dataset_tree_after" != "$dataset_tree_before" ]]; then
  echo "ARC dataset subject changed during qualification" >&2
  exit 2
fi
if [[ -n "$dataset_status_after" ]]; then
  echo "ARC dataset checkout became dirty during qualification" >&2
  printf '%s\n' "$dataset_status_after" >&2
  exit 2
fi

printf 'qualified ARC dataset subject %s tree %s split %s\n' \
  "$dataset_head_after" "$dataset_tree_after" "$SYMTHAEA_ARC_SPLIT"
