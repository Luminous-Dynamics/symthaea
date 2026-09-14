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
  SYMTHAEA_ARC_DATASET_REVISION \
  SYMTHAEA_ARC_DATASET_MANIFEST_PATH \
  SYMTHAEA_ARC_SOLVER_VIEW_ROOT \
  SYMTHAEA_ARC_SOLVER_VIEW_MANIFEST_PATH \
  SYMTHAEA_ARC_TARGET_BUNDLE_PATH \
  SYMTHAEA_ARC_POLICY_RESULTS_PATH \
  SYMTHAEA_ARC_ISOLATED_EVALUATION_PATH \
  SYMTHAEA_ARC_EXHAUSTIVE_VERIFY_PATH \
  SYMTHAEA_ARC_SMOKE_TASK_FILES \
  SYMTHAEA_ARC_CANDIDATE_BUDGET \
  SYMTHAEA_ARC_POLICY_SEED; do
  require_env "$name"
done

repo_root="$(git rev-parse --show-toplevel)"
subject_head_before="$(git rev-parse HEAD)"
subject_tree_before="$(git rev-parse 'HEAD^{tree}')"
subject_status_before="$(git status --porcelain=v1 --untracked-files=all)"
dataset_head_before="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" rev-parse HEAD)"
dataset_tree_before="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" rev-parse 'HEAD^{tree}')"
dataset_status_before="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" status --porcelain=v1 --untracked-files=all)"

if [[ "$subject_head_before" != "$SYMTHAEA_SUBJECT_REVISION" ]]; then
  echo "Symthaea HEAD does not match declared process-isolated subject" >&2
  exit 2
fi
if [[ -n "$subject_status_before" || -n "$dataset_status_before" ]]; then
  echo "both code and dataset checkouts must be clean before isolation run" >&2
  exit 2
fi
if [[ "$dataset_head_before" != "$SYMTHAEA_ARC_DATASET_REVISION" ]]; then
  echo "dataset HEAD does not match declared process-isolated subject" >&2
  exit 2
fi

projector_bin="${SYMTHAEA_ARC_PROJECTOR_BIN:-$repo_root/target/debug/arc_solver_view_projector}"
policy_bin="${SYMTHAEA_ARC_POLICY_BIN:-$repo_root/target/debug/arc_budgeted_policy}"
evaluator_bin="${SYMTHAEA_ARC_ISOLATED_EVALUATOR_BIN:-$repo_root/target/debug/arc_process_isolated_evaluator}"
exhaustive_bin="${SYMTHAEA_ARC_EXHAUSTIVE_VERIFY_BIN:-$repo_root/target/debug/arc_isolated_exhaustive_verifier}"
for binary in "$projector_bin" "$policy_bin" "$evaluator_bin" "$exhaustive_bin"; do
  if [[ ! -x "$binary" ]]; then
    echo "required prebuilt binary missing: $binary" >&2
    exit 2
  fi
done

# Stage A: evaluator-side projection. True targets exist only here and in the target artifact.
"$projector_bin"

if [[ ! -s "$SYMTHAEA_ARC_SOLVER_VIEW_MANIFEST_PATH" || ! -s "$SYMTHAEA_ARC_TARGET_BUNDLE_PATH" ]]; then
  echo "projection did not emit both solver-view and evaluator-target artifacts" >&2
  exit 2
fi

# Stage B: policy-only process. Copy the frozen executable into an isolated working directory and
# run with an empty environment except for solver-visible configuration. Real dataset root,
# manifest path, target bundle path/tree and evaluator outputs are absent from the process env.
sandbox="$repo_root/target/rq-006z/arc-policy-sandbox"
rm -rf "$sandbox"
mkdir -p "$sandbox/bin" "$sandbox/out"
cp "$policy_bin" "$sandbox/bin/arc_budgeted_policy"
chmod 0555 "$sandbox/bin/arc_budgeted_policy"
solver_root_abs="$(realpath "$SYMTHAEA_ARC_SOLVER_VIEW_ROOT")"
report_abs="$(realpath -m "$SYMTHAEA_ARC_POLICY_RESULTS_PATH")"

(
  cd "$sandbox"
  env -i \
    PATH="$PATH" \
    SYMTHAEA_SUBJECT_REVISION="$SYMTHAEA_SUBJECT_REVISION" \
    SYMTHAEA_ARC_DATASET_VERSION="$SYMTHAEA_ARC_DATASET_REVISION" \
    SYMTHAEA_ARC_SPLIT="training" \
    SYMTHAEA_ARC_DATA_DIR="$solver_root_abs" \
    SYMTHAEA_ARC_POLICY_RESULTS_PATH="$report_abs" \
    SYMTHAEA_ARC_MAX_TASKS="$SYMTHAEA_ARC_SMOKE_TASK_FILES" \
    SYMTHAEA_ARC_CANDIDATE_BUDGET="$SYMTHAEA_ARC_CANDIDATE_BUDGET" \
    SYMTHAEA_ARC_POLICY_SEED="$SYMTHAEA_ARC_POLICY_SEED" \
    ./bin/arc_budgeted_policy
)

if [[ ! -s "$SYMTHAEA_ARC_POLICY_RESULTS_PATH" ]]; then
  echo "isolated policy process did not emit a report" >&2
  exit 2
fi

# Stage C: evaluator-only processes. The primary evaluator gets real targets and independently
# replays each policy action/seal. A separate verifier recomputes the full exhaustive reference.
"$evaluator_bin"
"$exhaustive_bin"

for artifact in \
  "$SYMTHAEA_ARC_ISOLATED_EVALUATION_PATH" \
  "$SYMTHAEA_ARC_EXHAUSTIVE_VERIFY_PATH"; do
  if [[ ! -s "$artifact" ]]; then
    echo "missing process-isolated evaluation artifact: $artifact" >&2
    exit 2
  fi
done

# Independent outer sanity: the true target bundle must never have been copied into the policy
# sandbox. The solver-view files must contain the fixed public sentinel in every test output.
if find "$sandbox" -type f -name '*target*' -o -name '*manifest*' | grep -q .; then
  echo "policy sandbox unexpectedly contains evaluator provenance" >&2
  exit 2
fi
python3 - "$SYMTHAEA_ARC_SOLVER_VIEW_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "training"
paths = sorted(root.glob("*.json"))
if not paths:
    raise SystemExit("solver view contains no tasks")
for path in paths:
    task = json.loads(path.read_text(encoding="utf-8"))
    for pair in task["test"]:
        if pair.get("output") != [[0]]:
            raise SystemExit(f"non-sentinel test output reached solver view: {path}")
PY

subject_head_after="$(git rev-parse HEAD)"
subject_tree_after="$(git rev-parse 'HEAD^{tree}')"
subject_status_after="$(git status --porcelain=v1 --untracked-files=all)"
dataset_head_after="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" rev-parse HEAD)"
dataset_tree_after="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" rev-parse 'HEAD^{tree}')"
dataset_status_after="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" status --porcelain=v1 --untracked-files=all)"

if [[ "$subject_head_after" != "$subject_head_before" || "$subject_tree_after" != "$subject_tree_before" ]]; then
  echo "Symthaea checkout changed during process-isolated qualification" >&2
  exit 2
fi
if [[ "$dataset_head_after" != "$dataset_head_before" || "$dataset_tree_after" != "$dataset_tree_before" ]]; then
  echo "dataset checkout changed during process-isolated qualification" >&2
  exit 2
fi
if [[ -n "$subject_status_after" || -n "$dataset_status_after" ]]; then
  echo "code or dataset checkout became dirty during process-isolated qualification" >&2
  exit 2
fi

printf 'qualified process-isolated ARC policy subject %s dataset %s\n' \
  "$subject_head_after" "$dataset_head_after"
