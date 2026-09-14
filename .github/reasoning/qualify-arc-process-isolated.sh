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
  SYMTHAEA_ARC_EVALUATOR_ROOT \
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
for command in sudo useradd runuser unshare realpath install setpriv ip bash; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required isolation command is unavailable: $command" >&2
    exit 2
  fi
done
setpriv_bin="$(command -v setpriv)"
ip_bin="$(command -v ip)"
bash_bin="$(command -v bash)"
env_bin="$(command -v env)"

# Stage A: evaluator-side projection. True targets exist only here and in the target artifact.
"$projector_bin"

if [[ ! -s "$SYMTHAEA_ARC_SOLVER_VIEW_MANIFEST_PATH" || ! -s "$SYMTHAEA_ARC_TARGET_BUNDLE_PATH" ]]; then
  echo "projection did not emit both solver-view and evaluator-target artifacts" >&2
  exit 2
fi

evaluator_root_abs="$(realpath "$SYMTHAEA_ARC_EVALUATOR_ROOT")"
dataset_root_abs="$(realpath "$SYMTHAEA_ARC_DATASET_ROOT")"
manifest_abs="$(realpath "$SYMTHAEA_ARC_DATASET_MANIFEST_PATH")"
target_bundle_abs="$(realpath "$SYMTHAEA_ARC_TARGET_BUNDLE_PATH")"
for sensitive in "$dataset_root_abs" "$manifest_abs" "$target_bundle_abs"; do
  case "$sensitive" in
    "$evaluator_root_abs"/*) ;;
    *)
      echo "evaluator-only input escaped evaluator root: $sensitive" >&2
      exit 2
      ;;
  esac
done

# Harden the whole evaluator authority root. This prevents stale or future evaluator artifacts under
# the same directory from becoming readable simply because one known filename was omitted here.
chmod -R go-rwx "$evaluator_root_abs"
chmod 0700 "$evaluator_root_abs"
chmod -R a+rX "$SYMTHAEA_ARC_SOLVER_VIEW_ROOT"
chmod a+r "$SYMTHAEA_ARC_SOLVER_VIEW_MANIFEST_PATH"

policy_user="rq006zpolicy"
if ! id -u "$policy_user" >/dev/null 2>&1; then
  sudo useradd --system --no-create-home --shell /usr/sbin/nologin "$policy_user"
fi
policy_uid="$(id -u "$policy_user")"
policy_gid="$(id -g "$policy_user")"

# The exact UID that will run the policy must not be able to read or traverse the evaluator root.
if sudo -u "$policy_user" -- test -r "$evaluator_root_abs" \
  || sudo -u "$policy_user" -- test -x "$evaluator_root_abs"; then
  echo "policy UID can access evaluator-only root" >&2
  exit 2
fi
if ! sudo -u "$policy_user" -- test -r "$SYMTHAEA_ARC_SOLVER_VIEW_ROOT/training"; then
  echo "policy UID cannot read target-stripped solver view" >&2
  exit 2
fi

# Fail closed if the runner cannot actually create a route-empty network namespace.
routes="$(sudo unshare --net --fork -- "$ip_bin" route show)"
if [[ -n "$routes" ]]; then
  echo "fresh policy network namespace unexpectedly has routes" >&2
  printf '%s\n' "$routes" >&2
  exit 2
fi

# Stage B: policy-only process. Copy the frozen executable into a dedicated sandbox and run it as a
# separate unprivileged UID in a fresh route-empty network namespace. The real dataset, full
# manifest, evaluator targets, and any evaluator-only output are behind an owner-only directory.
sandbox="$repo_root/target/rq-006z/arc-policy-sandbox"
rm -rf "$sandbox"
mkdir -p "$sandbox/bin" "$sandbox/out"
cp "$policy_bin" "$sandbox/bin/arc_budgeted_policy"
chmod 0555 "$sandbox/bin/arc_budgeted_policy"
chmod 0755 "$sandbox" "$sandbox/bin"
sudo chown "$policy_uid:$policy_gid" "$sandbox/out"
sudo chmod 0700 "$sandbox/out"

solver_root_abs="$(realpath "$SYMTHAEA_ARC_SOLVER_VIEW_ROOT")"
sandbox_report="$sandbox/out/policy-report.json"
report_abs="$(realpath -m "$SYMTHAEA_ARC_POLICY_RESULTS_PATH")"
mkdir -p "$(dirname "$report_abs")"

# Re-prove the final filesystem boundary immediately before execution.
if sudo -u "$policy_user" -- test -r "$evaluator_root_abs" \
  || sudo -u "$policy_user" -- test -x "$evaluator_root_abs"; then
  echo "policy UID unexpectedly gained evaluator-root access" >&2
  exit 2
fi

sudo unshare --net --fork -- \
  runuser -u "$policy_user" -- \
  "$setpriv_bin" --no-new-privs \
  "$bash_bin" -c 'set -euo pipefail; sandbox="$1"; shift; cd "$sandbox"; umask 077; exec "$@"' \
  bash "$sandbox" \
  "$env_bin" -i \
    SYMTHAEA_SUBJECT_REVISION="$SYMTHAEA_SUBJECT_REVISION" \
    SYMTHAEA_ARC_DATASET_VERSION="$SYMTHAEA_ARC_DATASET_REVISION" \
    SYMTHAEA_ARC_SPLIT="training" \
    SYMTHAEA_ARC_DATA_DIR="$solver_root_abs" \
    SYMTHAEA_ARC_POLICY_RESULTS_PATH="out/policy-report.json" \
    SYMTHAEA_ARC_MAX_TASKS="$SYMTHAEA_ARC_SMOKE_TASK_FILES" \
    SYMTHAEA_ARC_CANDIDATE_BUDGET="$SYMTHAEA_ARC_CANDIDATE_BUDGET" \
    SYMTHAEA_ARC_POLICY_SEED="$SYMTHAEA_ARC_POLICY_SEED" \
    ./bin/arc_budgeted_policy

if ! sudo -u "$policy_user" -- test -s "$sandbox_report"; then
  echo "isolated policy process did not emit a report" >&2
  exit 2
fi
sudo install -o "$(id -u)" -g "$(id -g)" -m 0644 "$sandbox_report" "$report_abs"
if [[ ! -s "$SYMTHAEA_ARC_POLICY_RESULTS_PATH" ]]; then
  echo "policy report was not exported from the isolated sandbox" >&2
  exit 2
fi

# Prove the same UID still cannot access evaluator-only state after policy execution.
if sudo -u "$policy_user" -- test -r "$evaluator_root_abs" \
  || sudo -u "$policy_user" -- test -x "$evaluator_root_abs"; then
  echo "policy UID can access evaluator-only root after execution" >&2
  exit 2
fi

# Independent coverage theorem: target problem IDs and policy problem IDs must be exactly equal,
# with one canonical and one random row per target. Equal aggregate counts are not sufficient.
python3 - "$SYMTHAEA_ARC_TARGET_BUNDLE_PATH" "$SYMTHAEA_ARC_POLICY_RESULTS_PATH" <<'PY'
import collections
import json
import sys

target_path, report_path = sys.argv[1:]
with open(target_path, "r", encoding="utf-8") as handle:
    target_bundle = json.load(handle)
with open(report_path, "r", encoding="utf-8") as handle:
    report = json.load(handle)

target_ids = [row.get("problem_id") for row in target_bundle.get("targets", [])]
if not target_ids or any(not isinstance(value, str) or not value for value in target_ids):
    raise SystemExit("target bundle contains invalid problem IDs")
if len(target_ids) != len(set(target_ids)):
    raise SystemExit("target bundle contains duplicate problem IDs")
target_set = set(target_ids)

allowed = {"canonical-order-v1", "uniform-random-without-replacement-v1"}
pairs = collections.defaultdict(list)
for row in report.get("tasks", []):
    problem_id = row.get("problem_id")
    policy_id = row.get("policy_id")
    if problem_id not in target_set:
        raise SystemExit(f"policy report references non-target problem {problem_id!r}")
    if policy_id not in allowed:
        raise SystemExit(f"policy report contains unsupported policy {policy_id!r}")
    pairs[problem_id].append(policy_id)

if set(pairs) != target_set:
    missing = sorted(target_set.difference(pairs))
    extra = sorted(set(pairs).difference(target_set))
    raise SystemExit(f"target/policy problem set mismatch: missing={missing}, extra={extra}")
for problem_id in sorted(target_ids):
    policies = pairs[problem_id]
    if len(policies) != 2 or set(policies) != allowed:
        raise SystemExit(f"{problem_id} does not have exactly one row per frozen policy: {policies}")
if report.get("test_cases_evaluated") != len(target_ids):
    raise SystemExit("policy report test_cases_evaluated does not match target count")
if len(report.get("tasks", [])) != 2 * len(target_ids):
    raise SystemExit("policy report row count does not equal two rows per target")
PY

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

# Independent outer sanity: evaluator provenance was never copied into the policy sandbox, and the
# solver-view files contain only the fixed public sentinel in every test-output slot.
if sudo find "$sandbox" -type f \( -name '*target*' -o -name '*manifest*' \) | grep -q .; then
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
