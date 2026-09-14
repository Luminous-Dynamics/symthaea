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
  SYMTHAEA_ARC_STREAM_RESULTS_PATH \
  SYMTHAEA_ARC_STREAM_VERIFY_PATH \
  SYMTHAEA_ARC_SMOKE_TASK_FILES \
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
  echo "Symthaea HEAD does not match declared streaming subject" >&2
  exit 2
fi
if [[ "$dataset_head_before" != "$SYMTHAEA_ARC_DATASET_REVISION" ]]; then
  echo "dataset HEAD does not match declared streaming subject" >&2
  exit 2
fi
if [[ -n "$subject_status_before" || -n "$dataset_status_before" ]]; then
  echo "code and dataset checkouts must be clean before streaming isolation" >&2
  exit 2
fi

projector_bin="${SYMTHAEA_ARC_PROJECTOR_BIN:-$repo_root/target/debug/arc_solver_view_projector}"
stream_bin="${SYMTHAEA_ARC_STREAM_BIN:-$repo_root/target/debug/arc_streaming_policy}"
verify_bin="${SYMTHAEA_ARC_STREAM_VERIFY_BIN:-$repo_root/target/debug/arc_streaming_verifier}"
for binary in "$projector_bin" "$stream_bin" "$verify_bin"; do
  if [[ ! -x "$binary" ]]; then
    echo "required prebuilt binary missing: $binary" >&2
    exit 2
  fi
done
for command in sudo useradd runuser unshare realpath install setpriv ip bash env; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required isolation command is unavailable: $command" >&2
    exit 2
  fi
done
setpriv_bin="$(command -v setpriv)"
ip_bin="$(command -v ip)"
bash_bin="$(command -v bash)"
env_bin="$(command -v env)"

# Stage A: evaluator authority projects the exact manifest-bound task prefix into a target-stripped
# solver view and separately stores the true targets.
"$projector_bin"
if [[ ! -s "$SYMTHAEA_ARC_SOLVER_VIEW_MANIFEST_PATH" || ! -s "$SYMTHAEA_ARC_TARGET_BUNDLE_PATH" ]]; then
  echo "streaming projection did not emit both solver-view and target artifacts" >&2
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
if sudo -u "$policy_user" -- test -r "$evaluator_root_abs" \
  || sudo -u "$policy_user" -- test -x "$evaluator_root_abs"; then
  echo "streaming policy UID can access evaluator-only root" >&2
  exit 2
fi
if ! sudo -u "$policy_user" -- test -r "$SYMTHAEA_ARC_SOLVER_VIEW_ROOT/training"; then
  echo "streaming policy UID cannot read solver view" >&2
  exit 2
fi

routes="$(sudo unshare --net --fork -- "$ip_bin" route show)"
if [[ -n "$routes" ]]; then
  echo "fresh streaming policy network namespace unexpectedly has routes" >&2
  printf '%s\n' "$routes" >&2
  exit 2
fi

# Stage B: generate both complete streaming baselines in a target-blind sandbox.
sandbox="$repo_root/target/rq-006z/arc-streaming-sandbox"
rm -rf "$sandbox"
mkdir -p "$sandbox/bin" "$sandbox/out"
cp "$stream_bin" "$sandbox/bin/arc_streaming_policy"
chmod 0555 "$sandbox/bin/arc_streaming_policy"
chmod 0755 "$sandbox" "$sandbox/bin"
sudo chown "$policy_uid:$policy_gid" "$sandbox/out"
sudo chmod 0700 "$sandbox/out"

solver_root_abs="$(realpath "$SYMTHAEA_ARC_SOLVER_VIEW_ROOT")"
sandbox_report="$sandbox/out/streaming-report.json"
report_abs="$(realpath -m "$SYMTHAEA_ARC_STREAM_RESULTS_PATH")"
mkdir -p "$(dirname "$report_abs")"

sudo unshare --net --fork -- \
  runuser -u "$policy_user" -- \
  "$setpriv_bin" --no-new-privs \
  "$bash_bin" -c 'set -euo pipefail; sandbox="$1"; shift; cd "$sandbox"; umask 077; exec "$@"' \
  bash "$sandbox" \
  "$env_bin" -i \
    SYMTHAEA_SUBJECT_REVISION="$SYMTHAEA_SUBJECT_REVISION" \
    SYMTHAEA_ARC_SOLVER_VIEW_ROOT="$solver_root_abs" \
    SYMTHAEA_ARC_STREAM_RESULTS_PATH="out/streaming-report.json" \
    SYMTHAEA_ARC_POLICY_SEED="$SYMTHAEA_ARC_POLICY_SEED" \
    SYMTHAEA_ARC_MAX_TASKS="$SYMTHAEA_ARC_SMOKE_TASK_FILES" \
    ./bin/arc_streaming_policy

if ! sudo -u "$policy_user" -- test -s "$sandbox_report"; then
  echo "isolated streaming policy did not emit a report" >&2
  exit 2
fi
sudo install -o "$(id -u)" -g "$(id -g)" -m 0644 "$sandbox_report" "$report_abs"
if [[ ! -s "$SYMTHAEA_ARC_STREAM_RESULTS_PATH" ]]; then
  echo "streaming report was not exported from isolated sandbox" >&2
  exit 2
fi
if sudo -u "$policy_user" -- test -r "$evaluator_root_abs" \
  || sudo -u "$policy_user" -- test -x "$evaluator_root_abs"; then
  echo "streaming policy UID gained evaluator-root access after execution" >&2
  exit 2
fi

# Independent outer coverage check against the evaluator target set. The stream itself never sees
# target content, but it must cover exactly the same problem IDs with both frozen policies.
python3 - "$SYMTHAEA_ARC_TARGET_BUNDLE_PATH" "$SYMTHAEA_ARC_STREAM_RESULTS_PATH" <<'PY'
import collections
import json
import sys

target_path, stream_path = sys.argv[1:]
with open(target_path, "r", encoding="utf-8") as handle:
    target_bundle = json.load(handle)
with open(stream_path, "r", encoding="utf-8") as handle:
    report = json.load(handle)

target_ids = [row.get("problem_id") for row in target_bundle.get("targets", [])]
if not target_ids or len(target_ids) != len(set(target_ids)):
    raise SystemExit("target bundle problem IDs are empty or duplicated")
target_set = set(target_ids)
allowed = {"canonical-stream-v1", "semantic-hash-random-v2"}
pairs = collections.defaultdict(list)
for episode in report.get("episodes", []):
    problem_id = episode.get("problem_id")
    policy_id = episode.get("policy_id")
    if problem_id not in target_set:
        raise SystemExit(f"streaming report references non-target problem {problem_id!r}")
    if policy_id not in allowed:
        raise SystemExit(f"streaming report contains unsupported policy {policy_id!r}")
    pairs[problem_id].append(policy_id)
if set(pairs) != target_set:
    raise SystemExit("streaming report problem set does not equal evaluator target set")
for problem_id in target_ids:
    policies = pairs[problem_id]
    if len(policies) != 2 or set(policies) != allowed:
        raise SystemExit(f"{problem_id} does not have exactly one streaming episode per policy")
if report.get("test_cases_evaluated") != len(target_ids):
    raise SystemExit("streaming report test-case count does not match target set")
if len(report.get("episodes", [])) != 2 * len(target_ids):
    raise SystemExit("streaming report episode count is not two per target")
PY

# Stage C: only after the policy process exits, replay the report independently. The verifier needs
# only target-stripped solver-view bytes; it never receives the true target bundle.
"$verify_bin"
if [[ ! -s "$SYMTHAEA_ARC_STREAM_VERIFY_PATH" ]]; then
  echo "streaming verifier did not emit its evidence artifact" >&2
  exit 2
fi

if sudo find "$sandbox" -type f \( -name '*target*' -o -name '*manifest*' \) | grep -q .; then
  echo "streaming policy sandbox unexpectedly contains evaluator provenance" >&2
  exit 2
fi

subject_head_after="$(git rev-parse HEAD)"
subject_tree_after="$(git rev-parse 'HEAD^{tree}')"
subject_status_after="$(git status --porcelain=v1 --untracked-files=all)"
dataset_head_after="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" rev-parse HEAD)"
dataset_tree_after="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" rev-parse 'HEAD^{tree}')"
dataset_status_after="$(git -C "$SYMTHAEA_ARC_DATASET_ROOT" status --porcelain=v1 --untracked-files=all)"

if [[ "$subject_head_after" != "$subject_head_before" || "$subject_tree_after" != "$subject_tree_before" ]]; then
  echo "Symthaea checkout changed during streaming qualification" >&2
  exit 2
fi
if [[ "$dataset_head_after" != "$dataset_head_before" || "$dataset_tree_after" != "$dataset_tree_before" ]]; then
  echo "dataset checkout changed during streaming qualification" >&2
  exit 2
fi
if [[ -n "$subject_status_after" || -n "$dataset_status_after" ]]; then
  echo "code or dataset checkout became dirty during streaming qualification" >&2
  exit 2
fi

printf 'qualified isolated ARC streaming subject %s dataset %s\n' \
  "$subject_head_after" "$dataset_head_after"
