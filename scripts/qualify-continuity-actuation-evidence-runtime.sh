#!/usr/bin/env bash
set -uo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
runtime_subject="${RUNTIME_SUBJECT_SHA:-}"
qualifier_sha="${QUALIFIER_SHA:-}"
receipt="${CONTINUITY_RUNTIME_RECEIPT:-}"
package="symthaea-continuity"
qualification_scope="package:${package}"
manifest="crates/core/symthaea-continuity/Cargo.toml"
workflow_path=".github/workflows/continuity-actuation-evidence-runtime.yml"
qualifier_path="scripts/qualify-continuity-actuation-evidence-runtime.sh"

status="INDETERMINATE"
failure_class="NONE"
phase="bootstrap"
reason="NONE"
failed_command="NONE"
failed_exit_code="0"
closed_world_gate="NOT_RUN"
runtime_inputs_unchanged="NOT_RUN"
subject_worktree_verified="NOT_RUN"
dependency_fetch="NOT_RUN"
cargo_fmt="NOT_RUN"
cargo_check_all_targets="NOT_RUN"
cargo_test_lib="NOT_RUN"
cargo_test_doc="NOT_RUN"
cargo_clippy_all_targets_deny_warnings="NOT_RUN"
runtime_tree="UNAVAILABLE"
qualifier_tree="UNAVAILABLE"
changed_paths_sha256="UNAVAILABLE"
changed_path_count="0"
lock_blob="UNAVAILABLE"
manifest_blob="UNAVAILABLE"
toolchain_blob="UNAVAILABLE"
qualifier_blob="UNAVAILABLE"
workflow_blob="UNAVAILABLE"
rustc_version="UNAVAILABLE"
cargo_version="UNAVAILABLE"
subject_root=""
scratch_root=""

sanitize() {
  printf '%s' "$1" | tr '\t\r\n' '   '
}

write_receipt() {
  [[ -n "$receipt" ]] || return 0
  mkdir -p "$(dirname "$receipt")" || return 0
  {
    printf 'schema\tsymthaea-continuity-actuation-evidence-runtime-qualification-v2\n'
    printf 'status\t%s\n' "$(sanitize "$status")"
    printf 'failure_class\t%s\n' "$(sanitize "$failure_class")"
    printf 'phase\t%s\n' "$(sanitize "$phase")"
    printf 'reason\t%s\n' "$(sanitize "$reason")"
    printf 'failed_command\t%s\n' "$(sanitize "$failed_command")"
    printf 'failed_exit_code\t%s\n' "$failed_exit_code"
    printf 'qualification_scope\t%s\n' "$(sanitize "$qualification_scope")"
    printf 'runtime_subject_sha\t%s\n' "$(sanitize "$runtime_subject")"
    printf 'runtime_subject_tree\t%s\n' "$runtime_tree"
    printf 'qualifier_sha\t%s\n' "$(sanitize "$qualifier_sha")"
    printf 'qualifier_tree\t%s\n' "$qualifier_tree"
    printf 'changed_paths_sha256\t%s\n' "$changed_paths_sha256"
    printf 'changed_path_count\t%s\n' "$changed_path_count"
    printf 'cargo_lock_blob\t%s\n' "$lock_blob"
    printf 'crate_manifest_blob\t%s\n' "$manifest_blob"
    printf 'rust_toolchain_blob\t%s\n' "$toolchain_blob"
    printf 'qualifier_script_blob\t%s\n' "$qualifier_blob"
    printf 'workflow_blob\t%s\n' "$workflow_blob"
    printf 'rustc_version\t%s\n' "$(sanitize "$rustc_version")"
    printf 'cargo_version\t%s\n' "$(sanitize "$cargo_version")"
    printf 'runner_os\t%s\n' "$(sanitize "${RUNNER_OS:-UNAVAILABLE}")"
    printf 'runner_arch\t%s\n' "$(sanitize "${RUNNER_ARCH:-UNAVAILABLE}")"
    printf 'image_os\t%s\n' "$(sanitize "${ImageOS:-UNAVAILABLE}")"
    printf 'github_run_id\t%s\n' "$(sanitize "${GITHUB_RUN_ID:-UNAVAILABLE}")"
    printf 'github_run_attempt\t%s\n' "$(sanitize "${GITHUB_RUN_ATTEMPT:-UNAVAILABLE}")"
    printf 'github_job\t%s\n' "$(sanitize "${GITHUB_JOB:-UNAVAILABLE}")"
    printf 'github_sha\t%s\n' "$(sanitize "${GITHUB_SHA:-UNAVAILABLE}")"
    printf 'closed_world_repository_input_gate\t%s\n' "$closed_world_gate"
    printf 'runtime_inputs_unchanged\t%s\n' "$runtime_inputs_unchanged"
    printf 'subject_worktree_verified\t%s\n' "$subject_worktree_verified"
    printf 'dependency_fetch\t%s\n' "$dependency_fetch"
    printf 'cargo_fmt\t%s\n' "$cargo_fmt"
    printf 'cargo_check_all_targets\t%s\n' "$cargo_check_all_targets"
    printf 'cargo_test_lib\t%s\n' "$cargo_test_lib"
    printf 'cargo_test_doc\t%s\n' "$cargo_test_doc"
    printf 'cargo_clippy_all_targets_deny_warnings\t%s\n' "$cargo_clippy_all_targets_deny_warnings"
  } > "$receipt"
  cat "$receipt"
}

cleanup() {
  if [[ -n "$subject_root" && -d "$subject_root" && -n "$repo_root" ]]; then
    git -C "$repo_root" worktree remove --force "$subject_root" >/dev/null 2>&1 || true
  fi
  if [[ -n "$scratch_root" && -d "$scratch_root" ]]; then
    rm -rf "$scratch_root"
  fi
}
trap cleanup EXIT

fail_qualification() {
  failure_class="$1"
  phase="$2"
  reason="$3"
  failed_exit_code="${4:-1}"
  status="FAIL"
  write_receipt
  exit "$failed_exit_code"
}

if [[ -z "$receipt" ]]; then
  echo 'CONTINUITY_RUNTIME_RECEIPT is required' >&2
  exit 20
fi
mkdir -p "$(dirname "$receipt")" || {
  echo "cannot create receipt directory: $(dirname "$receipt")" >&2
  exit 20
}

[[ -n "$repo_root" ]] || fail_qualification QUALIFIER_INVALID bootstrap 'not inside a Git repository' 20
cd "$repo_root" || fail_qualification QUALIFIER_INVALID bootstrap 'cannot enter repository root' 20
[[ -n "$runtime_subject" ]] || fail_qualification QUALIFIER_INVALID preflight 'RUNTIME_SUBJECT_SHA is required' 20
[[ -n "$qualifier_sha" ]] || fail_qualification QUALIFIER_INVALID preflight 'QUALIFIER_SHA is required' 20

actual_head="$(git rev-parse HEAD 2>/dev/null || true)"
if [[ "$actual_head" != "$qualifier_sha" ]]; then
  fail_qualification QUALIFIER_INVALID preflight "qualifier HEAD mismatch: expected=$qualifier_sha actual=$actual_head" 20
fi

if ! git cat-file -e "${runtime_subject}^{commit}" 2>/dev/null; then
  fail_qualification QUALIFIER_INVALID preflight "runtime subject is not available as a commit: $runtime_subject" 20
fi
if ! git merge-base --is-ancestor "$runtime_subject" "$qualifier_sha"; then
  fail_qualification QUALIFIER_INVALID preflight 'runtime subject is not an ancestor of qualifier head' 20
fi

runtime_tree="$(git show -s --format=%T "$runtime_subject" 2>/dev/null || true)"
qualifier_tree="$(git show -s --format=%T "$qualifier_sha" 2>/dev/null || true)"
[[ -n "$runtime_tree" && -n "$qualifier_tree" ]] || fail_qualification QUALIFIER_INVALID preflight 'unable to resolve subject/qualifier tree identity' 20

changed_output="$(git diff --name-only "$runtime_subject" "$qualifier_sha" 2>/dev/null)" || \
  fail_qualification QUALIFIER_INVALID preflight 'unable to enumerate subject-to-qualifier path changes' 20
changed_paths=()
if [[ -n "$changed_output" ]]; then
  mapfile -t changed_paths <<< "$changed_output"
fi
changed_path_count="${#changed_paths[@]}"
changed_paths_sha256="$(printf '%s\n' "${changed_paths[@]}" | sha256sum | awk '{print $1}')"

allowed_paths=(
  "$workflow_path"
  "docs/release/evidence/continuity-actuation-enforcement-campaign-v1.md"
  "docs/release/evidence/continuity-actuation-enforcement-invalidation-v1.md"
  "$qualifier_path"
)

is_allowed_path() {
  local candidate="$1"
  local allowed
  for allowed in "${allowed_paths[@]}"; do
    [[ "$candidate" == "$allowed" ]] && return 0
  done
  return 1
}

for path in "${changed_paths[@]}"; do
  if ! is_allowed_path "$path"; then
    fail_qualification QUALIFIER_INVALID closed_world "runtime qualification input drift outside closed allowlist: $path" 20
  fi
done

for required in "${allowed_paths[@]}"; do
  if [[ ! -f "$required" ]]; then
    fail_qualification QUALIFIER_INVALID closed_world "required qualifier/evidence path missing: $required" 20
  fi
  found=0
  for path in "${changed_paths[@]}"; do
    if [[ "$path" == "$required" ]]; then
      found=1
      break
    fi
  done
  if [[ "$found" -ne 1 ]]; then
    fail_qualification QUALIFIER_INVALID closed_world "required closed-world path is not part of the frozen subject delta: $required" 20
  fi
done
closed_world_gate="PASS"

if ! git diff --quiet "$runtime_subject" "$qualifier_sha" -- \
  crates/core/symthaea-continuity \
  Cargo.toml \
  Cargo.lock \
  rust-toolchain.toml; then
  fail_qualification QUALIFIER_INVALID runtime_input_gate 'explicit Cargo/runtime input drift detected' 20
fi
runtime_inputs_unchanged="PASS"

lock_blob="$(git rev-parse "${runtime_subject}:Cargo.lock" 2>/dev/null || true)"
manifest_blob="$(git rev-parse "${runtime_subject}:${manifest}" 2>/dev/null || true)"
toolchain_blob="$(git rev-parse "${runtime_subject}:rust-toolchain.toml" 2>/dev/null || true)"
qualifier_blob="$(git rev-parse "${qualifier_sha}:${qualifier_path}" 2>/dev/null || true)"
workflow_blob="$(git rev-parse "${qualifier_sha}:${workflow_path}" 2>/dev/null || true)"
if [[ -z "$lock_blob" || -z "$manifest_blob" || -z "$toolchain_blob" || -z "$qualifier_blob" || -z "$workflow_blob" ]]; then
  fail_qualification QUALIFIER_INVALID provenance 'unable to bind one or more exact Git blobs' 20
fi

scratch_base="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
scratch_root="$(mktemp -d "${scratch_base%/}/continuity-runtime.XXXXXX" 2>/dev/null || true)"
[[ -n "$scratch_root" && -d "$scratch_root" ]] || fail_qualification QUALIFIER_INVALID subject_materialization 'unable to allocate exact-subject worktree scratch directory' 20
subject_root="$scratch_root/subject"
if ! git worktree add --detach "$subject_root" "$runtime_subject" >/dev/null 2>&1; then
  fail_qualification QUALIFIER_INVALID subject_materialization 'unable to materialize detached exact-subject worktree' 20
fi

materialized_head="$(git -C "$subject_root" rev-parse HEAD 2>/dev/null || true)"
materialized_tree="$(git -C "$subject_root" show -s --format=%T HEAD 2>/dev/null || true)"
if [[ "$materialized_head" != "$runtime_subject" || "$materialized_tree" != "$runtime_tree" ]]; then
  fail_qualification QUALIFIER_INVALID subject_materialization "materialized subject identity mismatch: head=$materialized_head tree=$materialized_tree" 20
fi
subject_worktree_verified="PASS"

rustc_version="$(cd "$subject_root" && rustc --version 2>/dev/null || true)"
cargo_version="$(cd "$subject_root" && cargo --version 2>/dev/null || true)"
if [[ "$rustc_version" != rustc\ 1.96.0* ]]; then
  fail_qualification QUALIFIER_INVALID toolchain "exact subject did not select rustc 1.96.0: $rustc_version" 20
fi
if [[ -z "$cargo_version" ]]; then
  fail_qualification QUALIFIER_INVALID toolchain 'cargo version unavailable in exact-subject worktree' 20
fi

run_subject() {
  local result_var="$1"
  local class="$2"
  local command_label="$3"
  shift 3
  phase="$command_label"
  failed_command="$(printf '%q ' "$@")"
  echo "[$command_label] $failed_command"
  (cd "$subject_root" && "$@")
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    printf -v "$result_var" '%s' FAIL
    if [[ "$class" == "INFRASTRUCTURE_INDETERMINATE" ]]; then
      fail_qualification "$class" "$command_label" 'dependency preparation failed before offline subject qualification' 30
    fi
    fail_qualification "$class" "$command_label" 'exact runtime subject command failed' 40
  fi
  printf -v "$result_var" '%s' PASS
  failed_command="NONE"
  failed_exit_code="0"
}

run_subject cargo_fmt FAIL_SUBJECT cargo_fmt \
  cargo fmt --package "$package" --manifest-path "$manifest" -- --check

run_subject dependency_fetch INFRASTRUCTURE_INDETERMINATE dependency_fetch \
  cargo fetch --locked --manifest-path "$manifest"

run_subject cargo_check_all_targets FAIL_SUBJECT cargo_check_all_targets \
  cargo check --offline --locked --package "$package" --manifest-path "$manifest" --all-targets

run_subject cargo_test_lib FAIL_SUBJECT cargo_test_lib \
  cargo test --offline --locked --package "$package" --manifest-path "$manifest" --lib

run_subject cargo_test_doc FAIL_SUBJECT cargo_test_doc \
  cargo test --offline --locked --package "$package" --manifest-path "$manifest" --doc

run_subject cargo_clippy_all_targets_deny_warnings FAIL_SUBJECT cargo_clippy_all_targets_deny_warnings \
  cargo clippy --offline --locked --package "$package" --manifest-path "$manifest" --all-targets -- -D warnings

status="PASS"
failure_class="NONE"
phase="complete"
reason="all exact-subject package-scoped Rust qualification commands passed"
failed_command="NONE"
failed_exit_code="0"
write_receipt
