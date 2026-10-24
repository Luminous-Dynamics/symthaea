#!/usr/bin/env bash
set -uo pipefail

out_dir="${1:-target/mel005a-qualification}"
expected_head="${MEL005A_EXPECTED_HEAD:-}"
actual_head="$(git rev-parse HEAD 2>&1 || true)"
tree_sha="$(git rev-parse 'HEAD^{tree}' 2>&1 || true)"
preflight_worktree_status="$(git status --porcelain --untracked-files=all 2>&1 || true)"
mkdir -p "$out_dir"
exec > >(tee "$out_dir/runner.log") 2>&1

commit_subject="$(git show -s --format=%s HEAD)"
rustc_version="$(rustc --version --verbose 2>&1)"
cargo_version="$(cargo --version --verbose 2>&1)"
rustup_toolchain="$(rustup show active-toolchain 2>&1 || true)"

printf 'qualification=MEL-005A-focused-v1\n'
printf 'utc_started=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'git_head=%s\n' "$actual_head"
printf 'git_tree_sha=%s\n' "$tree_sha"
printf 'git_subject=%s\n' "$commit_subject"
printf 'expected_head=%s\n' "$expected_head"
printf 'rustup_active_toolchain=%s\n' "$rustup_toolchain"
printf '%s\n' '--- rustc --version --verbose ---' "$rustc_version"
printf '%s\n' '--- cargo --version --verbose ---' "$cargo_version"
git status --short --branch

steps=(format importer_test studio_check state_space_test world_state_test)
for step in "${steps[@]}"; do
  printf 'NOT_RUN_FAIL_FAST\n' > "$out_dir/$step.exit_code"
done

if [[ -z "$expected_head" || "$actual_head" != "$expected_head" ]]; then
  printf 'qualification=FAIL\nreason=checkout_head_mismatch_or_missing_expected_head\n' | tee "$out_dir/result.txt"
  printf 'expected_head=%s\nactual_head=%s\n' "$expected_head" "$actual_head" > "$out_dir/head-mismatch.txt"
  for step in "${steps[@]}"; do
    printf 'NOT_RUN_HEAD_MISMATCH\n' > "$out_dir/$step.exit_code"
  done
  exit 78
fi

printf '%s\n' "$preflight_worktree_status" > "$out_dir/preflight-worktree-status.txt"
if [[ -n "$preflight_worktree_status" ]]; then
  printf 'qualification=FAIL\nreason=dirty_or_unverifiable_worktree_before_qualification\n' | tee "$out_dir/result.txt"
  printf '%s\n' "$preflight_worktree_status" > "$out_dir/worktree-dirty.txt"
  for step in "${steps[@]}"; do
    printf 'NOT_RUN_DIRTY_WORKTREE\n' > "$out_dir/$step.exit_code"
  done
  exit 79
fi

run_step() {
  local name="$1"
  shift
  local current_head
  local current_worktree_status
  local command_status
  local tee_status
  local post_head
  local post_worktree_status
  local -a pipeline_status
  current_head="$(git rev-parse HEAD)"
  if [[ "$current_head" != "$actual_head" ]]; then
    printf 'HEAD changed during qualification: initial=%s current=%s\n' "$actual_head" "$current_head" >&2
    printf 'HEAD_CHANGED_DURING_RUN\n' > "$out_dir/$name.exit_code"
    return 78
  fi
  current_worktree_status="$(git status --porcelain --untracked-files=all 2>&1 || true)"
  if [[ -n "$current_worktree_status" ]]; then
    printf 'Dirty/unverifiable worktree before step %s:\n%s\n' "$name" "$current_worktree_status" >&2
    printf 'NOT_RUN_DIRTY_WORKTREE\n' > "$out_dir/$name.exit_code"
    printf '%s\n' "$current_worktree_status" > "$out_dir/$name.worktree-dirty.txt"
    return 79
  fi

  {
    printf 'step=%s\n' "$name"
    printf 'git_head=%s\n' "$current_head"
    printf 'git_tree_sha=%s\n' "$tree_sha"
    printf 'commit_subject=%s\n' "$commit_subject"
    printf 'rustup_active_toolchain=%s\n' "$rustup_toolchain"
    printf 'rustc_version=%s\n' "$(rustc --version)"
    printf 'cargo_version=%s\n' "$(cargo --version)"
    printf 'command='
    printf '%q ' "$@"
    printf '\n'
  } > "$out_dir/$name.receipt.txt"

  printf '\n===== STEP %s =====\n' "$name"
  cat "$out_dir/$name.receipt.txt"
  printf '%q ' "$@" > "$out_dir/$name.command.txt"
  printf '\n' >> "$out_dir/$name.command.txt"

  "$@" 2>&1 | tee "$out_dir/$name.output.log"
  pipeline_status=("${PIPESTATUS[@]}")
  command_status="${pipeline_status[0]}"
  tee_status="${pipeline_status[1]:-1}"
  # The primary command is recorded independently; artifact capture failure is
  # also fatal because a pass without complete output is not valid evidence.
  if [[ "$command_status" -eq 0 && "$tee_status" -ne 0 ]]; then
    command_status=74
  fi
  post_head="$(git rev-parse HEAD 2>&1 || true)"
  if [[ "$post_head" != "$actual_head" ]]; then
    printf 'HEAD changed during step %s: initial=%s current=%s\n' "$name" "$actual_head" "$post_head" >&2
    printf 'initial=%s\ncurrent=%s\n' "$actual_head" "$post_head" > "$out_dir/$name.head-change.txt"
    if [[ "$command_status" -eq 0 ]]; then command_status=78; fi
  fi
  post_worktree_status="$(git status --porcelain --untracked-files=all 2>&1 || true)"
  if [[ -n "$post_worktree_status" ]]; then
    printf 'Worktree changed or became unverifiable during step %s:\n%s\n' "$name" "$post_worktree_status" >&2
    printf '%s\n' "$post_worktree_status" > "$out_dir/$name.worktree-dirty.txt"
    if [[ "$command_status" -eq 0 ]]; then command_status=79; fi
  fi
  printf '%s\n' "$command_status" > "$out_dir/$name.exit_code"
  printf 'step=%s\nexit_code=%s\n' "$name" "$command_status" | tee -a "$out_dir/$name.receipt.txt"
  return "$command_status"
}

run_step format cargo fmt -p symthaea-muse -p symthaea-music-theory -- --check
rc=$?
if [[ "$rc" -ne 0 ]]; then
  printf 'qualification=FAIL\nfailed_step=format\nexit_code=%s\n' "$rc" > "$out_dir/result.txt"
  exit "$rc"
fi

run_step importer_test cargo test --locked -p symthaea-muse --features theory --lib symbolic_import::tests::
rc=$?
if [[ "$rc" -ne 0 ]]; then
  printf 'qualification=FAIL\nfailed_step=importer_test\nexit_code=%s\n' "$rc" > "$out_dir/result.txt"
  exit "$rc"
fi

run_step studio_check cargo check --locked -p symthaea-muse --features studio --bin muse_studio
rc=$?
if [[ "$rc" -ne 0 ]]; then
  printf 'qualification=FAIL\nfailed_step=studio_check\nexit_code=%s\n' "$rc" > "$out_dir/result.txt"
  exit "$rc"
fi

run_step state_space_test cargo test --locked -p symthaea-music-theory --lib state_space::tests::
rc=$?
if [[ "$rc" -ne 0 ]]; then
  printf 'qualification=FAIL\nfailed_step=state_space_test\nexit_code=%s\n' "$rc" > "$out_dir/result.txt"
  exit "$rc"
fi

run_step world_state_test cargo test --locked -p symthaea-music-theory --lib world_state::tests::
rc=$?
if [[ "$rc" -ne 0 ]]; then
  printf 'qualification=FAIL\nfailed_step=world_state_test\nexit_code=%s\n' "$rc" > "$out_dir/result.txt"
  exit "$rc"
fi

printf 'qualification=PASS\nexact_head=%s\nsource_tree=%s\ncommands_passed=5\nutc_finished=%s\n' \
  "$actual_head" "$tree_sha" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$out_dir/result.txt"
