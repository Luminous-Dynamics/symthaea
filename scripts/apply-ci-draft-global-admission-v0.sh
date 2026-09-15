#!/usr/bin/env bash
set -euo pipefail

workflow='.github/workflows/ci.yml'
patch='docs/qualification-admissions/ci-draft-global-admission-v0.patch'
expected_workflow_blob='a48366076b30eb8e12d22c927a3b8bf333181409'
expected_patch_blob='763b3ef4d2def578e4859c315c2f196e33acc321'

[[ -f "$workflow" ]]
[[ -f "$patch" ]]

tracked_blob="$(git rev-parse "HEAD:$workflow")"
worktree_blob="$(git hash-object "$workflow")"
patch_blob="$(git hash-object "$patch")"

if [[ "$tracked_blob" != "$expected_workflow_blob" ]]; then
  echo "refusing: tracked ci.yml blob drifted: $tracked_blob" >&2
  exit 2
fi
if [[ "$worktree_blob" != "$expected_workflow_blob" ]]; then
  echo "refusing: worktree ci.yml bytes are not the audited source blob: $worktree_blob" >&2
  exit 3
fi
if [[ "$patch_blob" != "$expected_patch_blob" ]]; then
  echo "refusing: staged admission patch bytes drifted: $patch_blob" >&2
  exit 4
fi
if ! git diff --quiet -- "$workflow"; then
  echo 'refusing: ci.yml already has uncommitted changes' >&2
  exit 5
fi

git apply --check "$patch"
git apply "$patch"

git diff --check -- "$workflow"
mapfile -t changed < <(git diff --name-only)
if [[ "${#changed[@]}" -ne 1 || "${changed[0]}" != "$workflow" ]]; then
  printf 'refusing: patch changed unexpected paths: %s\n' "${changed[*]-}" >&2
  exit 6
fi

numstat="$(git diff --numstat -- "$workflow")"
if [[ "$numstat" != $'3\t2\t.github/workflows/ci.yml' ]]; then
  echo "refusing: unexpected ci.yml diff cardinality: $numstat" >&2
  exit 7
fi

grep -Fq 'types: [opened, synchronize, reopened, ready_for_review]' "$workflow"
grep -Fq "github.event_name == 'pull_request' && github.event.pull_request.draft == true && 'draft-global' || github.ref" "$workflow"
grep -Fq "cancel-in-progress: \${{ github.event_name != 'pull_request' || github.event.pull_request.draft != true }}" "$workflow"

echo 'ci_draft_global_admission_v0=APPLIED_TO_WORKTREE_NOT_COMMITTED'
echo "source_workflow_blob=$expected_workflow_blob"
echo "patch_blob=$expected_patch_blob"
echo 'changed_paths=1 additions=3 deletions=2'
echo 'draft_global_capacity=one_running_one_newest_pending'
