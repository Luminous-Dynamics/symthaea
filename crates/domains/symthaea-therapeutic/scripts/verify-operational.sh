#!/usr/bin/env bash
set -euo pipefail

package="symthaea-therapeutic"
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

scripts/verify-governance.sh

# Operational modules must compile and their regression tests must execute in
# the default, research-disabled build.
cargo test -p "$package" --no-default-features policy_runtime::tests
cargo test -p "$package" --no-default-features model_registry::tests
cargo test -p "$package" --no-default-features red_team::tests
cargo test -p "$package" --no-default-features observability::tests
cargo test -p "$package" --no-default-features resilience::tests
cargo test -p "$package" --no-default-features recovery::tests
cargo test -p "$package" --no-default-features release_evidence::tests
cargo test -p "$package" --no-default-features incident::tests

# Fail on accidentally committed conflict markers or secret-looking operational
# keys. This is intentionally conservative and does not replace secret scanning.
if git grep -nE '^(<<<<<<<|=======|>>>>>>>)' -- . ':!THERAPEUTIC_OPERATIONAL_MIGRATION.md'; then
  echo "merge conflict marker detected" >&2
  exit 1
fi

if git grep -nE '(audit_key|checkpoint_key|release_key|subject_salt)[[:space:]]*=[[:space:]]*\[[0-9]+;[[:space:]]*32\]' -- ':!src/**/tests/**' ':!**/*.md'; then
  echo "possible hard-coded operational secret detected" >&2
  exit 1
fi

# Produce stable, non-secret evidence inputs for the release envelope.
printf 'source_tree=%s\n' "$(git write-tree)"
printf 'head_commit=%s\n' "$(git rev-parse HEAD)"
printf 'crate=%s\n' "$package"
