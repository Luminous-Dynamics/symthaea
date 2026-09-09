#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Focused exact-head software-contract qualification for the shared evidence
# plane and psych-bench evidence/authority surfaces.
#
# This is deliberately NOT an empirical/scientific evidence generator and it
# must never create or update a regression baseline. Run locally from the repo
# root, preferably inside the pinned environment:
#
#   nix develop -c bash scripts/qualify-evidence-contract.sh
#
# CI may set QUALIFIED_SHA to require one exact commit. Local runs default to
# the current HEAD and refuse dirty/untracked source so HEAD really identifies
# the code being qualified.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -n "${UPDATE_SNAPSHOT+x}" ]]; then
    echo "error: UPDATE_SNAPSHOT is present; qualification must remain read-only" >&2
    exit 1
fi

actual_sha="$(git rev-parse HEAD)"
expected_sha="${QUALIFIED_SHA:-$actual_sha}"
if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "error: qualified head mismatch: expected=$expected_sha actual=$actual_sha" >&2
    exit 1
fi

# Exact-head means the commit must identify the source under qualification.
# Ignored build outputs are fine; tracked modifications, staged changes, and
# untracked source are not. A future execution-capsule path may bind a dirty or
# transformed tree explicitly, but this narrow verifier chooses clean-only.
if ! git diff --quiet --ignore-submodules --; then
    echo "error: tracked working-tree changes present; commit SHA does not identify executed source" >&2
    exit 1
fi
if ! git diff --cached --quiet --ignore-submodules --; then
    echo "error: staged changes present; commit SHA does not identify executed source" >&2
    exit 1
fi
untracked="$(git ls-files --others --exclude-standard)"
if [[ -n "$untracked" ]]; then
    echo "error: untracked files present; exact-head qualification requires a clean source tree" >&2
    printf '%s\n' "$untracked" >&2
    exit 1
fi

head_tree="$(git rev-parse 'HEAD^{tree}')"

echo "evidence-contract qualified_sha=$actual_sha"
echo "evidence-contract committed_tree=$head_tree"
echo "evidence-contract scope=software-contract-only"
rustc -Vv
cargo -V

cargo metadata --locked --no-deps --format-version 1 >/dev/null

cargo test --locked -p symthaea-evidence-plane
cargo test --locked -p symthaea-psych-bench
cargo test --locked -p symthaea-psych-bench --features symthaea-backend --lib -- butlin

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
        echo '## Evidence Contract Qualification'
        echo
        echo "- qualified SHA: \`$actual_sha\`"
        echo "- committed tree: \`$head_tree\`"
        echo '- source state: clean exact checkout'
        echo '- scope: software-contract qualification only'
        echo '- full repository CI: independent'
        echo '- empirical/scientific claim authority: none'
    } >> "$GITHUB_STEP_SUMMARY"
fi
