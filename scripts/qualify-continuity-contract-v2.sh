#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Exact-head software-contract qualification for symthaea-continuity.
#
# V2 never mutates the exact checkout. Repair diagnostics run in detached temporary
# worktrees and remain non-qualifying. Derived bytes therefore cannot overwrite the
# identity of the committed head that actually passed or failed.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

actual_sha="$(git rev-parse HEAD)"
committed_tree="$(git rev-parse 'HEAD^{tree}')"
expected_sha="${QUALIFIED_SHA:-$actual_sha}"
receipt_path="${CONTINUITY_CONTRACT_RECEIPT:-${TMPDIR:-/tmp}/symthaea-continuity-contract-qualification-v2.tsv}"
format_patch_path="${CONTINUITY_FORMAT_PATCH:-${TMPDIR:-/tmp}/symthaea-continuity-rustfmt-v2.patch}"
format_source_dir="${CONTINUITY_FORMAT_SOURCE_DIR:-${TMPDIR:-/tmp}/symthaea-continuity-rustfmt-v2-source}"
lock_patch_path="${CONTINUITY_LOCK_PATCH:-${TMPDIR:-/tmp}/symthaea-continuity-cargo-lock-v2.patch}"
workflow_path="${CONTINUITY_WORKFLOW_PATH:-.github/workflows/continuity-contract-v2.yml}"
script_path="scripts/qualify-continuity-contract-v2.sh"

status="FAIL"
stage="preflight"
source_state="unverified"
diagnostic_scope="none"
diagnostic_status="not-run"
diagnostic_stage="not-run"
format_patch_state="not-produced"
format_patch_sha256="unavailable"
format_source_state="not-produced"
lock_patch_state="not-produced"
lock_patch_sha256="unavailable"
lock_repair_snapshot_sha256="unavailable"
lock_repair_classification="not-run"
lock_patch_additions="0"
lock_patch_deletions="0"

committed_cargo_lock_sha256="unavailable"
workspace_manifest_sha256="unavailable"
continuity_manifest_sha256="unavailable"
rust_toolchain_sha256="unavailable"
qualifier_script_sha256="unavailable"
workflow_sha256="unavailable"

sha256_file() {
    local path="$1"
    if [[ -f "$path" ]]; then
        sha256sum "$path" | awk '{print $1}'
    else
        printf 'unavailable'
    fi
}

assert_exact_tree_clean() {
    if [[ "$(git rev-parse HEAD)" != "$actual_sha" ]]; then
        source_state="head-changed"
        return 1
    fi
    if ! git diff --quiet --ignore-submodules --; then
        source_state="tracked-modifications-present"
        return 1
    fi
    if ! git diff --cached --quiet --ignore-submodules --; then
        source_state="staged-modifications-present"
        return 1
    fi
    local untracked
    untracked="$(git ls-files --others --exclude-standard)"
    if [[ -n "$untracked" ]]; then
        source_state="untracked-source-present"
        printf '%s\n' "$untracked" >&2
        return 1
    fi
    source_state="clean-exact-checkout"
    return 0
}

write_receipt() {
    local exit_code="$1"
    local final_status="$status"
    local terminal_stage="$stage"
    local provider="local"
    local rustc_verbose=""
    local rustc_release="unavailable"
    local rustc_commit="unavailable"
    local rustc_host="unavailable"
    local cargo_version="unavailable"
    local tmp_path="${receipt_path}.tmp.$$"

    set +e
    if [[ "$exit_code" -ne 0 ]]; then
        final_status="FAIL"
    fi
    if [[ "$final_status" == "PASS" ]]; then
        terminal_stage="none"
    fi
    if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
        provider="github-actions"
    fi
    if command -v rustc >/dev/null 2>&1; then
        rustc_verbose="$(rustc -Vv 2>/dev/null)"
        rustc_release="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "release" {print $2; exit}')"
        rustc_commit="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "commit-hash" {print $2; exit}')"
        rustc_host="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "host" {print $2; exit}')"
        [[ -n "$rustc_release" ]] || rustc_release="unavailable"
        [[ -n "$rustc_commit" ]] || rustc_commit="unavailable"
        [[ -n "$rustc_host" ]] || rustc_host="unavailable"
    fi
    if command -v cargo >/dev/null 2>&1; then
        cargo_version="$(cargo -V 2>/dev/null || printf 'unavailable')"
    fi

    mkdir -p "$(dirname "$receipt_path")" || return 1
    {
        printf 'schema\tsymthaea-continuity-contract-qualification-v2\n'
        printf 'status\t%s\n' "$final_status"
        printf 'exit_code\t%s\n' "$exit_code"
        printf 'terminal_stage\t%s\n' "$terminal_stage"
        printf 'scope\tcontinuity-software-contract-only\n'
        printf 'real_world_availability_authority\tnone\n'
        printf 'scientific_authority\tnone\n'
        printf 'execution_authority\tnone\n'
        printf 'full_repository_ci\tindependent\n'
        printf 'receipt_attestation\tnone\n'
        printf 'qualified_sha\t%s\n' "$actual_sha"
        printf 'expected_sha\t%s\n' "$expected_sha"
        printf 'committed_tree\t%s\n' "$committed_tree"
        printf 'source_state\t%s\n' "$source_state"
        printf 'diagnostic_scope\t%s\n' "$diagnostic_scope"
        printf 'diagnostic_status\t%s\n' "$diagnostic_status"
        printf 'diagnostic_terminal_stage\t%s\n' "$diagnostic_stage"
        printf 'format_patch_state\t%s\n' "$format_patch_state"
        printf 'format_patch_sha256\t%s\n' "$format_patch_sha256"
        printf 'format_source_state\t%s\n' "$format_source_state"
        printf 'lock_patch_state\t%s\n' "$lock_patch_state"
        printf 'lock_patch_sha256\t%s\n' "$lock_patch_sha256"
        printf 'lock_patch_additions\t%s\n' "$lock_patch_additions"
        printf 'lock_patch_deletions\t%s\n' "$lock_patch_deletions"
        printf 'lock_repair_classification\t%s\n' "$lock_repair_classification"
        printf 'lock_repair_snapshot_sha256\t%s\n' "$lock_repair_snapshot_sha256"
        printf 'execution_provider\t%s\n' "$provider"
        printf 'runner_label\t%s\n' "${CONTINUITY_RUNNER_LABEL:-unknown}"
        printf 'runner_os\t%s\n' "${RUNNER_OS:-unknown}"
        printf 'runner_arch\t%s\n' "${RUNNER_ARCH:-unknown}"
        printf 'runner_image_os\t%s\n' "${ImageOS:-unknown}"
        printf 'runner_image_version\t%s\n' "${ImageVersion:-unknown}"
        printf 'os_release_sha256\t%s\n' "$(sha256_file /etc/os-release)"
        printf 'kernel_release\t%s\n' "$(uname -r 2>/dev/null || printf 'unavailable')"
        printf 'rustc_release\t%s\n' "$rustc_release"
        printf 'rustc_commit_hash\t%s\n' "$rustc_commit"
        printf 'rustc_host\t%s\n' "$rustc_host"
        printf 'cargo_version\t%s\n' "$cargo_version"
        printf 'checkout_action_sha\t%s\n' "${CHECKOUT_ACTION_SHA:-not-applicable}"
        printf 'upload_artifact_action_sha\t%s\n' "${UPLOAD_ARTIFACT_ACTION_SHA:-not-applicable}"
        printf 'committed_cargo_lock_sha256\t%s\n' "$committed_cargo_lock_sha256"
        printf 'cargo_lock_sha256\t%s\n' "$committed_cargo_lock_sha256"
        printf 'workspace_manifest_sha256\t%s\n' "$workspace_manifest_sha256"
        printf 'continuity_manifest_sha256\t%s\n' "$continuity_manifest_sha256"
        printf 'rust_toolchain_sha256\t%s\n' "$rust_toolchain_sha256"
        printf 'qualifier_script_sha256\t%s\n' "$qualifier_script_sha256"
        printf 'workflow_sha256\t%s\n' "$workflow_sha256"
        printf 'github_event_name\t%s\n' "${GITHUB_EVENT_NAME:-not-applicable}"
        printf 'github_repository\t%s\n' "${GITHUB_REPOSITORY:-not-applicable}"
        printf 'github_workflow_ref\t%s\n' "${GITHUB_WORKFLOW_REF:-not-applicable}"
        printf 'github_job\t%s\n' "${GITHUB_JOB:-not-applicable}"
        printf 'github_run_id\t%s\n' "${GITHUB_RUN_ID:-not-applicable}"
        printf 'github_run_attempt\t%s\n' "${GITHUB_RUN_ATTEMPT:-not-applicable}"
    } > "$tmp_path" || {
        rm -f "$tmp_path"
        return 1
    }
    mv "$tmp_path" "$receipt_path" || {
        rm -f "$tmp_path"
        return 1
    }

    echo "continuity-contract-v2 receipt=$receipt_path status=$final_status stage=$terminal_stage"
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        {
            echo '## Continuity Contract Qualification v2'
            echo
            echo "- status: **$final_status**"
            echo "- qualified SHA: \`$actual_sha\`"
            echo "- committed tree: \`$committed_tree\`"
            echo "- terminal stage: \`$terminal_stage\`"
            echo "- exact checkout: \`$source_state\`"
            echo "- diagnostic: \`$diagnostic_status\` / \`$diagnostic_stage\` (\`$diagnostic_scope\`)"
            echo "- lock repair classification: \`$lock_repair_classification\`"
            echo "- committed Cargo.lock SHA-256: \`$committed_cargo_lock_sha256\`"
            echo "- derived lock snapshot SHA-256: \`$lock_repair_snapshot_sha256\`"
            echo '- diagnostics run in detached worktrees and cannot change exact-head PASS/FAIL status'
            echo '- full repository CI: independent'
            echo '- real-world availability/scientific/execution authority: none'
        } >> "$GITHUB_STEP_SUMMARY" || true
    fi
    return 0
}

finish() {
    local exit_code=$?
    trap - EXIT

    if ! assert_exact_tree_clean; then
        stage="postflight_source_immutability"
        status="FAIL"
        exit_code=1
        echo 'error: exact checkout changed during qualification' >&2
    fi
    if [[ "$exit_code" -eq 0 && "$status" != "PASS" ]]; then
        echo "error: verifier exited without terminal PASS (stage=$stage)" >&2
        exit_code=1
    fi
    if ! write_receipt "$exit_code"; then
        echo "error: qualification receipt could not be persisted to $receipt_path" >&2
        [[ "$exit_code" -ne 0 ]] || exit_code=1
    fi
    exit "$exit_code"
}
trap finish EXIT

make_probe_worktree() {
    local root
    root="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/continuity-probe-v2.XXXXXX")"
    local worktree="$root/repo"
    git worktree add --quiet --detach "$worktree" "$actual_sha"
    printf '%s\n' "$worktree"
}

remove_probe_worktree() {
    local worktree="$1"
    local root
    root="$(dirname "$worktree")"
    git worktree remove --force "$worktree" >/dev/null 2>&1 || true
    rm -rf "$root"
}

classify_lock_repair() {
    local before="$1"
    local after="$2"
    python3 - "$before" "$after" <<'PY'
import collections
import json
import sys
import tomllib

with open(sys.argv[1], 'rb') as fh:
    before = tomllib.load(fh)
with open(sys.argv[2], 'rb') as fh:
    after = tomllib.load(fh)

def canon(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))

before_packages = collections.Counter(canon(p) for p in before.get('package', []))
after_packages = collections.Counter(canon(p) for p in after.get('package', []))
added = list((after_packages - before_packages).elements())
removed = list((before_packages - after_packages).elements())

classification = 'broad-workspace-reresolution'
if not removed and len(added) == 1:
    package = json.loads(added[0])
    expected = {
        'name': 'symthaea-continuity',
        'version': '0.1.0',
        'dependencies': ['blake3', 'serde', 'thiserror 1.0.69'],
    }
    if package == expected:
        classification = 'minimal-workspace-member-admission'
elif not removed and not added:
    classification = 'no-package-set-change'

print(classification)
PY
}

diagnose_lock_resolution() {
    diagnostic_scope="detached-worktree-lock-repair-non-qualifying"
    diagnostic_status="FAIL"
    diagnostic_stage="offline_metadata_repair"

    local worktree before_lock resolve_rc additions deletions
    worktree="$(make_probe_worktree)"
    before_lock="$(dirname "$worktree")/Cargo.lock.committed"
    cp Cargo.lock "$before_lock"

    set +e
    (
        cd "$worktree"
        cargo metadata --offline --format-version 1 >/dev/null
    )
    resolve_rc=$?
    set -e

    if ! git -C "$worktree" diff --quiet -- Cargo.lock; then
        git -C "$worktree" diff --binary -- Cargo.lock > "$lock_patch_path"
        lock_patch_state="generated-in-detached-worktree"
        lock_patch_sha256="$(sha256_file "$lock_patch_path")"
        lock_repair_snapshot_sha256="$(sha256_file "$worktree/Cargo.lock")"
        read -r additions deletions < <(git -C "$worktree" diff --numstat -- Cargo.lock)
        lock_patch_additions="${additions:--}"
        lock_patch_deletions="${deletions:--}"
        lock_repair_classification="$(classify_lock_repair "$before_lock" "$worktree/Cargo.lock")"
        echo "continuity-contract-v2 cargo_lock_patch_sha256=$lock_patch_sha256 classification=$lock_repair_classification"
        git -C "$worktree" diff --stat -- Cargo.lock >&2 || true
    else
        lock_patch_state="no-lock-diff"
        lock_repair_classification="no-lock-diff"
    fi

    if [[ "$resolve_rc" -eq 0 && "$lock_repair_classification" == "minimal-workspace-member-admission" ]]; then
        diagnostic_stage="check_all_targets_with_minimal_repaired_lock"
        set +e
        (
            cd "$worktree"
            cargo check --locked -p symthaea-continuity --all-targets
        )
        resolve_rc=$?
        if [[ "$resolve_rc" -eq 0 ]]; then
            diagnostic_stage="clippy_all_targets_with_minimal_repaired_lock"
            (
                cd "$worktree"
                cargo clippy --locked -p symthaea-continuity --all-targets -- -D warnings
            )
            resolve_rc=$?
        fi
        if [[ "$resolve_rc" -eq 0 ]]; then
            diagnostic_stage="unit_and_integration_tests_with_minimal_repaired_lock"
            (cd "$worktree" && cargo test --locked -p symthaea-continuity)
            resolve_rc=$?
        fi
        if [[ "$resolve_rc" -eq 0 ]]; then
            diagnostic_stage="doc_tests_with_minimal_repaired_lock"
            (cd "$worktree" && cargo test --locked -p symthaea-continuity --doc)
            resolve_rc=$?
        fi
        set -e
    elif [[ "$resolve_rc" -eq 0 ]]; then
        diagnostic_stage="lock_repair_not_minimal"
        resolve_rc=1
    fi

    if [[ "$resolve_rc" -eq 0 ]]; then
        diagnostic_status="PASS"
        diagnostic_stage="complete"
    fi

    remove_probe_worktree "$worktree"
    return 1
}

diagnose_format() {
    diagnostic_scope="detached-worktree-rustfmt-repair-non-qualifying"
    diagnostic_status="FAIL"
    diagnostic_stage="format_repair"

    local worktree probe_rc
    worktree="$(make_probe_worktree)"
    set +e
    (cd "$worktree" && cargo fmt -p symthaea-continuity)
    probe_rc=$?
    set -e

    if [[ "$probe_rc" -eq 0 ]] && ! git -C "$worktree" diff --quiet -- crates/core/symthaea-continuity; then
        git -C "$worktree" diff --binary -- crates/core/symthaea-continuity > "$format_patch_path"
        rm -rf "$format_source_dir"
        mkdir -p "$format_source_dir"
        cp -a "$worktree/crates/core/symthaea-continuity/src/." "$format_source_dir/"
        format_patch_state="generated-in-detached-worktree"
        format_patch_sha256="$(sha256_file "$format_patch_path")"
        format_source_state="generated-in-detached-worktree"
        diagnostic_status="PASS"
        diagnostic_stage="repair_artifact_ready"
        echo "continuity-contract-v2 rustfmt_patch_sha256=$format_patch_sha256"
    elif [[ "$probe_rc" -eq 0 ]]; then
        diagnostic_stage="format_failed_without_repair_diff"
    fi

    remove_probe_worktree "$worktree"
    return 1
}

stage="preflight_exact_head"
if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "error: qualified head mismatch: expected=$expected_sha actual=$actual_sha" >&2
    exit 1
fi

stage="preflight_crate_present"
test -f crates/core/symthaea-continuity/Cargo.toml || {
    echo 'error: symthaea-continuity manifest is absent from exact qualified head' >&2
    exit 1
}

stage="preflight_clean_tree"
assert_exact_tree_clean || {
    echo 'error: exact-head qualification requires a clean source tree' >&2
    exit 1
}

rm -f "$format_patch_path" "$lock_patch_path"
rm -rf "$format_source_dir"

# Snapshot immutable committed inputs before any diagnostic worktree exists.
committed_cargo_lock_sha256="$(sha256_file Cargo.lock)"
workspace_manifest_sha256="$(sha256_file Cargo.toml)"
continuity_manifest_sha256="$(sha256_file crates/core/symthaea-continuity/Cargo.toml)"
rust_toolchain_sha256="$(sha256_file rust-toolchain.toml)"
qualifier_script_sha256="$(sha256_file "$script_path")"
workflow_sha256="$(sha256_file "$workflow_path")"

echo "continuity-contract-v2 qualified_sha=$actual_sha"
echo "continuity-contract-v2 committed_tree=$committed_tree"
rustc -Vv
cargo -V

stage="cargo_metadata_locked"
if ! cargo metadata --locked --format-version 1 >/dev/null; then
    diagnose_lock_resolution
    exit 1
fi

stage="format"
if ! cargo fmt -p symthaea-continuity -- --check; then
    diagnose_format
    exit 1
fi

stage="check_all_targets"
cargo check --locked -p symthaea-continuity --all-targets

stage="clippy_all_targets"
cargo clippy --locked -p symthaea-continuity --all-targets -- -D warnings

stage="unit_and_integration_tests"
cargo test --locked -p symthaea-continuity

stage="doc_tests"
cargo test --locked -p symthaea-continuity --doc

stage="postflight_source_immutability"
assert_exact_tree_clean || {
    echo 'error: exact checkout changed during qualification' >&2
    exit 1
}

status="PASS"
stage="complete"
