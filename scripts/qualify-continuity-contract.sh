#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Focused exact-head software-contract qualification for symthaea-continuity.
#
# This lane proves only that one exact committed source tree satisfies the
# focused continuity crate's formatting/compiler/lint/test contracts. It does
# not establish real-world availability, empirical/scientific evidence, policy
# approval, or execution authority.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

actual_sha="$(git rev-parse HEAD)"
head_tree="$(git rev-parse 'HEAD^{tree}')"
expected_sha="${QUALIFIED_SHA:-$actual_sha}"
receipt_path="${CONTINUITY_CONTRACT_RECEIPT:-${TMPDIR:-/tmp}/symthaea-continuity-contract-qualification-v1.tsv}"
format_patch_path="${CONTINUITY_FORMAT_PATCH:-${TMPDIR:-/tmp}/symthaea-continuity-rustfmt.patch}"
status="FAIL"
stage="preflight"
source_state="unverified"
format_patch_state="not-produced"

sha256_file() {
    local path="$1"
    if [[ -f "$path" ]]; then
        sha256sum "$path" | awk '{print $1}'
    else
        printf 'unavailable'
    fi
}

write_receipt() {
    local exit_code="$1"
    local final_status="$status"
    local failure_stage="$stage"
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
        failure_stage="none"
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
        printf 'schema\tsymthaea-continuity-contract-qualification-v1\n'
        printf 'status\t%s\n' "$final_status"
        printf 'exit_code\t%s\n' "$exit_code"
        printf 'terminal_stage\t%s\n' "$failure_stage"
        printf 'scope\tcontinuity-software-contract-only\n'
        printf 'real_world_availability_authority\tnone\n'
        printf 'scientific_authority\tnone\n'
        printf 'execution_authority\tnone\n'
        printf 'full_repository_ci\tindependent\n'
        printf 'receipt_attestation\tnone\n'
        printf 'qualified_sha\t%s\n' "$actual_sha"
        printf 'expected_sha\t%s\n' "$expected_sha"
        printf 'committed_tree\t%s\n' "$head_tree"
        printf 'source_state\t%s\n' "$source_state"
        printf 'format_patch_state\t%s\n' "$format_patch_state"
        printf 'format_patch_sha256\t%s\n' "$(sha256_file "$format_patch_path")"
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
        printf 'cargo_lock_sha256\t%s\n' "$(sha256_file Cargo.lock)"
        printf 'workspace_manifest_sha256\t%s\n' "$(sha256_file Cargo.toml)"
        printf 'continuity_manifest_sha256\t%s\n' "$(sha256_file crates/core/symthaea-continuity/Cargo.toml)"
        printf 'rust_toolchain_sha256\t%s\n' "$(sha256_file rust-toolchain.toml)"
        printf 'qualifier_script_sha256\t%s\n' "$(sha256_file scripts/qualify-continuity-contract.sh)"
        printf 'workflow_sha256\t%s\n' "$(sha256_file .github/workflows/continuity-contract.yml)"
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

    echo "continuity-contract receipt=$receipt_path status=$final_status stage=$failure_stage"

    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        {
            echo '## Continuity Contract Qualification'
            echo
            echo "- status: **$final_status**"
            echo "- qualified SHA: \`$actual_sha\`"
            echo "- committed tree: \`$head_tree\`"
            echo "- terminal stage: \`$failure_stage\`"
            echo "- source state: \`$source_state\`"
            echo "- format repair artifact: \`$format_patch_state\`"
            echo '- scope: continuity software contracts only'
            echo '- full repository CI: independent'
            echo '- real-world availability/scientific/execution authority: none'
            echo '- receipt attestation: none (workflow run/artifact is external witness)'
        } >> "$GITHUB_STEP_SUMMARY" || true
    fi
    return 0
}

finish() {
    local exit_code=$?
    trap - EXIT

    if [[ "$exit_code" -eq 0 && "$status" != "PASS" ]]; then
        echo "error: verifier exited without terminal PASS (stage=$stage)" >&2
        exit_code=1
    fi

    if ! write_receipt "$exit_code"; then
        echo "error: qualification receipt could not be persisted to $receipt_path" >&2
        if [[ "$exit_code" -eq 0 ]]; then
            exit_code=1
        fi
    fi
    exit "$exit_code"
}
trap finish EXIT

stage="preflight_exact_head"
if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "error: qualified head mismatch: expected=$expected_sha actual=$actual_sha" >&2
    exit 1
fi

stage="preflight_crate_present"
if [[ ! -f crates/core/symthaea-continuity/Cargo.toml ]]; then
    echo 'error: symthaea-continuity manifest is absent from exact qualified head' >&2
    exit 1
fi

stage="preflight_clean_tree"
if ! git diff --quiet --ignore-submodules --; then
    source_state="tracked-modifications-present"
    echo 'error: tracked working-tree changes present' >&2
    exit 1
fi
if ! git diff --cached --quiet --ignore-submodules --; then
    source_state="staged-modifications-present"
    echo 'error: staged changes present' >&2
    exit 1
fi
untracked="$(git ls-files --others --exclude-standard)"
if [[ -n "$untracked" ]]; then
    source_state="untracked-source-present"
    echo 'error: untracked files present; exact-head qualification requires a clean source tree' >&2
    printf '%s\n' "$untracked" >&2
    exit 1
fi
source_state="clean-exact-checkout"

rm -f "$format_patch_path"

echo "continuity-contract qualified_sha=$actual_sha"
echo "continuity-contract committed_tree=$head_tree"
rustc -Vv
cargo -V

stage="cargo_metadata"
cargo metadata --locked --no-deps --format-version 1 >/dev/null

stage="format"
if ! cargo fmt -p symthaea-continuity -- --check; then
    mkdir -p "$(dirname "$format_patch_path")"
    cargo fmt -p symthaea-continuity
    if git diff --quiet -- crates/core/symthaea-continuity; then
        source_state="format-check-failed-without-repair-diff"
        echo 'error: rustfmt check failed but formatter produced no continuity diff' >&2
        exit 1
    fi
    git diff --binary -- crates/core/symthaea-continuity > "$format_patch_path"
    format_patch_state="generated-from-exact-head"
    source_state="formatter-derived-diff-from-exact-head"
    echo "continuity-contract rustfmt_patch=$format_patch_path"
    echo "continuity-contract rustfmt_patch_sha256=$(sha256_file "$format_patch_path")"
    git diff --stat -- crates/core/symthaea-continuity >&2 || true
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
if [[ "$(git rev-parse HEAD)" != "$actual_sha" ]]; then
    source_state="head-changed-during-tests"
    echo 'error: HEAD changed during qualification' >&2
    exit 1
fi
if ! git diff --quiet --ignore-submodules --; then
    source_state="tracked-source-mutated-during-tests"
    echo 'error: tracked source changed during qualification' >&2
    git diff --stat >&2 || true
    exit 1
fi
if ! git diff --cached --quiet --ignore-submodules --; then
    source_state="staged-source-mutated-during-tests"
    echo 'error: staged source appeared during qualification' >&2
    exit 1
fi
post_untracked="$(git ls-files --others --exclude-standard)"
if [[ -n "$post_untracked" ]]; then
    source_state="untracked-source-created-during-tests"
    echo 'error: untracked source appeared during qualification' >&2
    printf '%s\n' "$post_untracked" >&2
    exit 1
fi
source_state="clean-exact-checkout-postflight"

status="PASS"
stage="complete"
