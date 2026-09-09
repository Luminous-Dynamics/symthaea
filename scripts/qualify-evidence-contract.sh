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

actual_sha="$(git rev-parse HEAD)"
head_tree="$(git rev-parse 'HEAD^{tree}')"
expected_sha="${QUALIFIED_SHA:-$actual_sha}"
receipt_path="${EVIDENCE_CONTRACT_RECEIPT:-${TMPDIR:-/tmp}/symthaea-evidence-contract-qualification-v1.tsv}"
status="FAIL"
stage="preflight"
source_state="unverified"
authority_targets=()

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
    local target_list="none"

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
    if [[ "${#authority_targets[@]}" -gt 0 ]]; then
        target_list="$(IFS=,; printf '%s' "${authority_targets[*]}")"
    fi

    mkdir -p "$(dirname "$receipt_path")"
    {
        printf 'schema\tsymthaea-evidence-contract-qualification-v1\n'
        printf 'status\t%s\n' "$final_status"
        printf 'exit_code\t%s\n' "$exit_code"
        printf 'terminal_stage\t%s\n' "$failure_stage"
        printf 'scope\tsoftware-contract-only\n'
        printf 'scientific_authority\tnone\n'
        printf 'full_repository_ci\tindependent\n'
        printf 'baseline_generation\tforbidden\n'
        printf 'environment_authority\tobserved-not-capsule-qualified\n'
        printf 'receipt_attestation\tnone\n'
        printf 'provider_metadata_basis\tambient-runtime\n'
        printf 'qualified_sha\t%s\n' "$actual_sha"
        printf 'expected_sha\t%s\n' "$expected_sha"
        printf 'committed_tree\t%s\n' "$head_tree"
        printf 'source_state\t%s\n' "$source_state"
        printf 'execution_provider\t%s\n' "$provider"
        printf 'runner_label\t%s\n' "${EVIDENCE_RUNNER_LABEL:-unknown}"
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
        printf 'rust_toolchain_sha256\t%s\n' "$(sha256_file rust-toolchain.toml)"
        printf 'qualifier_script_sha256\t%s\n' "$(sha256_file scripts/qualify-evidence-contract.sh)"
        printf 'workflow_sha256\t%s\n' "$(sha256_file .github/workflows/evidence-contract.yml)"
        printf 'authority_integration_targets\t%s\n' "$target_list"
        printf 'github_event_name\t%s\n' "${GITHUB_EVENT_NAME:-not-applicable}"
        printf 'github_repository\t%s\n' "${GITHUB_REPOSITORY:-not-applicable}"
        printf 'github_workflow_ref\t%s\n' "${GITHUB_WORKFLOW_REF:-not-applicable}"
        printf 'github_job\t%s\n' "${GITHUB_JOB:-not-applicable}"
        printf 'github_run_id\t%s\n' "${GITHUB_RUN_ID:-not-applicable}"
        printf 'github_run_attempt\t%s\n' "${GITHUB_RUN_ATTEMPT:-not-applicable}"
    } > "$receipt_path"

    echo "evidence-contract receipt=$receipt_path status=$final_status stage=$failure_stage"

    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        {
            echo '## Evidence Contract Qualification'
            echo
            echo "- status: **$final_status**"
            echo "- qualified SHA: \`$actual_sha\`"
            echo "- committed tree: \`$head_tree\`"
            echo "- terminal stage: \`$failure_stage\`"
            echo "- source state: \`$source_state\`"
            echo '- scope: software-contract qualification only'
            echo '- full repository CI: independent'
            echo '- empirical/scientific claim authority: none'
            echo '- receipt attestation: none (run/artifact context is the external witness)'
        } >> "$GITHUB_STEP_SUMMARY"
    fi
}

finish() {
    local exit_code=$?
    trap - EXIT
    write_receipt "$exit_code"
    exit "$exit_code"
}
trap finish EXIT

stage="preflight_baseline_mode"
if [[ -n "${UPDATE_SNAPSHOT+x}" ]]; then
    echo "error: UPDATE_SNAPSHOT is present; qualification must remain read-only" >&2
    exit 1
fi

stage="preflight_exact_head"
if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "error: qualified head mismatch: expected=$expected_sha actual=$actual_sha" >&2
    exit 1
fi

# Exact-head means the commit must identify the source under qualification.
# Ignored build outputs are fine; tracked modifications, staged changes, and
# untracked source are not. A future execution-capsule path may bind a dirty or
# transformed tree explicitly, but this narrow verifier chooses clean-only.
stage="preflight_clean_tree"
if ! git diff --quiet --ignore-submodules --; then
    source_state="tracked-modifications-present"
    echo "error: tracked working-tree changes present; commit SHA does not identify executed source" >&2
    exit 1
fi
if ! git diff --cached --quiet --ignore-submodules --; then
    source_state="staged-modifications-present"
    echo "error: staged changes present; commit SHA does not identify executed source" >&2
    exit 1
fi
untracked="$(git ls-files --others --exclude-standard)"
if [[ -n "$untracked" ]]; then
    source_state="untracked-source-present"
    echo "error: untracked files present; exact-head qualification requires a clean source tree" >&2
    printf '%s\n' "$untracked" >&2
    exit 1
fi
source_state="clean-exact-checkout"

echo "evidence-contract qualified_sha=$actual_sha"
echo "evidence-contract committed_tree=$head_tree"
echo "evidence-contract scope=software-contract-only"
rustc -Vv
cargo -V

stage="cargo_metadata"
cargo metadata --locked --no-deps --format-version 1 >/dev/null

# The evidence plane is a small dedicated crate, so its whole test surface is
# part of this focused contract. Psych-bench is intentionally different: a
# package-wide `cargo test` also executes historical full-battery performance
# regression policy, which is a separate theorem and must not leak into this
# authority-contract lane.
stage="evidence_plane_contracts"
cargo test --locked -p symthaea-evidence-plane

stage="butlin_default_lib_contracts"
cargo test --locked -p symthaea-psych-bench --lib benchmarks::butlin

stage="butlin_structural_contract"
cargo test --locked -p symthaea-psych-bench --test butlin_regression

# Authority diagnostics use a naming convention so a PR that adds a focused
# fail-closed regression is automatically exercised without enumerating every
# future authority issue in this script. Only this narrow family is discovered;
# broad/full-battery regression tests remain outside this theorem.
mapfile -t authority_targets < <(
    find crates/domains/symthaea-psych-bench/tests \
        -maxdepth 1 -type f -name 'butlin_*authority*_regression.rs' -printf '%f\n' \
        | sed 's/\.rs$//' \
        | LC_ALL=C sort
)
for target in "${authority_targets[@]}"; do
    stage="authority_integration_${target}"
    cargo test --locked -p symthaea-psych-bench --test "$target"
done

stage="butlin_backend_contracts"
cargo test --locked -p symthaea-psych-bench --features symthaea-backend --lib -- butlin

status="PASS"
stage="complete"
