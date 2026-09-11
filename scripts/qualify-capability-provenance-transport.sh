#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Focused exact-head qualification for public capability-provenance transport.
# This is a software/source execution witness only. It grants no observation,
# availability, scientific, policy, recovery, ownership, or execution authority.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

actual_sha="$(git rev-parse HEAD)"
head_tree="$(git rev-parse 'HEAD^{tree}')"
expected_sha="${QUALIFIED_SHA:-$actual_sha}"
receipt_path="${CAPABILITY_TRANSPORT_RECEIPT:-${TMPDIR:-/tmp}/capability-provenance-transport-qualification-v1.tsv}"
status="FAIL"
stage="preflight"
source_state="unverified"
transport_test_count="not-run"
package_test_count="not-run"

sha256_file() {
    local path="$1"
    if [[ -f "$path" ]]; then sha256sum "$path" | awk '{print $1}'; else printf 'unavailable'; fi
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
    [[ "$exit_code" -eq 0 ]] || final_status="FAIL"
    [[ "$final_status" != "PASS" ]] || terminal_stage="none"
    [[ "${GITHUB_ACTIONS:-}" != "true" ]] || provider="github-actions"
    if command -v rustc >/dev/null 2>&1; then
        rustc_verbose="$(rustc -Vv 2>/dev/null)"
        rustc_release="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "release" {print $2; exit}')"
        rustc_commit="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "commit-hash" {print $2; exit}')"
        rustc_host="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "host" {print $2; exit}')"
    fi
    command -v cargo >/dev/null 2>&1 && cargo_version="$(cargo -V 2>/dev/null || printf 'unavailable')"

    mkdir -p "$(dirname "$receipt_path")" || return 1
    {
        printf 'schema\tsymthaea-capability-provenance-transport-qualification-v1\n'
        printf 'status\t%s\n' "$final_status"
        printf 'exit_code\t%s\n' "$exit_code"
        printf 'terminal_stage\t%s\n' "$terminal_stage"
        printf 'scope\tcapability-provenance-public-transport-software-contract-only\n'
        printf 'qualified_sha\t%s\n' "$actual_sha"
        printf 'expected_sha\t%s\n' "$expected_sha"
        printf 'committed_tree\t%s\n' "$head_tree"
        printf 'source_state\t%s\n' "$source_state"
        printf 'transport_test_count\t%s\n' "$transport_test_count"
        printf 'package_test_count\t%s\n' "$package_test_count"
        printf 'execution_provider\t%s\n' "$provider"
        printf 'runner_label\t%s\n' "${CAPABILITY_TRANSPORT_RUNNER_LABEL:-unknown}"
        printf 'runner_os\t%s\n' "${RUNNER_OS:-unknown}"
        printf 'runner_arch\t%s\n' "${RUNNER_ARCH:-unknown}"
        printf 'runner_image_os\t%s\n' "${ImageOS:-unknown}"
        printf 'runner_image_version\t%s\n' "${ImageVersion:-unknown}"
        printf 'kernel_release\t%s\n' "$(uname -r 2>/dev/null || printf 'unavailable')"
        printf 'os_release_sha256\t%s\n' "$(sha256_file /etc/os-release)"
        printf 'rustc_release\t%s\n' "$rustc_release"
        printf 'rustc_commit_hash\t%s\n' "$rustc_commit"
        printf 'rustc_host\t%s\n' "$rustc_host"
        printf 'cargo_version\t%s\n' "$cargo_version"
        printf 'checkout_action_sha\t%s\n' "${CHECKOUT_ACTION_SHA:-not-applicable}"
        printf 'upload_artifact_action_sha\t%s\n' "${UPLOAD_ARTIFACT_ACTION_SHA:-not-applicable}"
        printf 'cargo_lock_sha256\t%s\n' "$(sha256_file Cargo.lock)"
        printf 'continuity_manifest_sha256\t%s\n' "$(sha256_file crates/core/symthaea-continuity/Cargo.toml)"
        printf 'rust_toolchain_sha256\t%s\n' "$(sha256_file rust-toolchain.toml)"
        printf 'provenance_source_sha256\t%s\n' "$(sha256_file crates/core/symthaea-continuity/src/capability_analysis_provenance.rs)"
        printf 'transport_tests_sha256\t%s\n' "$(sha256_file crates/core/symthaea-continuity/tests/capability_analysis_provenance_transport.rs)"
        printf 'qualifier_script_sha256\t%s\n' "$(sha256_file scripts/qualify-capability-provenance-transport.sh)"
        printf 'workflow_sha256\t%s\n' "$(sha256_file .github/workflows/capability-provenance-transport.yml)"
        printf 'observation_authority\tnone\n'
        printf 'availability_authority\tnone\n'
        printf 'scientific_authority\tnone\n'
        printf 'policy_authority\tnone\n'
        printf 'recovery_authority\tnone\n'
        printf 'ownership_authority\tnone\n'
        printf 'execution_authority\tnone\n'
        printf 'receipt_attestation\tnone\n'
        printf 'github_run_id\t%s\n' "${GITHUB_RUN_ID:-not-applicable}"
        printf 'github_run_attempt\t%s\n' "${GITHUB_RUN_ATTEMPT:-not-applicable}"
    } >"$tmp_path" || { rm -f "$tmp_path"; return 1; }
    mv "$tmp_path" "$receipt_path" || { rm -f "$tmp_path"; return 1; }
    echo "capability-provenance-transport receipt=$receipt_path status=$final_status stage=$terminal_stage"
    return 0
}

finish() {
    local exit_code=$?
    trap - EXIT
    if [[ "$exit_code" -eq 0 && "$status" != "PASS" ]]; then
        echo "error: verifier exited without terminal PASS (stage=$stage)" >&2
        exit_code=1
    fi
    write_receipt "$exit_code" || { echo "error: receipt persistence failed" >&2; [[ "$exit_code" -ne 0 ]] || exit_code=1; }
    exit "$exit_code"
}
trap finish EXIT

stage="preflight_exact_head"
[[ "$actual_sha" == "$expected_sha" ]] || { echo "error: expected=$expected_sha actual=$actual_sha" >&2; exit 1; }

stage="preflight_files"
test -f crates/core/symthaea-continuity/tests/capability_analysis_provenance_transport.rs
test -f crates/core/symthaea-continuity/src/capability_analysis_provenance.rs

stage="preflight_clean_tree"
git diff --quiet --ignore-submodules -- || { source_state="tracked-modifications-present"; exit 1; }
git diff --cached --quiet --ignore-submodules -- || { source_state="staged-modifications-present"; exit 1; }
untracked="$(git ls-files --others --exclude-standard)"
[[ -z "$untracked" ]] || { source_state="untracked-source-present"; printf '%s\n' "$untracked" >&2; exit 1; }
source_state="clean-exact-checkout"

stage="dependency_boundary"
python3 - <<'PY'
import pathlib, tomllib
manifest = tomllib.loads(pathlib.Path('crates/core/symthaea-continuity/Cargo.toml').read_text())
if 'serde_json' in manifest.get('dependencies', {}):
    raise SystemExit('serde_json escaped into runtime dependencies')
if 'serde_json' not in manifest.get('dev-dependencies', {}):
    raise SystemExit('serde_json test dependency absent')
PY

stage="cargo_metadata"
cargo metadata --locked --no-deps --format-version 1 >/dev/null

stage="format_transport"
rustfmt --edition 2024 --check crates/core/symthaea-continuity/tests/capability_analysis_provenance_transport.rs

stage="check_all_targets"
cargo check --locked -p symthaea-continuity --all-targets

stage="transport_regressions"
transport_output="$(mktemp)"
cargo test --locked -p symthaea-continuity --test capability_analysis_provenance_transport -- --nocapture 2>&1 | tee "$transport_output"
transport_test_count="$(sed -nE 's/^test result: ok\. ([0-9]+) passed;.*/\1/p' "$transport_output" | tail -n1)"
rm -f "$transport_output"
[[ "$transport_test_count" == "13" ]] || { echo "error: expected exactly 13 transport tests, got $transport_test_count" >&2; exit 1; }

stage="package_tests"
package_output="$(mktemp)"
cargo test --locked -p symthaea-continuity 2>&1 | tee "$package_output"
package_test_count="$(sed -nE 's/^test result: ok\. ([0-9]+) passed;.*/\1/p' "$package_output" | awk '{s+=$1} END {print s+0}')"
rm -f "$package_output"
[[ "$package_test_count" -gt 0 ]] || { echo 'error: package test count not observed' >&2; exit 1; }

stage="doc_tests"
cargo test --locked -p symthaea-continuity --doc

stage="postflight_source_immutability"
[[ "$(git rev-parse HEAD)" == "$actual_sha" ]] || { source_state="head-changed-during-tests"; exit 1; }
git diff --quiet --ignore-submodules -- || { source_state="tracked-source-mutated-during-tests"; exit 1; }
git diff --cached --quiet --ignore-submodules -- || { source_state="staged-source-mutated-during-tests"; exit 1; }
post_untracked="$(git ls-files --others --exclude-standard)"
[[ -z "$post_untracked" ]] || { source_state="untracked-source-created-during-tests"; printf '%s\n' "$post_untracked" >&2; exit 1; }
source_state="clean-exact-checkout-postflight"

stage="complete"
status="PASS"
