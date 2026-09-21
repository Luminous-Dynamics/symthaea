#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Engineering qualification for the frozen Melothaea lockbox execution harness.
# This qualifier MUST NOT execute or unblind a valid lockbox subject. Its two
# runtime tests intentionally fail before protocol construction: missing explicit
# execution acknowledgement, then invalid qualification receipt.

set -euo pipefail

QUALIFIER_ID="melothaea-lockbox-runner-qualification-v1"
SUBJECT_SHA="95e5bd949033215d09b4b9edf4d0e490ba823a3c"
BASE_SHA="d888341323b6ee463007cf90d8032153039ec099"
PREREQUISITE_STREAM_SUBJECT_SHA="d888341323b6ee463007cf90d8032153039ec099"
EXPECTED_RUST="1.96.0"
EXPECTED_FILE="crates/domains/symthaea-muse/examples/prog_suite_contextual_harmony_lockbox_runner.rs"
EXAMPLE="prog_suite_contextual_harmony_lockbox_runner"

root="$(git rev-parse --show-toplevel)"
cd "$root"
qualifier_sha="$(git rev-parse HEAD)"
qualifier_tree="$(git rev-parse 'HEAD^{tree}')"
script_path=".github/qualification/qualify-melothaea-lockbox-runner-v1.sh"
receipt_path="${MEL_RUNNER_QUAL_RECEIPT:-${TMPDIR:-/tmp}/melothaea-lockbox-runner-qualification-v1.tsv}"
worktree=""
status="FAIL"
stage="preflight"
source_state="unverified"
subject_tree="unavailable"
subject_parent="unavailable"
rustc_release="unavailable"
rustc_commit="unavailable"
rustc_host="unavailable"
cargo_version="unavailable"
runner_source_sha256="unavailable"

exact_subject_gate="not-run"
surface_gate="not-run"
toolchain_gate="not-run"
metadata_gate="not-run"
fmt_gate="not-run"
check_runner_gate="not-run"
clippy_runner_gate="not-run"
missing_ack_gate="not-run"
invalid_receipt_gate="not-run"
postflight_gate="not-run"

sha256_file() {
    local path="$1"
    if [[ -f "$path" ]]; then
        sha256sum "$path" | awk '{print $1}'
    else
        printf 'unavailable'
    fi
}

write_receipt() {
    local rc="$1"
    local final_status="$status"
    local terminal_stage="$stage"
    local provider="local"
    local tmp="${receipt_path}.tmp.$$"
    local script_sha

    [[ "$rc" -eq 0 ]] || final_status="FAIL"
    [[ "$final_status" != "PASS" ]] || terminal_stage="none"
    [[ "${GITHUB_ACTIONS:-}" != "true" ]] || provider="github-actions"
    script_sha="$(sha256_file "$root/$script_path")"

    mkdir -p "$(dirname "$receipt_path")"
    {
        printf 'schema\tmelothaea-lockbox-runner-qualification-v1\n'
        printf 'qualifier_id\t%s\n' "$QUALIFIER_ID"
        printf 'status\t%s\n' "$final_status"
        printf 'exit_code\t%s\n' "$rc"
        printf 'terminal_stage\t%s\n' "$terminal_stage"
        printf 'authority_scope\tengineering-execution-harness-contract-only\n'
        printf 'scientific_lockbox_execution\tnot-performed\n'
        printf 'lockbox_subjects_observed\t0\n'
        printf 'human_perceptual_authority\tnone\n'
        printf 'artistic_quality_authority\tnone\n'
        printf 'product_authority\tnone\n'
        printf 'qualification_provider\t%s\n' "$provider"
        printf 'receipt_attestation\tnone\n'
        printf 'qualifier_checkout_sha\t%s\n' "$qualifier_sha"
        printf 'qualifier_checkout_tree\t%s\n' "$qualifier_tree"
        printf 'qualifier_script_sha256\t%s\n' "$script_sha"
        printf 'subject_sha\t%s\n' "$SUBJECT_SHA"
        printf 'subject_tree\t%s\n' "$subject_tree"
        printf 'subject_parent\t%s\n' "$subject_parent"
        printf 'base_sha\t%s\n' "$BASE_SHA"
        printf 'prerequisite_stream_subject_sha\t%s\n' "$PREREQUISITE_STREAM_SUBJECT_SHA"
        printf 'subject_changed_file_count\t1\n'
        printf 'subject_changed_file\t%s\n' "$EXPECTED_FILE"
        printf 'runner_source_sha256\t%s\n' "$runner_source_sha256"
        printf 'source_state\t%s\n' "$source_state"
        printf 'expected_rust_release\t%s\n' "$EXPECTED_RUST"
        printf 'rustc_release\t%s\n' "$rustc_release"
        printf 'rustc_commit_hash\t%s\n' "$rustc_commit"
        printf 'rustc_host\t%s\n' "$rustc_host"
        printf 'cargo_version\t%s\n' "$cargo_version"
        printf 'exact_subject_gate\t%s\n' "$exact_subject_gate"
        printf 'surface_gate\t%s\n' "$surface_gate"
        printf 'toolchain_gate\t%s\n' "$toolchain_gate"
        printf 'metadata_gate\t%s\n' "$metadata_gate"
        printf 'fmt_gate\t%s\n' "$fmt_gate"
        printf 'check_runner_gate\t%s\n' "$check_runner_gate"
        printf 'clippy_runner_gate\t%s\n' "$clippy_runner_gate"
        printf 'missing_ack_gate\t%s\n' "$missing_ack_gate"
        printf 'invalid_receipt_gate\t%s\n' "$invalid_receipt_gate"
        printf 'postflight_gate\t%s\n' "$postflight_gate"
        printf 'github_repository\t%s\n' "${GITHUB_REPOSITORY:-not-applicable}"
        printf 'github_run_id\t%s\n' "${GITHUB_RUN_ID:-not-applicable}"
        printf 'github_run_attempt\t%s\n' "${GITHUB_RUN_ATTEMPT:-not-applicable}"
    } > "$tmp"
    mv "$tmp" "$receipt_path"
}

cleanup() {
    cd "$root" 2>/dev/null || true
    if [[ -n "$worktree" && -d "$worktree" ]]; then
        git worktree remove --force "$worktree" >/dev/null 2>&1 || true
    fi
}

finish() {
    local rc=$?
    trap - EXIT
    if [[ "$rc" -eq 0 && "$status" != "PASS" ]]; then
        rc=1
    fi
    write_receipt "$rc" || rc=1
    cleanup
    exit "$rc"
}
trap finish EXIT

stage="qualifier_clean_checkout"
git diff --quiet --ignore-submodules --
git diff --cached --quiet --ignore-submodules --
[[ -z "$(git ls-files --others --exclude-standard)" ]] || {
    source_state="qualifier-untracked-source-present"
    exit 1
}

stage="exact_subject_identity"
git cat-file -e "${SUBJECT_SHA}^{commit}"
git cat-file -e "${BASE_SHA}^{commit}"
subject_tree="$(git rev-parse "${SUBJECT_SHA}^{tree}")"
subject_parent="$(git rev-parse "${SUBJECT_SHA}^")"
[[ "$subject_parent" == "$BASE_SHA" ]] || exit 1
exact_subject_gate="pass"

stage="incremental_surface"
mapfile -t changed < <(git diff --name-only "$BASE_SHA" "$SUBJECT_SHA")
[[ "${#changed[@]}" -eq 1 && "${changed[0]}" == "$EXPECTED_FILE" ]] || {
    printf 'error: noncanonical runner surface\n' >&2
    printf '%s\n' "${changed[@]}" >&2
    exit 1
}
surface_gate="pass"

stage="create_subject_worktree"
worktree="$(mktemp -d "${TMPDIR:-/tmp}/melothaea-runner-qual.XXXXXX")"
rmdir "$worktree"
git worktree add --detach "$worktree" "$SUBJECT_SHA" >/dev/null
cd "$worktree"
[[ "$(git rev-parse HEAD)" == "$SUBJECT_SHA" ]]
git diff --quiet --ignore-submodules --
git diff --cached --quiet --ignore-submodules --
[[ -z "$(git ls-files --others --exclude-standard)" ]]
source_state="clean-exact-subject-checkout"
runner_source_sha256="$(sha256_file "$EXPECTED_FILE")"

stage="toolchain_identity"
declared="$(sed -nE 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' rust-toolchain.toml | head -n1)"
[[ "$declared" == "$EXPECTED_RUST" ]]
rustc_verbose="$(rustc -Vv)"
rustc_release="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "release" {print $2; exit}')"
rustc_commit="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "commit-hash" {print $2; exit}')"
rustc_host="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "host" {print $2; exit}')"
cargo_version="$(cargo -V)"
[[ "$rustc_release" == "$EXPECTED_RUST" ]]
[[ "$cargo_version" == cargo\ 1.96.0\ * ]]
toolchain_gate="pass"

stage="cargo_metadata"
cargo metadata --locked --no-deps --format-version 1 >/dev/null
metadata_gate="pass"

stage="format"
cargo fmt -p symthaea-muse -- --check
fmt_gate="pass"

stage="check_runner"
cargo check --locked -p symthaea-muse --features theory --example "$EXAMPLE"
check_runner_gate="pass"

stage="clippy_runner"
cargo clippy --locked -p symthaea-muse --features theory --example "$EXAMPLE" --no-deps -- -D warnings
clippy_runner_gate="pass"

stage="missing_execution_ack_fails_closed"
missing_out="$(mktemp "${TMPDIR:-/tmp}/melothaea-runner-noack.XXXXXX")"
if cargo run --quiet --locked -p symthaea-muse --features theory --example "$EXAMPLE" -- \
    --output-dir "${TMPDIR:-/tmp}/melothaea-runner-noack-output.$$" >"$missing_out" 2>&1; then
    echo 'error: runner executed without explicit lockbox acknowledgement' >&2
    cat "$missing_out" >&2
    rm -f "$missing_out"
    exit 1
fi
grep -Fq 'lockbox execution is gated' "$missing_out" || {
    echo 'error: missing-ack failure was not the canonical gate' >&2
    cat "$missing_out" >&2
    rm -f "$missing_out"
    exit 1
}
rm -f "$missing_out"
missing_ack_gate="pass"

stage="invalid_qualification_receipt_fails_closed"
fake_dir="$(mktemp -d "${TMPDIR:-/tmp}/melothaea-runner-invalid-receipt.XXXXXX")"
fake_receipt="$fake_dir/fake.tsv"
printf 'schema\tinvalid\n' > "$fake_receipt"
fake_sha="$(sha256_file "$fake_receipt")"
invalid_out="$fake_dir/output.log"
set +e
MEL_CONTEXTUAL_HARMONY_LOCKBOX_EXECUTION="EXECUTE_FROZEN_64_SUBJECT_LOCKBOX" \
MEL_STREAM_QUAL_RECEIPT="$fake_receipt" \
MEL_STREAM_QUAL_RECEIPT_SHA256="$fake_sha" \
cargo run --quiet --locked -p symthaea-muse --features theory --example "$EXAMPLE" -- \
    --output-dir "$fake_dir/results" >"$invalid_out" 2>&1
invalid_rc=$?
set -e
if [[ "$invalid_rc" -eq 0 ]]; then
    echo 'error: runner accepted invalid qualification receipt' >&2
    cat "$invalid_out" >&2
    rm -rf "$fake_dir"
    exit 1
fi
grep -Fq 'qualification receipt' "$invalid_out" || {
    echo 'error: invalid-receipt failure did not occur at receipt gate' >&2
    cat "$invalid_out" >&2
    rm -rf "$fake_dir"
    exit 1
}
if [[ -f "$fake_dir/results/audio-protocol.json" || -d "$fake_dir/results/subjects/00" ]]; then
    echo 'error: runner reached protocol/subject execution before rejecting fake receipt' >&2
    rm -rf "$fake_dir"
    exit 1
fi
rm -rf "$fake_dir"
invalid_receipt_gate="pass"

stage="postflight_source_immutability"
[[ "$(git rev-parse HEAD)" == "$SUBJECT_SHA" ]]
git diff --quiet --ignore-submodules --
git diff --cached --quiet --ignore-submodules --
[[ -z "$(git ls-files --others --exclude-standard)" ]] || {
    source_state="subject-untracked-source-created"
    exit 1
}
source_state="clean-exact-subject-checkout-postflight"
postflight_gate="pass"

status="PASS"
stage="complete"
