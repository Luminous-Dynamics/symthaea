#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portable exact-subject engineering qualification for the Melothaea
# tonal-survival streaming-admission adapter.
#
# This verifies software behavior only. It does NOT execute/unblind the
# 64-subject lockbox and grants no perceptual, artistic, or product authority.

set -euo pipefail

QUALIFIER_ID="melothaea-tonal-survival-streaming-admission-qualification-v1"
SUBJECT_SHA="d888341323b6ee463007cf90d8032153039ec099"
BASE_SHA="f3a38ed769d5d2477e6ec5094919150e48638710"
EXPECTED_RUST="1.96.0"
EXPECTED_SUBJECT_FILES=(
  "crates/domains/symthaea-muse/src/evidence_digest.rs"
  "crates/domains/symthaea-muse/src/prog_suite_contextual_harmony_tonal_survival_stream.rs"
)
EXPECTED_QUALIFIER_FILES=(
  ".github/qualification/qualify-melothaea-tonal-survival-streaming-admission-v1.sh"
  ".github/qualification/verify-melothaea-tonal-survival-streaming-admission-receipt-v1.py"
)

outer_root="$(git rev-parse --show-toplevel)"
cd "$outer_root"
qualifier_sha="$(git rev-parse HEAD)"
qualifier_tree="$(git rev-parse 'HEAD^{tree}')"
qualifier_parent="$(git rev-parse 'HEAD^')"
script_path=".github/qualification/qualify-melothaea-tonal-survival-streaming-admission-v1.sh"
receipt_path="${MEL_STREAM_QUAL_RECEIPT:-${TMPDIR:-/tmp}/melothaea-tonal-survival-streaming-admission-qualification-v1.tsv}"
worktree=""
status="FAIL"
stage="preflight"
source_state="unverified"
subject_tree="unavailable"
subject_parent="unavailable"
rustc_release="unavailable"
rustc_commit="unavailable"
rustc_host="unavailable"
cargo_release="unavailable"
cargo_version="unavailable"
cargo_lock_sha256="unavailable"
rust_toolchain_sha256="unavailable"
muse_manifest_sha256="unavailable"

qualifier_identity_gate="not-run"
exact_subject_gate="not-run"
surface_gate="not-run"
toolchain_gate="not-run"
metadata_gate="not-run"
fmt_gate="not-run"
test_gate="not-run"
check_gate="not-run"
clippy_gate="not-run"
postflight_gate="not-run"

sha256_file() {
    local path="$1"
    if [[ -f "$path" ]]; then sha256sum "$path" | awk '{print $1}'; else printf 'unavailable'; fi
}

write_receipt() {
    local exit_code="$1"
    local final_status="$status"
    local terminal_stage="$stage"
    local provider="local"
    local script_sha="unavailable"
    local tmp_path="${receipt_path}.tmp.$$"
    [[ "$exit_code" -eq 0 ]] || final_status="FAIL"
    [[ "$final_status" == "PASS" ]] && terminal_stage="none"
    [[ "${GITHUB_ACTIONS:-}" == "true" ]] && provider="github-actions"
    script_sha="$(sha256_file "$outer_root/$script_path")"
    mkdir -p "$(dirname "$receipt_path")" || return 1
    {
        printf 'schema\tmelothaea-tonal-survival-streaming-admission-qualification-v1\n'
        printf 'qualifier_id\t%s\n' "$QUALIFIER_ID"
        printf 'status\t%s\n' "$final_status"
        printf 'exit_code\t%s\n' "$exit_code"
        printf 'terminal_stage\t%s\n' "$terminal_stage"
        printf 'authority_scope\tengineering-software-contract-only\n'
        printf 'scientific_lockbox_execution\tnot-performed\n'
        printf 'human_perceptual_authority\tnone\n'
        printf 'artistic_quality_authority\tnone\n'
        printf 'product_authority\tnone\n'
        printf 'qualification_provider\t%s\n' "$provider"
        printf 'environment_authority\tobserved-not-hermetic-capsule-qualified\n'
        printf 'receipt_attestation\tnone\n'
        printf 'qualifier_checkout_sha\t%s\n' "$qualifier_sha"
        printf 'qualifier_checkout_tree\t%s\n' "$qualifier_tree"
        printf 'qualifier_parent\t%s\n' "$qualifier_parent"
        printf 'qualifier_script_sha256\t%s\n' "$script_sha"
        printf 'subject_sha\t%s\n' "$SUBJECT_SHA"
        printf 'subject_tree\t%s\n' "$subject_tree"
        printf 'subject_parent\t%s\n' "$subject_parent"
        printf 'base_sha\t%s\n' "$BASE_SHA"
        printf 'source_state\t%s\n' "$source_state"
        printf 'expected_rust_release\t%s\n' "$EXPECTED_RUST"
        printf 'rustc_release\t%s\n' "$rustc_release"
        printf 'rustc_commit_hash\t%s\n' "$rustc_commit"
        printf 'rustc_host\t%s\n' "$rustc_host"
        printf 'cargo_release\t%s\n' "$cargo_release"
        printf 'cargo_version\t%s\n' "$cargo_version"
        printf 'cargo_lock_sha256\t%s\n' "$cargo_lock_sha256"
        printf 'rust_toolchain_sha256\t%s\n' "$rust_toolchain_sha256"
        printf 'muse_manifest_sha256\t%s\n' "$muse_manifest_sha256"
        printf 'qualifier_identity_gate\t%s\n' "$qualifier_identity_gate"
        printf 'exact_subject_gate\t%s\n' "$exact_subject_gate"
        printf 'surface_gate\t%s\n' "$surface_gate"
        printf 'toolchain_gate\t%s\n' "$toolchain_gate"
        printf 'metadata_gate\t%s\n' "$metadata_gate"
        printf 'fmt_gate\t%s\n' "$fmt_gate"
        printf 'test_gate\t%s\n' "$test_gate"
        printf 'check_gate\t%s\n' "$check_gate"
        printf 'clippy_gate\t%s\n' "$clippy_gate"
        printf 'postflight_gate\t%s\n' "$postflight_gate"
        printf 'github_repository\t%s\n' "${GITHUB_REPOSITORY:-not-applicable}"
        printf 'github_run_id\t%s\n' "${GITHUB_RUN_ID:-not-applicable}"
        printf 'github_run_attempt\t%s\n' "${GITHUB_RUN_ATTEMPT:-not-applicable}"
    } > "$tmp_path"
    mv "$tmp_path" "$receipt_path"
    printf 'melothaea streaming qualification receipt=%s status=%s stage=%s\n' "$receipt_path" "$final_status" "$terminal_stage"
}

cleanup_worktree() {
    cd "$outer_root" 2>/dev/null || true
    if [[ -n "$worktree" && -d "$worktree" ]]; then
        git worktree remove --force "$worktree" >/dev/null 2>&1 || true
    fi
}

finish() {
    local exit_code=$?
    trap - EXIT
    if [[ "$exit_code" -eq 0 && "$status" != "PASS" ]]; then exit_code=1; fi
    write_receipt "$exit_code" || { echo "error: receipt write failed" >&2; [[ "$exit_code" -ne 0 ]] || exit_code=1; }
    cleanup_worktree
    exit "$exit_code"
}
trap finish EXIT

stage="qualifier_clean_checkout"
git diff --quiet --ignore-submodules --
git diff --cached --quiet --ignore-submodules --
[[ -z "$(git ls-files --others --exclude-standard)" ]] || { source_state="qualifier-untracked-source-present"; exit 1; }

stage="qualifier_identity"
if [[ "$qualifier_parent" != "$SUBJECT_SHA" ]]; then
    echo "error: qualifier is not an exact one-commit child of frozen subject" >&2
    exit 1
fi
mapfile -t qualifier_files < <(git diff --name-only "$SUBJECT_SHA" "$qualifier_sha" | LC_ALL=C sort)
if [[ "${#qualifier_files[@]}" -ne "${#EXPECTED_QUALIFIER_FILES[@]}" ]]; then
    echo "error: unexpected qualifier file count" >&2; exit 1
fi
for i in "${!EXPECTED_QUALIFIER_FILES[@]}"; do
    [[ "${qualifier_files[$i]}" == "${EXPECTED_QUALIFIER_FILES[$i]}" ]] || { echo "error: unexpected qualifier path" >&2; exit 1; }
done
qualifier_identity_gate="pass"

stage="exact_subject_identity"
git cat-file -e "${SUBJECT_SHA}^{commit}"
git cat-file -e "${BASE_SHA}^{commit}"
subject_tree="$(git rev-parse "${SUBJECT_SHA}^{tree}")"
subject_parent="$(git rev-parse "${SUBJECT_SHA}^")"
[[ "$subject_parent" == "$BASE_SHA" ]] || { echo "error: subject parent mismatch" >&2; exit 1; }
exact_subject_gate="pass"

stage="incremental_surface"
mapfile -t changed_files < <(git diff --name-only "$BASE_SHA" "$SUBJECT_SHA" | LC_ALL=C sort)
if [[ "${#changed_files[@]}" -ne "${#EXPECTED_SUBJECT_FILES[@]}" ]]; then
    echo "error: unexpected subject file count" >&2; exit 1
fi
for i in "${!EXPECTED_SUBJECT_FILES[@]}"; do
    [[ "${changed_files[$i]}" == "${EXPECTED_SUBJECT_FILES[$i]}" ]] || { echo "error: unexpected subject path" >&2; exit 1; }
done
surface_gate="pass"

stage="create_subject_worktree"
worktree="$(mktemp -d "${TMPDIR:-/tmp}/melothaea-stream-qual.XXXXXX")"
rmdir "$worktree"
git worktree add --detach "$worktree" "$SUBJECT_SHA" >/dev/null
cd "$worktree"
[[ "$(git rev-parse HEAD)" == "$SUBJECT_SHA" ]]
git diff --quiet --ignore-submodules --
git diff --cached --quiet --ignore-submodules --
[[ -z "$(git ls-files --others --exclude-standard)" ]]
source_state="clean-exact-subject-checkout"

stage="toolchain_identity"
declared_rust="$(sed -nE 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' rust-toolchain.toml | head -n1)"
[[ "$declared_rust" == "$EXPECTED_RUST" ]] || { echo "error: frozen toolchain changed" >&2; exit 1; }
rustc_verbose="$(rustc -Vv)"
rustc_release="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "release" {print $2; exit}')"
rustc_commit="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "commit-hash" {print $2; exit}')"
rustc_host="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "host" {print $2; exit}')"
cargo_version="$(cargo -V)"
cargo_release="$(printf '%s\n' "$cargo_version" | awk '{print $2}')"
[[ "$rustc_release" == "$EXPECTED_RUST" ]] || { echo "error: rustc mismatch" >&2; exit 1; }
[[ "$cargo_release" == "$EXPECTED_RUST" ]] || { echo "error: cargo mismatch" >&2; exit 1; }
cargo_lock_sha256="$(sha256_file Cargo.lock)"
rust_toolchain_sha256="$(sha256_file rust-toolchain.toml)"
muse_manifest_sha256="$(sha256_file crates/domains/symthaea-muse/Cargo.toml)"
toolchain_gate="pass"

stage="cargo_metadata"
cargo metadata --locked --no-deps --format-version 1 >/dev/null
metadata_gate="pass"

stage="fmt_muse"
cargo fmt -p symthaea-muse -- --check
fmt_gate="pass"

stage="test_streaming_tonal_survival"
cargo test --locked -p symthaea-muse --features theory prog_suite_contextual_harmony_tonal_survival
test_gate="pass"

stage="check_muse"
cargo check --locked -p symthaea-muse --features theory --all-targets
check_gate="pass"

stage="clippy_muse"
cargo clippy --locked -p symthaea-muse --features theory --all-targets --no-deps -- -D warnings
clippy_gate="pass"

stage="postflight_source_immutability"
[[ "$(git rev-parse HEAD)" == "$SUBJECT_SHA" ]] || { source_state="subject-head-changed"; exit 1; }
git diff --quiet --ignore-submodules -- || { source_state="subject-tracked-source-mutated"; exit 1; }
git diff --cached --quiet --ignore-submodules -- || { source_state="subject-staged-source-mutated"; exit 1; }
[[ -z "$(git ls-files --others --exclude-standard)" ]] || { source_state="subject-untracked-source-created"; exit 1; }
source_state="clean-exact-subject-checkout-postflight"
postflight_gate="pass"
status="PASS"
stage="complete"
