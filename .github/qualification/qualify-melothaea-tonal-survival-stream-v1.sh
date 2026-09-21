#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portable exact-head engineering qualification for the low-memory ProgSuite
# tonal-survival streaming-admission tranche. This script qualifies the frozen
# product subject, never its own qualification-only checkout, and never executes
# or unblinds the 64-subject lockbox.

set -euo pipefail

QUALIFIER_ID="melothaea-tonal-survival-stream-qualification-v1"
SUBJECT_SHA="d888341323b6ee463007cf90d8032153039ec099"
BASE_SHA="f3a38ed769d5d2477e6ec5094919150e48638710"
EXPECTED_RUST="1.96.0"
EXPECTED_FILES=(
  "crates/domains/symthaea-muse/src/evidence_digest.rs"
  "crates/domains/symthaea-muse/src/prog_suite_contextual_harmony_tonal_survival_stream.rs"
)

root="$(git rev-parse --show-toplevel)"
cd "$root"
qualifier_sha="$(git rev-parse HEAD)"
qualifier_tree="$(git rev-parse 'HEAD^{tree}')"
script_path=".github/qualification/qualify-melothaea-tonal-survival-stream-v1.sh"
receipt_path="${MEL_STREAM_QUAL_RECEIPT:-${TMPDIR:-/tmp}/melothaea-tonal-survival-stream-qualification-v1.tsv}"
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
cargo_lock_sha256="unavailable"
rust_toolchain_sha256="unavailable"
muse_manifest_sha256="unavailable"

exact_subject_gate="not-run"
surface_gate="not-run"
toolchain_gate="not-run"
metadata_gate="not-run"
fmt_gate="not-run"
test_music_theory_gate="not-run"
test_stream_gate="not-run"
check_muse_gate="not-run"
clippy_muse_gate="not-run"
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
    local exit_code="$1"
    local final_status="$status"
    local terminal_stage="$stage"
    local provider="local"
    local script_sha
    local tmp="${receipt_path}.tmp.$$"

    [[ "$exit_code" -eq 0 ]] || final_status="FAIL"
    [[ "$final_status" != "PASS" ]] || terminal_stage="none"
    [[ "${GITHUB_ACTIONS:-}" != "true" ]] || provider="github-actions"
    script_sha="$(sha256_file "$root/$script_path")"

    mkdir -p "$(dirname "$receipt_path")"
    {
        printf 'schema\tmelothaea-tonal-survival-stream-qualification-v1\n'
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
        printf 'qualifier_script_sha256\t%s\n' "$script_sha"
        printf 'subject_sha\t%s\n' "$SUBJECT_SHA"
        printf 'subject_tree\t%s\n' "$subject_tree"
        printf 'subject_parent\t%s\n' "$subject_parent"
        printf 'base_sha\t%s\n' "$BASE_SHA"
        printf 'subject_changed_file_count\t%s\n' "${#EXPECTED_FILES[@]}"
        for path in "${EXPECTED_FILES[@]}"; do
            printf 'subject_changed_file\t%s\n' "$path"
        done
        printf 'source_state\t%s\n' "$source_state"
        printf 'expected_rust_release\t%s\n' "$EXPECTED_RUST"
        printf 'rustc_release\t%s\n' "$rustc_release"
        printf 'rustc_commit_hash\t%s\n' "$rustc_commit"
        printf 'rustc_host\t%s\n' "$rustc_host"
        printf 'cargo_version\t%s\n' "$cargo_version"
        printf 'cargo_lock_sha256\t%s\n' "$cargo_lock_sha256"
        printf 'rust_toolchain_sha256\t%s\n' "$rust_toolchain_sha256"
        printf 'muse_manifest_sha256\t%s\n' "$muse_manifest_sha256"
        printf 'exact_subject_gate\t%s\n' "$exact_subject_gate"
        printf 'surface_gate\t%s\n' "$surface_gate"
        printf 'toolchain_gate\t%s\n' "$toolchain_gate"
        printf 'metadata_gate\t%s\n' "$metadata_gate"
        printf 'fmt_gate\t%s\n' "$fmt_gate"
        printf 'test_music_theory_gate\t%s\n' "$test_music_theory_gate"
        printf 'test_stream_gate\t%s\n' "$test_stream_gate"
        printf 'check_muse_gate\t%s\n' "$check_muse_gate"
        printf 'clippy_muse_gate\t%s\n' "$clippy_muse_gate"
        printf 'postflight_gate\t%s\n' "$postflight_gate"
        printf 'github_repository\t%s\n' "${GITHUB_REPOSITORY:-not-applicable}"
        printf 'github_run_id\t%s\n' "${GITHUB_RUN_ID:-not-applicable}"
        printf 'github_run_attempt\t%s\n' "${GITHUB_RUN_ATTEMPT:-not-applicable}"
    } > "$tmp"
    mv "$tmp" "$receipt_path"
    printf 'melothaea streaming qualification receipt=%s status=%s stage=%s\n' \
        "$receipt_path" "$final_status" "$terminal_stage"
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
if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    source_state="qualifier-untracked-source-present"
    echo 'error: qualifier checkout contains untracked files' >&2
    exit 1
fi

stage="exact_subject_identity"
git cat-file -e "${SUBJECT_SHA}^{commit}"
git cat-file -e "${BASE_SHA}^{commit}"
subject_tree="$(git rev-parse "${SUBJECT_SHA}^{tree}")"
subject_parent="$(git rev-parse "${SUBJECT_SHA}^")"
[[ "$subject_parent" == "$BASE_SHA" ]] || {
    echo 'error: subject is not the exact child of declared base' >&2
    exit 1
}
exact_subject_gate="pass"

stage="incremental_surface"
mapfile -t changed < <(git diff --name-only "$BASE_SHA" "$SUBJECT_SHA" | LC_ALL=C sort)
[[ "${#changed[@]}" -eq "${#EXPECTED_FILES[@]}" ]] || {
    printf 'error: unexpected subject file count: %s\n' "${#changed[@]}" >&2
    exit 1
}
for i in "${!EXPECTED_FILES[@]}"; do
    [[ "${changed[$i]}" == "${EXPECTED_FILES[$i]}" ]] || {
        printf 'error: unexpected subject path: %s\n' "${changed[$i]}" >&2
        exit 1
    }
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
declared="$(sed -nE 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' rust-toolchain.toml | head -n1)"
[[ "$declared" == "$EXPECTED_RUST" ]] || {
    echo "error: declared Rust is $declared, expected $EXPECTED_RUST" >&2
    exit 1
}
rustc_verbose="$(rustc -Vv)"
rustc_release="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "release" {print $2; exit}')"
rustc_commit="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "commit-hash" {print $2; exit}')"
rustc_host="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "host" {print $2; exit}')"
cargo_version="$(cargo -V)"
[[ "$rustc_release" == "$EXPECTED_RUST" ]]
[[ "$cargo_version" == cargo\ 1.96.0\ * ]] || {
    echo "error: active Cargo is not 1.96.0: $cargo_version" >&2
    exit 1
}
cargo_lock_sha256="$(sha256_file Cargo.lock)"
rust_toolchain_sha256="$(sha256_file rust-toolchain.toml)"
muse_manifest_sha256="$(sha256_file crates/domains/symthaea-muse/Cargo.toml)"
toolchain_gate="pass"

stage="cargo_metadata"
cargo metadata --locked --no-deps --format-version 1 >/dev/null
metadata_gate="pass"

stage="format"
cargo fmt -p symthaea-muse -- --check
fmt_gate="pass"

stage="test_music_theory_prog_suite"
cargo test --locked -p symthaea-music-theory prog_suite
test_music_theory_gate="pass"

stage="test_streaming_admission"
cargo test --locked -p symthaea-muse --features theory prog_suite_contextual_harmony_tonal_survival_stream
test_stream_gate="pass"

stage="check_muse_all_targets"
cargo check --locked -p symthaea-muse --features theory --all-targets
check_muse_gate="pass"

stage="clippy_muse_all_targets"
cargo clippy --locked -p symthaea-muse --features theory --all-targets --no-deps -- -D warnings
clippy_muse_gate="pass"

stage="postflight_source_immutability"
[[ "$(git rev-parse HEAD)" == "$SUBJECT_SHA" ]]
git diff --quiet --ignore-submodules --
git diff --cached --quiet --ignore-submodules --
if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    source_state="subject-untracked-source-created"
    echo 'error: qualification created untracked subject files' >&2
    exit 1
fi
source_state="clean-exact-subject-checkout-postflight"
postflight_gate="pass"

status="PASS"
stage="complete"
