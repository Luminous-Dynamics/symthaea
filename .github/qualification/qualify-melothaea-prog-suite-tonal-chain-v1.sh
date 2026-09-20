#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portable exact-subject engineering qualification for the frozen Melothaea
# ProgSuite contextual-harmony machine-evidence chain.
#
# This script lives on a qualification-only child. It never treats its own
# checkout as the product/scientific subject. It creates a detached worktree at
# the frozen SUBJECT_SHA, runs bounded software gates there, and emits a
# fail-closed receipt. It does NOT execute or unblind the 64-subject lockbox.
#
# Preferred local invocation:
#
#   nix develop -c bash \
#     .github/qualification/qualify-melothaea-prog-suite-tonal-chain-v1.sh

set -euo pipefail

QUALIFIER_ID="melothaea-prog-suite-tonal-chain-qualification-v1"
SUBJECT_SHA="f3a38ed769d5d2477e6ec5094919150e48638710"
BASE_SHA="646b74d184ad908429956d17faaf949364311d1e"
EXPECTED_RUST="1.96.0"
EXPECTED_FILES=(
  "crates/domains/symthaea-muse/src/evidence_digest.rs"
  "crates/domains/symthaea-muse/src/prog_suite_contextual_harmony_tonal_survival_panel.rs"
)

outer_root="$(git rev-parse --show-toplevel)"
cd "$outer_root"

qualifier_sha="$(git rev-parse HEAD)"
qualifier_tree="$(git rev-parse 'HEAD^{tree}')"
script_path=".github/qualification/qualify-melothaea-prog-suite-tonal-chain-v1.sh"
receipt_path="${MEL_TONAL_QUAL_RECEIPT:-${TMPDIR:-/tmp}/melothaea-prog-suite-tonal-chain-qualification-v1.tsv}"
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
music_theory_manifest_sha256="unavailable"
muse_manifest_sha256="unavailable"

exact_subject_gate="not-run"
surface_gate="not-run"
toolchain_gate="not-run"
metadata_gate="not-run"
fmt_music_theory_gate="not-run"
fmt_muse_gate="not-run"
test_music_theory_gate="not-run"
test_muse_gate="not-run"
check_music_theory_gate="not-run"
check_muse_gate="not-run"
clippy_music_theory_gate="not-run"
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
    local script_sha="unavailable"
    local tmp_path="${receipt_path}.tmp.$$"

    if [[ "$exit_code" -ne 0 ]]; then
        final_status="FAIL"
    fi
    if [[ "$final_status" == "PASS" ]]; then
        terminal_stage="none"
    fi
    if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
        provider="github-actions"
    fi

    script_sha="$(sha256_file "$outer_root/$script_path")"
    mkdir -p "$(dirname "$receipt_path")" || return 1
    {
        printf 'schema\tmelothaea-prog-suite-tonal-chain-qualification-v1\n'
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
        printf 'music_theory_manifest_sha256\t%s\n' "$music_theory_manifest_sha256"
        printf 'muse_manifest_sha256\t%s\n' "$muse_manifest_sha256"
        printf 'exact_subject_gate\t%s\n' "$exact_subject_gate"
        printf 'surface_gate\t%s\n' "$surface_gate"
        printf 'toolchain_gate\t%s\n' "$toolchain_gate"
        printf 'metadata_gate\t%s\n' "$metadata_gate"
        printf 'fmt_music_theory_gate\t%s\n' "$fmt_music_theory_gate"
        printf 'fmt_muse_gate\t%s\n' "$fmt_muse_gate"
        printf 'test_music_theory_gate\t%s\n' "$test_music_theory_gate"
        printf 'test_muse_gate\t%s\n' "$test_muse_gate"
        printf 'check_music_theory_gate\t%s\n' "$check_music_theory_gate"
        printf 'check_muse_gate\t%s\n' "$check_muse_gate"
        printf 'clippy_music_theory_gate\t%s\n' "$clippy_music_theory_gate"
        printf 'clippy_muse_gate\t%s\n' "$clippy_muse_gate"
        printf 'postflight_gate\t%s\n' "$postflight_gate"
        printf 'github_repository\t%s\n' "${GITHUB_REPOSITORY:-not-applicable}"
        printf 'github_run_id\t%s\n' "${GITHUB_RUN_ID:-not-applicable}"
        printf 'github_run_attempt\t%s\n' "${GITHUB_RUN_ATTEMPT:-not-applicable}"
    } > "$tmp_path"
    mv "$tmp_path" "$receipt_path"
    printf 'melothaea tonal-chain qualification receipt=%s status=%s stage=%s\n' \
        "$receipt_path" "$final_status" "$terminal_stage"
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

    if [[ "$exit_code" -eq 0 && "$status" != "PASS" ]]; then
        echo "error: verifier exited without terminal PASS (stage=$stage)" >&2
        exit_code=1
    fi

    if ! write_receipt "$exit_code"; then
        echo "error: qualification receipt could not be written: $receipt_path" >&2
        if [[ "$exit_code" -eq 0 ]]; then
            exit_code=1
        fi
    fi

    cleanup_worktree
    exit "$exit_code"
}
trap finish EXIT

stage="qualifier_clean_checkout"
if ! git diff --quiet --ignore-submodules --; then
    source_state="qualifier-tracked-modifications-present"
    echo "error: qualification checkout has tracked modifications" >&2
    exit 1
fi
if ! git diff --cached --quiet --ignore-submodules --; then
    source_state="qualifier-staged-modifications-present"
    echo "error: qualification checkout has staged modifications" >&2
    exit 1
fi
outer_untracked="$(git ls-files --others --exclude-standard)"
if [[ -n "$outer_untracked" ]]; then
    source_state="qualifier-untracked-source-present"
    echo "error: qualification checkout has untracked files" >&2
    printf '%s\n' "$outer_untracked" >&2
    exit 1
fi

stage="exact_subject_identity"
git cat-file -e "${SUBJECT_SHA}^{commit}"
git cat-file -e "${BASE_SHA}^{commit}"
subject_tree="$(git rev-parse "${SUBJECT_SHA}^{tree}")"
subject_parent="$(git rev-parse "${SUBJECT_SHA}^")"
if [[ "$subject_parent" != "$BASE_SHA" ]]; then
    echo "error: frozen subject is not the exact one-commit child of the declared base" >&2
    exit 1
fi
exact_subject_gate="pass"

stage="incremental_surface"
mapfile -t changed_files < <(
    git diff --name-only "$BASE_SHA" "$SUBJECT_SHA" | LC_ALL=C sort
)
if [[ "${#changed_files[@]}" -ne "${#EXPECTED_FILES[@]}" ]]; then
    echo "error: unexpected frozen subject file count: ${#changed_files[@]}" >&2
    printf '%s\n' "${changed_files[@]}" >&2
    exit 1
fi
for i in "${!EXPECTED_FILES[@]}"; do
    if [[ "${changed_files[$i]}" != "${EXPECTED_FILES[$i]}" ]]; then
        echo "error: unexpected frozen subject path at index $i: ${changed_files[$i]}" >&2
        exit 1
    fi
done
surface_gate="pass"

stage="create_subject_worktree"
worktree="$(mktemp -d "${TMPDIR:-/tmp}/melothaea-tonal-qual.XXXXXX")"
rmdir "$worktree"
git worktree add --detach "$worktree" "$SUBJECT_SHA" >/dev/null
cd "$worktree"

if [[ "$(git rev-parse HEAD)" != "$SUBJECT_SHA" ]]; then
    echo "error: detached worktree did not resolve the frozen subject" >&2
    exit 1
fi
if ! git diff --quiet --ignore-submodules -- || ! git diff --cached --quiet --ignore-submodules --; then
    echo "error: frozen subject worktree is not clean" >&2
    exit 1
fi
subject_untracked="$(git ls-files --others --exclude-standard)"
if [[ -n "$subject_untracked" ]]; then
    echo "error: frozen subject worktree has untracked files" >&2
    printf '%s\n' "$subject_untracked" >&2
    exit 1
fi
source_state="clean-exact-subject-checkout"

stage="toolchain_identity"
declared_rust="$(
    sed -nE 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' \
        rust-toolchain.toml | head -n1
)"
if [[ "$declared_rust" != "$EXPECTED_RUST" ]]; then
    echo "error: frozen subject toolchain changed: expected=$EXPECTED_RUST declared=$declared_rust" >&2
    exit 1
fi

rustc_verbose="$(rustc -Vv)"
rustc_release="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "release" {print $2; exit}')"
rustc_commit="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "commit-hash" {print $2; exit}')"
rustc_host="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "host" {print $2; exit}')"
cargo_version="$(cargo -V)"
if [[ "$rustc_release" != "$EXPECTED_RUST" ]]; then
    echo "error: active rustc does not match frozen toolchain: expected=$EXPECTED_RUST actual=$rustc_release" >&2
    exit 1
fi

cargo_lock_sha256="$(sha256_file Cargo.lock)"
rust_toolchain_sha256="$(sha256_file rust-toolchain.toml)"
music_theory_manifest_sha256="$(sha256_file crates/domains/symthaea-music-theory/Cargo.toml)"
muse_manifest_sha256="$(sha256_file crates/domains/symthaea-muse/Cargo.toml)"
toolchain_gate="pass"

stage="cargo_metadata"
cargo metadata --locked --no-deps --format-version 1 >/dev/null
metadata_gate="pass"

stage="fmt_music_theory"
cargo fmt -p symthaea-music-theory -- --check
fmt_music_theory_gate="pass"

stage="fmt_muse"
cargo fmt -p symthaea-muse -- --check
fmt_muse_gate="pass"

stage="test_music_theory"
cargo test --locked -p symthaea-music-theory prog_suite
test_music_theory_gate="pass"

stage="test_muse_contextual_harmony"
cargo test --locked -p symthaea-muse --features theory prog_suite_contextual_harmony
test_muse_gate="pass"

stage="check_music_theory"
cargo check --locked -p symthaea-music-theory --all-targets
check_music_theory_gate="pass"

stage="check_muse"
cargo check --locked -p symthaea-muse --features theory --all-targets
check_muse_gate="pass"

stage="clippy_music_theory"
cargo clippy --locked -p symthaea-music-theory --all-targets --no-deps -- -D warnings
clippy_music_theory_gate="pass"

stage="clippy_muse"
cargo clippy --locked -p symthaea-muse --features theory --all-targets --no-deps -- -D warnings
clippy_muse_gate="pass"

stage="postflight_source_immutability"
if [[ "$(git rev-parse HEAD)" != "$SUBJECT_SHA" ]]; then
    source_state="subject-head-changed-during-qualification"
    echo "error: subject HEAD changed during qualification" >&2
    exit 1
fi
if ! git diff --quiet --ignore-submodules --; then
    source_state="subject-tracked-source-mutated"
    echo "error: qualification mutated tracked subject bytes" >&2
    git diff --stat >&2 || true
    exit 1
fi
if ! git diff --cached --quiet --ignore-submodules --; then
    source_state="subject-staged-source-mutated"
    echo "error: qualification created staged subject changes" >&2
    exit 1
fi
post_untracked="$(git ls-files --others --exclude-standard)"
if [[ -n "$post_untracked" ]]; then
    source_state="subject-untracked-source-created"
    echo "error: qualification created untracked source/evidence files" >&2
    printf '%s\n' "$post_untracked" >&2
    exit 1
fi
source_state="clean-exact-subject-checkout-postflight"
postflight_gate="pass"

status="PASS"
stage="complete"
