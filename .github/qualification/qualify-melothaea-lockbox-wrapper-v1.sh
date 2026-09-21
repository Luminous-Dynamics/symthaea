#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# No-lockbox engineering qualification for the exact-head execution wrapper.
# Valid qualification receipts are never supplied here. The deepest runtime
# probe deliberately supplies structurally fake receipts and must fail before
# admission or output-directory creation.

set -euo pipefail

QUALIFIER_ID="melothaea-lockbox-wrapper-qualification-v1"
SUBJECT_SHA="12b4a915b8ba94eda2691596bea0520a67c59364"
BASE_SHA="1b64a3e193086c99a99a606d7be9f1d93df52de4"
EXPECTED_FILE=".github/qualification/execute-melothaea-contextual-harmony-lockbox-v1.sh"
SCRIPT_PATH=".github/qualification/qualify-melothaea-lockbox-wrapper-v1.sh"
EXPECTED_STREAM_QUALIFIER_SHA="b9834ed86fc84506fdb1e42b421be339fd482bfb"
EXPECTED_RUNNER_QUALIFIER_SHA="1b64a3e193086c99a99a606d7be9f1d93df52de4"

root="$(git rev-parse --show-toplevel)"
cd "$root"
qualifier_sha="$(git rev-parse HEAD)"
qualifier_tree="$(git rev-parse 'HEAD^{tree}')"
receipt_path="${MEL_WRAPPER_QUAL_RECEIPT:-${TMPDIR:-/tmp}/melothaea-lockbox-wrapper-qualification-v1.tsv}"
worktree=""
status="FAIL"
stage="preflight"
source_state="unverified"
subject_tree="unavailable"
subject_parent="unavailable"
wrapper_source_sha256="unavailable"

exact_subject_gate="not-run"
surface_gate="not-run"
bash_syntax_gate="not-run"
missing_ack_gate="not-run"
invalid_receipts_gate="not-run"
postflight_gate="not-run"

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

write_receipt() {
    local rc="$1"
    local final_status="$status"
    local terminal_stage="$stage"
    local provider="local"
    local tmp="${receipt_path}.tmp.$$"
    local script_sha="unavailable"

    [[ "$rc" -eq 0 ]] || final_status="FAIL"
    [[ "$final_status" != "PASS" ]] || terminal_stage="none"
    [[ "${GITHUB_ACTIONS:-}" != "true" ]] || provider="github-actions"
    [[ ! -f "$root/$SCRIPT_PATH" ]] || script_sha="$(sha256_file "$root/$SCRIPT_PATH")"

    mkdir -p "$(dirname "$receipt_path")"
    {
        printf 'schema\tmelothaea-lockbox-wrapper-qualification-v1\n'
        printf 'qualifier_id\t%s\n' "$QUALIFIER_ID"
        printf 'status\t%s\n' "$final_status"
        printf 'exit_code\t%s\n' "$rc"
        printf 'terminal_stage\t%s\n' "$terminal_stage"
        printf 'authority_scope\tengineering-execution-wrapper-contract-only\n'
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
        printf 'subject_changed_file_count\t1\n'
        printf 'subject_changed_file\t%s\n' "$EXPECTED_FILE"
        printf 'wrapper_source_sha256\t%s\n' "$wrapper_source_sha256"
        printf 'expected_stream_qualifier_sha\t%s\n' "$EXPECTED_STREAM_QUALIFIER_SHA"
        printf 'expected_runner_qualifier_sha\t%s\n' "$EXPECTED_RUNNER_QUALIFIER_SHA"
        printf 'source_state\t%s\n' "$source_state"
        printf 'exact_subject_gate\t%s\n' "$exact_subject_gate"
        printf 'surface_gate\t%s\n' "$surface_gate"
        printf 'bash_syntax_gate\t%s\n' "$bash_syntax_gate"
        printf 'missing_ack_gate\t%s\n' "$missing_ack_gate"
        printf 'invalid_receipts_gate\t%s\n' "$invalid_receipts_gate"
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
[[ "$subject_parent" == "$BASE_SHA" ]] || {
    echo 'error: wrapper subject parent mismatch' >&2
    exit 1
}
exact_subject_gate="pass"

stage="incremental_surface"
mapfile -t changed < <(git diff --name-only "$BASE_SHA" "$SUBJECT_SHA")
[[ "${#changed[@]}" -eq 1 && "${changed[0]}" == "$EXPECTED_FILE" ]] || {
    echo 'error: wrapper subject surface is noncanonical' >&2
    printf '%s\n' "${changed[@]}" >&2
    exit 1
}
surface_gate="pass"

stage="create_subject_worktree"
worktree="$(mktemp -d "${TMPDIR:-/tmp}/mel-lockbox-wrapper-qual.XXXXXX")"
rmdir "$worktree"
git worktree add --detach "$worktree" "$SUBJECT_SHA" >/dev/null
cd "$worktree"
[[ "$(git rev-parse HEAD)" == "$SUBJECT_SHA" ]]
git diff --quiet --ignore-submodules --
git diff --cached --quiet --ignore-submodules --
[[ -z "$(git ls-files --others --exclude-standard)" ]]
source_state="clean-exact-subject-checkout"
wrapper_source_sha256="$(sha256_file "$EXPECTED_FILE")"

stage="bash_syntax"
bash -n "$EXPECTED_FILE"
bash_syntax_gate="pass"

stage="missing_ack_fails_closed"
probe_root="$(mktemp -d "${TMPDIR:-/tmp}/mel-lockbox-wrapper-noack.XXXXXX")"
noack_output="$probe_root/results"
noack_log="$probe_root/log"
set +e
MEL_LOCKBOX_OUTPUT_DIR="$noack_output" bash "$EXPECTED_FILE" >"$noack_log" 2>&1
noack_rc=$?
set -e
if [[ "$noack_rc" -eq 0 ]]; then
    echo 'error: wrapper accepted missing execution acknowledgement' >&2
    cat "$noack_log" >&2
    rm -rf "$probe_root"
    exit 1
fi
grep -Fq 'lockbox wrapper is gated' "$noack_log" || {
    echo 'error: missing-ack failure was not canonical' >&2
    cat "$noack_log" >&2
    rm -rf "$probe_root"
    exit 1
}
if [[ -e "$noack_output" || -e "${noack_output}.admission.tsv" ]]; then
    echo 'error: missing-ack probe created execution filesystem state' >&2
    rm -rf "$probe_root" "$noack_output" "${noack_output}.admission.tsv"
    exit 1
fi
rm -rf "$probe_root"
missing_ack_gate="pass"

stage="fake_receipts_fail_before_admission"
probe_root="$(mktemp -d "${TMPDIR:-/tmp}/mel-lockbox-wrapper-fake.XXXXXX")"
stream_fake="$probe_root/stream.tsv"
runner_fake="$probe_root/runner.tsv"
printf 'schema\tfake-stream\n' > "$stream_fake"
printf 'schema\tfake-runner\n' > "$runner_fake"
stream_fake_sha="$(sha256_file "$stream_fake")"
runner_fake_sha="$(sha256_file "$runner_fake")"
fake_output="$probe_root/results"
fake_log="$probe_root/log"
set +e
MEL_LOCKBOX_EXECUTION_ACK="OPEN_FROZEN_64_SUBJECT_LOCKBOX_V1" \
MEL_STREAM_QUAL_RECEIPT="$stream_fake" \
MEL_STREAM_QUAL_RECEIPT_SHA256="$stream_fake_sha" \
MEL_RUNNER_QUAL_RECEIPT="$runner_fake" \
MEL_RUNNER_QUAL_RECEIPT_SHA256="$runner_fake_sha" \
MEL_LOCKBOX_OUTPUT_DIR="$fake_output" \
bash "$EXPECTED_FILE" >"$fake_log" 2>&1
fake_rc=$?
set -e
if [[ "$fake_rc" -eq 0 ]]; then
    echo 'error: wrapper accepted fake qualification receipts' >&2
    cat "$fake_log" >&2
    rm -rf "$probe_root"
    exit 1
fi
if [[ -e "$fake_output" || -e "${fake_output}.admission.tsv" ]]; then
    echo 'error: fake-receipt probe reached admission/output creation' >&2
    rm -rf "$probe_root" "$fake_output" "${fake_output}.admission.tsv"
    exit 1
fi
if grep -Fq 'lockbox execution wrapper complete' "$fake_log"; then
    echo 'error: fake-receipt probe reached completion' >&2
    cat "$fake_log" >&2
    rm -rf "$probe_root"
    exit 1
fi
rm -rf "$probe_root"
invalid_receipts_gate="pass"

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
