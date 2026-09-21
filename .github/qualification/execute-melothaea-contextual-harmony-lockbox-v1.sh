#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Exact-head execution wrapper for the frozen 64-subject ProgSuite contextual-
# harmony lockbox. This wrapper does nothing unless two independent engineering
# PASS receipts validate in detached worktrees at their exact qualifier heads.
# It then executes the exact frozen runner commit, not the caller checkout.

set -euo pipefail

WRAPPER_SCHEMA="melothaea-contextual-harmony-lockbox-execution-wrapper-v1"
EXECUTION_ACK="OPEN_FROZEN_64_SUBJECT_LOCKBOX_V1"
STREAM_SUBJECT_SHA="d888341323b6ee463007cf90d8032153039ec099"
STREAM_QUALIFIER_SHA="b9834ed86fc84506fdb1e42b421be339fd482bfb"
RUNNER_SUBJECT_SHA="d5fc6185b471f091f098073a492d3ff42aeb0931"
RUNNER_QUALIFIER_SHA="1b64a3e193086c99a99a606d7be9f1d93df52de4"
EXPECTED_RUST="1.96.0"
RUNNER_EXAMPLE="prog_suite_contextual_harmony_lockbox_runner"
STREAM_VERIFIER=".github/qualification/verify-melothaea-tonal-survival-stream-receipt-v1.py"
RUNNER_VERIFIER=".github/qualification/verify-melothaea-lockbox-runner-receipt-v1.py"
SCRIPT_PATH=".github/qualification/execute-melothaea-contextual-harmony-lockbox-v1.sh"

root="$(git rev-parse --show-toplevel)"
cd "$root"
stream_worktree=""
runner_qual_worktree=""
runner_worktree=""
admission_path=""
execution_phase="pre-admission"
wrapper_checkout_sha="unavailable"
wrapper_checkout_tree="unavailable"
wrapper_script_sha256="unavailable"

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

is_sha256() {
    [[ "$1" =~ ^[0-9a-f]{64}$ ]]
}

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "error: required environment variable is missing: $name" >&2
        exit 1
    fi
}

write_admission() {
    local status="$1"
    local tmp="${admission_path}.tmp.$$"
    {
        printf 'schema\t%s\n' "$WRAPPER_SCHEMA"
        printf 'status\t%s\n' "$status"
        printf 'execution_phase\t%s\n' "$execution_phase"
        printf 'wrapper_checkout_sha\t%s\n' "$wrapper_checkout_sha"
        printf 'wrapper_checkout_tree\t%s\n' "$wrapper_checkout_tree"
        printf 'wrapper_script_sha256\t%s\n' "$wrapper_script_sha256"
        printf 'stream_subject_sha\t%s\n' "$STREAM_SUBJECT_SHA"
        printf 'stream_qualifier_sha\t%s\n' "$STREAM_QUALIFIER_SHA"
        printf 'stream_qualification_receipt_sha256\t%s\n' "$MEL_STREAM_QUAL_RECEIPT_SHA256"
        printf 'runner_subject_sha\t%s\n' "$RUNNER_SUBJECT_SHA"
        printf 'runner_qualifier_sha\t%s\n' "$RUNNER_QUALIFIER_SHA"
        printf 'runner_qualification_receipt_sha256\t%s\n' "$MEL_RUNNER_QUAL_RECEIPT_SHA256"
        printf 'lockbox_subjects_observed_before_launch\t0\n'
        printf 'authority_scope\tfrozen-lockbox-machine-evidence-execution-admission\n'
        printf 'human_perceptual_authority\tnone\n'
        printf 'artistic_quality_authority\tnone\n'
        printf 'product_authority\tnone\n'
    } > "$tmp"
    mv "$tmp" "$admission_path"
}

cleanup_worktrees() {
    cd "$root" 2>/dev/null || true
    for wt in "$stream_worktree" "$runner_qual_worktree" "$runner_worktree"; do
        if [[ -n "$wt" && -d "$wt" ]]; then
            git worktree remove --force "$wt" >/dev/null 2>&1 || true
        fi
    done
}

finish() {
    local rc=$?
    trap - EXIT
    if [[ "$rc" -ne 0 && -n "$admission_path" && -f "$admission_path" ]]; then
        case "$execution_phase" in
            runner-started)
                write_admission "FAILED_AFTER_EXECUTION_START" || true
                ;;
            post-runner-validation)
                write_admission "FAILED_POST_RUNNER_VALIDATION" || true
                ;;
        esac
    fi
    cleanup_worktrees
    exit "$rc"
}
trap finish EXIT

if [[ "${MEL_LOCKBOX_EXECUTION_ACK:-}" != "$EXECUTION_ACK" ]]; then
    echo "error: lockbox wrapper is gated; set MEL_LOCKBOX_EXECUTION_ACK=$EXECUTION_ACK only after both exact engineering qualifiers have PASS receipts" >&2
    exit 1
fi
require_env MEL_STREAM_QUAL_RECEIPT
require_env MEL_STREAM_QUAL_RECEIPT_SHA256
require_env MEL_RUNNER_QUAL_RECEIPT
require_env MEL_RUNNER_QUAL_RECEIPT_SHA256
require_env MEL_LOCKBOX_OUTPUT_DIR

stream_receipt="$(realpath "$MEL_STREAM_QUAL_RECEIPT")"
runner_receipt="$(realpath "$MEL_RUNNER_QUAL_RECEIPT")"
output_dir="$(realpath -m "$MEL_LOCKBOX_OUTPUT_DIR")"
admission_path="${output_dir}.admission.tsv"

[[ -f "$stream_receipt" ]] || { echo "error: stream qualification receipt missing" >&2; exit 1; }
[[ -f "$runner_receipt" ]] || { echo "error: runner qualification receipt missing" >&2; exit 1; }
is_sha256 "$MEL_STREAM_QUAL_RECEIPT_SHA256" || { echo "error: stream receipt SHA-256 is noncanonical" >&2; exit 1; }
is_sha256 "$MEL_RUNNER_QUAL_RECEIPT_SHA256" || { echo "error: runner receipt SHA-256 is noncanonical" >&2; exit 1; }
[[ "$(sha256_file "$stream_receipt")" == "$MEL_STREAM_QUAL_RECEIPT_SHA256" ]] || { echo "error: stream qualification receipt SHA-256 mismatch" >&2; exit 1; }
[[ "$(sha256_file "$runner_receipt")" == "$MEL_RUNNER_QUAL_RECEIPT_SHA256" ]] || { echo "error: runner qualification receipt SHA-256 mismatch" >&2; exit 1; }

if [[ -e "$output_dir" ]]; then
    echo "error: MEL_LOCKBOX_OUTPUT_DIR must not already exist" >&2
    exit 1
fi
if [[ -e "$admission_path" ]]; then
    echo "error: admission receipt already exists: $admission_path" >&2
    exit 1
fi

if ! git diff --quiet --ignore-submodules -- || ! git diff --cached --quiet --ignore-submodules --; then
    echo "error: wrapper checkout has tracked or staged modifications" >&2
    exit 1
fi
if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    echo "error: wrapper checkout has untracked source files" >&2
    exit 1
fi

for sha in "$STREAM_SUBJECT_SHA" "$STREAM_QUALIFIER_SHA" "$RUNNER_SUBJECT_SHA" "$RUNNER_QUALIFIER_SHA"; do
    git cat-file -e "${sha}^{commit}" || { echo "error: frozen commit unavailable: $sha" >&2; exit 1; }
done
[[ "$(git rev-parse "${STREAM_QUALIFIER_SHA}^")" == "$STREAM_SUBJECT_SHA" ]] || { echo "error: stream qualifier lineage mismatch" >&2; exit 1; }
[[ "$(git rev-parse "${RUNNER_QUALIFIER_SHA}^")" == "$RUNNER_SUBJECT_SHA" ]] || { echo "error: runner qualifier lineage mismatch" >&2; exit 1; }
[[ "$(git rev-parse "${RUNNER_SUBJECT_SHA}^")" == "$STREAM_SUBJECT_SHA" ]] || { echo "error: runner subject lineage mismatch" >&2; exit 1; }

stream_worktree="$(mktemp -d "${TMPDIR:-/tmp}/mel-lockbox-stream-qual.XXXXXX")"
rmdir "$stream_worktree"
git worktree add --detach "$stream_worktree" "$STREAM_QUALIFIER_SHA" >/dev/null
runner_qual_worktree="$(mktemp -d "${TMPDIR:-/tmp}/mel-lockbox-runner-qual.XXXXXX")"
rmdir "$runner_qual_worktree"
git worktree add --detach "$runner_qual_worktree" "$RUNNER_QUALIFIER_SHA" >/dev/null
runner_worktree="$(mktemp -d "${TMPDIR:-/tmp}/mel-lockbox-runner.XXXXXX")"
rmdir "$runner_worktree"
git worktree add --detach "$runner_worktree" "$RUNNER_SUBJECT_SHA" >/dev/null

(
    cd "$stream_worktree"
    [[ "$(git rev-parse HEAD)" == "$STREAM_QUALIFIER_SHA" ]]
    python3 "$STREAM_VERIFIER" "$stream_receipt"
)
(
    cd "$runner_qual_worktree"
    [[ "$(git rev-parse HEAD)" == "$RUNNER_QUALIFIER_SHA" ]]
    python3 "$RUNNER_VERIFIER" "$runner_receipt"
)

(
    cd "$runner_worktree"
    [[ "$(git rev-parse HEAD)" == "$RUNNER_SUBJECT_SHA" ]]
    declared="$(sed -nE 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' rust-toolchain.toml | head -n1)"
    [[ "$declared" == "$EXPECTED_RUST" ]] || { echo "error: runner subject declared Rust changed" >&2; exit 1; }
    active_release="$(rustc -Vv | awk -F ': ' '$1 == "release" {print $2; exit}')"
    [[ "$active_release" == "$EXPECTED_RUST" ]] || { echo "error: active rustc is $active_release, expected $EXPECTED_RUST" >&2; exit 1; }
    [[ "$(cargo -V)" == cargo\ 1.96.0\ * ]] || { echo "error: active Cargo is not 1.96.0" >&2; exit 1; }
)

wrapper_checkout_sha="$(git rev-parse HEAD)"
wrapper_checkout_tree="$(git rev-parse 'HEAD^{tree}')"
wrapper_script_sha256="$(sha256_file "$root/$SCRIPT_PATH")"
mkdir -p "$(dirname "$admission_path")"
execution_phase="admitted"
write_admission "ADMITTED_NOT_EXECUTED"

execution_phase="runner-started"
write_admission "EXECUTION_STARTED"
(
    cd "$runner_worktree"
    MEL_CONTEXTUAL_HARMONY_LOCKBOX_EXECUTION="EXECUTE_FROZEN_64_SUBJECT_LOCKBOX" \
    MEL_STREAM_QUAL_RECEIPT="$stream_receipt" \
    MEL_STREAM_QUAL_RECEIPT_SHA256="$MEL_STREAM_QUAL_RECEIPT_SHA256" \
    cargo run --locked -p symthaea-muse --features theory --example "$RUNNER_EXAMPLE" -- \
        --output-dir "$output_dir"
)

execution_phase="post-runner-validation"
write_admission "RUNNER_RETURNED_SUCCESS_PENDING_VALIDATION"
[[ -f "$output_dir/run-manifest.json" ]] || { echo "error: runner completed without run-manifest.json" >&2; exit 1; }
[[ -f "$output_dir/run-manifest.sha256" ]] || { echo "error: runner completed without run-manifest.sha256" >&2; exit 1; }
(
    cd "$output_dir"
    sha256sum -c run-manifest.sha256
)
run_manifest_sha256="$(awk 'NR == 1 {print $1}' "$output_dir/run-manifest.sha256")"
is_sha256 "$run_manifest_sha256" || { echo "error: run manifest sidecar is noncanonical" >&2; exit 1; }

write_admission "RUNNER_VALIDATED"
admission_sha256="$(sha256_file "$admission_path")"
{
    printf 'schema\t%s\n' "$WRAPPER_SCHEMA"
    printf 'status\tCOMPLETED\n'
    printf 'admission_receipt_sha256\t%s\n' "$admission_sha256"
    printf 'wrapper_checkout_sha\t%s\n' "$wrapper_checkout_sha"
    printf 'wrapper_checkout_tree\t%s\n' "$wrapper_checkout_tree"
    printf 'wrapper_script_sha256\t%s\n' "$wrapper_script_sha256"
    printf 'stream_subject_sha\t%s\n' "$STREAM_SUBJECT_SHA"
    printf 'stream_qualifier_sha\t%s\n' "$STREAM_QUALIFIER_SHA"
    printf 'stream_qualification_receipt_sha256\t%s\n' "$MEL_STREAM_QUAL_RECEIPT_SHA256"
    printf 'runner_subject_sha\t%s\n' "$RUNNER_SUBJECT_SHA"
    printf 'runner_qualifier_sha\t%s\n' "$RUNNER_QUALIFIER_SHA"
    printf 'runner_qualification_receipt_sha256\t%s\n' "$MEL_RUNNER_QUAL_RECEIPT_SHA256"
    printf 'run_manifest_sha256\t%s\n' "$run_manifest_sha256"
    printf 'raw_pcm_persisted\tfalse\n'
    printf 'primary_unit\tmotif-seed-subject\n'
    printf 'scientific_scope\tdescriptive-frozen-lockbox-machine-evidence-only\n'
    printf 'human_perceptual_authority\tnone\n'
    printf 'artistic_quality_authority\tnone\n'
    printf 'product_authority\tnone\n'
} > "$output_dir/execution-capsule.tsv"
execution_capsule_sha256="$(sha256_file "$output_dir/execution-capsule.tsv")"
printf '%s  execution-capsule.tsv\n' "$execution_capsule_sha256" > "$output_dir/execution-capsule.sha256"

execution_phase="complete"
printf 'lockbox execution wrapper complete\n'
printf 'admission_receipt=%s\n' "$admission_path"
printf 'run_manifest_sha256=%s\n' "$run_manifest_sha256"
printf 'execution_capsule_sha256=%s\n' "$execution_capsule_sha256"
