#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Final execution wrapper for the frozen ProgSuite contextual-harmony lockbox.
# It verifies both prerequisite qualification receipts in detached worktrees at
# their exact qualifier SHAs, enforces the same Rust/Cargo/lockfile identity for
# scientific execution, runs the exact frozen runner subject, verifies every
# compact result artifact, and emits an outer execution receipt.

set -euo pipefail

STREAM_QUALIFIER_SHA="b9834ed86fc84506fdb1e42b421be339fd482bfb"
STREAM_SUBJECT_SHA="d888341323b6ee463007cf90d8032153039ec099"
STREAM_VERIFIER=".github/qualification/verify-melothaea-tonal-survival-stream-receipt-v1.py"
RUNNER_QUALIFIER_SHA="0e042a07f1409d453d27c9530d5e90840d6dd281"
RUNNER_SUBJECT_SHA="95e5bd949033215d09b4b9edf4d0e490ba823a3c"
RUNNER_VERIFIER=".github/qualification/verify-melothaea-lockbox-runner-receipt-v1.py"
RUNNER_SOURCE="crates/domains/symthaea-muse/examples/prog_suite_contextual_harmony_lockbox_runner.rs"
EXAMPLE="prog_suite_contextual_harmony_lockbox_runner"
EXECUTION_ACK="EXECUTE_FROZEN_64_SUBJECT_LOCKBOX"
EXPECTED_RUST="1.96.0"

root="$(git rev-parse --show-toplevel)"
cd "$root"
wrapper_sha="$(git rev-parse HEAD)"
wrapper_tree="$(git rev-parse 'HEAD^{tree}')"
wrapper_path=".github/qualification/execute-melothaea-contextual-harmony-lockbox-v1.sh"
wrapper_source_sha256="$(sha256sum "$wrapper_path" | awk '{print $1}')"
receipt_path="${MEL_LOCKBOX_EXECUTION_RECEIPT:-${TMPDIR:-/tmp}/melothaea-contextual-harmony-lockbox-execution-v1.tsv}"

stream_receipt=""
runner_receipt=""
output_dir=""
preflight_only="false"
stream_receipt_sha256="unavailable"
runner_receipt_sha256="unavailable"
runner_source_sha256="unavailable"
run_manifest_sha256="unavailable"
panel_sha256="unavailable"
persisted_subject_receipt_count="0"
rustc_release="unavailable"
rustc_commit="unavailable"
rustc_host="unavailable"
cargo_version="unavailable"
cargo_lock_sha256="unavailable"
rust_toolchain_sha256="unavailable"
execution_host="unavailable"
status="FAIL"
stage="parse_arguments"
stream_wt=""
runner_qual_wt=""
runner_wt=""

usage() {
    cat >&2 <<'EOF'
usage: execute-melothaea-contextual-harmony-lockbox-v1.sh \
  --stream-receipt PATH \
  --runner-receipt PATH \
  --output-dir FRESH_PATH \
  [--preflight-only]

--preflight-only performs all receipt/source/toolchain/lockfile checks but never
sets the lockbox execution acknowledgement and never invokes the runner.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stream-receipt)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            stream_receipt="$2"; shift 2 ;;
        --runner-receipt)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            runner_receipt="$2"; shift 2 ;;
        --output-dir)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            output_dir="$2"; shift 2 ;;
        --preflight-only)
            preflight_only="true"; shift ;;
        *)
            usage
            printf 'error: unknown argument: %s\n' "$1" >&2
            exit 2 ;;
    esac
done

[[ -n "$stream_receipt" && -n "$runner_receipt" && -n "$output_dir" ]] || {
    usage
    exit 2
}
stream_receipt="$(realpath "$stream_receipt")"
runner_receipt="$(realpath "$runner_receipt")"
output_dir="$(realpath -m "$output_dir")"
receipt_path="$(realpath -m "$receipt_path")"
for candidate in "$output_dir" "$receipt_path"; do
    case "$candidate" in
        "$root"|"$root"/*)
            echo 'error: execution outputs/receipts must live outside the repository checkout' >&2
            exit 2 ;;
    esac
done

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
receipt_value() {
    local key="$1" file="$2"
    awk -F '\t' -v key="$key" '$1 == key {print $2}' "$file"
}
count_subject_receipts() {
    if [[ -d "$output_dir/subjects" ]]; then
        find "$output_dir/subjects" -type f -name subject-receipt.json -print 2>/dev/null \
            | wc -l | tr -d ' '
    else
        printf '0'
    fi
}

write_receipt() {
    local rc="$1" final_status="$status" authority="none" lockbox_state="not-completed"
    local tmp="${receipt_path}.tmp.$$"
    persisted_subject_receipt_count="$(count_subject_receipts)"
    [[ "$rc" -eq 0 ]] || final_status="FAIL"
    if [[ "$final_status" == "PASS" ]]; then
        authority="frozen-lockbox-machine-evidence-execution"
        lockbox_state="completed-64-subjects"
    elif [[ "$final_status" == "PREFLIGHT_PASS" ]]; then
        authority="execution-preflight-only"
        lockbox_state="not-performed"
    fi
    mkdir -p "$(dirname "$receipt_path")" || return 1
    {
        printf 'schema\tmelothaea-contextual-harmony-lockbox-execution-receipt-v1\n'
        printf 'status\t%s\n' "$final_status"
        printf 'exit_code\t%s\n' "$rc"
        printf 'terminal_stage\t%s\n' "$stage"
        printf 'authority_scope\t%s\n' "$authority"
        printf 'lockbox_execution\t%s\n' "$lockbox_state"
        printf 'persisted_subject_receipt_count\t%s\n' "$persisted_subject_receipt_count"
        printf 'primary_unit\tmotif-seed-subject\n'
        printf 'human_perceptual_authority\tnone\n'
        printf 'artistic_quality_authority\tnone\n'
        printf 'product_authority\tnone\n'
        printf 'wrapper_checkout_sha\t%s\n' "$wrapper_sha"
        printf 'wrapper_checkout_tree\t%s\n' "$wrapper_tree"
        printf 'wrapper_source_sha256\t%s\n' "$wrapper_source_sha256"
        printf 'stream_qualifier_sha\t%s\n' "$STREAM_QUALIFIER_SHA"
        printf 'stream_subject_sha\t%s\n' "$STREAM_SUBJECT_SHA"
        printf 'stream_receipt_sha256\t%s\n' "$stream_receipt_sha256"
        printf 'runner_qualifier_sha\t%s\n' "$RUNNER_QUALIFIER_SHA"
        printf 'runner_subject_sha\t%s\n' "$RUNNER_SUBJECT_SHA"
        printf 'runner_source_sha256\t%s\n' "$runner_source_sha256"
        printf 'runner_receipt_sha256\t%s\n' "$runner_receipt_sha256"
        printf 'expected_rust_release\t%s\n' "$EXPECTED_RUST"
        printf 'rustc_release\t%s\n' "$rustc_release"
        printf 'rustc_commit_hash\t%s\n' "$rustc_commit"
        printf 'rustc_host\t%s\n' "$rustc_host"
        printf 'cargo_version\t%s\n' "$cargo_version"
        printf 'cargo_lock_sha256\t%s\n' "$cargo_lock_sha256"
        printf 'rust_toolchain_sha256\t%s\n' "$rust_toolchain_sha256"
        printf 'execution_host\t%s\n' "$execution_host"
        printf 'run_manifest_sha256\t%s\n' "$run_manifest_sha256"
        printf 'tonal_panel_sha256\t%s\n' "$panel_sha256"
        printf 'output_directory\t%s\n' "$output_dir"
    } > "$tmp"
    mv "$tmp" "$receipt_path"
    printf 'melothaea lockbox execution receipt=%s status=%s stage=%s\n' \
        "$receipt_path" "$final_status" "$stage"
}

cleanup() {
    cd "$root" 2>/dev/null || true
    for wt in "$stream_wt" "$runner_qual_wt" "$runner_wt"; do
        if [[ -n "$wt" && -d "$wt" ]]; then
            git worktree remove --force "$wt" >/dev/null 2>&1 || true
        fi
    done
}
finish() {
    local rc=$?
    trap - EXIT
    write_receipt "$rc" || rc=1
    cleanup
    exit "$rc"
}
trap finish EXIT

stage="clean_wrapper_checkout"
git diff --quiet --ignore-submodules --
git diff --cached --quiet --ignore-submodules --
[[ -z "$(git ls-files --others --exclude-standard)" ]] || {
    echo 'error: wrapper checkout contains untracked files' >&2
    exit 1
}

stage="required_git_objects"
for sha in "$STREAM_QUALIFIER_SHA" "$STREAM_SUBJECT_SHA" "$RUNNER_QUALIFIER_SHA" "$RUNNER_SUBJECT_SHA"; do
    git cat-file -e "${sha}^{commit}" || {
        echo "error: required frozen commit unavailable locally: $sha" >&2
        echo 'fetch the corresponding qualification/execution branches before retrying' >&2
        exit 1
    }
done

stage="verify_streaming_library_receipt"
stream_wt="$(mktemp -d "${TMPDIR:-/tmp}/mel-stream-receipt.XXXXXX")"; rmdir "$stream_wt"
git worktree add --detach "$stream_wt" "$STREAM_QUALIFIER_SHA" >/dev/null
python3 "$stream_wt/$STREAM_VERIFIER" "$stream_receipt" --repo "$stream_wt"
stream_receipt_sha256="$(sha256_file "$stream_receipt")"
qualified_lock_sha="$(receipt_value cargo_lock_sha256 "$stream_receipt")"
qualified_toolchain_sha="$(receipt_value rust_toolchain_sha256 "$stream_receipt")"
[[ "$qualified_lock_sha" =~ ^[0-9a-f]{64}$ && "$qualified_toolchain_sha" =~ ^[0-9a-f]{64}$ ]] || {
    echo 'error: verified streaming receipt lacks canonical environment digests' >&2
    exit 1
}
git worktree remove --force "$stream_wt" >/dev/null; stream_wt=""

stage="verify_runner_receipt"
runner_qual_wt="$(mktemp -d "${TMPDIR:-/tmp}/mel-runner-receipt.XXXXXX")"; rmdir "$runner_qual_wt"
git worktree add --detach "$runner_qual_wt" "$RUNNER_QUALIFIER_SHA" >/dev/null
python3 "$runner_qual_wt/$RUNNER_VERIFIER" "$runner_receipt" --repo "$runner_qual_wt"
runner_receipt_sha256="$(sha256_file "$runner_receipt")"
runner_source_sha256="$(receipt_value runner_source_sha256 "$runner_receipt")"
[[ "$runner_source_sha256" =~ ^[0-9a-f]{64}$ ]] || {
    echo 'error: verified runner receipt lacks canonical source digest' >&2
    exit 1
}
git worktree remove --force "$runner_qual_wt" >/dev/null; runner_qual_wt=""

stage="prepare_exact_runner_worktree"
runner_wt="$(mktemp -d "${TMPDIR:-/tmp}/mel-lockbox-runner.XXXXXX")"; rmdir "$runner_wt"
git worktree add --detach "$runner_wt" "$RUNNER_SUBJECT_SHA" >/dev/null
[[ "$(git -C "$runner_wt" rev-parse HEAD)" == "$RUNNER_SUBJECT_SHA" ]]
[[ "$(sha256_file "$runner_wt/$RUNNER_SOURCE")" == "$runner_source_sha256" ]] || {
    echo 'error: runner source bytes differ from qualified runner receipt' >&2
    exit 1
}

stage="execution_environment_identity"
rustc_verbose="$(rustc -Vv)"
rustc_release="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "release" {print $2; exit}')"
rustc_commit="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "commit-hash" {print $2; exit}')"
rustc_host="$(printf '%s\n' "$rustc_verbose" | awk -F ': ' '$1 == "host" {print $2; exit}')"
cargo_version="$(cargo -V)"
[[ "$rustc_release" == "$EXPECTED_RUST" ]] || {
    echo "error: execution rustc=$rustc_release expected=$EXPECTED_RUST" >&2
    exit 1
}
[[ "$cargo_version" == cargo\ 1.96.0\ * ]] || {
    echo "error: execution Cargo is not 1.96.0: $cargo_version" >&2
    exit 1
}
cargo_lock_sha256="$(sha256_file "$runner_wt/Cargo.lock")"
rust_toolchain_sha256="$(sha256_file "$runner_wt/rust-toolchain.toml")"
[[ "$cargo_lock_sha256" == "$qualified_lock_sha" ]] || {
    echo 'error: execution Cargo.lock differs from qualified streaming subject' >&2
    exit 1
}
[[ "$rust_toolchain_sha256" == "$qualified_toolchain_sha" ]] || {
    echo 'error: execution rust-toolchain.toml differs from qualified streaming subject' >&2
    exit 1
}
execution_host="$(uname -srm | tr '\t\n' '  ')"

if [[ "$preflight_only" == "true" ]]; then
    stage="preflight_complete"
    status="PREFLIGHT_PASS"
    exit 0
fi

stage="execute_frozen_lockbox"
MEL_CONTEXTUAL_HARMONY_LOCKBOX_EXECUTION="$EXECUTION_ACK" \
MEL_STREAM_QUAL_RECEIPT="$stream_receipt" \
MEL_STREAM_QUAL_RECEIPT_SHA256="$stream_receipt_sha256" \
cargo --manifest-path "$runner_wt/Cargo.toml" run --quiet --locked \
    -p symthaea-muse --features theory --example "$EXAMPLE" -- \
    --output-dir "$output_dir"

stage="verify_output_artifacts"
python3 - "$output_dir" "$stream_receipt_sha256" "$runner_source_sha256" <<'PY'
import hashlib
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
stream_receipt_sha = sys.argv[2]
runner_source_sha = sys.argv[3]


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))

expected_top = {
    "audio-protocol.json", "run-start.json", "subjects",
    "tonal-survival-panel.json", "run-manifest.json", "run-manifest.sha256",
}
actual_top = {p.name for p in out.iterdir()}
if actual_top != expected_top:
    raise SystemExit(f"noncanonical top-level output set: {sorted(actual_top)}")

manifest_path = out / "run-manifest.json"
manifest = load(manifest_path)
manifest_sha = digest(manifest_path)
sidecar = (out / "run-manifest.sha256").read_text(encoding="utf-8").strip().split()
if sidecar != [manifest_sha, "run-manifest.json"]:
    raise SystemExit("run-manifest SHA-256 sidecar mismatch")
if manifest.get("schema") != "melothaea-prog-suite-contextual-harmony-lockbox-execution-v1":
    raise SystemExit("wrong run manifest schema")
if manifest.get("qualified_stream_subject_sha") != "d888341323b6ee463007cf90d8032153039ec099":
    raise SystemExit("run manifest stream subject mismatch")
if manifest.get("qualification_receipt_sha256") != stream_receipt_sha:
    raise SystemExit("run manifest qualification receipt mismatch")
if manifest.get("runner_source_sha256") != runner_source_sha:
    raise SystemExit("run manifest runner source mismatch")
if manifest.get("subject_count") != 64 or len(manifest.get("subject_artifacts", [])) != 64:
    raise SystemExit("run manifest does not contain exactly 64 subjects")
if manifest.get("raw_pcm_persisted") is not False:
    raise SystemExit("run manifest claims raw PCM persistence")
for key in ("perceptual_authority", "artistic_quality_authority", "product_authority"):
    if manifest.get(key) != "none":
        raise SystemExit(f"unexpected authority escalation: {key}")

if digest(out / "audio-protocol.json") != manifest.get("audio_protocol_sha256"):
    raise SystemExit("audio protocol digest mismatch")
if digest(out / "tonal-survival-panel.json") != manifest.get("tonal_panel_sha256"):
    raise SystemExit("tonal panel digest mismatch")
run_start = load(out / "run-start.json")
if run_start.get("qualification_receipt_sha256") != stream_receipt_sha:
    raise SystemExit("run-start qualification receipt mismatch")
if run_start.get("runner_source_sha256") != runner_source_sha:
    raise SystemExit("run-start runner source mismatch")
if run_start.get("intended_subject_count") != 64:
    raise SystemExit("run-start subject count mismatch")

subject_root = out / "subjects"
expected_dirs = {f"{i:02d}" for i in range(64)}
actual_dirs = {p.name for p in subject_root.iterdir() if p.is_dir()}
if actual_dirs != expected_dirs or any(p.is_file() for p in subject_root.iterdir()):
    raise SystemExit("subject directory roster mismatch")

seen = []
for item in manifest["subject_artifacts"]:
    idx = item.get("subject_index")
    if not isinstance(idx, int) or not 0 <= idx < 64:
        raise SystemExit("invalid subject index")
    seen.append(idx)
    directory = subject_root / f"{idx:02d}"
    expected_files = {"symbolic-comparison.json", "tonal-evidence.json", "subject-receipt.json"}
    if {p.name for p in directory.iterdir()} != expected_files:
        raise SystemExit(f"subject {idx} file set mismatch")
    if digest(directory / "symbolic-comparison.json") != item.get("comparison_sha256"):
        raise SystemExit(f"subject {idx} comparison digest mismatch")
    if digest(directory / "tonal-evidence.json") != item.get("tonal_evidence_sha256"):
        raise SystemExit(f"subject {idx} tonal evidence digest mismatch")
    if load(directory / "subject-receipt.json") != item:
        raise SystemExit(f"subject {idx} compact receipt mismatch")
if seen != list(range(64)):
    raise SystemExit("manifest subject ordering is noncanonical")

print(f"verified lockbox output manifest_sha256={manifest_sha}")
print(f"verified tonal panel sha256={manifest['tonal_panel_sha256']}")
PY

run_manifest_sha256="$(sha256_file "$output_dir/run-manifest.json")"
panel_sha256="$(sha256_file "$output_dir/tonal-survival-panel.json")"
persisted_subject_receipt_count="$(count_subject_receipts)"
[[ "$persisted_subject_receipt_count" == "64" ]] || {
    echo 'error: output does not contain exactly 64 admitted subject receipts' >&2
    exit 1
}

stage="complete"
status="PASS"
