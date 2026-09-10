#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Focused exact-head software-contract qualification for the typed continuity
# subject/snapshot lineage. This establishes software properties only; it does
# not establish physical existence, current availability, ownership, policy,
# recovery sufficiency, or execution authority.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

actual_sha="$(git rev-parse HEAD)"
head_tree="$(git rev-parse 'HEAD^{tree}')"
expected_sha="${QUALIFIED_SHA:-$actual_sha}"
receipt_path="${SUBJECT_SNAPSHOT_RECEIPT:-${TMPDIR:-/tmp}/continuity-subject-snapshot-qualification-v1.tsv}"
status="FAIL"
stage="preflight"
source_state="unverified"
strict_clippy_exit="not-run"
legacy_lint_diagnostic_count="not-run"
legacy_lint_allowlist="dead_code@compose.rs,exact_policy.rs,verifier.rs,witness.rs;clippy::too_many_arguments@observation.rs"

scope_source="crates/core/symthaea-continuity/src/scope.rs"
snapshot_source="crates/core/symthaea-continuity/src/subject_snapshot.rs"
format_scope="$scope_source;$snapshot_source"

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
    local rustc_verbose=""
    local rustc_release="unavailable"
    local rustc_commit="unavailable"
    local rustc_host="unavailable"
    local cargo_version="unavailable"
    local rustfmt_version="unavailable"
    local tmp_path="${receipt_path}.tmp.$$"

    set +e
    [[ "$exit_code" -eq 0 ]] || final_status="FAIL"
    [[ "$final_status" != "PASS" ]] || terminal_stage="none"

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
    if command -v rustfmt >/dev/null 2>&1; then
        rustfmt_version="$(rustfmt -V 2>/dev/null || printf 'unavailable')"
    fi

    mkdir -p "$(dirname "$receipt_path")" || return 1
    {
        printf 'schema\tcontinuity-subject-snapshot-qualification-v1\n'
        printf 'status\t%s\n' "$final_status"
        printf 'exit_code\t%s\n' "$exit_code"
        printf 'terminal_stage\t%s\n' "$terminal_stage"
        printf 'scope\ttyped-subject-snapshot-software-contract-only\n'
        printf 'format_scope\t%s\n' "$format_scope"
        printf 'format_edition\t2024\n'
        printf 'clippy_policy\tstrict-inventory-plus-explicit-legacy-baseline\n'
        printf 'strict_clippy_exit\t%s\n' "$strict_clippy_exit"
        printf 'legacy_lint_allowlist\t%s\n' "$legacy_lint_allowlist"
        printf 'legacy_lint_diagnostic_count\t%s\n' "$legacy_lint_diagnostic_count"
        printf 'physical_existence_authority\tnone\n'
        printf 'availability_authority\tnone\n'
        printf 'ownership_authority\tnone\n'
        printf 'policy_authority\tnone\n'
        printf 'recovery_sufficiency_authority\tnone\n'
        printf 'execution_authority\tnone\n'
        printf 'full_repository_ci\tindependent\n'
        printf 'receipt_attestation\tnone\n'
        printf 'qualified_sha\t%s\n' "$actual_sha"
        printf 'expected_sha\t%s\n' "$expected_sha"
        printf 'committed_tree\t%s\n' "$head_tree"
        printf 'source_state\t%s\n' "$source_state"
        printf 'runner_label\t%s\n' "${SUBJECT_RUNNER_LABEL:-unknown}"
        printf 'runner_os\t%s\n' "${RUNNER_OS:-unknown}"
        printf 'runner_arch\t%s\n' "${RUNNER_ARCH:-unknown}"
        printf 'runner_image_os\t%s\n' "${ImageOS:-unknown}"
        printf 'runner_image_version\t%s\n' "${ImageVersion:-unknown}"
        printf 'kernel_release\t%s\n' "$(uname -r 2>/dev/null || printf 'unavailable')"
        printf 'rustc_release\t%s\n' "$rustc_release"
        printf 'rustc_commit_hash\t%s\n' "$rustc_commit"
        printf 'rustc_host\t%s\n' "$rustc_host"
        printf 'cargo_version\t%s\n' "$cargo_version"
        printf 'rustfmt_version\t%s\n' "$rustfmt_version"
        printf 'cargo_lock_sha256\t%s\n' "$(sha256_file Cargo.lock)"
        printf 'workspace_manifest_sha256\t%s\n' "$(sha256_file Cargo.toml)"
        printf 'continuity_manifest_sha256\t%s\n' "$(sha256_file crates/core/symthaea-continuity/Cargo.toml)"
        printf 'subject_scope_source_sha256\t%s\n' "$(sha256_file "$scope_source")"
        printf 'subject_snapshot_source_sha256\t%s\n' "$(sha256_file "$snapshot_source")"
        printf 'rust_toolchain_sha256\t%s\n' "$(sha256_file rust-toolchain.toml)"
        printf 'qualifier_script_sha256\t%s\n' "$(sha256_file scripts/qualify-continuity-subject-snapshot.sh)"
        printf 'workflow_sha256\t%s\n' "$(sha256_file .github/workflows/continuity-subject-snapshot.yml)"
        printf 'github_run_id\t%s\n' "${GITHUB_RUN_ID:-not-applicable}"
        printf 'github_run_attempt\t%s\n' "${GITHUB_RUN_ATTEMPT:-not-applicable}"
    } > "$tmp_path" || { rm -f "$tmp_path"; return 1; }
    mv "$tmp_path" "$receipt_path" || { rm -f "$tmp_path"; return 1; }

    echo "continuity-subject-snapshot receipt=$receipt_path status=$final_status stage=$terminal_stage"
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        {
            echo '## Continuity Subject Snapshot Qualification'
            echo
            echo "- status: **$final_status**"
            echo "- qualified SHA: \`$actual_sha\`"
            echo "- committed tree: \`$head_tree\`"
            echo "- terminal stage: \`$terminal_stage\`"
            echo "- source state: \`$source_state\`"
            echo "- format scope: \`$format_scope\` (Edition 2024)"
            echo "- strict Clippy exit: \`$strict_clippy_exit\`"
            echo "- accepted legacy lint diagnostics: \`$legacy_lint_diagnostic_count\`"
            echo '- scope: typed subject/snapshot software contracts only'
            echo '- full repository CI: independent'
            echo '- physical existence/availability/ownership/policy/recovery/execution authority: none'
        } >> "$GITHUB_STEP_SUMMARY" || true
    fi
}

finish() {
    local exit_code=$?
    trap - EXIT
    if [[ "$exit_code" -eq 0 && "$status" != "PASS" ]]; then
        echo "error: verifier exited without terminal PASS (stage=$stage)" >&2
        exit_code=1
    fi
    write_receipt "$exit_code" || {
        echo "error: qualification receipt could not be persisted" >&2
        [[ "$exit_code" -ne 0 ]] || exit_code=1
    }
    exit "$exit_code"
}
trap finish EXIT

stage="preflight_exact_head"
if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "error: exact-head mismatch expected=$expected_sha actual=$actual_sha" >&2
    exit 1
fi

stage="preflight_sources_present"
for path in \
    crates/core/symthaea-continuity/Cargo.toml \
    "$scope_source" \
    "$snapshot_source"; do
    [[ -f "$path" ]] || { echo "error: required source absent: $path" >&2; exit 1; }
done

stage="preflight_manifest_contract"
grep -Eq '^edition[[:space:]]*=[[:space:]]*"2024"[[:space:]]*$' crates/core/symthaea-continuity/Cargo.toml || {
    echo 'error: focused formatter contract expects symthaea-continuity Edition 2024' >&2
    exit 1
}

stage="preflight_clean_tree"
if ! git diff --quiet --ignore-submodules -- || ! git diff --cached --quiet --ignore-submodules --; then
    source_state="tracked-or-staged-modifications-present"
    exit 1
fi
untracked="$(git ls-files --others --exclude-standard)"
if [[ -n "$untracked" ]]; then
    source_state="untracked-source-present"
    printf '%s\n' "$untracked" >&2
    exit 1
fi
source_state="clean-exact-checkout"

stage="cargo_metadata"
cargo metadata --locked --no-deps --format-version 1 >/dev/null

# Deliberately format only the sources named by this theorem. Package-wide
# `cargo fmt -p` would make unrelated continuity modules part of this
# qualification boundary and can therefore create false failures.
stage="format_subject_sources"
rustfmt --edition 2024 --check "$scope_source" "$snapshot_source"

stage="check_all_targets"
cargo check --locked -p symthaea-continuity --all-targets

# First run Clippy with the strictest policy and parse its diagnostics. Only the
# independently bounded legacy baseline already qualified by the continuity
# contract lane may be accepted. Any lint from scope.rs or subject_snapshot.rs,
# or any new lint elsewhere, fails this stage rather than being globally hidden.
stage="clippy_strict_inventory"
strict_clippy_json="${TMPDIR:-/tmp}/symthaea-subject-snapshot-clippy-${actual_sha}.jsonl"
set +e
cargo clippy --locked -p symthaea-continuity --all-targets --message-format=json -- -D warnings >"$strict_clippy_json" 2>&1
strict_clippy_exit=$?
set -e

if [[ "$strict_clippy_exit" -eq 0 ]]; then
    legacy_lint_diagnostic_count="0"
else
    legacy_lint_diagnostic_count="$(python3 - "$strict_clippy_json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
allowed_dead_code_files = {"compose.rs", "exact_policy.rs", "verifier.rs", "witness.rs"}
allowed_pairs = {("clippy::too_many_arguments", "observation.rs")}
count = 0
unexpected = []
for raw in path.read_text(errors="replace").splitlines():
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        continue
    if event.get("reason") != "compiler-message":
        continue
    message = event.get("message") or {}
    if message.get("level") != "error":
        continue
    code = (message.get("code") or {}).get("code")
    primary = next((span for span in message.get("spans", []) if span.get("is_primary")), None)
    file_name = pathlib.PurePosixPath((primary or {}).get("file_name", "")).name
    rendered = (message.get("message") or "").replace("\n", " ")
    count += 1
    allowed = code == "dead_code" and file_name in allowed_dead_code_files
    allowed = allowed or (code, file_name) in allowed_pairs
    if not allowed:
        unexpected.append((code or "<none>", file_name or "<none>", rendered))
if count == 0:
    print("error: strict Clippy failed without parseable error diagnostics", file=sys.stderr)
    sys.exit(2)
if unexpected:
    print("error: strict Clippy produced diagnostics outside explicit legacy baseline", file=sys.stderr)
    for code, file_name, rendered in unexpected:
        print(f"  {code}@{file_name}: {rendered}", file=sys.stderr)
    sys.exit(3)
print(count)
PY
)"
fi

stage="clippy_all_targets_with_legacy_baseline"
cargo clippy --locked -p symthaea-continuity --all-targets -- \
    -A dead_code \
    -A clippy::too_many_arguments \
    -D warnings

stage="subject_snapshot_regressions"
cargo test --locked -p symthaea-continuity subject_snapshot -- --nocapture
cargo test --locked -p symthaea-continuity --lib 'subject_snapshot::tests::parent_structure_fails_closed_without_closure_precondition' -- --exact --nocapture

stage="package_tests"
cargo test --locked -p symthaea-continuity

stage="doc_tests"
cargo test --locked -p symthaea-continuity --doc

stage="postflight_source_immutability"
if [[ "$(git rev-parse HEAD)" != "$actual_sha" ]]; then
    source_state="head-changed-during-tests"
    exit 1
fi
if ! git diff --quiet --ignore-submodules -- || ! git diff --cached --quiet --ignore-submodules --; then
    source_state="tracked-or-staged-source-mutated-during-tests"
    exit 1
fi
post_untracked="$(git ls-files --others --exclude-standard)"
if [[ -n "$post_untracked" ]]; then
    source_state="untracked-source-created-during-tests"
    printf '%s\n' "$post_untracked" >&2
    exit 1
fi
source_state="clean-exact-checkout-postflight"

status="PASS"
stage="complete"
