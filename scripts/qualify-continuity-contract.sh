#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Focused exact-head software-contract qualification for symthaea-continuity.
#
# This lane proves only that one exact committed source tree satisfies the
# focused continuity crate's formatting/compiler/lint/test contracts. It does
# not establish real-world availability, empirical/scientific evidence, policy
# approval, or execution authority.
#
# The crate currently contains staged pre-capability code that is intentionally
# not wired into its public runtime path yet. Strict Clippy diagnostics are
# inventoried first and may be baselined only when both their lint code and
# source file match the narrow legacy allowlist below. Every other diagnostic
# remains fatal. The exact baseline and observed diagnostic count are recorded
# in the qualification receipt.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

actual_sha="$(git rev-parse HEAD)"
head_tree="$(git rev-parse 'HEAD^{tree}')"
expected_sha="${QUALIFIED_SHA:-$actual_sha}"
receipt_path="${CONTINUITY_CONTRACT_RECEIPT:-${TMPDIR:-/tmp}/symthaea-continuity-contract-qualification-v1.tsv}"
status="FAIL"
stage="preflight"
source_state="unverified"
strict_clippy_exit="not-run"
legacy_lint_diagnostic_count="not-run"
legacy_lint_allowlist="dead_code@compose.rs,exact_policy.rs,verifier.rs,witness.rs;clippy::too_many_arguments@observation.rs"

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
    local tmp_path="${receipt_path}.tmp.$$"

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

    mkdir -p "$(dirname "$receipt_path")" || return 1
    {
        printf 'schema\tsymthaea-continuity-contract-qualification-v1\n'
        printf 'status\t%s\n' "$final_status"
        printf 'exit_code\t%s\n' "$exit_code"
        printf 'terminal_stage\t%s\n' "$failure_stage"
        printf 'scope\tcontinuity-software-contract-only\n'
        printf 'real_world_availability_authority\tnone\n'
        printf 'scientific_authority\tnone\n'
        printf 'execution_authority\tnone\n'
        printf 'full_repository_ci\tindependent\n'
        printf 'receipt_attestation\tnone\n'
        printf 'clippy_policy\tstrict-inventory-plus-explicit-legacy-baseline\n'
        printf 'strict_clippy_exit\t%s\n' "$strict_clippy_exit"
        printf 'legacy_lint_allowlist\t%s\n' "$legacy_lint_allowlist"
        printf 'legacy_lint_diagnostic_count\t%s\n' "$legacy_lint_diagnostic_count"
        printf 'qualified_sha\t%s\n' "$actual_sha"
        printf 'expected_sha\t%s\n' "$expected_sha"
        printf 'committed_tree\t%s\n' "$head_tree"
        printf 'source_state\t%s\n' "$source_state"
        printf 'execution_provider\t%s\n' "$provider"
        printf 'runner_label\t%s\n' "${CONTINUITY_RUNNER_LABEL:-unknown}"
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
        printf 'continuity_manifest_sha256\t%s\n' "$(sha256_file crates/core/symthaea-continuity/Cargo.toml)"
        printf 'rust_toolchain_sha256\t%s\n' "$(sha256_file rust-toolchain.toml)"
        printf 'qualifier_script_sha256\t%s\n' "$(sha256_file scripts/qualify-continuity-contract.sh)"
        printf 'workflow_sha256\t%s\n' "$(sha256_file .github/workflows/continuity-contract.yml)"
        printf 'github_event_name\t%s\n' "${GITHUB_EVENT_NAME:-not-applicable}"
        printf 'github_repository\t%s\n' "${GITHUB_REPOSITORY:-not-applicable}"
        printf 'github_workflow_ref\t%s\n' "${GITHUB_WORKFLOW_REF:-not-applicable}"
        printf 'github_job\t%s\n' "${GITHUB_JOB:-not-applicable}"
        printf 'github_run_id\t%s\n' "${GITHUB_RUN_ID:-not-applicable}"
        printf 'github_run_attempt\t%s\n' "${GITHUB_RUN_ATTEMPT:-not-applicable}"
    } > "$tmp_path" || {
        rm -f "$tmp_path"
        return 1
    }
    mv "$tmp_path" "$receipt_path" || {
        rm -f "$tmp_path"
        return 1
    }

    echo "continuity-contract receipt=$receipt_path status=$final_status stage=$failure_stage"

    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        {
            echo '## Continuity Contract Qualification'
            echo
            echo "- status: **$final_status**"
            echo "- qualified SHA: \`$actual_sha\`"
            echo "- committed tree: \`$head_tree\`"
            echo "- terminal stage: \`$failure_stage\`"
            echo "- source state: \`$source_state\`"
            echo "- strict Clippy exit: \`$strict_clippy_exit\`"
            echo "- accepted legacy lint diagnostics: \`$legacy_lint_diagnostic_count\`"
            echo "- legacy lint allowlist: \`$legacy_lint_allowlist\`"
            echo '- scope: continuity software contracts only'
            echo '- full repository CI: independent'
            echo '- real-world availability/scientific/execution authority: none'
            echo '- receipt attestation: none (workflow run/artifact is external witness)'
        } >> "$GITHUB_STEP_SUMMARY" || true
    fi
    return 0
}

finish() {
    local exit_code=$?
    trap - EXIT

    if [[ "$exit_code" -eq 0 && "$status" != "PASS" ]]; then
        echo "error: verifier exited without terminal PASS (stage=$stage)" >&2
        exit_code=1
    fi

    if ! write_receipt "$exit_code"; then
        echo "error: qualification receipt could not be persisted to $receipt_path" >&2
        if [[ "$exit_code" -eq 0 ]]; then
            exit_code=1
        fi
    fi
    exit "$exit_code"
}
trap finish EXIT

stage="preflight_exact_head"
if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "error: qualified head mismatch: expected=$expected_sha actual=$actual_sha" >&2
    exit 1
fi

stage="preflight_crate_present"
if [[ ! -f crates/core/symthaea-continuity/Cargo.toml ]]; then
    echo 'error: symthaea-continuity manifest is absent from exact qualified head' >&2
    exit 1
fi

stage="preflight_clean_tree"
if ! git diff --quiet --ignore-submodules --; then
    source_state="tracked-modifications-present"
    echo 'error: tracked working-tree changes present' >&2
    exit 1
fi
if ! git diff --cached --quiet --ignore-submodules --; then
    source_state="staged-modifications-present"
    echo 'error: staged changes present' >&2
    exit 1
fi
untracked="$(git ls-files --others --exclude-standard)"
if [[ -n "$untracked" ]]; then
    source_state="untracked-source-present"
    echo 'error: untracked files present; exact-head qualification requires a clean source tree' >&2
    printf '%s\n' "$untracked" >&2
    exit 1
fi
source_state="clean-exact-checkout"

echo "continuity-contract qualified_sha=$actual_sha"
echo "continuity-contract committed_tree=$head_tree"
rustc -Vv
cargo -V

stage="cargo_metadata"
cargo metadata --locked --no-deps --format-version 1 >/dev/null

stage="format"
cargo fmt -p symthaea-continuity -- --check

stage="check_all_targets"
cargo check --locked -p symthaea-continuity --all-targets

stage="clippy_strict_inventory"
strict_clippy_json="${TMPDIR:-/tmp}/symthaea-continuity-strict-clippy-${actual_sha}.jsonl"
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
allowed_dead_code_files = {
    "compose.rs",
    "exact_policy.rs",
    "verifier.rs",
    "witness.rs",
}
allowed_pairs = {
    ("clippy::too_many_arguments", "observation.rs"),
}
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

    allowed = False
    if code == "dead_code" and file_name in allowed_dead_code_files:
        allowed = True
    if (code, file_name) in allowed_pairs:
        allowed = True
    if not allowed:
        unexpected.append((code or "<none>", file_name or "<none>", rendered))

if count == 0:
    print("error: strict Clippy failed without parseable error diagnostics", file=sys.stderr)
    sys.exit(2)
if unexpected:
    print("error: strict Clippy produced diagnostics outside the explicit legacy baseline", file=sys.stderr)
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

stage="unit_and_integration_tests"
cargo test --locked -p symthaea-continuity

stage="doc_tests"
cargo test --locked -p symthaea-continuity --doc

stage="postflight_source_immutability"
if [[ "$(git rev-parse HEAD)" != "$actual_sha" ]]; then
    source_state="head-changed-during-tests"
    echo 'error: HEAD changed during qualification' >&2
    exit 1
fi
if ! git diff --quiet --ignore-submodules --; then
    source_state="tracked-source-mutated-during-tests"
    echo 'error: tracked source changed during qualification' >&2
    git diff --stat >&2 || true
    exit 1
fi
if ! git diff --cached --quiet --ignore-submodules --; then
    source_state="staged-source-mutated-during-tests"
    echo 'error: staged source appeared during qualification' >&2
    exit 1
fi
post_untracked="$(git ls-files --others --exclude-standard)"
if [[ -n "$post_untracked" ]]; then
    source_state="untracked-source-created-during-tests"
    echo 'error: untracked source appeared during qualification' >&2
    printf '%s\n' "$post_untracked" >&2
    exit 1
fi
source_state="clean-exact-checkout-postflight"

status="PASS"
stage="complete"
