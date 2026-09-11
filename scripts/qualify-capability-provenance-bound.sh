#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Exact-head qualification for the final capability-provenance transport/binding stack.
# This proves a software contract only. It grants no observation, availability,
# scientific, policy, recovery, ownership, or execution authority.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

actual_sha="$(git rev-parse HEAD)"
actual_tree="$(git rev-parse 'HEAD^{tree}')"
subject_sha="${QUALIFIED_SUBJECT_SHA:?QUALIFIED_SUBJECT_SHA is required}"
expected_qualifier_sha="${QUALIFIER_SHA:-$actual_sha}"
subject_tree="$(git rev-parse "${subject_sha}^{tree}")"
receipt_path="${CAPABILITY_BOUND_RECEIPT:-${TMPDIR:-/tmp}/capability-provenance-bound-qualification-v1.tsv}"
status="FAIL"
stage="preflight"
source_state="unverified"
transport_test_count="not-run"
unit_test_count="not-run"
strict_clippy_exit="not-run"
legacy_lint_diagnostic_count="not-run"
legacy_lint_allowlist="dead_code@compose.rs,exact_policy.rs,verifier.rs,witness.rs;clippy::too_many_arguments@observation.rs"
workflow_path='.github/workflows/capability-provenance-bound.yml'
script_path='scripts/qualify-capability-provenance-bound.sh'
provenance_source='crates/core/symthaea-continuity/src/capability_analysis_provenance.rs'
lib_source='crates/core/symthaea-continuity/src/lib.rs'
transport_tests='crates/core/symthaea-continuity/tests/capability_analysis_provenance_transport.rs'

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
        printf 'schema\tsymthaea-capability-provenance-bound-qualification-v1\n'
        printf 'status\t%s\n' "$final_status"
        printf 'exit_code\t%s\n' "$exit_code"
        printf 'terminal_stage\t%s\n' "$terminal_stage"
        printf 'scope\tcapability-provenance-final-transport-and-runtime-binding-software-contract\n'
        printf 'qualifier_sha\t%s\n' "$actual_sha"
        printf 'expected_qualifier_sha\t%s\n' "$expected_qualifier_sha"
        printf 'qualifier_tree\t%s\n' "$actual_tree"
        printf 'qualified_subject_sha\t%s\n' "$subject_sha"
        printf 'qualified_subject_tree\t%s\n' "$subject_tree"
        printf 'projection_exclusions\t%s,%s\n' "$workflow_path" "$script_path"
        printf 'projection_contract\texactly-two-harness-paths-no-other-diff\n'
        printf 'source_state\t%s\n' "$source_state"
        printf 'transport_test_count\t%s\n' "$transport_test_count"
        printf 'unit_test_count\t%s\n' "$unit_test_count"
        printf 'strict_clippy_exit\t%s\n' "$strict_clippy_exit"
        printf 'legacy_lint_allowlist\t%s\n' "$legacy_lint_allowlist"
        printf 'legacy_lint_diagnostic_count\t%s\n' "$legacy_lint_diagnostic_count"
        printf 'decode_contract\tclosed-wire-plus-canonical-try-from\n'
        printf 'runtime_binding_contract\tnonserializable-borrowed-typestate\n'
        printf 'execution_provider\t%s\n' "$provider"
        printf 'runner_label\t%s\n' "${CAPABILITY_BOUND_RUNNER_LABEL:-unknown}"
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
        printf 'continuity_manifest_sha256\t%s\n' "$(sha256_file crates/core/symthaea-continuity/Cargo.toml)"
        printf 'rust_toolchain_sha256\t%s\n' "$(sha256_file rust-toolchain.toml)"
        printf 'provenance_source_sha256\t%s\n' "$(sha256_file "$provenance_source")"
        printf 'lib_source_sha256\t%s\n' "$(sha256_file "$lib_source")"
        printf 'transport_tests_sha256\t%s\n' "$(sha256_file "$transport_tests")"
        printf 'qualifier_script_sha256\t%s\n' "$(sha256_file "$script_path")"
        printf 'workflow_sha256\t%s\n' "$(sha256_file "$workflow_path")"
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
    echo "capability-provenance-bound receipt=$receipt_path status=$final_status stage=$terminal_stage"
    return 0
}

finish() {
    local exit_code=$?
    trap - EXIT
    if [[ "$exit_code" -eq 0 && "$status" != "PASS" ]]; then
        echo "error: verifier exited without terminal PASS (stage=$stage)" >&2
        exit_code=1
    fi
    write_receipt "$exit_code" || { echo 'error: receipt persistence failed' >&2; [[ "$exit_code" -ne 0 ]] || exit_code=1; }
    exit "$exit_code"
}
trap finish EXIT

stage="preflight_exact_qualifier_head"
[[ "$actual_sha" == "$expected_qualifier_sha" ]] || { echo "error: expected qualifier=$expected_qualifier_sha actual=$actual_sha" >&2; exit 1; }

stage="preflight_subject_ancestry"
git merge-base --is-ancestor "$subject_sha" "$actual_sha" || { echo 'error: qualified subject is not an ancestor of qualifier' >&2; exit 1; }

stage="qualification_subject_projection"
mapfile -t projected_delta < <(git diff --name-only "$subject_sha" "$actual_sha")
if [[ "${#projected_delta[@]}" -ne 2 ]]; then
    echo 'error: qualifier differs from subject by unexpected path count' >&2
    printf '%s\n' "${projected_delta[@]}" >&2
    exit 1
fi
printf '%s\n' "${projected_delta[@]}" | grep -Fxq "$workflow_path"
printf '%s\n' "${projected_delta[@]}" | grep -Fxq "$script_path"
# Excluding the two explicit harness files must leave an empty diff, including modes.
git diff --exit-code "$subject_sha" "$actual_sha" -- . ":(exclude)$workflow_path" ":(exclude)$script_path" >/dev/null
# Harness exclusions must not mask any subject-owned path.
for path in "$workflow_path" "$script_path"; do
    if git cat-file -e "${subject_sha}:${path}" 2>/dev/null; then
        echo "error: projection exclusion masks a subject-owned path: $path" >&2
        exit 1
    fi
done

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

stage="static_provenance_contract"
python3 - <<'PY'
from pathlib import Path
source = Path('crates/core/symthaea-continuity/src/capability_analysis_provenance.rs').read_text()
lib = Path('crates/core/symthaea-continuity/src/lib.rs').read_text()
tests = Path('crates/core/symthaea-continuity/tests/capability_analysis_provenance_transport.rs').read_text()
if source.count('#[serde(try_from = "CapabilityActivationProvenanceWireV1")]') != 1:
    raise SystemExit('activation validated-decode contract absent')
if source.count('#[serde(try_from = "CapabilityCounterfactualProvenanceWireV1")]') != 1:
    raise SystemExit('counterfactual validated-decode contract absent')
if source.count('#[serde(deny_unknown_fields)]') < 2:
    raise SystemExit('closed wire contract absent')
if source.count('pub fn bind_to') != 2:
    raise SystemExit('expected exactly two runtime bind_to methods')
for name in ('BoundCapabilityActivationProvenanceV1', 'BoundCapabilityCounterfactualProvenanceV1'):
    if source.count(f'pub struct {name}') != 1:
        raise SystemExit(f'{name} declaration missing or duplicated')
    pos = source.index(f'pub struct {name}')
    prefix = source[max(0, pos - 260):pos]
    if '#[derive(Debug)]' not in prefix:
        raise SystemExit(f'{name} is not Debug-only')
    if 'Serialize' in prefix or 'Deserialize' in prefix or '#[serde' in prefix:
        raise SystemExit(f'{name} became transportable')
    if lib.count(name) != 1:
        raise SystemExit(f'{name} public export missing or duplicated')
if tests.count('provenance_binding_is_runtime_specific') != 2:
    raise SystemExit('runtime-specific binding regressions missing')
PY

stage="cargo_metadata"
cargo metadata --locked --no-deps --format-version 1 >/dev/null

stage="format"
rustfmt --edition 2024 --check "$provenance_source" "$transport_tests"
rustfmt --edition 2024 --config skip_children=true --check "$lib_source"

stage="check_all_targets"
cargo check --locked -p symthaea-continuity --all-targets

stage="clippy_strict_inventory"
strict_clippy_json="${TMPDIR:-/tmp}/symthaea-capability-bound-clippy-${actual_sha}.jsonl"
set +e
cargo clippy --locked -p symthaea-continuity --all-targets --message-format=json -- -D warnings >"$strict_clippy_json" 2>&1
strict_clippy_exit=$?
set -e
if [[ "$strict_clippy_exit" -eq 0 ]]; then
    legacy_lint_diagnostic_count="0"
else
    legacy_lint_diagnostic_count="$(python3 - "$strict_clippy_json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
allowed_dead_code_files = {'compose.rs', 'exact_policy.rs', 'verifier.rs', 'witness.rs'}
allowed_pairs = {('clippy::too_many_arguments', 'observation.rs')}
count = 0
unexpected = []
for raw in path.read_text(errors='replace').splitlines():
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        continue
    if event.get('reason') != 'compiler-message':
        continue
    message = event.get('message') or {}
    if message.get('level') != 'error':
        continue
    code = (message.get('code') or {}).get('code')
    primary = next((s for s in message.get('spans', []) if s.get('is_primary')), None)
    file_name = pathlib.PurePosixPath((primary or {}).get('file_name', '')).name
    rendered = (message.get('message') or '').replace('\n', ' ')
    count += 1
    allowed = code == 'dead_code' and file_name in allowed_dead_code_files
    allowed = allowed or (code, file_name) in allowed_pairs
    if not allowed:
        unexpected.append((code or '<none>', file_name or '<none>', rendered))
if count == 0:
    print('error: strict Clippy failed without parseable error diagnostics', file=sys.stderr)
    sys.exit(2)
if unexpected:
    print('error: strict Clippy produced diagnostics outside explicit legacy baseline', file=sys.stderr)
    for code, file_name, rendered in unexpected:
        print(f'  {code}@{file_name}: {rendered}', file=sys.stderr)
    sys.exit(3)
print(count)
PY
)"
fi

stage="clippy_with_bounded_legacy_baseline"
cargo clippy --locked -p symthaea-continuity --all-targets -- -A dead_code -A clippy::too_many_arguments -D warnings

stage="transport_regressions"
transport_output="$(mktemp)"
cargo test --locked -p symthaea-continuity --test capability_analysis_provenance_transport -- --nocapture 2>&1 | tee "$transport_output"
transport_test_count="$(sed -nE 's/^test result: ok\. ([0-9]+) passed;.*/\1/p' "$transport_output" | tail -n1)"
rm -f "$transport_output"
[[ "$transport_test_count" == '19' ]] || { echo "error: expected 19 transport tests, got $transport_test_count" >&2; exit 1; }

stage="unit_tests"
unit_output="$(mktemp)"
cargo test --locked -p symthaea-continuity --lib 2>&1 | tee "$unit_output"
unit_test_count="$(sed -nE 's/^test result: ok\. ([0-9]+) passed;.*/\1/p' "$unit_output" | tail -n1)"
rm -f "$unit_output"
[[ "$unit_test_count" == '88' ]] || { echo "error: expected 88 unit tests, got $unit_test_count" >&2; exit 1; }

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
