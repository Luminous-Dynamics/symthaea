#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Exact-head qualification for the frozen V1 capability-analysis algorithm
# identity-preimage contract. Software/source evidence only; no authority grant.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

actual_sha="$(git rev-parse HEAD)"
actual_tree="$(git rev-parse 'HEAD^{tree}')"
subject_sha="${QUALIFIED_SUBJECT_SHA:?QUALIFIED_SUBJECT_SHA is required}"
expected_qualifier_sha="${QUALIFIER_SHA:-$actual_sha}"
subject_tree="$(git rev-parse "${subject_sha}^{tree}")"
receipt_path="${CAPABILITY_IDENTITY_RECEIPT:-${TMPDIR:-/tmp}/capability-algorithm-identity-preimage-qualification-v1.tsv}"
status="FAIL"
stage="preflight"
source_state="unverified"
focused_test_count="not-run"
unit_test_count="not-run"
transport_test_count="not-run"
strict_clippy_exit="not-run"
legacy_lint_diagnostic_count="not-run"

workflow_path='.github/workflows/capability-algorithm-identity-preimage.yml'
script_path='scripts/qualify-capability-algorithm-identity-preimage.sh'
identity_source='crates/core/symthaea-continuity/src/capability_algorithm_identity.rs'
provenance_source='crates/core/symthaea-continuity/src/capability_analysis_provenance.rs'
lib_source='crates/core/symthaea-continuity/src/lib.rs'
transport_tests='crates/core/symthaea-continuity/tests/capability_analysis_provenance_transport.rs'

schema='symthaea-continuity-capability-algorithm-identity-preimage-v1'
digest_algorithm='blake3-256'
semantics_encoding='utf-8-exact-no-normalization-no-length-prefix'
version_suffix='.v1\0'
activation_domain='symthaea.continuity.capability-activation.algorithm-semantics.v1\0'
counterfactual_domain='symthaea.continuity.capability-counterfactual.algorithm-semantics.v1\0'
activation_semantics='symthaea-continuity-capability-activation-monotone-fixed-point-v1'
counterfactual_semantics='symthaea-continuity-capability-counterfactual-bounded-support-frontier-v1'
activation_golden='37c419594ff686d875b9d74c46fe5067af58ddd934d5f22df2aa28ac0b03c891'
counterfactual_golden='64ce9f30f849b20735c610888553498f12355e1b9b38417dd33380f1c42b3174'

sha256_file() {
    [[ -f "$1" ]] && sha256sum "$1" | awk '{print $1}' || printf 'unavailable'
}

write_receipt() {
    local exit_code="$1" final_status="$status" terminal_stage="$stage" provider="local"
    local rustc_verbose="" rustc_release="unavailable" rustc_commit="unavailable" rustc_host="unavailable"
    local cargo_version="unavailable" tmp_path="${receipt_path}.tmp.$$"
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
        printf 'schema\tsymthaea-capability-algorithm-identity-preimage-qualification-v1\n'
        printf 'status\t%s\n' "$final_status"
        printf 'exit_code\t%s\n' "$exit_code"
        printf 'terminal_stage\t%s\n' "$terminal_stage"
        printf 'scope\tcapability-analysis-algorithm-identity-preimage-v1-software-contract\n'
        printf 'qualifier_sha\t%s\n' "$actual_sha"
        printf 'expected_qualifier_sha\t%s\n' "$expected_qualifier_sha"
        printf 'qualifier_tree\t%s\n' "$actual_tree"
        printf 'qualified_subject_sha\t%s\n' "$subject_sha"
        printf 'qualified_subject_tree\t%s\n' "$subject_tree"
        printf 'projection_contract\texactly-two-harness-paths-no-other-diff\n'
        printf 'source_state\t%s\n' "$source_state"
        printf 'preimage_schema\t%s\n' "$schema"
        printf 'digest_algorithm\t%s\n' "$digest_algorithm"
        printf 'semantic_name_encoding\t%s\n' "$semantics_encoding"
        printf 'domain_version_suffix_rust_literal\t%s\n' "$version_suffix"
        printf 'activation_domain_rust_literal\t%s\n' "$activation_domain"
        printf 'counterfactual_domain_rust_literal\t%s\n' "$counterfactual_domain"
        printf 'activation_semantics\t%s\n' "$activation_semantics"
        printf 'counterfactual_semantics\t%s\n' "$counterfactual_semantics"
        printf 'activation_algorithm_id_v1_golden\t%s\n' "$activation_golden"
        printf 'counterfactual_algorithm_id_v1_golden\t%s\n' "$counterfactual_golden"
        printf 'external_digest_oracle\tstandalone-rust-blake3-over-independent-literals\n'
        printf 'focused_identity_test_count\t%s\n' "$focused_test_count"
        printf 'unit_test_count\t%s\n' "$unit_test_count"
        printf 'transport_test_count\t%s\n' "$transport_test_count"
        printf 'strict_clippy_exit\t%s\n' "$strict_clippy_exit"
        printf 'legacy_lint_diagnostic_count\t%s\n' "$legacy_lint_diagnostic_count"
        printf 'execution_provider\t%s\n' "$provider"
        printf 'runner_label\t%s\n' "${CAPABILITY_IDENTITY_RUNNER_LABEL:-unknown}"
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
        printf 'identity_source_sha256\t%s\n' "$(sha256_file "$identity_source")"
        printf 'provenance_source_sha256\t%s\n' "$(sha256_file "$provenance_source")"
        printf 'lib_source_sha256\t%s\n' "$(sha256_file "$lib_source")"
        printf 'transport_tests_sha256\t%s\n' "$(sha256_file "$transport_tests")"
        printf 'qualifier_script_sha256\t%s\n' "$(sha256_file "$script_path")"
        printf 'workflow_sha256\t%s\n' "$(sha256_file "$workflow_path")"
        printf 'observation_authority\tnone\navailability_authority\tnone\nscientific_authority\tnone\n'
        printf 'policy_authority\tnone\nrecovery_authority\tnone\nownership_authority\tnone\nexecution_authority\tnone\n'
        printf 'receipt_attestation\tnone\n'
        printf 'github_run_id\t%s\n' "${GITHUB_RUN_ID:-not-applicable}"
        printf 'github_run_attempt\t%s\n' "${GITHUB_RUN_ATTEMPT:-not-applicable}"
    } >"$tmp_path" || { rm -f "$tmp_path"; return 1; }
    mv "$tmp_path" "$receipt_path" || { rm -f "$tmp_path"; return 1; }
    echo "capability-algorithm-identity-preimage receipt=$receipt_path status=$final_status stage=$terminal_stage"
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
[[ "$actual_sha" == "$expected_qualifier_sha" ]] || { echo "error: qualifier head mismatch" >&2; exit 1; }
stage="preflight_subject_ancestry"
git merge-base --is-ancestor "$subject_sha" "$actual_sha" || { echo 'error: subject is not an ancestor' >&2; exit 1; }
[[ "$(git rev-list --count "${subject_sha}..${actual_sha}")" -eq 1 ]] || { echo 'error: qualifier must be exactly one commit above subject' >&2; exit 1; }

stage="qualification_subject_projection"
mapfile -t projected_delta < <(git diff --name-only "$subject_sha" "$actual_sha")
[[ "${#projected_delta[@]}" -eq 2 ]] || { printf '%s\n' "${projected_delta[@]}" >&2; exit 1; }
printf '%s\n' "${projected_delta[@]}" | grep -Fxq "$workflow_path"
printf '%s\n' "${projected_delta[@]}" | grep -Fxq "$script_path"
git diff --exit-code "$subject_sha" "$actual_sha" -- . ":(exclude)$workflow_path" ":(exclude)$script_path" >/dev/null
for path in "$workflow_path" "$script_path"; do
    git cat-file -e "${subject_sha}:${path}" 2>/dev/null && { echo "error: exclusion masks subject path $path" >&2; exit 1; } || true
done

stage="preflight_clean_tree"
git diff --quiet --ignore-submodules -- || { source_state="tracked-modifications-present"; exit 1; }
git diff --cached --quiet --ignore-submodules -- || { source_state="staged-modifications-present"; exit 1; }
untracked="$(git ls-files --others --exclude-standard)"
[[ -z "$untracked" ]] || { source_state="untracked-source-present"; printf '%s\n' "$untracked" >&2; exit 1; }
source_state="clean-exact-checkout"

stage="static_preimage_contract"
SCHEMA="$schema" DIGEST_ALGORITHM="$digest_algorithm" SEMANTICS_ENCODING="$semantics_encoding" \
VERSION_SUFFIX="$version_suffix" ACTIVATION_DOMAIN="$activation_domain" COUNTERFACTUAL_DOMAIN="$counterfactual_domain" \
ACTIVATION_SEMANTICS="$activation_semantics" COUNTERFACTUAL_SEMANTICS="$counterfactual_semantics" python3 - <<'PY'
from pathlib import Path
import os, re
identity = Path('crates/core/symthaea-continuity/src/capability_algorithm_identity.rs').read_text()
provenance = Path('crates/core/symthaea-continuity/src/capability_analysis_provenance.rs').read_text()
lib = Path('crates/core/symthaea-continuity/src/lib.rs').read_text()
for name, value in {
    'CAPABILITY_ALGORITHM_IDENTITY_PREIMAGE_SCHEMA_V1': os.environ['SCHEMA'],
    'CAPABILITY_ALGORITHM_IDENTITY_DIGEST_V1': os.environ['DIGEST_ALGORITHM'],
    'CAPABILITY_ALGORITHM_IDENTITY_SEMANTICS_ENCODING_V1': os.environ['SEMANTICS_ENCODING'],
}.items():
    if not re.search(rf'pub const {name}: &str\s*=\s*"{re.escape(value)}";', identity):
        raise SystemExit(f'exact string constant drift: {name}')
for name, literal in {
    'CAPABILITY_ALGORITHM_IDENTITY_DOMAIN_VERSION_SUFFIX_V1': os.environ['VERSION_SUFFIX'],
    'CAPABILITY_ACTIVATION_ALGORITHM_DOMAIN_SEPARATOR_V1': os.environ['ACTIVATION_DOMAIN'],
    'CAPABILITY_COUNTERFACTUAL_ALGORITHM_DOMAIN_SEPARATOR_V1': os.environ['COUNTERFACTUAL_DOMAIN'],
}.items():
    if not re.search(rf'pub const {name}: &\[u8\]\s*=\s*b"{re.escape(literal)}";', identity):
        raise SystemExit(f'exact byte constant drift: {name}')
for public_name in (
    'CapabilityAlgorithmIdentityKindV1', 'CapabilityAlgorithmIdentityPreimageSpecV1',
    'capability_activation_algorithm_identity_preimage_spec_v1',
    'capability_counterfactual_algorithm_identity_preimage_spec_v1',
):
    if lib.count(public_name) != 1:
        raise SystemExit(f'public export missing or duplicated: {public_name}')
for declaration in (
    'pub struct CapabilityAlgorithmIdentityPreimageSpecV1 {',
    'pub enum CapabilityAlgorithmIdentityKindV1 {',
    'pub fn canonical_preimage(&self) -> Vec<u8>',
    'pub fn digest_bytes(&self) -> [u8; 32]',
    'pub fn matches_public_v1_algorithm_id(&self) -> bool',
):
    if identity.count(declaration) != 1:
        raise SystemExit(f'declaration missing or duplicated: {declaration}')
for semantic in (os.environ['ACTIVATION_SEMANTICS'], os.environ['COUNTERFACTUAL_SEMANTICS']):
    if provenance.count(f'"{semantic}"') != 1:
        raise SystemExit(f'legacy semantic source drift: {semantic}')
for domain in (os.environ['ACTIVATION_DOMAIN'], os.environ['COUNTERFACTUAL_DOMAIN']):
    if provenance.count(f'b"{domain}"') != 1:
        raise SystemExit(f'legacy domain source drift: {domain}')
if 'AlgorithmIdentityPreimage != AlgorithmExecution != ResultCorrectness != Authority' not in identity:
    raise SystemExit('layer-separation theorem missing')
PY

stage="cargo_metadata"
cargo metadata --locked --no-deps --format-version 1 >/dev/null
stage="format"
rustfmt --edition 2024 --check "$identity_source"
rustfmt --edition 2024 --config skip_children=true --check "$lib_source"
stage="check_all_targets"
cargo check --locked -p symthaea-continuity --all-targets

stage="independent_blake3_oracle"
cargo build --locked -p symthaea-continuity >/dev/null
blake3_rlib="$(find target/debug/deps -maxdepth 1 -type f -name 'libblake3-*.rlib' | sort | head -n1)"
[[ -n "$blake3_rlib" && -f "$blake3_rlib" ]] || { echo 'error: compiled blake3 rlib not found' >&2; exit 1; }
oracle_rs="${TMPDIR:-/tmp}/capability-identity-oracle-${actual_sha}.rs"
oracle_bin="${TMPDIR:-/tmp}/capability-identity-oracle-${actual_sha}"
cat >"$oracle_rs" <<'RS'
fn main() {
    const A: &[u8] = b"symthaea.continuity.capability-activation.algorithm-semantics.v1\0symthaea-continuity-capability-activation-monotone-fixed-point-v1";
    const C: &[u8] = b"symthaea.continuity.capability-counterfactual.algorithm-semantics.v1\0symthaea-continuity-capability-counterfactual-bounded-support-frontier-v1";
    const AE: [u8; 32] = [0x37,0xc4,0x19,0x59,0x4f,0xf6,0x86,0xd8,0x75,0xb9,0xd7,0x4c,0x46,0xfe,0x50,0x67,0xaf,0x58,0xdd,0xd9,0x34,0xd5,0xf2,0x2d,0xf2,0xaa,0x28,0xac,0x0b,0x03,0xc8,0x91];
    const CE: [u8; 32] = [0x64,0xce,0x9f,0x30,0xf8,0x49,0xb2,0x07,0x35,0xc6,0x10,0x88,0x85,0x53,0x49,0x8f,0x12,0x35,0x5e,0x1b,0x9b,0x38,0x41,0x7d,0xd3,0x33,0x80,0xf1,0xc4,0x2b,0x31,0x74];
    assert_eq!(*blake3::hash(A).as_bytes(), AE);
    assert_eq!(*blake3::hash(C).as_bytes(), CE);
    assert_ne!(AE, CE);
}
RS
rustc --edition 2024 "$oracle_rs" --extern "blake3=$blake3_rlib" -L dependency=target/debug/deps -o "$oracle_bin"
"$oracle_bin"
rm -f "$oracle_rs" "$oracle_bin"

stage="focused_identity_tests"
focused_list="${TMPDIR:-/tmp}/capability-identity-focused-${actual_sha}.list"
cargo test --locked -p symthaea-continuity capability_algorithm_identity::tests:: -- --list >"$focused_list"
focused_test_count="$(grep -Ec ': test$' "$focused_list" || true)"
[[ "$focused_test_count" == "5" ]] || { echo "error: expected 5 focused identity tests, got $focused_test_count" >&2; cat "$focused_list" >&2; exit 1; }
cargo test --locked -p symthaea-continuity capability_algorithm_identity::tests::

stage="transport_regression_inventory"
transport_list="${TMPDIR:-/tmp}/capability-identity-transport-${actual_sha}.list"
cargo test --locked -p symthaea-continuity --test capability_analysis_provenance_transport -- --list >"$transport_list"
transport_test_count="$(grep -Ec ': test$' "$transport_list" || true)"
[[ "$transport_test_count" == "21" ]] || { echo "error: expected 21 provenance transport tests, got $transport_test_count" >&2; cat "$transport_list" >&2; exit 1; }
cargo test --locked -p symthaea-continuity --test capability_analysis_provenance_transport

stage="unit_test_inventory"
unit_list="${TMPDIR:-/tmp}/capability-identity-unit-${actual_sha}.list"
cargo test --locked -p symthaea-continuity --lib -- --list >"$unit_list"
unit_test_count="$(grep -Ec ': test$' "$unit_list" || true)"
[[ "$unit_test_count" == "93" ]] || { echo "error: expected 93 unit tests, got $unit_test_count" >&2; cat "$unit_list" >&2; exit 1; }
cargo test --locked -p symthaea-continuity --lib
stage="doc_tests"
cargo test --locked -p symthaea-continuity --doc

stage="clippy_strict_inventory"
strict_clippy_json="${TMPDIR:-/tmp}/symthaea-capability-identity-clippy-${actual_sha}.jsonl"
set +e
cargo clippy --locked -p symthaea-continuity --all-targets --message-format=json -- -D warnings >"$strict_clippy_json" 2>&1
strict_clippy_exit=$?
set -e
if [[ "$strict_clippy_exit" -eq 0 ]]; then
    legacy_lint_diagnostic_count="0"
else
    legacy_lint_diagnostic_count="$(python3 - "$strict_clippy_json" <<'PY'
import json, pathlib, sys
allowed_dead = {'compose.rs', 'exact_policy.rs', 'verifier.rs', 'witness.rs'}
allowed_pairs = {('clippy::too_many_arguments', 'observation.rs')}
count = 0
unexpected = []
for raw in pathlib.Path(sys.argv[1]).read_text(errors='replace').splitlines():
    try: event = json.loads(raw)
    except json.JSONDecodeError: continue
    if event.get('reason') != 'compiler-message': continue
    message = event.get('message') or {}
    if message.get('level') != 'error': continue
    code = (message.get('code') or {}).get('code')
    primary = next((s for s in message.get('spans', []) if s.get('is_primary')), None)
    filename = pathlib.PurePosixPath((primary or {}).get('file_name', '')).name
    if (code == 'dead_code' and filename in allowed_dead) or ((code, filename) in allowed_pairs): count += 1
    else: unexpected.append((code, filename, message.get('message')))
if unexpected:
    for item in unexpected: print(f'unexpected clippy diagnostic: {item}', file=sys.stderr)
    raise SystemExit(1)
print(count)
PY
)" || { cat "$strict_clippy_json" >&2; exit 1; }
    [[ "$legacy_lint_diagnostic_count" == "5" ]] || { echo "error: expected 5 bounded legacy clippy diagnostics, got $legacy_lint_diagnostic_count" >&2; cat "$strict_clippy_json" >&2; exit 1; }
fi

stage="postflight_clean_tree"
git diff --quiet --ignore-submodules -- || { source_state="tracked-modifications-after-tests"; exit 1; }
git diff --cached --quiet --ignore-submodules -- || { source_state="staged-modifications-after-tests"; exit 1; }
untracked="$(git ls-files --others --exclude-standard)"
[[ -z "$untracked" ]] || { source_state="untracked-source-after-tests"; printf '%s\n' "$untracked" >&2; exit 1; }
[[ "$(git rev-parse HEAD)" == "$actual_sha" ]] || { echo 'error: HEAD changed during qualification' >&2; exit 1; }
source_state="clean-exact-checkout"
stage="terminal_pass"
status="PASS"
