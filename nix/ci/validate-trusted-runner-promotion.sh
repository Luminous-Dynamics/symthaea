#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Queue-neutral Stage-C verifier for the trusted CPU runner promotion boundary.

set -euo pipefail
umask 077

REPOSITORY_URL='https://github.com/Luminous-Dynamics/symthaea.git'
RECOVERY_BRANCH='ci/nixos-ephemeral-runner-v1'
MANIFEST_PATH="${SYMTHAEA_TRUSTED_BOOTSTRAP_MANIFEST_PATH:-}"
EXPECTED_MANIFEST_SHA256="${SYMTHAEA_TRUSTED_BOOTSTRAP_MANIFEST_SHA256:-}"

if [[ "${GITHUB_ACTIONS:-}" == 'true' ]]; then
  echo 'promotion verifier must run directly under operator control, not inside GitHub Actions' >&2
  exit 1
fi
[[ -n "$MANIFEST_PATH" && -f "$MANIFEST_PATH" ]] || {
  echo 'SYMTHAEA_TRUSTED_BOOTSTRAP_MANIFEST_PATH must name the exact retained Stage-A manifest' >&2
  exit 1
}
[[ "$EXPECTED_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo 'SYMTHAEA_TRUSTED_BOOTSTRAP_MANIFEST_SHA256 must be the independently recorded 64-hex Stage-A manifest SHA-256' >&2
  exit 1
}
for command in git sha256sum awk sort mktemp; do
  command -v "$command" >/dev/null 2>&1 || exit 1
done

actual_manifest_sha256="$(sha256sum "$MANIFEST_PATH" | awk '{print $1}')"
[[ "$actual_manifest_sha256" == "$EXPECTED_MANIFEST_SHA256" ]] || {
  echo 'Stage-A manifest bytes do not match the independently recorded SHA-256' >&2
  exit 1
}

manifest_value() {
  local key="$1" value count
  value="$(awk -v k="$key" 'index($0, k "=") == 1 { sub(/^[^=]*=/, ""); print }' "$MANIFEST_PATH")"
  count="$(awk -v k="$key" 'index($0, k "=") == 1 { n += 1 } END { print n + 0 }' "$MANIFEST_PATH")"
  [[ "$count" == '1' && -n "$value" ]] || {
    echo "manifest key must occur exactly once with a non-empty value: $key" >&2
    exit 1
  }
  printf '%s' "$value"
}

schema="$(manifest_value schema)"
result="$(manifest_value result)"
repository="$(manifest_value repository)"
recovery_branch="$(manifest_value recovery_branch)"
operator_authorized_head="$(manifest_value operator_authorized_head)"
recovery_head="$(manifest_value recovery_head)"
source_tree="$(manifest_value source_tree)"
main_head="$(manifest_value main_head)"
main_tree="$(manifest_value main_tree)"
promotion_required_ancestor="$(manifest_value promotion_required_ancestor)"
promotion_expected_main_tree="$(manifest_value promotion_expected_main_tree)"
recovery_diff_paths_sha256="$(manifest_value recovery_diff_paths_sha256)"
runner_module_blob="$(manifest_value runner_module_blob)"
routing_policy_blob="$(manifest_value routing_policy_blob)"
smoke_workflow_blob="$(manifest_value smoke_workflow_blob)"
bootstrap_validator_blob="$(manifest_value bootstrap_validator_blob)"
promotion_verifier_blob="$(manifest_value promotion_verifier_blob)"
recovery_eligibility_verifier_blob="$(manifest_value recovery_eligibility_verifier_blob)"
host_lifecycle_contract_blob="$(manifest_value host_lifecycle_contract_blob)"

[[ "$schema" == 'symthaea.trusted-runner.bootstrap.v6' ]]
[[ "$result" == 'PASS' ]]
[[ "$repository" == "$REPOSITORY_URL" ]]
[[ "$recovery_branch" == "$RECOVERY_BRANCH" ]]
[[ "$operator_authorized_head" =~ ^[0-9a-f]{40}$ ]]
[[ "$recovery_head" == "$operator_authorized_head" ]]
[[ "$promotion_required_ancestor" == "$operator_authorized_head" ]]
[[ "$source_tree" =~ ^[0-9a-f]{40}$ ]]
[[ "$promotion_expected_main_tree" == "$source_tree" ]]
[[ "$main_head" =~ ^[0-9a-f]{40}$ ]]
[[ "$main_tree" =~ ^[0-9a-f]{40}$ ]]
[[ "$recovery_diff_paths_sha256" =~ ^[0-9a-f]{64}$ ]]
for blob in "$runner_module_blob" "$routing_policy_blob" "$smoke_workflow_blob" "$bootstrap_validator_blob" "$promotion_verifier_blob" "$recovery_eligibility_verifier_blob" "$host_lifecycle_contract_blob"; do
  [[ "$blob" =~ ^[0-9a-f]{40}$ ]]
done
for pass_key in operator_authorization_checked runner_policy_eval routing_policy_eval minimal_locked_rust_check refs_unchanged_during_validation; do
  [[ "$(manifest_value "$pass_key")" == 'PASS' ]]
done
[[ "$(manifest_value evidence_scope)" == 'runner-bootstrap-correctness-only' ]]

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
origin_url="$(git remote get-url origin)"
if [[ "$origin_url" != "$REPOSITORY_URL" && "$origin_url" != 'https://github.com/Luminous-Dynamics/symthaea' ]]; then
  echo "origin must be canonical public HTTPS Symthaea repository; found: $origin_url" >&2
  exit 1
fi
local_head="$(git rev-parse HEAD)"
local_tree="$(git rev-parse HEAD^{tree})"
git diff --exit-code
git diff --cached --exit-code
test -z "$(git status --porcelain=v1 --untracked-files=all --ignored=matching)"

git -c protocol.version=2 fetch --no-tags origin \
  "+refs/heads/main:refs/remotes/origin/main" \
  "+refs/heads/${RECOVERY_BRANCH}:refs/remotes/origin/${RECOVERY_BRANCH}"
public_main="$(git rev-parse refs/remotes/origin/main)"
public_main_tree="$(git rev-parse refs/remotes/origin/main^{tree})"
public_recovery="$(git rev-parse "refs/remotes/origin/${RECOVERY_BRANCH}")"

[[ "$local_head" == "$public_main" && "$local_tree" == "$public_main_tree" ]] || {
  echo 'promotion verifier must execute from a pristine detached checkout of exact current public main' >&2
  exit 1
}
[[ "$(git rev-parse HEAD:nix/ci/validate-trusted-runner-promotion.sh)" == "$promotion_verifier_blob" ]]
[[ "$public_recovery" == "$operator_authorized_head" ]]
git merge-base --is-ancestor "$main_head" "$public_main"
git merge-base --is-ancestor "$promotion_required_ancestor" "$public_main"
[[ "$public_main_tree" == "$promotion_expected_main_tree" ]]

actual_paths="$(git diff --name-only "$main_head" "$public_main" | LC_ALL=C sort)"
actual_paths_sha256="$(printf '%s\n' "$actual_paths" | sha256sum | awk '{print $1}')"
[[ "$actual_paths_sha256" == "$recovery_diff_paths_sha256" ]]

verify_blob() {
  local path="$1" expected="$2"
  [[ "$(git rev-parse "$public_main:$path")" == "$expected" ]] || {
    echo "promoted artifact blob differs from Stage-A-qualified artifact: $path" >&2
    exit 1
  }
}
verify_blob nix/modules/github-actions-runner.nix "$runner_module_blob"
verify_blob nix/tests/eval-trusted-runner-routing.nix "$routing_policy_blob"
verify_blob .github/workflows/self-hosted-runner-smoke.yml "$smoke_workflow_blob"
verify_blob nix/ci/validate-trusted-runner-bootstrap.sh "$bootstrap_validator_blob"
verify_blob nix/ci/validate-trusted-runner-promotion.sh "$promotion_verifier_blob"
verify_blob nix/ci/validate-trusted-runner-recovery-eligibility.sh "$recovery_eligibility_verifier_blob"
verify_blob docs/operations/TRUSTED_CPU_RUNNER_HOST_LIFECYCLE.md "$host_lifecycle_contract_blob"

git -c protocol.version=2 fetch --no-tags origin \
  "+refs/heads/main:refs/remotes/origin/main" \
  "+refs/heads/${RECOVERY_BRANCH}:refs/remotes/origin/${RECOVERY_BRANCH}"
[[ "$(git rev-parse refs/remotes/origin/main)" == "$public_main" ]]
[[ "$(git rev-parse "refs/remotes/origin/${RECOVERY_BRANCH}")" == "$public_recovery" ]]
[[ "$(git rev-parse HEAD)" == "$local_head" ]]
[[ "$(git rev-parse HEAD^{tree})" == "$local_tree" ]]
git diff --exit-code
git diff --cached --exit-code
test -z "$(git status --porcelain=v1 --untracked-files=all --ignored=matching)"

promotion_manifest="$(mktemp /tmp/symthaea-trusted-runner-promotion-v2.XXXXXX)"
cat > "$promotion_manifest" <<EOF
schema=symthaea.trusted-runner.promotion.v2
result=PASS
bootstrap_manifest_sha256=$actual_manifest_sha256
authorized_recovery_head=$operator_authorized_head
promoted_main_head=$public_main
promoted_main_tree=$public_main_tree
smoke_workflow_blob=$smoke_workflow_blob
runner_module_blob=$runner_module_blob
routing_policy_blob=$routing_policy_blob
promotion_verifier_blob=$promotion_verifier_blob
recovery_eligibility_verifier_blob=$recovery_eligibility_verifier_blob
promotion_ancestry_checked=PASS
promotion_tree_identity_checked=PASS
promotion_diff_surface_checked=PASS
promotion_artifact_blobs_checked=PASS
promotion_local_checkout_checked=PASS
promotion_refs_stable=PASS
evidence_scope=stage-d-smoke-eligibility-only
EOF
promotion_manifest_sha256="$(sha256sum "$promotion_manifest" | awk '{print $1}')"
cat "$promotion_manifest"
printf 'promotion_manifest_sha256=%s\n' "$promotion_manifest_sha256"
printf 'promotion_manifest_path=%s\n' "$promotion_manifest"
