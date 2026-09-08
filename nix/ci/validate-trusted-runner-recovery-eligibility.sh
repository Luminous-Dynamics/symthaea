#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Queue-neutral Stage-E eligibility verifier. It joins the exact promotion.v2
# and smoke.v1 evidence objects before any trusted recovery workload is eligible.

set -euo pipefail
umask 077

REPOSITORY_URL='https://github.com/Luminous-Dynamics/symthaea.git'
RECOVERY_BRANCH='ci/nixos-ephemeral-runner-v1'
PROMOTION_MANIFEST_PATH="${SYMTHAEA_TRUSTED_PROMOTION_MANIFEST_PATH:-}"
PROMOTION_MANIFEST_SHA256="${SYMTHAEA_TRUSTED_PROMOTION_MANIFEST_SHA256:-}"
SMOKE_MANIFEST_PATH="${SYMTHAEA_TRUSTED_SMOKE_MANIFEST_PATH:-}"
SMOKE_MANIFEST_SHA256="${SYMTHAEA_TRUSTED_SMOKE_MANIFEST_SHA256:-}"

if [[ "${GITHUB_ACTIONS:-}" == 'true' ]]; then
  echo 'recovery eligibility verifier must run directly under operator control' >&2
  exit 1
fi

for path in "$PROMOTION_MANIFEST_PATH" "$SMOKE_MANIFEST_PATH"; do
  [[ -n "$path" && -f "$path" ]] || {
    echo 'promotion and smoke manifest paths must name retained evidence files' >&2
    exit 1
  }
done
for digest in "$PROMOTION_MANIFEST_SHA256" "$SMOKE_MANIFEST_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
    echo 'promotion and smoke manifest SHA-256 values must be explicit 64-hex digests' >&2
    exit 1
  }
done

for command in git sha256sum awk mktemp; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "required command missing: $command" >&2
    exit 1
  }
done

verify_manifest_hash() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "manifest hash mismatch: $path" >&2
    printf 'expected=%s\nactual=%s\n' "$expected" "$actual" >&2
    exit 1
  }
}
verify_manifest_hash "$PROMOTION_MANIFEST_PATH" "$PROMOTION_MANIFEST_SHA256"
verify_manifest_hash "$SMOKE_MANIFEST_PATH" "$SMOKE_MANIFEST_SHA256"

manifest_value() {
  local file="$1" key="$2" value count
  value="$(awk -v k="$key" 'index($0, k "=") == 1 { sub(/^[^=]*=/, ""); print }' "$file")"
  count="$(awk -v k="$key" 'index($0, k "=") == 1 { n += 1 } END { print n + 0 }' "$file")"
  if [[ "$count" != '1' || -z "$value" ]]; then
    echo "manifest key must occur exactly once with a non-empty value: $key" >&2
    exit 1
  fi
  printf '%s' "$value"
}

promotion_schema="$(manifest_value "$PROMOTION_MANIFEST_PATH" schema)"
promotion_result="$(manifest_value "$PROMOTION_MANIFEST_PATH" result)"
authorized_recovery_head="$(manifest_value "$PROMOTION_MANIFEST_PATH" authorized_recovery_head)"
promoted_main_head="$(manifest_value "$PROMOTION_MANIFEST_PATH" promoted_main_head)"
promoted_main_tree="$(manifest_value "$PROMOTION_MANIFEST_PATH" promoted_main_tree)"
promotion_smoke_workflow_blob="$(manifest_value "$PROMOTION_MANIFEST_PATH" smoke_workflow_blob)"
promotion_runner_module_blob="$(manifest_value "$PROMOTION_MANIFEST_PATH" runner_module_blob)"
promotion_routing_policy_blob="$(manifest_value "$PROMOTION_MANIFEST_PATH" routing_policy_blob)"
recovery_eligibility_verifier_blob="$(manifest_value "$PROMOTION_MANIFEST_PATH" recovery_eligibility_verifier_blob)"

[[ "$promotion_schema" == 'symthaea.trusted-runner.promotion.v2' ]]
[[ "$promotion_result" == 'PASS' ]]
for value in "$authorized_recovery_head" "$promoted_main_head" "$promoted_main_tree" "$promotion_smoke_workflow_blob" "$promotion_runner_module_blob" "$promotion_routing_policy_blob" "$recovery_eligibility_verifier_blob"; do
  [[ "$value" =~ ^[0-9a-f]{40}$ ]]
done
for pass_key in promotion_ancestry_checked promotion_tree_identity_checked promotion_diff_surface_checked promotion_artifact_blobs_checked promotion_local_checkout_checked promotion_refs_stable; do
  [[ "$(manifest_value "$PROMOTION_MANIFEST_PATH" "$pass_key")" == 'PASS' ]]
done
[[ "$(manifest_value "$PROMOTION_MANIFEST_PATH" evidence_scope)" == 'stage-d-smoke-eligibility-only' ]]

smoke_schema="$(manifest_value "$SMOKE_MANIFEST_PATH" schema)"
smoke_result="$(manifest_value "$SMOKE_MANIFEST_PATH" result)"
smoke_repository="$(manifest_value "$SMOKE_MANIFEST_PATH" github_repository)"
smoke_ref="$(manifest_value "$SMOKE_MANIFEST_PATH" github_ref)"
smoke_run_id="$(manifest_value "$SMOKE_MANIFEST_PATH" github_run_id)"
smoke_run_attempt="$(manifest_value "$SMOKE_MANIFEST_PATH" github_run_attempt)"
smoke_main_head="$(manifest_value "$SMOKE_MANIFEST_PATH" github_sha)"
smoke_main_tree="$(manifest_value "$SMOKE_MANIFEST_PATH" source_tree)"
smoke_workflow_blob="$(manifest_value "$SMOKE_MANIFEST_PATH" smoke_workflow_blob)"
smoke_runner_module_blob="$(manifest_value "$SMOKE_MANIFEST_PATH" runner_module_blob)"
smoke_routing_policy_blob="$(manifest_value "$SMOKE_MANIFEST_PATH" routing_policy_blob)"
runner_name="$(manifest_value "$SMOKE_MANIFEST_PATH" runner_name)"
runner_os="$(manifest_value "$SMOKE_MANIFEST_PATH" runner_os)"
runner_arch="$(manifest_value "$SMOKE_MANIFEST_PATH" runner_arch)"

[[ "$smoke_schema" == 'symthaea.trusted-runner.smoke.v1' ]]
[[ "$smoke_result" == 'PASS' ]]
[[ "$smoke_repository" == 'Luminous-Dynamics/symthaea' ]]
[[ "$smoke_ref" == 'refs/heads/main' ]]
[[ "$smoke_run_id" =~ ^[0-9]+$ ]]
[[ "$smoke_run_attempt" =~ ^[0-9]+$ ]]
for value in "$smoke_main_head" "$smoke_main_tree" "$smoke_workflow_blob" "$smoke_runner_module_blob" "$smoke_routing_policy_blob"; do
  [[ "$value" =~ ^[0-9a-f]{40}$ ]]
done
for pass_key in runner_policy_eval routing_policy_eval minimal_locked_rust_check source_immutability_checked; do
  [[ "$(manifest_value "$SMOKE_MANIFEST_PATH" "$pass_key")" == 'PASS' ]]
done
[[ "$(manifest_value "$SMOKE_MANIFEST_PATH" evidence_scope)" == 'trusted-cpu-correctness-smoke-only' ]]

[[ "$smoke_main_head" == "$promoted_main_head" ]]
[[ "$smoke_main_tree" == "$promoted_main_tree" ]]
[[ "$smoke_workflow_blob" == "$promotion_smoke_workflow_blob" ]]
[[ "$smoke_runner_module_blob" == "$promotion_runner_module_blob" ]]
[[ "$smoke_routing_policy_blob" == "$promotion_routing_policy_blob" ]]

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
origin_url="$(git remote get-url origin)"
if [[ "$origin_url" != "$REPOSITORY_URL" && "$origin_url" != 'https://github.com/Luminous-Dynamics/symthaea' ]]; then
  echo "origin must be canonical public HTTPS Symthaea repository; found: $origin_url" >&2
  exit 1
fi

git diff --exit-code
git diff --cached --exit-code
test -z "$(git status --porcelain=v1 --untracked-files=all --ignored=matching)"

git -c protocol.version=2 fetch --no-tags origin \
  "+refs/heads/main:refs/remotes/origin/main" \
  "+refs/heads/${RECOVERY_BRANCH}:refs/remotes/origin/${RECOVERY_BRANCH}"

public_main="$(git rev-parse refs/remotes/origin/main)"
public_main_tree="$(git rev-parse refs/remotes/origin/main^{tree})"
public_recovery="$(git rev-parse "refs/remotes/origin/${RECOVERY_BRANCH}")"
local_head="$(git rev-parse HEAD)"
local_tree="$(git rev-parse HEAD^{tree})"

[[ "$public_main" == "$smoke_main_head" ]]
[[ "$public_main_tree" == "$smoke_main_tree" ]]
[[ "$public_recovery" == "$authorized_recovery_head" ]]
[[ "$local_head" == "$public_main" ]]
[[ "$local_tree" == "$public_main_tree" ]]

local_verifier_blob="$(git rev-parse HEAD:nix/ci/validate-trusted-runner-recovery-eligibility.sh)"
[[ "$local_verifier_blob" == "$recovery_eligibility_verifier_blob" ]]
[[ "$(git rev-parse HEAD:.github/workflows/self-hosted-runner-smoke.yml)" == "$smoke_workflow_blob" ]]
[[ "$(git rev-parse HEAD:nix/modules/github-actions-runner.nix)" == "$smoke_runner_module_blob" ]]
[[ "$(git rev-parse HEAD:nix/tests/eval-trusted-runner-routing.nix)" == "$smoke_routing_policy_blob" ]]

git -c protocol.version=2 fetch --no-tags origin \
  "+refs/heads/main:refs/remotes/origin/main" \
  "+refs/heads/${RECOVERY_BRANCH}:refs/remotes/origin/${RECOVERY_BRANCH}"
[[ "$(git rev-parse refs/remotes/origin/main)" == "$public_main" ]]
[[ "$(git rev-parse "refs/remotes/origin/${RECOVERY_BRANCH}")" == "$public_recovery" ]]

manifest="$(mktemp /tmp/symthaea-trusted-runner-recovery-eligibility-v1.XXXXXX)"
cat > "$manifest" <<EOF
schema=symthaea.trusted-runner.recovery-eligibility.v1
result=PASS
promotion_manifest_sha256=$PROMOTION_MANIFEST_SHA256
smoke_manifest_sha256=$SMOKE_MANIFEST_SHA256
authorized_recovery_head=$authorized_recovery_head
qualified_main_head=$public_main
qualified_main_tree=$public_main_tree
smoke_run_id=$smoke_run_id
smoke_run_attempt=$smoke_run_attempt
runner_name=$runner_name
runner_os=$runner_os
runner_arch=$runner_arch
recovery_eligibility_verifier_blob=$recovery_eligibility_verifier_blob
promotion_smoke_join_checked=PASS
current_main_identity_checked=PASS
current_recovery_identity_checked=PASS
local_verifier_identity_checked=PASS
refs_stable=PASS
evidence_scope=trusted-cpu-recovery-eligibility-only
EOF
eligibility_sha256="$(sha256sum "$manifest" | awk '{print $1}')"
cat "$manifest"
printf 'recovery_eligibility_manifest_sha256=%s\n' "$eligibility_sha256"
printf 'recovery_eligibility_manifest_path=%s\n' "$manifest"
