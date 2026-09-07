#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Queue-neutral host-side bootstrap validator for the trusted CPU runner.
# This script is deliberately NOT a GitHub Actions qualification path. It proves
# only that the recovery branch's runner/routing policy and minimal correctness
# shell evaluate on the isolated host before the infrastructure exists on main.

set -euo pipefail
umask 077

REPOSITORY_URL='https://github.com/Luminous-Dynamics/symthaea.git'
RECOVERY_BRANCH='ci/nixos-ephemeral-runner-v1'
EXPECTED_HEAD="${SYMTHAEA_TRUSTED_RECOVERY_EXPECTED_HEAD:-}"

if [[ "${GITHUB_ACTIONS:-}" == 'true' ]]; then
  echo 'bootstrap validator must run directly on the isolated host, not inside GitHub Actions' >&2
  exit 1
fi

if [[ ! "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'SYMTHAEA_TRUSTED_RECOVERY_EXPECTED_HEAD must be an explicitly operator-authorized 40-hex commit SHA' >&2
  exit 1
fi

for command in git nix sha256sum mktemp awk bash; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "required command missing: $command" >&2
    exit 1
  }
done

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

origin_url="$(git remote get-url origin)"
if [[ "$origin_url" != "$REPOSITORY_URL" && "$origin_url" != 'https://github.com/Luminous-Dynamics/symthaea' ]]; then
  echo "origin must be canonical public HTTPS Symthaea repository; found: $origin_url" >&2
  exit 1
fi

initial_head="$(git rev-parse HEAD)"
initial_tree="$(git rev-parse HEAD^{tree})"

if [[ "$initial_head" != "$EXPECTED_HEAD" ]]; then
  echo 'checked-out recovery generation is not the operator-authorized generation' >&2
  printf 'authorized_head=%s\nworking_head=%s\n' "$EXPECTED_HEAD" "$initial_head" >&2
  exit 1
fi

# Bootstrap evaluation must start from a pristine tracked/untracked/ignored
# source tree. All Cargo/Nix build state belongs outside the repository.
git diff --exit-code
git diff --cached --exit-code
test -z "$(git status --porcelain=v1 --untracked-files=all --ignored=matching)"

refresh_public_refs() {
  git -c protocol.version=2 fetch --no-tags origin \
    "+refs/heads/main:refs/remotes/origin/main" \
    "+refs/heads/${RECOVERY_BRANCH}:refs/remotes/origin/${RECOVERY_BRANCH}"
}

refresh_public_refs
main_head_start="$(git rev-parse refs/remotes/origin/main)"
main_tree_start="$(git rev-parse refs/remotes/origin/main^{tree})"
recovery_head_start="$(git rev-parse "refs/remotes/origin/${RECOVERY_BRANCH}")"

if [[ "$recovery_head_start" != "$EXPECTED_HEAD" ]]; then
  echo 'published recovery branch does not equal the operator-authorized generation' >&2
  printf 'authorized_head=%s\nrecovery_head=%s\n' "$EXPECTED_HEAD" "$recovery_head_start" >&2
  exit 1
fi

if [[ "$initial_head" != "$recovery_head_start" ]]; then
  echo 'working tree is not the current published trusted-runner recovery head' >&2
  printf 'working_head=%s\nrecovery_head=%s\n' "$initial_head" "$recovery_head_start" >&2
  exit 1
fi

git merge-base --is-ancestor "$main_head_start" "$recovery_head_start" || {
  echo 'trusted-runner recovery branch is behind/diverged from current main' >&2
  printf 'main_head=%s\nrecovery_head=%s\n' "$main_head_start" "$recovery_head_start" >&2
  exit 1
}

# Freeze the bootstrap merge surface. Stage A must fail if application code,
# scientific-result code, root Cargo/toolchain state, unrelated Nix modules, or
# any unreviewed workflow enters the recovery branch.
expected_paths="$(cat <<'EOF'
.github/workflows/self-hosted-ai-assurance-foundation-recovery.yml
.github/workflows/self-hosted-rca-canonical-lineage-recovery.yml
.github/workflows/self-hosted-runner-smoke.yml
.github/workflows/self-hosted-sym-arch-002a-core-recovery.yml
docs/operations/AI_ASSURANCE_TRUSTED_RECOVERY.md
docs/operations/GITHUB_ACTIONS_NIXOS_RUNNER.md
docs/operations/RCA_CANONICAL_LINEAGE_TRUSTED_RECOVERY.md
docs/operations/SYM_ARCH_002A_TRUSTED_RECOVERY.md
docs/operations/TRUSTED_CPU_RUNNER_BOOTSTRAP.md
docs/operations/TRUSTED_CPU_RUNNER_HOST_LIFECYCLE.md
nix/ci-rust-shell.nix
nix/ci/validate-trusted-runner-bootstrap.sh
nix/ci/validate-trusted-runner-promotion.sh
nix/modules/default.nix
nix/modules/github-actions-runner.nix
nix/tests/eval-github-actions-runner.nix
nix/tests/eval-trusted-runner-routing.nix
EOF
)"
actual_paths="$(git diff --name-only "$main_head_start" "$recovery_head_start" | LC_ALL=C sort)"
if [[ "$actual_paths" != "$expected_paths" ]]; then
  echo 'trusted-runner bootstrap branch diff surface changed' >&2
  echo 'expected:' >&2
  printf '%s\n' "$expected_paths" >&2
  echo 'actual:' >&2
  printf '%s\n' "$actual_paths" >&2
  exit 1
fi

diff_paths_sha256="$(printf '%s\n' "$actual_paths" | sha256sum | awk '{print $1}')"
runner_module_blob="$(git rev-parse "$recovery_head_start:nix/modules/github-actions-runner.nix")"
routing_policy_blob="$(git rev-parse "$recovery_head_start:nix/tests/eval-trusted-runner-routing.nix")"
smoke_workflow_blob="$(git rev-parse "$recovery_head_start:.github/workflows/self-hosted-runner-smoke.yml")"
bootstrap_validator_blob="$(git rev-parse "$recovery_head_start:nix/ci/validate-trusted-runner-bootstrap.sh")"
promotion_verifier_blob="$(git rev-parse "$recovery_head_start:nix/ci/validate-trusted-runner-promotion.sh")"
host_lifecycle_contract_blob="$(git rev-parse "$recovery_head_start:docs/operations/TRUSTED_CPU_RUNNER_HOST_LIFECYCLE.md")"

flake_lock_sha256="$(sha256sum flake.lock | awk '{print $1}')"
rust_toolchain_sha256="$(sha256sum rust-toolchain.toml | awk '{print $1}')"
nixpkgs_rev="$(nix eval --raw --expr 'let l = builtins.fromJSON (builtins.readFile ./flake.lock); n = l.nodes.root.inputs.nixpkgs; in (builtins.getAttr n l.nodes).locked.rev')"
rust_channel="$(nix eval --raw --expr '(builtins.fromTOML (builtins.readFile ./rust-toolchain.toml)).toolchain.channel')"
host_nix_system="$(nix eval --raw --impure --expr builtins.currentSystem)"
host_nix_version="$(nix --version | awk '{print $3}')"

printf 'bootstrap_authorized_head=%s\n' "$EXPECTED_HEAD"
printf 'bootstrap_recovery_head=%s\n' "$recovery_head_start"
printf 'bootstrap_main_head=%s\n' "$main_head_start"
printf 'bootstrap_main_tree=%s\n' "$main_tree_start"
printf 'bootstrap_source_tree=%s\n' "$initial_tree"
printf 'bootstrap_promotion_required_ancestor=%s\n' "$EXPECTED_HEAD"
printf 'bootstrap_promotion_expected_main_tree=%s\n' "$initial_tree"
printf 'bootstrap_diff_paths_sha256=%s\n' "$diff_paths_sha256"
printf 'bootstrap_nixpkgs_rev=%s\n' "$nixpkgs_rev"
printf 'bootstrap_rust_channel=%s\n' "$rust_channel"
printf 'bootstrap_host_nix_system=%s\n' "$host_nix_system"
printf 'bootstrap_host_nix_version=%s\n' "$host_nix_version"
printf 'bootstrap_host_lifecycle_contract_blob=%s\n' "$host_lifecycle_contract_blob"
printf 'bootstrap_promotion_verifier_blob=%s\n' "$promotion_verifier_blob"

# These evaluations contact no GitHub API and consume no runner credential.
nix build --no-link --no-write-lock-file \
  --impure \
  --expr 'let f = builtins.getFlake (toString ./.); pkgs = import f.inputs.nixpkgs { system = builtins.currentSystem; }; in import ./nix/tests/eval-github-actions-runner.nix { inherit pkgs; }'

nix build --no-link --no-write-lock-file \
  --impure \
  --expr 'let f = builtins.getFlake (toString ./.); pkgs = import f.inputs.nixpkgs { system = builtins.currentSystem; }; in import ./nix/tests/eval-trusted-runner-routing.nix { inherit pkgs; }'

state_dir="$(mktemp -d)"
cleanup() {
  rm -rf -- "$state_dir"
}
trap cleanup EXIT

export CARGO_HOME="$state_dir/cargo-home"
export CARGO_TARGET_DIR="$state_dir/cargo-target"
mkdir -p "$CARGO_HOME" "$CARGO_TARGET_DIR"

nix develop --no-write-lock-file \
  --impure \
  --expr 'let f = builtins.getFlake (toString ./.); in import ./nix/ci-rust-shell.nix { nixpkgs = f.inputs.nixpkgs; rust-overlay = f.inputs.rust-overlay; system = builtins.currentSystem; }' \
  --command bash -c '
    set -euo pipefail
    rustc --version
    cargo --version
    rustfmt --version
    cargo clippy --version
    cargo metadata --locked --format-version 1 > /dev/null
    cargo check --locked -p symthaea-psych-bench --lib
  '

# Bootstrap validation may populate only temp/Nix state, never the repository.
test "$(git rev-parse HEAD)" = "$initial_head"
test "$(git rev-parse HEAD^{tree})" = "$initial_tree"
git diff --exit-code
git diff --cached --exit-code
test -z "$(git status --porcelain=v1 --untracked-files=all --ignored=matching)"

# Close the ref-movement window. A PASS is valid only if both public refs are
# unchanged across the complete validation interval. If main or the recovery
# branch moved, rerun Stage A on the new exact head rather than carrying a stale
# PASS forward.
refresh_public_refs
main_head_end="$(git rev-parse refs/remotes/origin/main)"
recovery_head_end="$(git rev-parse "refs/remotes/origin/${RECOVERY_BRANCH}")"

if [[ "$main_head_start" != "$main_head_end" || "$recovery_head_start" != "$recovery_head_end" ]]; then
  echo 'public branch state changed during bootstrap validation; refusing stale PASS' >&2
  printf 'main_start=%s\nmain_end=%s\n' "$main_head_start" "$main_head_end" >&2
  printf 'recovery_start=%s\nrecovery_end=%s\n' "$recovery_head_start" "$recovery_head_end" >&2
  exit 1
fi

if [[ "$recovery_head_end" != "$EXPECTED_HEAD" ]]; then
  echo 'published recovery branch no longer equals the operator-authorized generation' >&2
  printf 'authorized_head=%s\nrecovery_head=%s\n' "$EXPECTED_HEAD" "$recovery_head_end" >&2
  exit 1
fi

if [[ "$initial_head" != "$recovery_head_end" ]]; then
  echo 'working tree no longer matches current published recovery head' >&2
  exit 1
fi

git merge-base --is-ancestor "$main_head_end" "$recovery_head_end" || {
  echo 'recovery branch ceased to contain current main during validation' >&2
  exit 1
}

manifest="$(mktemp /tmp/symthaea-trusted-runner-bootstrap-v5.XXXXXX)"
cat > "$manifest" <<EOF
schema=symthaea.trusted-runner.bootstrap.v5
result=PASS
repository=$REPOSITORY_URL
recovery_branch=$RECOVERY_BRANCH
operator_authorized_head=$EXPECTED_HEAD
recovery_head=$recovery_head_end
source_tree=$initial_tree
main_head=$main_head_end
main_tree=$main_tree_start
promotion_required_ancestor=$EXPECTED_HEAD
promotion_expected_main_tree=$initial_tree
recovery_diff_paths_sha256=$diff_paths_sha256
runner_module_blob=$runner_module_blob
routing_policy_blob=$routing_policy_blob
smoke_workflow_blob=$smoke_workflow_blob
bootstrap_validator_blob=$bootstrap_validator_blob
promotion_verifier_blob=$promotion_verifier_blob
host_lifecycle_contract_blob=$host_lifecycle_contract_blob
nixpkgs_rev=$nixpkgs_rev
rust_channel=$rust_channel
host_nix_system=$host_nix_system
host_nix_version=$host_nix_version
flake_lock_sha256=$flake_lock_sha256
rust_toolchain_sha256=$rust_toolchain_sha256
operator_authorization_checked=PASS
runner_policy_eval=PASS
routing_policy_eval=PASS
minimal_locked_rust_check=PASS
refs_unchanged_during_validation=PASS
evidence_scope=runner-bootstrap-correctness-only
EOF

manifest_sha256="$(sha256sum "$manifest" | awk '{print $1}')"
cat "$manifest"
printf 'bootstrap_manifest_sha256=%s\n' "$manifest_sha256"
printf 'bootstrap_manifest_path=%s\n' "$manifest"
