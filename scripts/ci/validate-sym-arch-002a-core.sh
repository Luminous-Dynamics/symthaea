#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# One-purpose, operator-run correctness validator for the frozen SYM-ARCH-002A
# PR #57 tree. This script intentionally accepts no target SHA/ref arguments.

set -euo pipefail
IFS=$'\n\t'
umask 077

REPO_URL="https://github.com/Luminous-Dynamics/symthaea.git"
TARGET_COMMIT="f61f5ca04700db90a4f33baca5e58cd1daf068c9"
TARGET_TREE="ca6cfd632f614144f3be1362d51b2222d60a664f"
TARGET_BASE="e143c61110a70a361111bda91a986caa2924489a"
TARGET_HOSTED_WORKFLOW_BLOB="c48b655bfb1a233e29d4a1bf478985d9acb37c11"
EXPECTED_SYSTEM="x86_64-linux"
SCHEMA="symthaea.sym-arch-002a.offline-validator.v1"

if (( $# > 1 )); then
  printf 'usage: %s [NEW_EVIDENCE_DIRECTORY]\n' "$0" >&2
  exit 2
fi

for tool in git nix sha256sum awk sort tee mktemp date readlink uname; do
  command -v "$tool" >/dev/null 2>&1 || {
    printf 'missing required tool: %s\n' "$tool" >&2
    exit 2
  }
done

if [[ ! -r /etc/os-release ]]; then
  printf 'validator requires NixOS; /etc/os-release is unavailable\n' >&2
  exit 2
fi
# shellcheck disable=SC1091
. /etc/os-release
if [[ "${ID:-}" != "nixos" ]]; then
  printf 'validator requires NixOS; found ID=%s\n' "${ID:-unknown}" >&2
  exit 2
fi

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
HARNESS_COMMIT="$(git -C "$HARNESS_DIR" rev-parse HEAD)"
HARNESS_TREE="$(git -C "$HARNESS_DIR" rev-parse HEAD^{tree})"
HARNESS_REMOTE="$(git -C "$HARNESS_DIR" remote get-url origin 2>/dev/null || true)"
SANDBOX_HELPER="$HARNESS_DIR/scripts/ci/run-sym-arch-002a-sandbox.sh"
CI_SHELL="$HARNESS_DIR/nix/ci-rust-shell.nix"

case "$HARNESS_REMOTE" in
  *github.com:Luminous-Dynamics/symthaea.git|*github.com/Luminous-Dynamics/symthaea.git|*github.com/Luminous-Dynamics/symthaea)
    ;;
  *)
    printf 'harness origin is not Luminous-Dynamics/symthaea: %s\n' "$HARNESS_REMOTE" >&2
    exit 2
    ;;
esac

[[ -f "$SANDBOX_HELPER" ]] || {
  printf 'missing reviewed sandbox helper: %s\n' "$SANDBOX_HELPER" >&2
  exit 2
}
[[ -f "$CI_SHELL" ]] || {
  printf 'missing reviewed CI shell: %s\n' "$CI_SHELL" >&2
  exit 2
}

# The validator itself must come from a clean reviewed checkout. Evidence output
# is written outside this repository so these checks remain meaningful.
git -C "$HARNESS_DIR" diff --exit-code
git -C "$HARNESS_DIR" diff --cached --exit-code
test -z "$(git -C "$HARNESS_DIR" status --porcelain=v1 --untracked-files=all --ignored=matching)"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
requested_output="${1:-${XDG_STATE_HOME:-$HOME/.local/state}/symthaea-validation/sym-arch-002a-${timestamp}}"
OUTPUT_DIR="$(readlink -m "$requested_output")"
case "$OUTPUT_DIR/" in
  "$HARNESS_DIR/"*)
    printf 'evidence directory must be outside the validator checkout: %s\n' "$OUTPUT_DIR" >&2
    exit 2
    ;;
esac
if [[ -e "$OUTPUT_DIR" ]]; then
  printf 'refusing to reuse existing evidence path: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

WORK_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/symthaea-002a-offline.XXXXXX")"
SOURCE_DIR="$WORK_ROOT/frozen-source"
CARGO_HOME_DIR="$WORK_ROOT/cargo-home"
CARGO_TARGET_DIR="$WORK_ROOT/cargo-target"
SANDBOX_HOME="$WORK_ROOT/home"
SANDBOX_TMP="$WORK_ROOT/tmp"
XDG_CONFIG_DIR="$WORK_ROOT/xdg-config"
XDG_CACHE_DIR="$WORK_ROOT/xdg-cache"
XDG_DATA_DIR="$WORK_ROOT/xdg-data"
trap 'rm -rf -- "$WORK_ROOT"' EXIT
mkdir -p \
  "$SOURCE_DIR" "$CARGO_HOME_DIR" "$CARGO_TARGET_DIR" "$SANDBOX_HOME" \
  "$SANDBOX_TMP" "$XDG_CONFIG_DIR" "$XDG_CACHE_DIR" "$XDG_DATA_DIR"

# Public Git only. Move HOME/XDG into the disposable workspace before any
# network fetch so ~/.netrc, credential helpers, askpass programs, SSH-agent
# state, and operator-local Git configuration are not part of the fetch path.
export HOME="$SANDBOX_HOME"
export XDG_CONFIG_HOME="$XDG_CONFIG_DIR"
export XDG_CACHE_HOME="$XDG_CACHE_DIR"
export XDG_DATA_HOME="$XDG_DATA_DIR"
export GIT_TERMINAL_PROMPT=0
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export GCM_INTERACTIVE=never
unset SSH_AUTH_SOCK GITHUB_TOKEN GH_TOKEN GIT_ASKPASS SSH_ASKPASS || true

printf 'Fetching frozen source identity...\n'
git -C "$SOURCE_DIR" init -q
git -C "$SOURCE_DIR" remote add origin "$REPO_URL"
git -C "$SOURCE_DIR" -c protocol.version=2 fetch --no-tags origin "$TARGET_BASE" "$TARGET_COMMIT"

test "$(git -C "$SOURCE_DIR" rev-parse "$TARGET_COMMIT^{commit}")" = "$TARGET_COMMIT"
test "$(git -C "$SOURCE_DIR" rev-parse "$TARGET_COMMIT^{tree}")" = "$TARGET_TREE"
git -C "$SOURCE_DIR" merge-base --is-ancestor "$TARGET_BASE" "$TARGET_COMMIT"
test "$(git -C "$SOURCE_DIR" rev-parse "$TARGET_COMMIT:.github/workflows/sym-arch-002a-core.yml")" = "$TARGET_HOSTED_WORKFLOW_BLOB"

actual_paths="$(git -C "$SOURCE_DIR" diff --name-only "$TARGET_BASE" "$TARGET_COMMIT" | LC_ALL=C sort)"
expected_paths="$(cat <<'EOF'
.github/workflows/sym-arch-002a-core.yml
crates/domains/symthaea-psych-bench/src/experiment/confirmatory.rs
crates/domains/symthaea-psych-bench/src/experiment/mod.rs
crates/domains/symthaea-psych-bench/src/lib.rs
docs/research/SYM_ARCH_002A_EXPERIMENTAL_CORE_V1.md
EOF
)"
test "$actual_paths" = "$expected_paths"
printf '%s\n' "$actual_paths" > "$OUTPUT_DIR/target-paths.txt"

# Before evaluating Nix or compiling anything, prove that #57 did not change the
# dependency/toolchain/control-plane files used by this validator.
for path in flake.nix flake.lock rust-toolchain.toml Cargo.toml Cargo.lock; do
  base_blob="$(git -C "$SOURCE_DIR" rev-parse "$TARGET_BASE:$path")"
  target_blob="$(git -C "$SOURCE_DIR" rev-parse "$TARGET_COMMIT:$path")"
  test "$base_blob" = "$target_blob" || {
    printf 'frozen target unexpectedly changes trusted control file: %s\n' "$path" >&2
    exit 3
  }
done

git -C "$SOURCE_DIR" checkout -q --detach "$TARGET_COMMIT"
test "$(git -C "$SOURCE_DIR" rev-parse HEAD)" = "$TARGET_COMMIT"
test "$(git -C "$SOURCE_DIR" rev-parse HEAD^{tree})" = "$TARGET_TREE"

NIX_BIN="$(command -v nix)"
SAFE_PATH="/run/current-system/sw/bin"
NIX_CONFIG_VALUE="experimental-features = nix-command flakes"

safe_nix() {
  env -i \
    PATH="$SAFE_PATH" \
    HOME="$SANDBOX_HOME" \
    USER=validator \
    LOGNAME=validator \
    TMPDIR="$SANDBOX_TMP" \
    XDG_CONFIG_HOME="$XDG_CONFIG_DIR" \
    XDG_CACHE_HOME="$XDG_CACHE_DIR" \
    XDG_DATA_HOME="$XDG_DATA_DIR" \
    GIT_TERMINAL_PROMPT=0 \
    GIT_CONFIG_NOSYSTEM=1 \
    GIT_CONFIG_GLOBAL=/dev/null \
    GCM_INTERACTIVE=never \
    NIX_CONFIG="$NIX_CONFIG_VALUE" \
    HARNESS_DIR="$HARNESS_DIR" \
    CARGO_HOME="$CARGO_HOME_DIR" \
    CARGO_TARGET_DIR="$CARGO_TARGET_DIR" \
    "$NIX_BIN" "$@"
}

cd "$SOURCE_DIR"
system="$(safe_nix eval --impure --raw --expr builtins.currentSystem)"
if [[ "$system" != "$EXPECTED_SYSTEM" ]]; then
  printf 'validator requires %s; found %s\n' "$EXPECTED_SYSTEM" "$system" >&2
  exit 2
fi
nixpkgs_rev="$(safe_nix eval --raw --expr 'let l = builtins.fromJSON (builtins.readFile ./flake.lock); n = l.nodes.root.inputs.nixpkgs; in (builtins.getAttr n l.nodes).locked.rev')"
rust_channel="$(safe_nix eval --raw --expr '(builtins.fromTOML (builtins.readFile ./rust-toolchain.toml)).toolchain.channel')"
flake_lock_sha="$(sha256sum flake.lock | awk '{print $1}')"
rust_toolchain_sha="$(sha256sum rust-toolchain.toml | awk '{print $1}')"
cargo_lock_sha="$(sha256sum Cargo.lock | awk '{print $1}')"
hosted_workflow_sha="$(sha256sum .github/workflows/sym-arch-002a-core.yml | awk '{print $1}')"
sandbox_helper_sha="$(sha256sum "$SANDBOX_HELPER" | awk '{print $1}')"
ci_shell_sha="$(sha256sum "$CI_SHELL" | awk '{print $1}')"
nix_version="$(safe_nix --version)"
git_version="$(git --version)"
kernel="$(uname -srmo)"

PROVENANCE_FILE="$OUTPUT_DIR/provenance.txt"
cat > "$PROVENANCE_FILE" <<EOF
validator_started_utc=$timestamp
os_id=nixos
nix_system=$system
kernel=$kernel
nix_version=$nix_version
git_version=$git_version
harness_commit=$HARNESS_COMMIT
harness_tree=$HARNESS_TREE
sandbox_helper_sha256=$sandbox_helper_sha
ci_shell_sha256=$ci_shell_sha
target_commit=$TARGET_COMMIT
target_tree=$TARGET_TREE
target_base=$TARGET_BASE
nixpkgs_rev=$nixpkgs_rev
rust_channel=$rust_channel
flake_lock_sha256=$flake_lock_sha
cargo_lock_sha256=$cargo_lock_sha
rust_toolchain_sha256=$rust_toolchain_sha
hosted_workflow_blob=$TARGET_HOSTED_WORKFLOW_BLOB
hosted_workflow_sha256=$hosted_workflow_sha
EOF

GATE_LOG="$OUTPUT_DIR/gate.log"
printf 'Running frozen correctness gate on %s inside Bubblewrap sandbox...\n' "$TARGET_COMMIT"
(
  safe_nix develop --no-write-lock-file --impure \
    --expr 'let
      f = builtins.getFlake (toString ./.);
      harness = builtins.getEnv "HARNESS_DIR";
      shellFile = builtins.toPath (harness + "/nix/ci-rust-shell.nix");
    in import shellFile {
      nixpkgs = f.inputs.nixpkgs;
      rust-overlay = f.inputs.rust-overlay;
      toolchainFile = ./rust-toolchain.toml;
      system = builtins.currentSystem;
    }' \
    --command bash "$SANDBOX_HELPER" \
      "$SOURCE_DIR" "$CARGO_HOME_DIR" "$CARGO_TARGET_DIR" "$SANDBOX_HOME" "$SANDBOX_TMP"
) 2>&1 | tee "$GATE_LOG"

# Target and harness must remain immutable. Cargo/Nix mutable state lives outside
# both worktrees; target execution saw the source only through a read-only bind.
test "$(git -C "$SOURCE_DIR" rev-parse HEAD)" = "$TARGET_COMMIT"
test "$(git -C "$SOURCE_DIR" rev-parse HEAD^{tree})" = "$TARGET_TREE"
git -C "$SOURCE_DIR" diff --exit-code
git -C "$SOURCE_DIR" diff --cached --exit-code
test -z "$(git -C "$SOURCE_DIR" status --porcelain=v1 --untracked-files=all --ignored=matching)"

git -C "$HARNESS_DIR" diff --exit-code
git -C "$HARNESS_DIR" diff --cached --exit-code
test -z "$(git -C "$HARNESS_DIR" status --porcelain=v1 --untracked-files=all --ignored=matching)"

gate_log_sha="$(sha256sum "$GATE_LOG" | awk '{print $1}')"
paths_sha="$(sha256sum "$OUTPUT_DIR/target-paths.txt" | awk '{print $1}')"
provenance_sha="$(sha256sum "$PROVENANCE_FILE" | awk '{print $1}')"
manifest="$OUTPUT_DIR/manifest.txt"
completed_utc="$(date -u +%Y%m%dT%H%M%SZ)"
cat > "$manifest" <<EOF
schema=$SCHEMA
result=PASS
executor=operator-nixos
correctness_only=true
validator_started_utc=$timestamp
validator_completed_utc=$completed_utc
harness_commit=$HARNESS_COMMIT
harness_tree=$HARNESS_TREE
sandbox_helper_sha256=$sandbox_helper_sha
ci_shell_sha256=$ci_shell_sha
target_commit=$TARGET_COMMIT
target_tree=$TARGET_TREE
target_base=$TARGET_BASE
hosted_workflow_blob=$TARGET_HOSTED_WORKFLOW_BLOB
hosted_workflow_sha256=$hosted_workflow_sha
nix_system=$system
nixpkgs_rev=$nixpkgs_rev
rust_channel=$rust_channel
flake_lock_sha256=$flake_lock_sha
cargo_lock_sha256=$cargo_lock_sha
rust_toolchain_sha256=$rust_toolchain_sha
target_paths_sha256=$paths_sha
provenance_sha256=$provenance_sha
gate_log_sha256=$gate_log_sha
gate=dependency-prefetch+bubblewrap-sandbox+rustfmt+experiment-tests+psych-bench-lib-check
cargo_dependency_mode=locked+offline-after-fetch
sandbox=bubblewrap-unshare-all
sandbox_network=unshared-after-prefetch
sandbox_source=read-only
sandbox_nix_store=read-only
sandbox_capabilities=dropped
sandbox_nested_userns=disabled
EOF

manifest_sha="$(sha256sum "$manifest" | awk '{print $1}')"
printf '%s  %s\n' "$manifest_sha" "$(basename "$manifest")" > "$OUTPUT_DIR/manifest.sha256"
sha256sum \
  "$GATE_LOG" "$OUTPUT_DIR/target-paths.txt" "$PROVENANCE_FILE" "$manifest" \
  > "$OUTPUT_DIR/SHA256SUMS"

printf '\nSYM-ARCH-002A offline validator: PASS\n'
printf 'target commit: %s\n' "$TARGET_COMMIT"
printf 'target tree:   %s\n' "$TARGET_TREE"
printf 'harness:       %s\n' "$HARNESS_COMMIT"
printf 'manifest SHA:  %s\n' "$manifest_sha"
printf 'evidence dir:  %s\n' "$OUTPUT_DIR"
printf '\nCorrectness-only evidence; not architecture-performance evidence.\n'
