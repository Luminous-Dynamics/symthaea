#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Executes the already-authenticated SYM-ARCH-002A correctness gate inside a
# Bubblewrap namespace. Dependency prefetch happens before the sandbox; target
# formatting/build/test execution then has no network and a read-only source.

set -euo pipefail
IFS=$'\n\t'
umask 077
ulimit -c 0

if (( $# != 5 )); then
  printf 'usage: %s SOURCE_DIR CARGO_HOME CARGO_TARGET SANDBOX_HOME SANDBOX_TMP\n' "$0" >&2
  exit 2
fi

SOURCE_DIR="$(readlink -f "$1")"
CARGO_HOME_DIR="$(readlink -f "$2")"
CARGO_TARGET_DIR="$(readlink -f "$3")"
SANDBOX_HOME="$(readlink -f "$4")"
SANDBOX_TMP="$(readlink -f "$5")"

for tool in bwrap cargo rustfmt bash touch readlink awk timeout; do
  command -v "$tool" >/dev/null 2>&1 || {
    printf 'sandbox gate missing required tool: %s\n' "$tool" >&2
    exit 2
  }
done

for dir in "$SOURCE_DIR" "$CARGO_HOME_DIR" "$CARGO_TARGET_DIR" "$SANDBOX_HOME" "$SANDBOX_TMP"; do
  [[ -d "$dir" ]] || {
    printf 'sandbox gate requires existing directory: %s\n' "$dir" >&2
    exit 2
  }
done

case "$CARGO_HOME_DIR/" in "$SOURCE_DIR/"*) echo 'CARGO_HOME must be outside source' >&2; exit 2;; esac
case "$CARGO_TARGET_DIR/" in "$SOURCE_DIR/"*) echo 'CARGO_TARGET_DIR must be outside source' >&2; exit 2;; esac
case "$SANDBOX_HOME/" in "$SOURCE_DIR/"*) echo 'sandbox HOME must be outside source' >&2; exit 2;; esac
case "$SANDBOX_TMP/" in "$SOURCE_DIR/"*) echo 'sandbox TMP must be outside source' >&2; exit 2;; esac

printf 'bubblewrap_version=%s\n' "$(bwrap --version)"
printf 'cargo_version=%s\n' "$(cargo --version)"
printf 'rustfmt_version=%s\n' "$(rustfmt --version)"
printf 'sandbox_timeout=20m\n'

# Resolve/download the locked dependency graph before target code executes.
# `cargo fetch` does not compile dependencies or run build scripts/tests.
cd "$SOURCE_DIR"
cargo fetch --locked

# The parent Nix environment was created through safe_nix/env -i, so PATH begins
# from a sanitized base. Retain only Nix-store entries inside the target sandbox.
sandbox_path=''
old_ifs="$IFS"
IFS=':'
for entry in $PATH; do
  case "$entry" in
    /nix/store/*)
      if [[ -z "$sandbox_path" ]]; then
        sandbox_path="$entry"
      else
        sandbox_path="$sandbox_path:$entry"
      fi
      ;;
  esac
done
IFS="$old_ifs"
[[ -n "$sandbox_path" ]] || {
  echo 'sandbox PATH has no Nix-store entries' >&2
  exit 2
}

# Bubblewrap starts with an empty mount namespace. Expose only the Nix store,
# authenticated source (read-only), and disposable Cargo/home/tmp state. Bind
# mounts precede --proc intentionally; Bubblewrap has had order-sensitive /proc
# behavior, so the new procfs is created only after all host binds are fixed.
# The 20-minute ceiling mirrors the hosted SYM-ARCH-002A workflow.
timeout --signal=TERM --kill-after=30s 20m \
  bwrap \
    --die-with-parent \
    --new-session \
    --unshare-all \
    --disable-userns \
    --assert-userns-disabled \
    --cap-drop ALL \
    --dir /nix \
    --ro-bind /nix/store /nix/store \
    --ro-bind "$SOURCE_DIR" /workspace \
    --bind "$CARGO_HOME_DIR" /cargo-home \
    --bind "$CARGO_TARGET_DIR" /cargo-target \
    --bind "$SANDBOX_HOME" /home \
    --bind "$SANDBOX_TMP" /tmp \
    --proc /proc \
    --dev /dev \
    --tmpfs /dev/shm \
    --chdir /workspace \
    --setenv PATH "$sandbox_path" \
    --setenv HOME /home \
    --setenv TMPDIR /tmp \
    --setenv CARGO_HOME /cargo-home \
    --setenv CARGO_TARGET_DIR /cargo-target \
    --setenv CARGO_NET_OFFLINE true \
    --unsetenv HARNESS_DIR \
    --unsetenv SSH_AUTH_SOCK \
    --unsetenv GITHUB_TOKEN \
    --unsetenv GH_TOKEN \
    -- \
    bash -c '
      set -euo pipefail

      # Fail closed if the namespace unexpectedly has usable external networking.
      if bash -c "exec 3<>/dev/tcp/1.1.1.1/53" 2>/dev/null; then
        echo "sandbox unexpectedly has external network access" >&2
        exit 4
      fi

      # Effective capabilities must be empty after --cap-drop ALL.
      cap_eff="$(awk "/^CapEff:/ { print \$2 }" /proc/self/status)"
      [[ "$cap_eff" =~ ^0+$ ]] || {
        printf "sandbox retained effective capabilities: %s\n" "$cap_eff" >&2
        exit 4
      }

      # Host identity/credential roots are deliberately absent from the namespace.
      for forbidden in /root /etc/shadow /run/current-system /sys; do
        if [[ -e "$forbidden" ]]; then
          printf "sandbox unexpectedly exposes host path: %s\n" "$forbidden" >&2
          exit 4
        fi
      done

      # The authenticated target must be physically read-only inside the sandbox.
      if touch /workspace/.symthaea-validator-write-probe 2>/dev/null; then
        echo "sandbox source mount is unexpectedly writable" >&2
        rm -f /workspace/.symthaea-validator-write-probe || true
        exit 4
      fi

      rustfmt --edition 2024 --check \
        crates/domains/symthaea-psych-bench/src/experiment/mod.rs \
        crates/domains/symthaea-psych-bench/src/experiment/confirmatory.rs \
        crates/domains/symthaea-psych-bench/src/lib.rs

      cargo test --locked --offline -p symthaea-psych-bench --lib experiment -- --nocapture
      cargo check --locked --offline -p symthaea-psych-bench --lib
    '
