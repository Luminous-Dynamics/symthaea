#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Focused qualification lane for the Spore boot stack.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PACKAGES=(
  symthaea-boot-protocol
  symthaea-boot-observer
  symthaea-quicken-fb
  symthaea-boot-control
  symthaea-boot-input
)

run_cargo_for_each() {
    local command="$1"
    shift
    for package in "${PACKAGES[@]}"; do
        echo "== cargo $command -p $package $* =="
        cargo "$command" -p "$package" "$@"
    done
}

for package in "${PACKAGES[@]}"; do
    echo "== cargo fmt -p $package --check =="
    cargo fmt -p "$package" --check
 done

run_cargo_for_each check --all-targets
run_cargo_for_each test

for package in "${PACKAGES[@]}"; do
    echo "== cargo clippy -p $package --all-targets -- -D warnings =="
    cargo clippy -p "$package" --all-targets -- -D warnings
 done

echo "== deterministic headless smoke =="
cargo run -p symthaea-quicken-fb --release --bin spore-boot-bench -- \
    --width 320 \
    --height 180 \
    --warmup-frames 2 \
    --frames 8 \
    --seed qualification-v1 >/tmp/spore-boot-bench-smoke.json

if command -v nix-instantiate >/dev/null 2>&1; then
    echo "== Nix parse: quicken-fb module =="
    nix-instantiate --parse nix/modules/quicken-fb.nix >/dev/null

    echo "== Nix parse: boot observer module =="
    nix-instantiate --parse nix/modules/symthaea-boot-observer.nix >/dev/null
else
    echo "SKIP: nix-instantiate not available"
fi

if [[ "${1:-}" == "--vm" ]]; then
    if ! command -v nix >/dev/null 2>&1; then
        echo "ERROR: --vm requested but nix is unavailable" >&2
        exit 1
    fi
    echo "== NixOS quicken VM build =="
    nix build .#vm-test
fi

echo "PASS: focused Spore boot stack qualification"
