#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Focused qualification lane for the Spore boot/lifecycle stack.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PACKAGES=(
  symthaea-boot-protocol
  symthaea-boot-observer
  symthaea-quicken-fb
  symthaea-boot-control
  symthaea-boot-input
  symthaea-boot-ecology-live
  symthaea-boot-visual-clock
  symthaea-boot-presentation
  symthaea-spore-continuity
)

# Projection is the final renderer-independent seam and is present only on the
# descendant integration branch. Keep one qualification entrypoint usable on
# both the convergence base and that descendant without duplicating package
# lists in workflow YAML.
if [[ -f crates/domains/symthaea-boot-render-projection/Cargo.toml ]]; then
    PACKAGES+=(symthaea-boot-render-projection)
fi

format_packages() {
    local mode="${1:-check}"
    for package in "${PACKAGES[@]}"; do
        if [[ "$mode" == "apply" ]]; then
            echo "== cargo fmt -p $package =="
            cargo fmt -p "$package"
        else
            echo "== cargo fmt -p $package --check =="
            cargo fmt -p "$package" --check
        fi
    done
}

# CI uses this mode to produce an exact rustfmt candidate without pretending
# that a formatter-mutated checkout is the committed state being qualified.
if [[ "${SPORE_BOOT_FORMAT_ONLY:-0}" == "1" ]]; then
    format_packages apply
    exit 0
fi

run_cargo_for_each() {
    local command="$1"
    shift
    for package in "${PACKAGES[@]}"; do
        echo "== cargo $command --locked -p $package $* =="
        cargo "$command" --locked -p "$package" "$@"
    done
}

# Fail immediately if Cargo.toml/workspace membership and Cargo.lock disagree.
# Nix builds use the workspace lock, so an unlocked success is not qualification.
echo "== cargo metadata --locked =="
cargo metadata --locked --no-deps --format-version 1 >/dev/null

format_packages check

run_cargo_for_each check --all-targets
run_cargo_for_each test

for package in "${PACKAGES[@]}"; do
    echo "== cargo clippy --locked -p $package --all-targets -- -D warnings =="
    cargo clippy --locked -p "$package" --all-targets -- -D warnings
done

echo "== deterministic headless smoke =="
BENCH_SMOKE="$(mktemp /tmp/spore-boot-bench-smoke.XXXXXX.json)"
trap 'rm -f "$BENCH_SMOKE"' EXIT
cargo run --locked -p symthaea-quicken-fb --release --bin spore-boot-bench -- \
    --width 320 \
    --height 180 \
    --warmup-frames 2 \
    --frames 8 \
    --seed qualification-v1 >"$BENCH_SMOKE"
grep -q '"schema": "spore-boot-headless-benchmark-v1"' "$BENCH_SMOKE"
grep -q '"total_cpu_frame"' "$BENCH_SMOKE"

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

echo "PASS: focused Spore boot/lifecycle stack qualification"
