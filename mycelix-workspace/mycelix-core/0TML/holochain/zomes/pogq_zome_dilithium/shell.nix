# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Rust toolchain
    rustc
    cargo
    rustfmt
    clippy

    # Build dependencies
    pkg-config
    openssl
  ];

  shellHook = ''
    echo "🧬 Holochain WASM Build Environment (Standalone)"
    echo "═══════════════════════════════════════════════"
    echo "Rust: $(rustc --version)"
    echo ""
    echo "⚠️  IMPORTANT: Install WASM target with:"
    echo "    rustup target add wasm32-unknown-unknown"
    echo ""
    echo "Build command:"
    echo "  cargo build --target wasm32-unknown-unknown --release"
  '';
}
