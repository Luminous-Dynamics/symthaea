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
    rust-analyzer

    # Build dependencies
    gcc
    pkg-config
    openssl
    zlib
    stdenv.cc.cc.lib

    # WASM linker (required for wasm32-unknown-unknown target)
    lld

    # Optional: Holochain CLI for DNA packing
    # holochain
  ];

  shellHook = ''
    echo "🍄 Mycelix Mail Development Environment"
    echo "Rust version: $(rustc --version)"
    echo "Targets: $(rustc --print target-list | grep wasm32)"
  '';
}
