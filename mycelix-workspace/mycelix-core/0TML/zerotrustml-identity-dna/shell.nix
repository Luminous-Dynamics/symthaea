# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{ pkgs ? import <nixpkgs> {
    overlays = [ (import (builtins.fetchTarball https://github.com/oxalica/rust-overlay/archive/master.tar.gz)) ];
  }
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Rust with WASM support
    (rust-bin.stable.latest.default.override {
      extensions = [ "rust-src" ];
      targets = [ "wasm32-unknown-unknown" ];
    })

    # Build dependencies
    gcc
    pkg-config
    openssl
    perl
  ];

  shellHook = ''
    echo "🍄 Holochain DNA Build Environment"
    echo "   Rust: $(rustc --version)"
    echo "   Cargo: $(cargo --version)"
    echo "   WASM target: available"
    echo ""
    echo "📦 To build all zomes:"
    echo "   cargo build --release --target wasm32-unknown-unknown"
    echo ""
    echo "🔧 To pack DNA:"
    echo "   hc dna pack ."
    echo ""
  '';
}
