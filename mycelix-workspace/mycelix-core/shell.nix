# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Rust toolchain
    cargo
    rustc
    rustfmt
    rust-analyzer
    
    # WASM tools
    wasm-pack
    wasm-bindgen-cli
    
    # Build dependencies
    pkg-config
    openssl
    protobuf
    
    # Node.js
    nodejs_20
    nodePackages.pnpm
    
    # Python for FL
    python311
    python311Packages.pip
    
    # Utilities
    jq
    curl
    wget
  ];
  
  shellHook = ''
    echo "🧬 Holochain Development Environment"
    echo "===================================="
    
    # Install Holochain via cargo if not present
    if ! command -v holochain &> /dev/null; then
      echo "📦 Installing Holochain via cargo..."
      cargo install holochain --version 0.5.6 2>/dev/null || true
      cargo install holochain_cli --version 0.5.0 2>/dev/null || true
      cargo install lair_keystore --version 0.5.4 2>/dev/null || true
    fi
    
    # Check versions
    echo "Installed versions:"
    holochain --version 2>/dev/null || echo "  Holochain: Not installed"
    hc --version 2>/dev/null || echo "  hc CLI: Not installed"
    cargo --version
    node --version
    python3 --version
    
    export PATH=$HOME/.cargo/bin:$PATH
  '';
}
