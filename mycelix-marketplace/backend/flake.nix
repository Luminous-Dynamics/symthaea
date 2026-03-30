# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "Mycelix Marketplace - P2P Marketplace on Holochain";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # Rust overlay for proper Rust toolchain with WASM
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";

    # Holochain official flake - pinned to 0.6.0 to match our hApp
    holochain-flake.url = "github:holochain/holochain?ref=holochain-0.6.0";
    holochain-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, holochain-flake }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Get Holochain binaries from the official flake
        holochainPkgs = holochain-flake.packages.${system};

        # Rust toolchain with WASM target using rust-overlay
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
          targets = [ "wasm32-unknown-unknown" ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Rust toolchain with WASM support (from rust-overlay)
            rustToolchain

            # Rust tools (already included in rustToolchain, but keeping for completeness)
            # rust-analyzer  # Already in rustToolchain extensions
            rustfmt
            clippy

            # WASM build dependencies
            gcc          # C compiler for build scripts
            lld          # LLVM linker for WASM (CRITICAL!)
            binaryen     # wasm-opt for optimizing WASM
            wasm-pack    # WASM build tool

            # Holochain tools from official flake
            holochainPkgs.holochain  # Holochain conductor (includes hc CLI)
            holochainPkgs.lair-keystore  # Lair keystore for key management

            # Development tools
            nodejs_20    # For frontend development
            pkg-config
            openssl

            # Optional but useful
            jq           # JSON processing
            just         # Command runner (optional)
          ];

          shellHook = ''
            echo "🚀 Mycelix Marketplace Development Environment (rust-overlay)"
            echo ""

            echo "Rust Version: $(rustc --version)"
            echo "Cargo Version: $(cargo --version)"
            echo "WASM Target: wasm32-unknown-unknown (built-in)"
            echo "LLD Linker: $(which lld || echo 'NOT FOUND')"
            echo ""
            echo "Available commands:"
            echo "  cargo build --release --target wasm32-unknown-unknown  # Build WASM zomes"
            echo "  cargo build --release --workspace                      # Build all zomes"
            echo ""
            echo "Note: WASM target is pre-installed via rust-overlay"
            echo ""
          '';

          # Set WASM target as default for easier builds
          CARGO_BUILD_TARGET = "wasm32-unknown-unknown";
        };
      }
    );
}
