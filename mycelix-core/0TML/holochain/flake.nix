# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "Holochain Workspace with WASM Support for PoGQ Zome";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Use stable Rust with WASM target for Holochain DNA compilation
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          targets = [ "wasm32-unknown-unknown" ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustToolchain
            cargo
            rust-analyzer

            # Build dependencies
            pkg-config
            openssl

            # Go compiler (required for Holochain networking)
            go

            # Development tools
            cargo-watch
            cargo-edit
          ];

          shellHook = ''
            echo "🧬 Holochain WASM Build Environment"
            echo "════════════════════════════════════"
            echo "Rust: $(rustc --version)"
            echo "Cargo: $(cargo --version)"
            echo "Go: $(go version)"
            echo ""
            echo "WASM target: $(rustc --print target-list | grep wasm32-unknown-unknown)"
            echo ""
            echo "Build commands:"
            echo "  cargo build --release --target wasm32-unknown-unknown -p pogq_zome"
            echo "  cargo test                                    # Run tests"
            echo "  cargo test --test byzantine_resistance       # Run integration tests"
            echo ""
            echo "Zomes in workspace:"
            echo "  - gradient_storage"
            echo "  - reputation_tracker"
            echo "  - zerotrustml_credits"
            echo "  - pogq_zome ⭐ (M0 Phase 2)"
          '';

          # Ensure Rust toolchain paths are set
          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";

          # Allow more stack for RISC Zero proof generation
          RUST_MIN_STACK = "8388608";  # 8MB
        };
      });
}
