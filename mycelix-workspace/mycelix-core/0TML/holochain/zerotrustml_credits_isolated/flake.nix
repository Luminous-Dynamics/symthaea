# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "ZeroTrustML Credits - Isolated Holochain DNA";

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

        # Use stable Rust with WASM target
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

            # Holochain tooling (if available)
            # holochain
            # hc
          ];

          shellHook = ''
            echo "🧬 ZeroTrustML Credits DNA Build Environment"
            echo "════════════════════════════════════════"
            echo "Rust: $(rustc --version)"
            echo "Cargo: $(cargo --version)"
            echo ""
            echo "WASM target installed: $(rustc --print target-list | grep wasm32-unknown-unknown)"
            echo ""
            echo "Build commands:"
            echo "  cargo check                    - Verify compilation"
            echo "  cargo build --release --target wasm32-unknown-unknown"
            echo "  # hc dna pack .                - Package DNA (needs hc)"
          '';

          # Ensure Rust toolchain paths are set
          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
        };
      });
}