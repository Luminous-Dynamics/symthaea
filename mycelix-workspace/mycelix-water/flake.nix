# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root# Mycelix Water - Holochain Development Environment
# Community water management: FLOW, PURITY, CAPTURE, STEWARD, WISDOM
#
# Usage:
#   nix develop              # Enter dev shell
#   nix develop .#ci         # CI environment (minimal)
{
  description = "Mycelix Water - Community water stewardship on Holochain";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    holonix = {
      url = "github:holochain/holonix/d21b3543"; # pinned to fixed commit, matches mycelix-workspace root (was moving branch main-0.6)
      inputs.nixpkgs.follows = "nixpkgs";
    };

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, holonix, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };

        holochainPackages = holonix.packages.${system};

        holochainBase = import ../nix/modules/holochain-base.nix {
          inherit pkgs system;
          holochainPackages = holochainPackages;
        };

      in {
        devShells = {
          default = holochainBase.mkHolochainShell {
            name = "water";
            extraBuildInputs = with pkgs; [ nodejs_20 ];
            extraShellHook = ''
              echo "Zomes:"
              echo "  flow/       - Water allocation & H2O credits"
              echo "  purity/     - Quality monitoring & remediation"
              echo "  capture/    - Harvesting & storage systems"
              echo "  steward/    - Watershed governance & rights"
              echo "  wisdom/     - Traditional knowledge & conservation"
              echo "  bridge/     - Cross-hApp integration"
              echo ""
              echo "Commands:"
              echo "  cargo build              - Build library crates"
              echo "  cargo test               - Run unit tests"
              echo ""
            '';
          };

          ci = pkgs.mkShell {
            name = "mycelix-water-ci";
            buildInputs = with pkgs; [
              holochainPackages.holochain
              holochainPackages.hc
              holochainBase.rustToolchain
              pkg-config
              openssl
              openssl.dev
            ];

            inherit (holochainBase.envVars)
              LIBCLANG_PATH BINDGEN_EXTRA_CLANG_ARGS
              OPENSSL_DIR OPENSSL_LIB_DIR OPENSSL_INCLUDE_DIR;
          };
        };

        packages = {
          zomes = pkgs.stdenv.mkDerivation {
            name = "mycelix-water-zomes";
            src = ./.;
            nativeBuildInputs = [ holochainBase.rustToolchain pkgs.pkg-config ];
            buildInputs = [ pkgs.openssl ];
            buildPhase = ''
              export HOME=$TMPDIR
              cargo build --release --target wasm32-unknown-unknown
            '';
            installPhase = ''
              mkdir -p $out/lib
              find target/wasm32-unknown-unknown/release -name "*.wasm" -exec cp {} $out/lib/ \;
            '';
          };
        };
      }
    );
}
