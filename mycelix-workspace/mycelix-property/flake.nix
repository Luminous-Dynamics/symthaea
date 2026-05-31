# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root# Mycelix Property - Holochain Development Environment
# Registry, Transfer, Commons, Disputes, Bridge
#
# Usage:
#   nix develop              # Enter dev shell
#   nix develop .#ci         # CI environment (minimal)
{
  description = "Mycelix Property - Decentralized property registry on Holochain";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    holonix = {
      url = "github:holochain/holonix/main-0.6";
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
            name = "property";
            extraBuildInputs = with pkgs; [ nodejs_20 ];
            extraShellHook = ''
              echo "Zomes:"
              echo "  registry/            - Property registration"
              echo "  transfer/            - Ownership transfer"
              echo "  commons/             - Common property management"
              echo "  disputes/            - Property disputes"
              echo "  bridge/              - Cross-zome integration"
              echo ""
              echo "Commands:"
              echo "  cargo build              - Build library crates"
              echo "  cargo test               - Run unit tests"
              echo ""
            '';
          };

          ci = pkgs.mkShell {
            name = "mycelix-property-ci";
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
            name = "mycelix-property-zomes";
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
