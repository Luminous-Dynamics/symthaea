# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Mycelix Personal - Sovereign Agent Cluster
# Private data vault: identity + health + credentials + selective disclosure
#
# Usage:
#   nix develop              # Enter dev shell
#   nix develop .#ci         # CI environment (minimal)
{
  description = "Mycelix Personal - Sovereign agent data vault on Holochain";

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
            name = "personal";
            extraBuildInputs = with pkgs; [ nodejs_20 ];
            extraShellHook = ''
              echo "Mycelix Personal — Sovereign Agent Data Vault"
              echo ""
              echo "Domains:"
              echo "  identity-vault/      - Profile, avatar, master keys (private)"
              echo "  health-vault/        - Medical records, biometrics (private)"
              echo "  credential-wallet/   - Verifiable credentials (private, provable)"
              echo "  personal-bridge/     - Selective disclosure to other clusters"
              echo ""
              echo "Commands:"
              echo "  cargo build                                     - Build library crates"
              echo "  cargo build --release --target wasm32-unknown-unknown - Build WASM zomes"
              echo "  cargo test                                      - Run unit tests"
              echo ""
            '';
          };

          ci = pkgs.mkShell {
            name = "mycelix-personal-ci";
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
            name = "mycelix-personal-zomes";
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
