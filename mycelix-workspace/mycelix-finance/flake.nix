# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root# Mycelix Finance - Holochain Development Environment
# CGC (Civic Gifting Credits), TEND (Time Exchange), HEARTH (Commons Pool)
# Treasury, Payments, Lending, Staking, Credit Scoring
#
# Usage:
#   nix develop              # Enter dev shell
#   nix develop .#ci         # CI environment (minimal)
#   nix develop .#test       # Test environment (with sweettest deps)
#
# Build:
#   nix build .#zomes        # Build all zomes to WASM
{
  description = "Mycelix Finance - Commons-based financial infrastructure on Holochain";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # Holochain from holonix
    holonix = {
      url = "github:holochain/holonix/main";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Rust toolchain
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

        # Holochain packages from holonix
        holochainPackages = holonix.packages.${system};

        # Import shared Holochain base configuration
        holochainBase = import ../nix/modules/holochain-base.nix {
          inherit pkgs system;
          holochainPackages = holochainPackages;
        };

        # Finance-specific dependencies
        financeExtraInputs = with pkgs; [
          # Node.js for integration tests
          nodejs_20
          nodePackages.pnpm

          # Database for local testing
          sqlite
        ];

      in {
        devShells = {
          # Default development environment
          default = holochainBase.mkHolochainShell {
            name = "finance";
            extraBuildInputs = financeExtraInputs;
            extraShellHook = ''
              echo "Zomes:"
              echo "  cgc/      - Civic Gifting Credits (10/month recognition)"
              echo "  tend/     - Time Exchange (mutual credit, +-40 limit)"
              echo "  hearth/   - Commons Pool (democratic allocation)"
              echo "  payments/ - P2P payments & channels"
              echo "  lending/  - Trust-based lending"
              echo "  treasury/ - DAO treasury management"
              echo "  staking/  - Tri-asset staking"
              echo ""
              echo "Commands:"
              echo "  cargo build              - Build library crates"
              echo "  cargo build --release    - Release build"
              echo "  cargo test               - Run unit tests"
              echo "  cd tests && cargo test   - Run integration tests"
              echo ""
            '';
          };

          # CI environment (minimal, fast to build)
          ci = pkgs.mkShell {
            name = "mycelix-finance-ci";
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

          # Test environment (full sweettest support)
          test = holochainBase.mkHolochainShell {
            name = "finance-test";
            extraBuildInputs = financeExtraInputs ++ (with pkgs; [
              # Additional test dependencies
              cargo-nextest
            ]);
            extraShellHook = ''
              echo "Test Environment - sweettest configured"
              echo ""
              echo "Run tests:"
              echo "  cargo nextest run        - Fast parallel tests"
              echo "  cd tests && cargo test   - Integration tests"
              echo ""
            '';
          };
        };

        # Packages (zome WASM builds)
        packages = {
          # Build all zomes
          zomes = pkgs.stdenv.mkDerivation {
            name = "mycelix-finance-zomes";
            src = ./.;

            nativeBuildInputs = [
              holochainBase.rustToolchain
              pkgs.pkg-config
            ];

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

        # Checks (run in CI)
        checks = {
          format = pkgs.runCommand "check-format" {
            buildInputs = [ holochainBase.rustToolchain ];
          } ''
            cd ${self}
            cargo fmt --check
            touch $out
          '';

          clippy = pkgs.runCommand "check-clippy" {
            buildInputs = [ holochainBase.rustToolchain pkgs.pkg-config pkgs.openssl ];
          } ''
            cd ${self}
            cargo clippy -- -D warnings
            touch $out
          '';
        };
      }
    );
}
