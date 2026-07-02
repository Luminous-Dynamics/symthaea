# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "Mycelix Attribution Registry — Decentralized Proof of Utility on Holochain";

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
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };

        holochainBase = import ../nix/modules/holochain-base.nix {
          inherit pkgs system;
          holochainPackages = holonix.packages.${system};
        };

      in {
        devShells = {
          default = holochainBase.mkHolochainShell {
            name = "attribution";
            extraBuildInputs = [ pkgs.nodejs_20 ];
            extraShellHook = ''
              echo "Attribution Registry zomes:"
              echo "  registry   — DependencyIdentity (register, query, verify)"
              echo "  usage      — UsageReceipt + UsageAttestation (ZK-STARK)"
              echo "  reciprocity — ReciprocityPledge + StewardshipScore"
              echo ""
              echo "Commands:"
              echo "  cargo test --lib                              # Unit tests (35)"
              echo "  cargo build --release --target wasm32-unknown-unknown  # WASM build"
              echo "  hc dna pack dna/                              # Pack DNA bundle"
              echo "  cd tests && cargo test -- --ignored           # Integration tests"
              echo "  cd cli && cargo run -- --help                 # CLI scanner"
              echo ""
            '';
          };

          ci = pkgs.mkShell {
            name = "mycelix-attribution-ci";
            buildInputs = with pkgs; [
              holochainBase.rustToolchain
              pkg-config
              openssl
              openssl.dev
            ] ++ [
              holonix.packages.${system}.holochain
              holonix.packages.${system}.hc
            ];

            inherit (holochainBase.envVars)
              LIBCLANG_PATH BINDGEN_EXTRA_CLANG_ARGS
              OPENSSL_DIR OPENSSL_LIB_DIR OPENSSL_INCLUDE_DIR;
          };
        };

        packages = {
          zomes = pkgs.stdenv.mkDerivation {
            name = "mycelix-attribution-zomes";
            src = ./.;
            nativeBuildInputs = with pkgs; [
              holochainBase.rustToolchain
              pkg-config
              openssl
              openssl.dev
            ];

            inherit (holochainBase.envVars)
              LIBCLANG_PATH BINDGEN_EXTRA_CLANG_ARGS
              OPENSSL_DIR OPENSSL_LIB_DIR OPENSSL_INCLUDE_DIR;

            buildPhase = ''
              cargo build --release --target wasm32-unknown-unknown
            '';

            installPhase = ''
              mkdir -p $out/lib
              cp target/wasm32-unknown-unknown/release/*.wasm $out/lib/
            '';
          };
        };
      }
    );
}
