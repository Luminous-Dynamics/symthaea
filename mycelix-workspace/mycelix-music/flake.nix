# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Mycelix Music - Development Environment
#
# Usage:
#   nix develop              # Enter dev shell (Holochain + Node.js)
#   nix develop .#node       # Node.js only (UI/services development)
{
  description = "Mycelix Music - Zero-cost streaming platform on Holochain";

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
          # Full development shell with Holochain + Rust + Node.js
          default = holochainBase.mkHolochainShell {
            name = "music";
            extraBuildInputs = with pkgs; [
              nodejs_20
              nodePackages.npm
              nodePackages.typescript
              nodePackages.typescript-language-server
              jq
              curl
            ];
            extraShellHook = ''
              echo "Mycelix Music — Zero-Cost Streaming Platform"
              echo ""
              echo "Holochain DNA:"
              echo "  dnas/mycelix-music/  - 5 zomes (catalog, plays, balances, trust, music-bridge)"
              echo ""
              echo "Commands:"
              echo "  cargo build --release --target wasm32-unknown-unknown  - Build WASM zomes"
              echo "  hc dna pack dnas/mycelix-music/                       - Pack DNA bundle"
              echo "  hc app pack .                                         - Pack hApp bundle"
              echo "  cargo test                                            - Run unit tests"
              echo "  cd tests && cargo test --release -- --ignored         - Run sweettests"
              echo ""
            '';
          };

          # Node.js-only shell for UI/services development
          node = pkgs.mkShell {
            name = "mycelix-music-node";
            buildInputs = with pkgs; [
              nodejs_20
              nodePackages.npm
              nodePackages.typescript
              nodePackages.typescript-language-server
              git
              jq
              curl
            ];

            shellHook = ''
              echo "Mycelix Music — Node.js Development"
              echo ""
              echo "Quick start:"
              echo "  cd apps/web && npm run dev    - Start frontend"
              echo "  cd apps/api && npm run dev    - Start API server"
              echo ""
            '';
          };
        };

        packages = {
          zomes = pkgs.stdenv.mkDerivation {
            name = "mycelix-music-zomes";
            src = ./.;
            nativeBuildInputs = [ holochainBase.rustToolchain pkgs.pkg-config ];
            buildInputs = [ pkgs.openssl ];
            buildPhase = ''
              export HOME=$TMPDIR
              cd dnas/mycelix-music
              cargo build --release --target wasm32-unknown-unknown
            '';
            installPhase = ''
              mkdir -p $out/lib
              find dnas/mycelix-music/target/wasm32-unknown-unknown/release -name "*.wasm" -exec cp {} $out/lib/ \;
            '';
          };
        };
      }
    );
}
