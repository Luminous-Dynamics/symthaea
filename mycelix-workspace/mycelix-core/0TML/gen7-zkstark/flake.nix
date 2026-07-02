# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "VSV-STARK: Verifiable Self-Validation with ZK-STARKs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Nix store paths for RISC Zero toolchain
        zlibPath = "${pkgs.zlib}/lib";
        stdenvPath = "${pkgs.stdenv.cc.cc.lib}/lib";
      in
      {
        devShells = {
          # Default: Standard Rust development (Day 2)
          default = pkgs.mkShell {
            packages = with pkgs; [
              rustc
              cargo
              rust-analyzer
              rustfmt
              clippy
            ];

            shellHook = ''
              echo "🦀 VSV-STARK Standard Rust Environment"
              echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
              echo "📦 Rust: $(rustc --version)"
              echo "🔨 Cargo: $(cargo --version)"
              echo ""
              echo "For zkVM builds, use: nix develop 'path:$(pwd)'#fhs"
              echo ""
            '';
          };

          # FHS: RISC Zero zkVM environment (Day 3+)
          fhs = pkgs.buildFHSEnv {
            name = "vsv-stark-fhs";

            targetPkgs = pkgs: with pkgs; [
              rustup
              zlib
              stdenv.cc.cc.lib
              gcc
              clang
              pkg-config
            ];

            runScript = pkgs.writeScript "fhs-init" ''
              #!${pkgs.bash}/bin/bash

              # Set library paths for RISC Zero toolchain
              export LD_LIBRARY_PATH="${zlibPath}:${stdenvPath}:$LD_LIBRARY_PATH"

              echo "⚡ VSV-STARK zkVM Environment (FHS)"
              echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
              echo "🔧 LD_LIBRARY_PATH configured for RISC Zero"
              echo "📍 zlib: ${zlibPath}"
              echo "📍 stdenv: ${stdenvPath}"
              echo ""
              echo "Quick Start:"
              echo "  ~/.risc0/bin/rzup install  # One-time setup"
              echo "  cargo build                # Build with zkVM guest"
              echo "  cargo run                  # Generate STARK proofs"
              echo ""

              exec bash
            '';
          };
        };

        # Package outputs (for CI/CD)
        packages = {
          # Build the project without zkVM (standard Rust)
          vsv-stark-lib = pkgs.rustPlatform.buildRustPackage {
            pname = "vsv-stark";
            version = "0.1.0";
            src = ./.;

            cargoLock = {
              lockFile = ./Cargo.lock;
            };

            buildInputs = with pkgs; [ ];

            # Skip zkVM guest builds in CI (standard Rust only)
            buildPhase = ''
              cargo build --lib --release
            '';
          };
        };
      }
    );
}
