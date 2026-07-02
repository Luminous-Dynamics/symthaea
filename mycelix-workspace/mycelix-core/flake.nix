# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "Zero-TrustML workspace – Poetry + Nix development environments";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = inputs@{ self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        overlays = [
          (import rust-overlay)
        ];

        pkgs = import nixpkgs {
          inherit system overlays;
          config = {
            allowUnfree = true;
          };
        };

        pythonEnv = pkgs.python313.withPackages (ps: with ps; [
          numpy
          torch-bin
          torchvision-bin
          matplotlib
          scikit-learn
          pandas
          jupyter
          ipython
          web3
          asyncpg
          websockets
          msgpack
          aiohttp
          cryptography
          zstandard
          redis
          pytest
          pytest-asyncio
          pytest-cov
          black
          mypy
          ruff
        ]);
      in {
        packages = {
          zerotrustml-python = pythonEnv;
        };

        devShells = {
          default = pkgs.mkShell {
            packages = with pkgs; [
              pythonEnv
              poetry
              rust-bin.stable.latest.default
              solc
              foundry
              nodejs_20
              gcc
              pkg-config
              zlib
              # Go is required for tx5-go-pion-sys (Holochain networking)
              go
            ];
            shellHook = ''
              echo "🧠 Zero-TrustML dev shell (Poetry + Nix)"
              echo "Run 'poetry install' to create a project virtualenv if needed."
              export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.zlib pkgs.stdenv.cc.cc ]}:$LD_LIBRARY_PATH
            '';
          };

          ci = pkgs.mkShell {
            packages = [ pythonEnv pkgs.poetry ];
          };
        };
      });
}
