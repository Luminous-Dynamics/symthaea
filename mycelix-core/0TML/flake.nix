# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "Zero-TrustML - Byzantine-Resistant Federated Learning";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;  # Required for PyTorch's Triton dependency
          };
        };

        pythonEnv = pkgs.python313.withPackages (ps: with ps; [
          numpy
          scipy
          matplotlib
          torch-bin          # Pre-compiled binary (fast!)
          torchvision-bin    # Pre-compiled binary (fast!)
          # For adversarial testing
          scikit-learn
          pandas
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonEnv

            # Development tools
            black
            ruff

            # System dependencies
            gcc
            pkg-config

            # Go is required for tx5-go-pion-sys (Holochain networking)
            go

            # Rust for Holochain zomes
            cargo
            rustc

            # Optional: Holochain tooling
            # (uncomment if needed)
            # holochain
            # lair-keystore
          ];

          shellHook = ''
            # Add Gen7 zkSTARK bindings to Python path
            export PYTHONPATH="$PWD/gen7-zkstark/bindings:$PYTHONPATH"

            echo "🎯 Zero-TrustML Development Environment"
            echo "   Python: $(python --version)"
            echo "   NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
            echo "   PyTorch: $(python -c 'import torch; print(torch.__version__)')"
            echo ""
            echo "📊 Ready to run adversarial tests!"
            echo "   Run: python tests/test_adversarial_detection_rates.py"
            echo ""
            echo "🔐 Gen7 zkSTARK + Dilithium available"
            echo "   import gen7_zkstark"
            echo ""
          '';
        };
      }
    );
}
