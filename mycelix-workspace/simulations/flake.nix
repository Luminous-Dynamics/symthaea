# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "Mycelix/Symthaea Simulation Suite — analysis & visualization";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          matplotlib
          numpy
          seaborn
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          name = "simulation-analysis";

          buildInputs = [
            pythonEnv
            pkgs.cargo
            pkgs.rustc
          ];

          shellHook = ''
            echo "Simulation Analysis Environment"
            echo "  Python: $(python --version)"
            echo "  matplotlib + numpy + seaborn available"
            echo ""
            echo "Usage:"
            echo "  bash scripts/sweep.sh                    # Run parameter sweeps"
            echo "  python scripts/analyze_simulations.py    # Generate figures"
          '';
        };
      }
    );
}
