# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Lightweight shell.nix for quick test runs
# Use: nix-shell test-shell.nix --run "pytest src/tests/ -v"
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    (python3.withPackages (ps: with ps; [
      numpy
      scipy
      pytest
      hypothesis
    ]))
  ];

  shellHook = ''
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
    echo "🧪 Test Environment Ready"
    echo "   pytest src/tests/ -v --tb=short"
  '';
}
