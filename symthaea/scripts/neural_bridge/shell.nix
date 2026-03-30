# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{ pkgs ? import <nixpkgs> {} }:

let
  # FHS-compatible environment for pip-installed packages
  fhs = pkgs.buildFHSEnv {
    name = "neural-bridge-env";
    targetPkgs = pkgs: with pkgs; [
      # Python
      python311
      python311Packages.pip
      python311Packages.virtualenv

      # Required system libraries
      zlib
      stdenv.cc.cc.lib
      libffi
      openssl
      git
    ];
    runScript = "bash";
  };
in
pkgs.mkShell {
  buildInputs = [ fhs pkgs.python311 pkgs.python311Packages.numpy ];

  shellHook = ''
    echo "Neural Bridge v2 Training Environment"
    echo ""
    echo "Run: neural-bridge-env"
    echo "Then inside: pip install sentence-transformers && python extract_bge_m3_embeddings.py"
  '';
}
