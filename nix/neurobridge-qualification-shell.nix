# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Minimal locked shell for exact-head local NeuroBridge qualification.
# The caller constructs `pkgs` from the repository's locked nixpkgs input and
# `rustToolchain` from rust-toolchain.toml through the locked rust-overlay.
{ pkgs, rustToolchain }:

pkgs.mkShellNoCC {
  packages = with pkgs; [
    rustToolchain
    stdenv.cc
    pkg-config
    python311
    nix
    git
    coreutils
    findutils
    gnugrep
    gawk
    gnused
    gnutar
    gzip
    which
    cacert
  ];

  shellHook = ''
    export LANG=C
    export LC_ALL=C
    export RUST_BACKTRACE=1
    export PYTHONDONTWRITEBYTECODE=1

    # The outer bootstrap has already selected an exact detached HEAD. Keep the
    # reviewed lock read-only while the inner qualification executes.
    if [ -f flake.lock ]; then
      chmod a-w flake.lock
    fi
  '';
}
