# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Minimal exact-tooling shell for local Agency TPM2 qualification.
#
# The caller must construct `pkgs` from this repository's locked nixpkgs input
# and `rustToolchain` from rust-toolchain.toml through the locked rust-overlay.
{ pkgs, rustToolchain }:

pkgs.mkShellNoCC {
  packages = with pkgs; [
    rustToolchain
    git
    python3
    swtpm
    tpm2-tools
    binutils
    file
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
  '';
}
