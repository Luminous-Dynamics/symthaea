# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Minimal Rust shell for CPU-only correctness validation. This intentionally
# avoids Symthaea's general-purpose audio/GUI/GPU development dependencies.

{
  nixpkgs,
  rust-overlay,
  toolchainFile ? ../rust-toolchain.toml,
  system ? builtins.currentSystem,
}:

let
  overlays = [ (import rust-overlay) ];
  pkgs = import nixpkgs {
    inherit system overlays;
  };

  toolchainToml = builtins.fromTOML (builtins.readFile toolchainFile);
  rustChannel = toolchainToml.toolchain.channel;
  rustToolchain = pkgs.rust-bin.stable.${rustChannel}.default.override {
    extensions = [ "rustfmt" ];
  };
in
pkgs.mkShell {
  packages = with pkgs; [
    rustToolchain
    bubblewrap
    coreutils
    gawk
    cacert
    gcc
    openssl
    openssl.dev
    pkg-config
  ];

  CARGO_TERM_COLOR = "always";
  RUST_BACKTRACE = "1";

  OPENSSL_DIR = "${pkgs.openssl.dev}";
  OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
  OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
  SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
}
