# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Minimal system/toolchain environment for EUREKA-002 V2 backend
# qualification. Cargo command authority remains exclusively in
# .github/eureka/eureka-v2-backend-qualification-contract.sh.
#
# This file deliberately resolves the repository root's exact nixpkgs and
# rust-overlay nodes from flake.lock. It does not call builtins.getFlake on the
# mutable local checkout and does not introduce a second package lock universe.

let
  lock = builtins.fromJSON (builtins.readFile ../flake.lock);
  rootNode = lock.nodes.${lock.root};

  lockedRootInput = inputName:
    let
      nodeName = rootNode.inputs.${inputName};
      locked = lock.nodes.${nodeName}.locked;
    in
    builtins.fetchTree (builtins.removeAttrs locked [ "lastModified" ]);

  nixpkgsSource = lockedRootInput "nixpkgs";
  rustOverlaySource = lockedRootInput "rust-overlay";

  # The trusted qualifier workflow currently runs only on GitHub's x86_64
  # Ubuntu runner lane. Keep the system explicit so the environment derivation
  # does not silently depend on builtins.currentSystem.
  system = "x86_64-linux";
  pkgs = import nixpkgsSource.outPath {
    inherit system;
    overlays = [ (import rustOverlaySource.outPath) ];
  };

  rustToolchainToml = builtins.fromTOML (builtins.readFile ../rust-toolchain.toml);
  rustChannel = rustToolchainToml.toolchain.channel;
  rustToolchain = pkgs.rust-bin.stable.${rustChannel}.default.override {
    extensions = [ "clippy" ];
  };
in
pkgs.mkShell {
  name = "symthaea-eureka-v2-qualifier";

  nativeBuildInputs = with pkgs; [
    rustToolchain
    pkg-config
    cmake
  ];

  buildInputs = with pkgs; [
    openssl
    openssl.dev
    llvmPackages.libclang
    alsa-lib
    cacert
  ];

  OPENSSL_DIR = "${pkgs.openssl.dev}";
  OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
  OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
  LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
  RUSTC_WRAPPER = "";
  SCCACHE_DISABLE = "1";
  RUST_BACKTRACE = "1";

  shellHook = ''
    export BINDGEN_EXTRA_CLANG_ARGS="$(< ${pkgs.stdenv.cc}/nix-support/libc-cflags) $(< ${pkgs.stdenv.cc}/nix-support/cc-cflags)"
  '';
}
