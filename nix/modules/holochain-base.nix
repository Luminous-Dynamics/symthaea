# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Mycelix Ecosystem - Shared Holochain Development Base
# This module provides a consistent Holochain development environment
# for all Mycelix hApps with proper bindgen/libclang configuration.
#
# Usage in flake.nix:
#   let
#     holochainBase = import ../nix/modules/holochain-base.nix { inherit pkgs system; };
#   in
#   pkgs.mkShell (holochainBase.shellConfig // { ... });

{ pkgs, system, holochainPackages ? null }:

let
  # Holochain packages (passed in or use holonix)
  hcPkgs = holochainPackages;

  # Rust toolchain with WASM target
  # Using latest stable - see docs/technical/RUST_COMPATIBILITY_ISSUES.md for mio/wait-timeout workarounds
  rustToolchain = pkgs.rust-bin.stable.latest.default.override {
    targets = [ "wasm32-unknown-unknown" ];
    extensions = [ "rust-src" "rust-analyzer" "clippy" ];
  };

  # Common build inputs for all Holochain projects
  commonBuildInputs = with pkgs; [
    # Rust toolchain
    rustToolchain
    bacon
    cargo-watch
    cargo-edit
    cargo-expand
    cargo-nextest
    deploy-rs
    colmena

    # Build essentials
    pkg-config
    openssl
    openssl.dev
    cmake
    gnumake

    # For bindgen (datachannel-sys, etc.)
    llvmPackages.libclang
    llvmPackages.clang
    glibc.dev

    # For C++ builds (datachannel-sys CMake build)
    # Use stdenv.cc to get proper include paths
    pkgs.stdenv.cc

    # Utilities
    just
    jq
    yq-go

    # === Tauri 2.0 Desktop Dependencies (Linux) ===
    # GTK3 for native UI
    gtk3
    gtk3.dev
    glib
    glib.dev
    gdk-pixbuf
    gdk-pixbuf.dev

    # WebKit for webview rendering
    webkitgtk_4_1  # Tauri 2.0 uses webkit2gtk-4.1
    libsoup_3      # Required by webkitgtk
    libsoup_3.dev

    # System tray support
    libappindicator-gtk3

    # SVG rendering for icons
    librsvg
    librsvg.dev

    # Additional Tauri deps (with dev packages for pkg-config)
    cairo
    cairo.dev
    pango
    pango.dev
    atk
    atk.dev
    harfbuzz
    harfbuzz.dev

    # GIO networking for HTTPS/TLS in WebKit
    glib-networking
    dbus

  ] ++ pkgs.lib.optionals (hcPkgs != null) [
    hcPkgs.holochain
    hcPkgs.lair-keystore
    hcPkgs.hc
    hcPkgs.bootstrap-srv
  ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
    pkgs.darwin.apple_sdk.frameworks.Security
    pkgs.darwin.apple_sdk.frameworks.SystemConfiguration
    pkgs.libiconv
  ];

  # Environment variables for proper C/C++ compilation
  # These are set directly as mkShell attributes (not via inherit)
  libclangPath = "${pkgs.llvmPackages.libclang.lib}/lib";

  # Clang resource root varies by system - find it dynamically
  clangResourceDir = "${pkgs.llvmPackages.clang.cc}/lib/clang/${pkgs.lib.versions.major pkgs.llvmPackages.clang.version}/include";

  bindgenArgs = builtins.concatStringsSep " " [
    "-I${pkgs.glibc.dev}/include"
    "-I${clangResourceDir}"
  ];

  # PKG_CONFIG_PATH with all required library paths
  pkgConfigPaths = pkgs.lib.concatStringsSep ":" [
    "${pkgs.openssl.dev}/lib/pkgconfig"
    "${pkgs.gtk3.dev}/lib/pkgconfig"
    "${pkgs.glib.dev}/lib/pkgconfig"
    "${pkgs.gdk-pixbuf.dev}/lib/pkgconfig"
    "${pkgs.webkitgtk_4_1}/lib/pkgconfig"
    "${pkgs.libsoup_3.dev}/lib/pkgconfig"
    "${pkgs.cairo.dev}/lib/pkgconfig"
    "${pkgs.pango.dev}/lib/pkgconfig"
    "${pkgs.atk.dev}/lib/pkgconfig"
    "${pkgs.harfbuzz.dev}/lib/pkgconfig"
    "${pkgs.librsvg.dev}/lib/pkgconfig"
    "${pkgs.libappindicator-gtk3}/lib/pkgconfig"
  ];

  # Environment variables attribute set for export
  # Note: C/C++ compiler paths are handled by stdenv.cc via inputsFrom
  envVars = {
    RUST_BACKTRACE = "1";
    RUST_LOG = "info";
    HC_ADMIN_PORT = "4444";
    HC_APP_PORT = "8888";
    LIBCLANG_PATH = libclangPath;
    BINDGEN_EXTRA_CLANG_ARGS = bindgenArgs;
    OPENSSL_DIR = "${pkgs.openssl.dev}";
    OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
    OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
    PKG_CONFIG_PATH = pkgConfigPaths;
    # GIO modules for WebKit HTTPS/TLS support
    GIO_MODULE_DIR = "${pkgs.glib-networking}/lib/gio/modules";
    GIO_EXTRA_MODULES = "${pkgs.glib-networking}/lib/gio/modules";
  };

  # Shell hook for Holochain projects
  shellHook = projectName: ''
    # Fix NixOS GCC 15 #include_next <stdlib.h> failure:
    # Too many -isystem entries from buildInputs push glibc headers before
    # GCC's include-fixed, breaking #include_next in <cstdlib>.
    # We preserve only -frandom-seed and let pkg-config/env vars handle paths.
    export NIX_CFLAGS_COMPILE="$(echo "$NIX_CFLAGS_COMPILE" | grep -oP '\-frandom-seed=\S+')"
    export NIX_CFLAGS_COMPILE_FOR_TARGET="$NIX_CFLAGS_COMPILE"

    echo ""
    echo "========================================"
    echo "  Mycelix ${projectName}"
    echo "  Holochain Development Environment"
    echo "========================================"
    echo ""
    ${if hcPkgs != null then ''
    echo "Holochain: $(holochain --version 2>/dev/null || echo 'loading...')"
    echo "hc:        $(hc --version 2>/dev/null || echo 'loading...')"
    '' else ''
    echo "Note: Holochain binaries not included (add holonix to your flake)"
    ''}
    echo "Rust:      $(rustc --version)"
    echo "Cargo:     $(cargo --version)"
    echo ""
    echo "Environment configured with:"
    echo "  - LIBCLANG_PATH for bindgen"
    echo "  - System includes for C/C++ deps"
    echo "  - OpenSSL paths"
    echo "  - bacon / cargo-nextest / deploy-rs / Colmena"
    echo ""
  '';

in {
  # Export for use in mkShell
  inherit commonBuildInputs envVars rustToolchain;

  # Complete shell configuration
  shellConfig = {
    name = "mycelix-holochain";
    buildInputs = commonBuildInputs;

    # Spread environment variables
    inherit (envVars)
      RUST_BACKTRACE RUST_LOG
      HC_ADMIN_PORT HC_APP_PORT
      LIBCLANG_PATH BINDGEN_EXTRA_CLANG_ARGS
      OPENSSL_DIR OPENSSL_LIB_DIR OPENSSL_INCLUDE_DIR
      PKG_CONFIG_PATH
      GIO_MODULE_DIR GIO_EXTRA_MODULES;

    shellHook = shellHook "hApp";
  };

  # Function to create a customized shell
  mkHolochainShell = { name, extraBuildInputs ? [], extraEnvVars ? {}, extraShellHook ? "" }:
    pkgs.mkShell ({
      name = "mycelix-${name}";
      buildInputs = commonBuildInputs ++ extraBuildInputs;

      inherit (envVars)
        RUST_BACKTRACE RUST_LOG
        HC_ADMIN_PORT HC_APP_PORT
        LIBCLANG_PATH BINDGEN_EXTRA_CLANG_ARGS
        OPENSSL_DIR OPENSSL_LIB_DIR OPENSSL_INCLUDE_DIR
        PKG_CONFIG_PATH
        GIO_MODULE_DIR GIO_EXTRA_MODULES;

      shellHook = (shellHook name) + extraShellHook;
    } // extraEnvVars);
}
