# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "Mycelix Desktop - P2P Consciousness Network";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" ];
          targets = [ "wasm32-unknown-unknown" ];
        };

        # Common dependencies for Tauri
        libraries = with pkgs; [
          webkitgtk_4_1
          gtk3
          cairo
          gdk-pixbuf
          glib
          dbus
          openssl
          librsvg
          xz  # liblzma for Holochain
        ];

        packages = with pkgs; [
          rustToolchain
          pkg-config
          dbus
          openssl
          glib
          gtk3
          libsoup_3  # Updated to v3 (secure, modern)
          webkitgtk_4_1
          librsvg

          # Tauri dependencies
          cargo-tauri
          nodejs_20
          nodePackages.npm

          # Holochain
          # (we'll build from source or use binary)

          # Development tools
          sqlitebrowser
          git
        ];

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = packages;

          shellHook = ''
            echo "🍄 Mycelix Desktop Development Environment"
            echo ""
            echo "Versions:"
            echo "  Rust: $(rustc --version | cut -d' ' -f2)"
            echo "  Node: $(node --version)"
            echo "  Cargo: $(cargo --version | cut -d' ' -f2)"
            echo ""
            echo "Available commands:"
            echo "  cargo tauri --help   - Tauri CLI"
            echo "  npm create tauri-app - Create new Tauri project"
            echo ""
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath libraries}:$LD_LIBRARY_PATH
            export XDG_DATA_DIRS=${pkgs.gsettings-desktop-schemas}/share/gsettings-schemas/${pkgs.gsettings-desktop-schemas.name}:${pkgs.gtk3}/share/gsettings-schemas/${pkgs.gtk3.name}:$XDG_DATA_DIRS
          '';
        };

        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "mycelix-desktop";
          version = "0.1.0";

          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          buildInputs = libraries;
          nativeBuildInputs = [ pkgs.pkg-config ];
        };
      }
    );
}