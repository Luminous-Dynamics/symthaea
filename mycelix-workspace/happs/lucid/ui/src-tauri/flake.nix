# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "LUCID Desktop - Tauri 2.0 development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Rust toolchain
            rustc cargo rustfmt clippy rust-analyzer

            # Tauri 2.0 system dependencies
            pkg-config
            openssl
            glib
            gtk3
            webkitgtk_4_1
            libsoup_3
            cairo
            pango
            gdk-pixbuf
            atk
            harfbuzz

            # Additional Tauri deps
            librsvg
            dbus

            # Node.js for frontend
            nodejs_20
            nodePackages.npm
          ];

          shellHook = ''
            echo "LUCID Tauri dev environment loaded"
          '';

          PKG_CONFIG_PATH = pkgs.lib.makeSearchPath "lib/pkgconfig" [
            pkgs.openssl.dev
            pkgs.glib.dev
            pkgs.gtk3.dev
            pkgs.webkitgtk_4_1.dev
            pkgs.libsoup_3.dev
            pkgs.cairo.dev
            pkgs.pango.dev
            pkgs.gdk-pixbuf.dev
            pkgs.atk.dev
            pkgs.harfbuzz.dev
            pkgs.dbus.dev
          ];
        };
      });
}
