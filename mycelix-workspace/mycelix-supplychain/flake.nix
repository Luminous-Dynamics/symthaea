# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "Mycelix ERP - Blockchain-Auditable Enterprise Resource Planning";

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
          extensions = [ "rust-src" "rust-analyzer" ];
        };

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Rust toolchain
            rustToolchain
            cargo
            cargo-watch
            cargo-edit

            # Build dependencies
            pkg-config
            openssl
            openssl.dev

            # PostgreSQL
            postgresql
            postgresql.lib

            # Database tools
            sqlx-cli

            # Development tools
            git
            jq

            # System libraries
            gcc
            cmake
            zlib
            zlib.dev
          ];

          shellHook = ''
            echo "🚀 Mycelix ERP Development Environment"
            echo "======================================"
            echo "Rust version: $(rustc --version)"
            echo "Cargo version: $(cargo --version)"
            echo "PostgreSQL: $(psql --version | head -n1)"
            echo ""
            echo "Quick commands:"
            echo "  cargo build --release  # Build ERP service"
            echo "  cargo test             # Run all tests"
            echo "  cargo run              # Start service on :8080"
            echo "  mycelix --help         # CLI commands"
            echo ""
            echo "Modules:"
            echo "  ✅ SCM - Supply Chain Management"
            echo "  ✅ FIN - Finance & Accounting"
            echo "  🚧 CRM - Coming soon"
            echo ""
          '';

          # Environment variables for compilation
          PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig:${pkgs.postgresql.lib}/lib/pkgconfig";
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [ pkgs.openssl pkgs.postgresql.lib pkgs.zlib ]}";
          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
        };

        # Package the service (for production builds)
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "mycelix-service";
          version = "0.1.0";

          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          nativeBuildInputs = with pkgs; [
            pkg-config
            rustToolchain
          ];

          buildInputs = with pkgs; [
            openssl
            postgresql
          ];

          PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
        };
      }
    );
}
