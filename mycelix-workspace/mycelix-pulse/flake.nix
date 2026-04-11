# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "Mycelix Pulse - Decentralized communication on Holochain";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";

    # Holochain development tools
    holochain-flake = {
      url = "github:holochain/holochain";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Rust toolchain
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, holochain-flake, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [
          (import rust-overlay)
        ];

        pkgs = import nixpkgs {
          inherit system overlays;
        };

        holochainPkg = holochain-flake.packages.${system}.holochain;

        # Rust toolchain with wasm target for Holochain DNA compilation
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
          targets = [ "wasm32-unknown-unknown" ];
        };

        # Common build inputs
        buildInputs = with pkgs; [
          # Rust
          rustToolchain
          cargo-watch
          cargo-edit
          cargo-tauri

          # Holochain
          holochainPkg

          # Node.js for frontend
          nodejs_20
          nodePackages.npm
          nodePackages.typescript
          nodePackages.typescript-language-server

          # IPFS for content storage
          kubo  # IPFS daemon

          # Build tools
          pkg-config
          openssl
          zlib
          glib
          gtk3
          webkitgtk_4_1
          libsoup_3

          # Development utilities
          just  # Command runner
          jq
          curl

          # Database tools (for development)
          sqlite
        ];

        # Native build inputs (platform-specific)
        nativeBuildInputs = with pkgs; [
          pkg-config
        ];

      in {
        devShells.default = pkgs.mkShell {
          inherit buildInputs nativeBuildInputs;

          shellHook = ''
            echo "🍄 Mycelix Pulse Development Environment"
            echo ""
            echo "Available commands:"
            echo "  just dev       - Start all services for development"
            echo "  just build     - Build all components"
            echo "  just test      - Run all tests"
            echo "  just dna       - Build Holochain DNA"
            echo "  just backend   - Run Rust backend"
            echo "  just frontend  - Run frontend dev server"
            echo "  cargo tauri dev --config desktop/src-tauri/tauri.conf.json"
            echo ""
            echo "Components:"
            echo "  - Holochain: $(holochain --version 2>/dev/null || echo 'not found')"
            echo "  - Rust: $(rustc --version 2>/dev/null || echo 'not found')"
            echo "  - Node: $(node --version 2>/dev/null || echo 'not found')"
            echo "  - Cargo Tauri: $(cargo tauri --version 2>/dev/null || echo 'not found')"
            echo "  - IPFS: $(ipfs --version 2>/dev/null || echo 'not found')"
            echo ""

            # Set up environment variables
            export RUST_BACKTRACE=1
            export RUST_LOG=info

            # Holochain configuration
            export HC_ADMIN_PORT=4444
            export HC_APP_PORT=4445

            # Backend configuration
            export HOST=0.0.0.0
            export PORT=3001
            export HOLOCHAIN_URL=ws://localhost:4444
            export JWT_SECRET=dev-secret-change-in-production
            export CORS_ORIGINS=http://localhost:8117,http://127.0.0.1:8117,http://localhost:1420,http://127.0.0.1:1420

            # IPFS configuration
            export IPFS_API_URL=http://localhost:5001

            # Frontend configuration
            export TRUNK_SERVE_ADDRESS=127.0.0.1
            export TRUNK_SERVE_PORT=8117
          '';

          # Environment variables for OpenSSL (needed by some Rust crates)
          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
          PKG_CONFIG_PATH = pkgs.lib.makeSearchPath "lib/pkgconfig" [
            pkgs.openssl.dev
            pkgs.zlib.dev
            pkgs.glib.dev
            pkgs.gtk3.dev
            pkgs.webkitgtk_4_1.dev
            pkgs.libsoup_3.dev
          ];
        };

        # Package definitions
        packages = {
          # Backend binary
          backend = pkgs.rustPlatform.buildRustPackage {
            pname = "mycelix-mail-backend";
            version = "0.1.0";
            src = ./happ/backend-rs;
            cargoLock.lockFile = ./happ/backend-rs/Cargo.lock;

            nativeBuildInputs = with pkgs; [ pkg-config ];
            buildInputs = with pkgs; [ openssl ];

            meta = {
              description = "Mycelix-Mail Axum backend";
              homepage = "https://github.com/Luminous-Dynamics/Mycelix-Mail";
              license = pkgs.lib.licenses.mit;
            };
          };

          # DNA package (would need hc to build)
          dna = pkgs.stdenv.mkDerivation {
            pname = "mycelix-mail-dna";
            version = "0.1.0";
            src = ./happ/dna;

            nativeBuildInputs = [ rustToolchain pkgs.hc ];

            buildPhase = ''
              cd integrity && cargo build --release --target wasm32-unknown-unknown
              cd ../zomes/mail_messages && cargo build --release --target wasm32-unknown-unknown
              cd ../trust_filter && cargo build --release --target wasm32-unknown-unknown
            '';

            installPhase = ''
              mkdir -p $out
              # DNA packaging would go here
            '';
          };
        };

        # Development scripts
        apps = {
          dev = flake-utils.lib.mkApp {
            drv = pkgs.writeShellScriptBin "mycelix-dev" ''
              echo "Starting Mycelix Pulse development environment..."

              # Start IPFS daemon if not running
              if ! pgrep -x "ipfs" > /dev/null; then
                echo "Starting IPFS daemon..."
                ipfs daemon &
                sleep 2
              fi

              # Start Holochain conductor if not running
              if ! pgrep -x "holochain" > /dev/null; then
                echo "Starting Holochain conductor..."
                ${holochainPkg}/bin/holochain -c ~/.config/holochain/conductor-config.yaml &
                sleep 3
              fi

              # Start backend
              echo "Starting Rust backend..."
              cd ${toString ./happ/backend-rs}
              cargo run &

              # Start frontend
              echo "Starting Leptos frontend dev server..."
              cd ${toString ./apps/leptos}
              env -u NO_COLOR RUSTC_WRAPPER= SCCACHE_DISABLE=1 ~/.cargo/bin/trunk serve --address 127.0.0.1 --port 8117 &

              echo ""
              echo "Services running:"
              echo "  - IPFS API: http://localhost:5001"
              echo "  - Holochain Admin: ws://localhost:4444"
              echo "  - Backend API: http://localhost:3001"
              echo "  - Frontend: http://127.0.0.1:8117"
              echo ""
              echo "Press Ctrl+C to stop all services"

              wait
            '';
          };
        };
      }
    );
}
