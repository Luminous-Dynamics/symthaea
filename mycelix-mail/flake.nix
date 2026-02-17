{
  description = "Mycelix-Mail - Decentralized email on Holochain with MATL trust filtering";

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
          holochain-flake.overlays.holochain
        ];

        pkgs = import nixpkgs {
          inherit system overlays;
        };

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

          # Holochain
          holochain
          hc
          lair-keystore

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
            echo "🍄 Mycelix-Mail Development Environment"
            echo ""
            echo "Available commands:"
            echo "  just dev       - Start all services for development"
            echo "  just build     - Build all components"
            echo "  just test      - Run all tests"
            echo "  just dna       - Build Holochain DNA"
            echo "  just backend   - Run Rust backend"
            echo "  just frontend  - Run frontend dev server"
            echo ""
            echo "Components:"
            echo "  - Holochain: $(holochain --version 2>/dev/null || echo 'not found')"
            echo "  - Rust: $(rustc --version 2>/dev/null || echo 'not found')"
            echo "  - Node: $(node --version 2>/dev/null || echo 'not found')"
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
            export CORS_ORIGINS=http://localhost:5173,http://localhost:3000

            # IPFS configuration
            export IPFS_API_URL=http://localhost:5001

            # Frontend configuration
            export VITE_API_URL=http://localhost:3001
            export VITE_WS_URL=ws://localhost:3001
          '';

          # Environment variables for OpenSSL (needed by some Rust crates)
          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
          PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
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
              echo "Starting Mycelix-Mail development environment..."

              # Start IPFS daemon if not running
              if ! pgrep -x "ipfs" > /dev/null; then
                echo "Starting IPFS daemon..."
                ipfs daemon &
                sleep 2
              fi

              # Start Holochain conductor if not running
              if ! pgrep -x "holochain" > /dev/null; then
                echo "Starting Holochain conductor..."
                holochain -c ~/.config/holochain/conductor-config.yaml &
                sleep 3
              fi

              # Start backend
              echo "Starting Rust backend..."
              cd ${toString ./happ/backend-rs}
              cargo run &

              # Start frontend
              echo "Starting frontend dev server..."
              cd ${toString ./ui/frontend}
              npm run dev &

              echo ""
              echo "Services running:"
              echo "  - IPFS API: http://localhost:5001"
              echo "  - Holochain Admin: ws://localhost:4444"
              echo "  - Backend API: http://localhost:3001"
              echo "  - Frontend: http://localhost:5173"
              echo ""
              echo "Press Ctrl+C to stop all services"

              wait
            '';
          };
        };
      }
    );
}
