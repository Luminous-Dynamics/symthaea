# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "Mycelix-DeSci - Reproducible Decentralized Science Platform";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
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

        # Pinned Rust toolchain for maximum reproducibility
        # Updated to 1.83.0 for edition2024 support (required by base64ct)
        rustToolchain = pkgs.rust-bin.stable."1.83.0".default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        # Common build dependencies
        commonBuildInputs = with pkgs; [
          openssl
        ];

        commonNativeBuildInputs = with pkgs; [
          rustToolchain
          pkg-config
        ];

        # Core library package
        mycelix-core = pkgs.rustPlatform.buildRustPackage {
          pname = "mycelix-desci-core";
          version = "0.1.0";
          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          buildInputs = commonBuildInputs;
          nativeBuildInputs = commonNativeBuildInputs;

          # Build only core package
          cargoBuildFlags = [ "--package" "mycelix-desci-core" ];

          # Run tests
          doCheck = true;
          checkFlags = [ "--package" "mycelix-desci-core" ];

          meta = with pkgs.lib; {
            description = "Core library for Mycelix-DeSci decentralized science platform";
            homepage = "https://github.com/Luminous-Dynamics/mycelix-desci";
            license = licenses.mit;
            maintainers = [ ];
            platforms = platforms.unix;
          };
        };

        # API server package
        mycelix-api = pkgs.rustPlatform.buildRustPackage {
          pname = "mycelix-api";
          version = "0.1.0";
          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          buildInputs = commonBuildInputs;
          nativeBuildInputs = commonNativeBuildInputs;

          cargoBuildFlags = [ "--package" "mycelix-desci-api" ];

          # Install documentation
          postInstall = ''
            mkdir -p $out/share/doc/mycelix-api
            cp README.md $out/share/doc/mycelix-api/ || true
            cp -r docs $out/share/doc/mycelix-api/ || true
            cp -r examples $out/share/doc/mycelix-api/ || true
          '';

          meta = with pkgs.lib; {
            description = "REST API server for Mycelix-DeSci";
            homepage = "https://github.com/Luminous-Dynamics/mycelix-desci";
            license = licenses.mit;
            mainProgram = "mycelix-api";
          };
        };

        # CLI tool package
        mycelix-cli = pkgs.rustPlatform.buildRustPackage {
          pname = "mycelix";
          version = "0.1.0";
          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          buildInputs = commonBuildInputs;
          nativeBuildInputs = commonNativeBuildInputs;

          cargoBuildFlags = [ "--package" "mycelix-cli" ];

          meta = with pkgs.lib; {
            description = "Command-line interface for Mycelix-DeSci";
            homepage = "https://github.com/Luminous-Dynamics/mycelix-desci";
            license = licenses.mit;
            mainProgram = "mycelix";
          };
        };

        # Reproducible Docker image built with Nix!
        dockerImage = pkgs.dockerTools.buildLayeredImage {
          name = "mycelix-api";
          tag = "latest";

          contents = with pkgs; [
            mycelix-api
            cacert  # SSL certificates for HTTPS
            coreutils
            bash
          ];

          config = {
            Cmd = [ "${mycelix-api}/bin/mycelix-api" ];
            ExposedPorts = {
              "8080/tcp" = {};
            };
            Env = [
              "PORT=8080"
              "RUST_LOG=mycelix_api=info"
              "CORS_ORIGINS=*"
            ];
            WorkingDir = "/app";
            User = "nobody";
          };

          # Metadata
          created = "now";
          maxLayers = 100;
        };

      in {
        # Packages available via `nix build`
        packages = {
          inherit mycelix-core mycelix-api mycelix-cli dockerImage;
          default = mycelix-cli;

          # Convenience: build all
          all = pkgs.symlinkJoin {
            name = "mycelix-desci-all";
            paths = [ mycelix-core mycelix-api mycelix-cli ];
          };
        };

        # Apps available via `nix run`
        apps = {
          # Run API server
          api = {
            type = "app";
            program = "${mycelix-api}/bin/mycelix-api";
          };

          # Run CLI tool
          cli = {
            type = "app";
            program = "${mycelix-cli}/bin/mycelix";
          };

          # Default: CLI tool
          default = self.apps.${system}.cli;
        };

        # Development shell with all tools
        devShells.default = pkgs.mkShell {
          buildInputs = commonBuildInputs ++ commonNativeBuildInputs ++ (with pkgs; [
            # Development tools
            rust-analyzer
            cargo-watch
            cargo-edit
            cargo-audit
            cargo-tarpaulin
            cargo-outdated

            # Utilities
            jq
            curl
            httpie

            # Docker (optional)
            docker
            docker-compose
          ]);

          shellHook = ''
            echo ""
            echo "╔══════════════════════════════════════════════════════════╗"
            echo "║   🔬 Mycelix-DeSci Development Environment (Nix)       ║"
            echo "╚══════════════════════════════════════════════════════════╝"
            echo ""
            echo "Rust version: $(rustc --version)"
            echo "Cargo version: $(cargo --version)"
            echo ""
            echo "📦 Available commands:"
            echo "  cargo build --release        # Build all packages"
            echo "  cargo test --all             # Run all tests"
            echo "  cargo bench                  # Run benchmarks"
            echo ""
            echo "🚀 Run applications:"
            echo "  cargo run --bin mycelix-api  # Start API server"
            echo "  cargo run --bin mycelix      # Run CLI tool"
            echo ""
            echo "🔧 Development tools:"
            echo "  cargo watch -x check         # Auto-check on changes"
            echo "  cargo audit                  # Security audit"
            echo "  cargo outdated               # Check for updates"
            echo ""
            echo "📖 Documentation:"
            echo "  cargo doc --open             # Open documentation"
            echo ""
            echo "🐳 Docker (Nix-built):"
            echo "  nix build .#dockerImage      # Build Docker image"
            echo "  docker load < result         # Load into Docker"
            echo ""
          '';

          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
          RUST_BACKTRACE = "1";
        };

        # Checks (run via `nix flake check`)
        checks = {
          # Build checks
          build-core = mycelix-core;
          build-api = mycelix-api;
          build-cli = mycelix-cli;

          # Format check
          format = pkgs.runCommand "check-format" {
            nativeBuildInputs = [ rustToolchain ];
          } ''
            cd ${./.}
            cargo fmt --all -- --check
            touch $out
          '';

          # Clippy check
          clippy = pkgs.runCommand "check-clippy" {
            nativeBuildInputs = commonNativeBuildInputs ++ commonBuildInputs;
          } ''
            cd ${./.}
            cargo clippy --all-targets --all-features -- -D warnings
            touch $out
          '';
        };
      }
    ) // {
      # NixOS module (for system-wide installation)
      nixosModules.default = import ./nixos-module.nix;

      # Overlay (for adding to your own nixpkgs)
      overlays.default = final: prev: {
        mycelix-desci-core = self.packages.${final.system}.mycelix-core;
        mycelix-api = self.packages.${final.system}.mycelix-api;
        mycelix-cli = self.packages.${final.system}.mycelix-cli;
      };
    };
}
