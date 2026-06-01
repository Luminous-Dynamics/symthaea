# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "Luminous Dynamics Monorepo - Common Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # Holochain from holonix (needed for sweettest builds)
    holonix = {
      url = "github:holochain/holonix/d21b3543";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Rust overlay (needed by holochain-base module)
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, holonix, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };

        # Holochain packages + base module (for .#holochain shell)
        holochainPackages = holonix.packages.${system};
        holochainBase = import ./nix/modules/holochain-base.nix {
          inherit pkgs system;
          holochainPackages = holochainPackages;
        };
      in
      {
        # ── Docker Images ──────────────────────────────────────────────
        # Build: nix build .#symthaea-docker
        # Load: docker load < result
        packages = let
          commonBuildInputs = with pkgs; [ openssl ];
          commonNativeBuildInputs = with pkgs; [ pkg-config ];
        in {
          symthaea-docker = let
            symthaea-api = pkgs.rustPlatform.buildRustPackage {
              pname = "symthaea-api";
              version = "2.0.0";
              src = ./symthaea;

              cargoLock = {
                lockFile = ./symthaea/Cargo.lock;
              };

              buildInputs = commonBuildInputs;
              nativeBuildInputs = commonNativeBuildInputs;

              cargoBuildFlags = [ "--bin" "symthaea-api" ];
              buildFeatures = [ "api_module" ];

              # Skip tests during Docker image build (run separately)
              doCheck = false;

              meta = with pkgs.lib; {
                description = "Symthaea Holographic Liquid Brain — REST API server";
                homepage = "https://luminousdynamics.org";
                license = licenses.mit;
                mainProgram = "symthaea-api";
              };
            };
          in pkgs.dockerTools.buildLayeredImage {
            name = "symthaea-api";
            tag = "latest";

            contents = with pkgs; [
              symthaea-api
              cacert        # SSL certificates for HTTPS
              coreutils
              bash
            ];

            config = {
              Cmd = [ "${symthaea-api}/bin/symthaea-api" ];
              ExposedPorts = {
                "8080/tcp" = {};
              };
              Env = [
                "SYMTHAEA_HOST=0.0.0.0"
                "SYMTHAEA_PORT=8080"
                "RUST_LOG=symthaea=info"
              ];
              WorkingDir = "/app";
              User = "nobody";
            };

            created = "now";
            maxLayers = 100;
          };
        };

        devShells = {
        default = pkgs.mkShell {
          name = "luminous-dynamics-dev";

          buildInputs = [
            # Rust 1.95.0 via rust-overlay (consistent with holochain shell)
            (pkgs.rust-bin.stable.latest.default.override {
              targets = [ "wasm32-unknown-unknown" ];
              extensions = [ "rust-src" "rust-analyzer" "clippy" ];
            })
            ] ++ (with pkgs; [
            # Common tools across all projects
            nodejs_22
            bacon

            cargo-nextest

            # Linker (required by symthaea/.cargo/config.toml: -fuse-ld=mold)
            mold

            # Database (PostgreSQL already in your environment)
            postgresql_15

            # Common utilities
            git
            curl
            jq

            # Nix tooling
            nixfmt
            nil  # Nix LSP
            deploy-rs
            colmena

            # libclang (needed by sweettest / bindgen for Holochain WASM builds)
            llvmPackages.libclang
            llvmPackages.clang

            # LaTeX (papers — HAI, psych-bench, stewardship, ALIFE 2026)
            (texlive.combine {
              inherit (texlive)
                scheme-medium
                changepage
                marvosym
                cm-super
                booktabs
                enumitem
                titlesec
                epigraph
                nextpage
                csquotes
                natbib
                hyperref
                ;
            })

            # Python + matplotlib (paper figures commented out due to upstream nixpkgs sphinx-9.1.0 python3.11 build conflict)
            # (python3.withPackages (ps: with ps; [
            #   matplotlib
            #   numpy
            # ]))

            # Symtropy game runtime dependencies (Bevy + Vulkan + X11)
            xorg.libX11
            xorg.libXcursor
            xorg.libXi
            xorg.libXrandr
            xorg.libxcb
            libxkbcommon
            vulkan-loader
            libGL
          ]);

          shellHook = ''
            echo "🌟 Luminous-Dynamics Development Environment"
            echo "📦 Node.js: $(node --version)"
            echo "🦀 Rust: $(rustc --version)"
            echo "❄️  Nix: $(nix --version)"
            echo ""
            echo "Commands:"
            echo "  lum-start  - Start all services"
            echo "  lum-status - Connect to overmind"
            echo "  lum-stop   - Stop all services"
            echo "  bacon      - Run repo watch jobs from ./bacon.toml"
            echo "  cargo nextest run - Faster Rust test runner"
            echo ""
            echo "🌊 We flow with Nix!"

            # Symtropy/Bevy runtime: graphics libraries on LD_LIBRARY_PATH
            export LD_LIBRARY_PATH="${pkgs.xorg.libX11}/lib:${pkgs.xorg.libXcursor}/lib:${pkgs.xorg.libXi}/lib:${pkgs.xorg.libXrandr}/lib:${pkgs.vulkan-loader}/lib:${pkgs.libxkbcommon}/lib:${pkgs.libGL}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            # libclang for bindgen (sweettest / Holochain WASM builds)
            export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"

            # Project-specific setup
            export LUMINOUS_DEV=true
            export LUMINOUS_ROOT="$PWD"
            export NODE_ENV=development

            # PostgreSQL setup
            export PGDATA="$PWD/.postgres"
            export PGHOST="localhost"
            export PGUSER="$USER"
            export PGDATABASE="luminous"

            lum_check_sensorium() {
              cargo check --manifest-path "$LUMINOUS_ROOT/mycelix-sensorium/Cargo.toml" "$@"
            }

            lum_check_personal() {
              cargo check --manifest-path "$LUMINOUS_ROOT/mycelix-personal/apps/leptos/Cargo.toml" "$@"
            }

            lum_check_commons() {
              cargo check --manifest-path "$LUMINOUS_ROOT/mycelix-commons/apps/leptos/Cargo.toml" "$@"
            }

            lum_check_health() {
              cargo check --manifest-path "$LUMINOUS_ROOT/mycelix-health/Cargo.toml" "$@"
            }

            lum_check_finance() {
              cargo check --manifest-path "$LUMINOUS_ROOT/mycelix-finance/Cargo.toml" "$@"
            }

            lum_check_knowledge() {
              cargo check --manifest-path "$LUMINOUS_ROOT/mycelix-knowledge/Cargo.toml" "$@"
            }

            lum_check_pulse() {
              cargo check --manifest-path "$LUMINOUS_ROOT/mycelix-workspace/mycelix-pulse/apps/leptos/Cargo.toml" "$@"
            }
          '';
        };

        # Holochain development shell (sweettests, zome builds, DNA packing)
        # Usage: nix develop .#holochain
        holochain = holochainBase.mkHolochainShell {
          name = "monorepo-holochain";
          extraShellHook = ''
            echo "Holochain shell from monorepo root."
            echo "Use this for sweettest builds and DNA packing."
            echo ""
          '';
        };
      };
      });
}
