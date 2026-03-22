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
            # Rust 1.94 via rust-overlay (consistent with holochain shell)
            (pkgs.rust-bin.stable.latest.default.override {
              targets = [ "wasm32-unknown-unknown" ];
              extensions = [ "rust-src" "rust-analyzer" "clippy" ];
            })
          ] ++ (with pkgs; [
            # Common tools across all projects
            nodejs_20

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

            # libclang (needed by sweettest / bindgen for Holochain WASM builds)
            llvmPackages.libclang
            llvmPackages.clang

            # LaTeX (papers — HAI, psych-bench, stewardship)
            (texlive.combine {
              inherit (texlive)
                scheme-medium
                changepage
                marvosym
                cm-super
                ;
            })
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
            echo ""
            echo "🌊 We flow with Nix!"

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
