# Mycelix Ecosystem - Unified Development Environment
# Orchestrates all Mycelix hApps with shared Holochain infrastructure
#
# Usage:
#   nix develop              # Full dev environment (all hApps)
#   nix develop .#holochain  # Holochain-only (zome development)
#   nix develop .#ml         # Python ML/FL environment
#   nix develop .#ci         # CI environment (minimal)
#
# Individual hApps (use their own flakes for focused work):
#   cd ../mycelix-finance && nix develop
#   cd ../mycelix-identity && nix develop
#   cd ../mycelix-governance && nix develop
#   cd ../mycelix-knowledge && nix develop
{
  description = "Mycelix Ecosystem - Unified development environment for all hApps";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # Holochain from holonix
    holonix = {
      url = "github:holochain/holonix/d21b3543";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Rust toolchain
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, holonix, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [
          (import rust-overlay)
          # Fix flaky test failures in nixpkgs python packages
          (final: prev: {
            python311 = prev.python311.override {
              packageOverrides = pyself: pysuper: {
                websockets = pysuper.websockets.overridePythonAttrs (old: {
                  doCheck = false;
                });
              };
            };
          })
        ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };

        # Holochain packages from holonix
        holochainPackages = holonix.packages.${system};

        # Import shared Holochain base configuration
        holochainBase = import ../nix/modules/holochain-base.nix {
          inherit pkgs system;
          holochainPackages = holochainPackages;
        };

        # Python environment for 0TML (Federated Learning)
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          numpy
          scipy
          pandas
          torch
          torchvision
          scikit-learn
          matplotlib
          pytest
          pytest-asyncio
          black
          mypy
          ruff
          # Async
          aiohttp
          websockets
        ]);

        # Node.js packages
        nodeEnv = with pkgs; [
          nodejs_20
          nodePackages.pnpm
          nodePackages.typescript
          nodePackages.typescript-language-server
        ];

      in {
        devShells = {
          # Full development environment (all tools)
          default = holochainBase.mkHolochainShell {
            name = "mycelix-workspace";
            extraBuildInputs = nodeEnv ++ [ pythonEnv ] ++ (with pkgs; [
              # Documentation
              mdbook
              graphviz

              # Additional dev tools
              httpie
              websocat
              gh
            ]);
            extraShellHook = ''
              echo ""
              echo "Mycelix Ecosystem Workspaces:"
              echo ""
              echo "  Core hApps (use individual flakes for focused work):"
              echo "    ../mycelix-finance/    - CGC, TEND, HEARTH, Treasury"
              echo "    ../mycelix-identity/   - DID, Verifiable Credentials"
              echo "    ../mycelix-governance/ - Proposals, Voting, Constitution"
              echo "    ../mycelix-knowledge/  - Epistemic Claims, Knowledge Graph"
              echo ""
              echo "  This Workspace:"
              echo "    ./sdk/                 - Shared Mycelix SDK"
              echo "    ./happs/               - Application hApps"
              echo "    ./core/0tml/           - Byzantine FL (Zero-TrustML)"
              echo "    ./observatory/         - Unified Dashboard"
              echo ""
              echo "Quick Commands:"
              echo "  just build               - Build all zomes"
              echo "  just test                - Run all tests"
              echo "  just dev                 - Start development servers"
              echo ""

              # Python venv for 0TML
              if [ -d "core/0tml" ] && [ ! -d "core/0tml/.venv" ]; then
                echo "Tip: Run 'python -m venv core/0tml/.venv' for 0TML development"
              fi
            '';
          };

          # CI environment (minimal, fast to build)
          ci = pkgs.mkShell {
            name = "mycelix-ci";
            buildInputs = with pkgs; [
              holochainPackages.holochain
              holochainPackages.hc
              holochainBase.rustToolchain
              nodejs_20
              nodePackages.pnpm
              pythonEnv
              just
              pkg-config
              openssl
              openssl.dev
            ];

            inherit (holochainBase.envVars)
              LIBCLANG_PATH BINDGEN_EXTRA_CLANG_ARGS
              OPENSSL_DIR OPENSSL_LIB_DIR OPENSSL_INCLUDE_DIR;
          };

          # Holochain-only environment (focused zome development)
          holochain = holochainBase.mkHolochainShell {
            name = "mycelix-holochain";
            extraShellHook = ''
              echo "Holochain-focused development environment"
              echo "For full tools, use 'nix develop' (default shell)"
            '';
          };

          # Python ML environment (0TML development)
          ml = pkgs.mkShell {
            name = "mycelix-ml";
            buildInputs = [ pythonEnv pkgs.just ];

            shellHook = ''
              echo "=========================================="
              echo "  Mycelix ML/FL Development Environment"
              echo "=========================================="
              echo ""
              echo "Python: $(python --version)"
              echo ""
              echo "0TML (Zero-TrustML) development:"
              echo "  cd core/0tml"
              echo "  pytest                   - Run FL tests"
              echo "  python -m mycelix_fl     - Run FL node"
              echo ""
            '';
          };

          # Documentation environment
          docs = pkgs.mkShell {
            name = "mycelix-docs";
            buildInputs = with pkgs; [
              nodejs_20
              nodePackages.pnpm
              mdbook
              graphviz
            ];

            shellHook = ''
              echo "Documentation environment ready!"
              echo "  mdbook serve docs/       - Serve Rust docs"
              echo "  cd observatory && pnpm dev - Dev dashboard"
            '';
          };
        };

        # Packages
        packages = {
          # Build all core zomes from workspace
          all-zomes = pkgs.stdenv.mkDerivation {
            name = "mycelix-all-zomes";
            src = ../.;

            nativeBuildInputs = [
              holochainBase.rustToolchain
              pkgs.pkg-config
            ];

            buildInputs = [ pkgs.openssl ];

            buildPhase = ''
              export HOME=$TMPDIR

              # Build each hApp's zomes
              for dir in mycelix-finance mycelix-identity mycelix-governance mycelix-knowledge; do
                if [ -d "$dir" ] && [ -f "$dir/Cargo.toml" ]; then
                  echo "Building $dir..."
                  (cd "$dir" && cargo build --release --target wasm32-unknown-unknown) || true
                fi
              done
            '';

            installPhase = ''
              mkdir -p $out/lib
              find . -path "*/target/wasm32-unknown-unknown/release/*.wasm" -exec cp {} $out/lib/ \;
            '';
          };
        };
      }
    );
}
