# Mycelix Hearth - Family/Household/Kinship Coordination
# HEARTH tier: shared family space between ME (Personal) and WE (Civic)
#
# Usage:
#   nix develop              # Enter dev shell
#   nix develop .#ci         # CI environment (minimal)
{
  description = "Mycelix Hearth - Family/household coordination on Holochain";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    holonix = {
      url = "github:holochain/holonix/main-0.6";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, holonix, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };

        holochainPackages = holonix.packages.${system};

        holochainBase = import ../nix/modules/holochain-base.nix {
          inherit pkgs system;
          holochainPackages = holochainPackages;
        };

      in {
        devShells = {
          default = holochainBase.mkHolochainShell {
            name = "hearth";
            extraBuildInputs = with pkgs; [ nodejs_20 ];
            extraShellHook = ''
              echo "Mycelix Hearth — Family/Household/Kinship Coordination"
              echo ""
              echo "Domains:"
              echo "  hearth-kinship/     - Core membership + kinship bonds"
              echo "  hearth-gratitude/   - Gratitude expressions + circles"
              echo "  hearth-stories/     - Family stories + traditions"
              echo "  hearth-care/        - Care schedules + meal plans"
              echo "  hearth-autonomy/    - Graduated autonomy for minors"
              echo "  hearth-emergency/   - Emergency plans + alerts"
              echo "  hearth-decisions/   - Family decisions + voting"
              echo "  hearth-resources/   - Shared resources + budgets"
              echo "  hearth-milestones/  - Life milestones + transitions"
              echo "  hearth-rhythms/     - Daily/weekly rhythms + presence"
              echo "  hearth-bridge/      - Cross-cluster integration"
              echo ""
              echo "Commands:"
              echo "  cargo build                                     - Build library crates"
              echo "  cargo build --release --target wasm32-unknown-unknown - Build WASM zomes"
              echo "  cargo test                                      - Run unit tests"
              echo ""
            '';
          };

          ci = pkgs.mkShell {
            name = "mycelix-hearth-ci";
            buildInputs = with pkgs; [
              holochainPackages.holochain
              holochainPackages.hc
              holochainBase.rustToolchain
              pkg-config
              openssl
              openssl.dev
            ];

            inherit (holochainBase.envVars)
              LIBCLANG_PATH BINDGEN_EXTRA_CLANG_ARGS
              OPENSSL_DIR OPENSSL_LIB_DIR OPENSSL_INCLUDE_DIR;
          };
        };

        packages = {
          zomes = pkgs.stdenv.mkDerivation {
            name = "mycelix-hearth-zomes";
            src = ./.;
            nativeBuildInputs = [ holochainBase.rustToolchain pkgs.pkg-config ];
            buildInputs = [ pkgs.openssl ];
            buildPhase = ''
              export HOME=$TMPDIR
              cargo build --release --target wasm32-unknown-unknown
            '';
            installPhase = ''
              mkdir -p $out/lib
              find target/wasm32-unknown-unknown/release -name "*.wasm" -exec cp {} $out/lib/ \;
            '';
          };
        };
      }
    );
}
