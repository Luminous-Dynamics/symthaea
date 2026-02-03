# Mycelix Governance - Holochain Development Environment
# Proposals, Voting, Constitution, Execution, Threshold Signing
#
# Usage:
#   nix develop              # Enter dev shell
#   nix develop .#ci         # CI environment (minimal)
{
  description = "Mycelix Governance - Decentralized governance with Phi-weighted voting on Holochain";

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
            name = "governance";
            extraBuildInputs = with pkgs; [ nodejs_20 ];
            extraShellHook = ''
              echo "Zomes:"
              echo "  proposals/           - Proposal creation & lifecycle"
              echo "  voting/              - Phi-weighted voting mechanisms"
              echo "  constitution/        - Constitutional amendments"
              echo "  execution/           - Approved proposal execution"
              echo "  threshold-signing/   - Multi-sig & DKG"
              echo "  bridge/              - Cross-zome integration"
              echo ""
              echo "Governance Tiers (Phi thresholds):"
              echo "  Basic:         Phi >= 0.3"
              echo "  Major:         Phi >= 0.4"
              echo "  Constitutional: Phi >= 0.6"
              echo ""
              echo "Commands:"
              echo "  cargo build              - Build library crates"
              echo "  cargo test               - Run unit tests"
              echo ""
            '';
          };

          ci = pkgs.mkShell {
            name = "mycelix-governance-ci";
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
            name = "mycelix-governance-zomes";
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
