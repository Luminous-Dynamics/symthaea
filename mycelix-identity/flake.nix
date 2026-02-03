# Mycelix Identity - Holochain Development Environment
# DID Registry, Verifiable Credentials, Trust Credentials, Recovery
#
# Usage:
#   nix develop              # Enter dev shell
#   nix develop .#ci         # CI environment (minimal)
{
  description = "Mycelix Identity - Decentralized identity and verifiable credentials on Holochain";

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
            name = "identity";
            extraBuildInputs = with pkgs; [ nodejs_20 ];
            extraShellHook = ''
              echo "Zomes:"
              echo "  did_registry/        - W3C DID method (did:mycelix:)"
              echo "  verifiable_credential/ - VC issuance & verification"
              echo "  trust_credential/    - Trust-weighted credentials"
              echo "  credential_schema/   - Schema definitions"
              echo "  recovery/            - Social recovery"
              echo "  revocation/          - Credential revocation"
              echo "  bridge/              - Cross-zome integration"
              echo ""
              echo "Commands:"
              echo "  cargo build              - Build library crates"
              echo "  cargo test               - Run unit tests"
              echo ""
            '';
          };

          ci = pkgs.mkShell {
            name = "mycelix-identity-ci";
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
            name = "mycelix-identity-zomes";
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
