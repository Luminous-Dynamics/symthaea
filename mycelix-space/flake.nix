{
  description = "Mycelix Space - Decentralized Space Domain Awareness Network";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # Holochain development environment - use holonix for full tooling
    holochain = {
      url = "github:holochain/holochain?ref=holochain-0.4.1";
    };

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, holochain, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Rust toolchain with wasm target
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          targets = [ "wasm32-unknown-unknown" ];
        };

        # Get holochain's devShell for full tooling (hc, holochain, lair-keystore)
        holochainDevShell = holochain.devShells.${system}.default or null;

      in
      {
        # Development shell with Holochain tools from upstream
        devShells.default = if holochainDevShell != null then
          pkgs.mkShell {
            inputsFrom = [ holochainDevShell ];
            buildInputs = with pkgs; [
              # Additional tools
              cargo-watch
              cargo-expand
            ];
            shellHook = ''
              echo "=== Mycelix Space Development Environment ==="
              echo ""
              echo "Rust: $(rustc --version 2>/dev/null || echo 'from holochain')"
              echo "Cargo: $(cargo --version 2>/dev/null || echo 'from holochain')"
              if command -v hc &> /dev/null; then
                echo "Holochain CLI: $(hc --version 2>/dev/null || echo 'available')"
              fi
              echo ""
              echo "Commands:"
              echo "  ./scripts/build-happ.sh  - Build WASM and package hApp"
              echo "  cargo check --workspace  - Check all code"
              echo "  cargo test --workspace   - Run tests"
              echo ""
            '';
          }
        else
          # Fallback shell without holochain tools
          pkgs.mkShell {
            buildInputs = with pkgs; [
              rustToolchain
              cargo-watch
              cargo-expand
              pkg-config
              openssl
            ];
            shellHook = ''
              echo "=== Mycelix Space Development Environment (No Holochain) ==="
              echo ""
              echo "Rust: $(rustc --version)"
              echo "Cargo: $(cargo --version)"
              echo ""
              echo "WARNING: Holochain tools not available."
              echo "WASM builds will work, but DNA/hApp packaging requires hc."
              echo ""
              echo "To get full Holochain tools, use:"
              echo "  nix develop .#holonix"
              echo ""
            '';
            RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
          };

        # Alternative shell using holochain's holonix directly
        devShells.holonix = if holochainDevShell != null then holochainDevShell else
          pkgs.mkShell {
            buildInputs = [ pkgs.coreutils ];
            shellHook = ''
              echo "ERROR: Holochain devShell not available for this system"
              exit 1
            '';
          };

        # Rust-only shell (faster to enter, for pure Rust development)
        devShells.rust = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustToolchain
            cargo-watch
            cargo-expand
            pkg-config
            openssl
          ];
          shellHook = ''
            echo "=== Mycelix Space Rust Development ==="
            echo "Rust: $(rustc --version)"
            echo ""
          '';
          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
        };

        # Package the orbital-mechanics library (non-Holochain, can be used standalone)
        packages.orbital-mechanics = pkgs.rustPlatform.buildRustPackage {
          pname = "orbital-mechanics";
          version = "0.1.0";
          src = ./lib/orbital-mechanics;
          cargoLock.lockFile = ./Cargo.lock;
        };
      }
    );
}
