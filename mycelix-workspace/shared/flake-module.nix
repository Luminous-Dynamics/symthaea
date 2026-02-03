# Mycelix Shared Flake Module
# Import this in your app's flake.nix for consistent tooling
#
# Usage in flake.nix:
#   inputs.mycelix-shared.url = "path:../mycelix-workspace/shared";
#   devShells.default = mycelix-shared.lib.mkDevShell { inherit pkgs; };

{ pkgs, holonixPkgs ? null, ... }:

let
  # Holochain 0.6 tools (if available)
  holochainTools = if holonixPkgs != null then [
    holonixPkgs.holochain
    holonixPkgs.hc
    holonixPkgs.lair-keystore
  ] else [];

  # Rust toolchain with WASM support
  rustToolchain = pkgs.rust-bin.stable.latest.default.override {
    extensions = [ "rust-src" "rust-analyzer" ];
    targets = [ "wasm32-unknown-unknown" ];
  };

  # Standard development tools
  devTools = with pkgs; [
    # Rust
    rustToolchain
    cargo-watch
    cargo-edit
    wasm-pack
    wasm-bindgen-cli

    # Node.js
    nodejs_20
    nodePackages.pnpm
    nodePackages.typescript
    nodePackages.typescript-language-server

    # Build tools
    just
    jq
    yq

    # Testing
    # tryorama via npm

    # Documentation
    mdbook
  ];

in {
  # Create a development shell with all Mycelix tools
  mkDevShell = { extraPackages ? [] }: pkgs.mkShell {
    packages = holochainTools ++ devTools ++ extraPackages;

    shellHook = ''
      echo "🍄 Mycelix Development Environment"
      echo "   Holochain: $(hc --version 2>/dev/null || echo 'not available')"
      echo "   Rust: $(rustc --version)"
      echo "   Node: $(node --version)"
      echo ""
      echo "Commands:"
      echo "   just          - Show available tasks"
      echo "   just dev      - Start development mode"
      echo "   just test     - Run all tests"
      echo "   just build    - Build for production"
    '';

    # Environment variables
    RUST_LOG = "info";
    CARGO_TARGET_DIR = "target";
  };

  # Create a minimal shell for CI
  mkCIShell = pkgs.mkShell {
    packages = with pkgs; [
      rustToolchain
      nodejs_20
      just
    ] ++ holochainTools;
  };
}
