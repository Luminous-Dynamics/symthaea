{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Rust toolchain with rustup for target management
    rustup

    # Rust tools
    rust-analyzer
    rustfmt
    clippy

    # WASM build dependencies
    lld          # LLVM linker for WASM (CRITICAL!)
    binaryen     # wasm-opt for optimizing WASM
    wasm-pack    # WASM build tool

    # Holochain tools - using packages from nixpkgs
    # Note: May not be latest version, but good for testing
    # For latest, use the holochain flake

    # Development tools
    nodejs_20    # For frontend development
    pkg-config
    openssl

    # Utilities
    jq           # JSON processing
  ];

  shellHook = ''
    echo "🚀 Mycelix Marketplace Development Environment (shell.nix)"
    echo ""

    # Ensure Rust toolchain is installed
    if ! rustup show &>/dev/null; then
      echo "📦 Installing Rust stable toolchain..."
      rustup default stable
    fi

    # Ensure wasm32 target is installed
    if ! rustup target list | grep -q "wasm32-unknown-unknown (installed)"; then
      echo "📦 Installing wasm32-unknown-unknown target..."
      rustup target add wasm32-unknown-unknown
    fi

    echo "Rust Version: $(rustc --version)"
    echo "Cargo Version: $(cargo --version)"
    echo "LLD Available: $(which lld || echo 'NOT FOUND')"
    echo ""
    echo "Available commands:"
    echo "  cargo build --release --target wasm32-unknown-unknown"
    echo "  ./build-happ.sh"
    echo ""
  '';

  # Set WASM target as default for easier builds
  CARGO_BUILD_TARGET = "wasm32-unknown-unknown";
}
