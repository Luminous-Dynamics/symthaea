{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Rust toolchain
    rustc
    cargo
    rustfmt
    rust-analyzer

    # Build dependencies
    gcc
    pkg-config
    openssl
    zlib
    stdenv.cc.cc.lib

    # WASM linker (required for wasm32-unknown-unknown target)
    lld

    # Optional: Holochain CLI for DNA packing
    # holochain
  ];

  shellHook = ''
    echo "🍄 Mycelix Mail Development Environment"
    echo "Rust version: $(rustc --version)"
    echo "Targets: $(rustc --print target-list | grep wasm32)"
  '';
}
