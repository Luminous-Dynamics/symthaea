{
  description = "Symthaea HLB - Holographic Liquid Brain";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Rust toolchain with all components
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        # Python environment with PyPhi for validation
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          # PyPhi and dependencies for exact IIT 3.0 Φ calculation
          numpy
          scipy
          networkx
          # pyphi - Note: May need to install via pip if not in nixpkgs

          # Additional scientific computing
          matplotlib
          pandas
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Rust toolchain
            rustToolchain
            cargo
            rustc

            # Python environment with scientific packages
            pythonEnv
            python311Packages.pip
            python311Packages.virtualenv

            # Build dependencies for Rust crates
            pkg-config
            openssl

            # System libraries
            gfortran  # For ndarray-linalg
            openblas  # For ndarray-linalg (BLAS)
            lapack    # For ndarray-linalg (LAPACK)
            alsa-lib  # For rodio (audio playback)

            # Development tools
            rust-analyzer
            clippy
            rustfmt

            # Optional: for benchmarking
            criterion
          ];

          # Environment variables for OpenSSL and BLAS
          shellHook = ''
            echo "🌟 Entering Symthaea HLB development environment"
            echo "📦 Rust version: $(rustc --version)"
            echo "🔧 Cargo version: $(cargo --version)"
            echo "🐍 Python version: $(python --version)"
            echo ""
            echo "Week 4: PyPhi Validation + Topology Analysis"
            echo "Ready to validate HDC-based Φ calculation! 🧬"
            echo ""
            echo "To install PyPhi: pip install --user pyphi"

            # Set OpenSSL paths for rust crates
            export OPENSSL_DIR="${pkgs.openssl.dev}"
            export OPENSSL_LIB_DIR="${pkgs.openssl.out}/lib"
            export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"

            # Set BLAS/LAPACK library paths for ndarray-linalg
            export LD_LIBRARY_PATH="${pkgs.openblas}/lib:${pkgs.gfortran.cc.lib}/lib:$LD_LIBRARY_PATH"
            export LIBRARY_PATH="${pkgs.openblas}/lib:${pkgs.gfortran.cc.lib}/lib:$LIBRARY_PATH"

            # Tell Rust linker where to find BLAS libraries
            export RUSTFLAGS="-C link-arg=-L${pkgs.openblas}/lib -C link-arg=-lopenblas"

            # Add user site-packages to PYTHONPATH for PyPhi
            export PYTHONPATH="$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH"
          '';

          # Prevent cargo from using vendored OpenSSL
          OPENSSL_NO_VENDOR = 1;
        };

        # Optional: Package for building release
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "symthaea-hlb";
          version = "0.1.0";
          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          nativeBuildInputs = with pkgs; [
            pkg-config
          ];

          buildInputs = with pkgs; [
            openssl
            gfortran
            openblas
            lapack
          ];
        };
      }
    );
}
