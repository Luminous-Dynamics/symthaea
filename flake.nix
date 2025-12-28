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
          # Core scientific computing dependencies
          numpy
          scipy
          networkx
          matplotlib
          pandas

          # PyPhi built from source with Python 3.11 fixes
          (ps.buildPythonPackage {
            pname = "pyphi";
            version = "1.2.1.dev1470";
            pyproject = true;
            src = pkgs.fetchFromGitHub {
              owner = "wmayner";
              repo = "pyphi";
              rev = "b78d0e342d37175cbd55cf35a6d52ae035b4c50f";
              hash = "sha256-qh5U0ToJ3fZ87m4gDD+YZmgSwPd215Hbw75KkCC1MGk=";
            };

            # Build system (PyPhi uses hatchling with hatch-vcs for version)
            build-system = with ps; [ hatchling hatch-vcs ];

            # PyPhi dependencies (from pip list when installed)
            dependencies = with ps; [
              numpy
              scipy
              networkx
              joblib
              more-itertools
              ordered-set
              psutil
              pyyaml
              toolz
              tqdm
              # graphillion - may not be in nixpkgs, will try
              # tblib - may not be in nixpkgs, will try
            ];

            # Skip tests and checks during build (they require special setup)
            doCheck = false;
            dontCheckRuntimeDeps = true;
            pythonImportsCheck = [];  # Skip import check due to optional deps
          })
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
            echo "✅ PyPhi built from source (Python 3.11 compatible)"

            # Set OpenSSL paths for rust crates
            export OPENSSL_DIR="${pkgs.openssl.dev}"
            export OPENSSL_LIB_DIR="${pkgs.openssl.out}/lib"
            export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"

            # Set BLAS/LAPACK library paths for ndarray-linalg
            export LD_LIBRARY_PATH="${pkgs.openblas}/lib:${pkgs.gfortran.cc.lib}/lib:$LD_LIBRARY_PATH"
            export LIBRARY_PATH="${pkgs.openblas}/lib:${pkgs.gfortran.cc.lib}/lib:$LIBRARY_PATH"

            # Tell Rust linker where to find BLAS libraries
            export RUSTFLAGS="-C link-arg=-L${pkgs.openblas}/lib -C link-arg=-lopenblas"
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
