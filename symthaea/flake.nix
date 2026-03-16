{
  description = "Symthaea HLB - Holographic Liquid Brain: Consciousness-first AI in Rust";

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
          config.android_sdk.accept_license = true;
          config.allowUnfree = true;
        };

        # Rust toolchain - stable with extensions
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
        };

        # Python for PyPhi integration, Neural Bridge training, and ML
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          numpy
          scipy
          networkx
          # pyphi  # If available in nixpkgs, otherwise install via pip

          # Neural Bridge: LLM activation extraction and probe training
          torch
          torchvision
          transformers
          safetensors
          huggingface-hub
          accelerate
          sentencepiece  # For tokenizers
        ]);

        # MuJoCo 3.3.7 pre-built binary (matches mujoco-rs 2.3.3+mj-3.3.7)
        mujoco337 = pkgs.stdenv.mkDerivation {
          pname = "mujoco";
          version = "3.3.7";
          src = pkgs.fetchurl {
            url = "https://github.com/google-deepmind/mujoco/releases/download/3.3.7/mujoco-3.3.7-linux-x86_64.tar.gz";
            sha256 = "075y1niyrg1slzwmdb0551whgbrjmqgxrsq3cw1adnc63vvs558q";
          };
          dontBuild = true;
          installPhase = ''
            mkdir -p $out
            cp -r lib $out/
            cp -r include $out/
            cp -r bin $out/ 2>/dev/null || true
          '';
          fixupPhase = ''
            patchelf --set-rpath "${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.libGL pkgs.xorg.libX11 ]}" $out/lib/libmujoco.so* 2>/dev/null || true
          '';
        };

        # Common build inputs
        buildInputs = with pkgs; [
          # Rust toolchain
          rustToolchain
          cargo-watch
          cargo-edit
          cargo-expand

          # System libraries
          pkg-config
          openssl
          openssl.dev
          cacert

          # Audio (for voice-tts, voice-stt features)
          alsa-lib
          libpulseaudio
          portaudio

          # Audio file formats
          flac
          libvorbis
          libogg
          libsndfile

          # FFT for signal processing
          fftw

          # Image processing (for vision feature)
          libpng
          libjpeg

          # D-Bus for desktop notifications (zbus)
          dbus

          # SQLite (bundled in rusqlite, but system lib can help)
          sqlite

          # libp2p dependencies
          protobuf

          # Tree-sitter for code parsing
          tree-sitter

          # Python for PyPhi integration
          pythonEnv

          # ONNX Runtime (for embeddings, vision, TTS)
          # Note: ort crate uses load-dynamic, so we need the shared lib
          onnxruntime

          # MuJoCo 3.3.7 physics engine (for symthaea-flight mujoco feature)
          mujoco337
          glfw
          libGL

          # C++ standard library (libstdc++.so.6 for neural bridge tests)
          stdenv.cc.cc.lib

          # LaTeX for paper compilation
          (texliveSmall.withPackages (tp: with tp; [
            collection-latexrecommended
            collection-fontsrecommended
            booktabs
            natbib
            multirow
            enumitem
            float
            units        # provides nicefrac.sty
            algorithms   # provides algorithm.sty and algorithmic.sty
            xcolor
            microtype
          ]))

          # Development tools
          cmake
          gnuplot
          graphviz  # For visualizing consciousness graphs
        ];

        nativeBuildInputs = with pkgs; [
          pkg-config
          cmake
          protobuf
        ];

        # Library paths
        libPath = pkgs.lib.makeLibraryPath buildInputs;

        # ONNX Runtime path for dynamic loading
        onnxPath = "${pkgs.onnxruntime}/lib";

        # MuJoCo library path (3.3.7 for mujoco-rs compatibility)
        mujocoPath = "${mujoco337}/lib";

        # Android NDK for mobile cross-compilation (Pixel 8 Pro target)
        androidComposition = pkgs.androidenv.composeAndroidPackages {
          platformVersions = [ "34" ];       # Android 14
          includeNDK = true;
          ndkVersions = [ "27.0.12077973" ]; # NDK r27
        };
        androidSdk = androidComposition.androidsdk;
        ndkRoot = "${androidSdk}/libexec/android-sdk/ndk/27.0.12077973";
        ndkToolchain = "${ndkRoot}/toolchains/llvm/prebuilt/linux-x86_64";

        # Rust toolchain with mobile targets
        rustToolchainMobile = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
          targets = [ "aarch64-linux-android" "aarch64-apple-ios" ];
        };

      in {
        devShells.default = pkgs.mkShell {
          inherit buildInputs nativeBuildInputs;

          shellHook = ''
            export LD_LIBRARY_PATH="${libPath}:${onnxPath}:${mujocoPath}:$LD_LIBRARY_PATH"
            export MUJOCO_PATH="${mujoco337}"
            export MUJOCO_DYNAMIC_LINK_DIR="${mujoco337}/lib"
            export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:${pkgs.alsa-lib}/lib/pkgconfig:${pkgs.dbus}/lib/pkgconfig:$PKG_CONFIG_PATH"

            # ONNX Runtime dynamic loading
            export ORT_DYLIB_PATH="${onnxPath}/libonnxruntime.so"

            # Rust environment
            export RUST_BACKTRACE=1
            export RUST_LOG=info

            # Python path for PyPhi
            export PYTHONPATH="${pythonEnv}/${pythonEnv.sitePackages}:$PYTHONPATH"

            # Data paths
            export SYMTHAEA_DATA_PATH="$PWD/data"
            export EEG_DATA_PATH="$PWD/data/sleep-edf"

            # LibriSpeech (if using external location)
            if [ -d "/home/tstoltz/Downloads/symthaea_stt/data/librispeech/LibriSpeech" ]; then
              export LIBRISPEECH_PATH="/home/tstoltz/Downloads/symthaea_stt/data/librispeech/LibriSpeech"
            fi

            echo ""
            echo "╔═══════════════════════════════════════════════════════════════╗"
            echo "║     SYMTHAEA HLB - Holographic Liquid Brain                   ║"
            echo "║     Consciousness-first AI Development Environment            ║"
            echo "╚═══════════════════════════════════════════════════════════════╝"
            echo ""
            echo "  Rust: $(rustc --version)"
            echo "  Python: $(python --version 2>&1)"
            echo ""
            echo "  Build commands:"
            echo "    cargo build                    # Debug build"
            echo "    cargo build --release          # Release build"
            echo "    cargo build --features full    # All features"
            echo ""
            echo "  Feature flags:"
            echo "    --features service             # symthaea service binary"
            echo "    --features shell               # TUI shell"
            echo "    --features gui                 # GUI application"
            echo "    --features voice-tts           # Text-to-speech (Kokoro)"
            echo "    --features embeddings          # Qwen3/BGE embeddings"
            echo "    --features perception          # Full multimodal perception"
            echo "    --features pyphi               # PyPhi IIT integration"
            echo ""
            echo "  Paper:"
            echo "    cd papers/latex && pdflatex hai_paper && bibtex hai_paper && pdflatex hai_paper && pdflatex hai_paper"
            echo ""
            echo "  Data available:"
            [ -d "$PWD/data/sleep-edf" ] && echo "    - Sleep EDF: $PWD/data/sleep-edf"
            [ -d "$PWD/data/meditation-eeg" ] && echo "    - Meditation EEG: $PWD/data/meditation-eeg"
            [ -d "$LIBRISPEECH_PATH" ] && echo "    - LibriSpeech: $LIBRISPEECH_PATH"
            echo ""
          '';

          # Environment variables for build scripts
          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
        };

        # Mobile development shell — Android NDK + aarch64 targets
        # Usage: nix develop .#mobile
        devShells.mobile = pkgs.mkShell {
          buildInputs = [
            rustToolchainMobile
            androidSdk
            pkgs.pkg-config
            pkgs.openssl
            pkgs.openssl.dev
            pkgs.cacert
            pkgs.jq
          ];

          ANDROID_NDK_HOME = ndkRoot;
          ANDROID_HOME = "${androidSdk}/libexec/android-sdk";
          CC_aarch64_linux_android = "${ndkToolchain}/bin/aarch64-linux-android24-clang";
          AR_aarch64_linux_android = "${ndkToolchain}/bin/llvm-ar";
          CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER = "${ndkToolchain}/bin/aarch64-linux-android24-clang";
          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";

          shellHook = ''
            echo ""
            echo "╔═══════════════════════════════════════════════════════════════╗"
            echo "║     SYMTHAEA MOBILE - ARM64 Cross-Compilation                ║"
            echo "║     Target: Pixel 8 Pro (Tensor G3, aarch64-linux-android)   ║"
            echo "╚═══════════════════════════════════════════════════════════════╝"
            echo ""
            echo "  Rust: $(rustc --version)"
            echo "  NDK:  ${ndkRoot}"
            echo "  CC:   ${ndkToolchain}/bin/aarch64-linux-android24-clang"
            echo ""
            echo "  Build commands:"
            echo "    cargo build --target aarch64-linux-android --release -p symthaea-spore --features native-ffi"
            echo "    ./crates/symthaea-spore/build-android.sh"
            echo ""
            echo "  Deploy to Pixel:"
            echo "    adb push target/aarch64-linux-android/release/libsymthaea_spore.so /data/local/tmp/"
            echo ""
          '';
        };

        # Package definition
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "symthaea";
          version = "0.2.0";
          src = ./.;
          cargoLock = {
            lockFile = ./Cargo.lock;
            # Allow the build to proceed without path dependencies
            # that aren't available in the Nix sandbox
            allowBuiltinFetchGit = true;
          };

          inherit buildInputs nativeBuildInputs;

          # Build with minimal features by default
          buildFeatures = [ "service" ];

          doCheck = false;

          meta = with pkgs.lib; {
            description = "Consciousness-first AI system with HDC and integrated information";
            homepage = "https://luminousdynamics.org";
            license = licenses.mit;
          };
        };

        # Apps
        apps = {
          default = flake-utils.lib.mkApp {
            drv = self.packages.${system}.default;
            name = "symthaea";
          };
        };

        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
