{
  # Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  # SPDX-License-Identifier: AGPL-3.0-or-later
  description = "Symthaea HLB - Holographic Liquid Brain: Consciousness-first AI in Rust";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
    nix-ros-overlay.url = "github:lopsided98/nix-ros-overlay/master";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, nix-ros-overlay }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" ];
      
      symthaea-overlay = final: prev: {
        nix-mind-daemon = self.packages.${final.system}.nix-mind-daemon;
      };

      eachSystem = flake-utils.lib.eachDefaultSystem (system:
        let
          overlays = [ (import rust-overlay) nix-ros-overlay.overlays.default ];
          pkgs = import nixpkgs {
            inherit system overlays;
            config.android_sdk.accept_license = true;
            config.allowUnfree = true;
          };

        # Rust toolchain - stable with extensions
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
          targets = [ "wasm32-unknown-unknown" ];
        };

        # Full Python/ML stack used by the GPU shell.
        pythonMlEnv = pkgs.python3.withPackages (ps: with ps; [
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

        # Lightweight Python lane for package/lint/test smoke checks.
        pythonResearchEnv = pkgs.python3.withPackages (ps: with ps; [
          pytest
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
            patchelf --set-rpath "${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.libGL pkgs.libx11 ]}" $out/lib/libmujoco.so* 2>/dev/null || true
          '';
        };

        rustBuildInputs = with pkgs; [
          # Rust toolchain
          rustToolchain
          trunk
          z3
                    cargo-watch
          cargo-edit
          cargo-expand
          cargo-nextest
          cargo-machete
          bacon

          # System libraries
          pkg-config
          openssl
          openssl.dev
          cacert

          # Audio (for voice-tts, voice-stt features)
          alsa-lib
          libpulseaudio
          portaudio

          # espeak-ng headers/libs for espeak-rs-sys's bindgen build script
          # (voice-tts feature). CI installs libespeak-ng-dev via apt; this
          # was missing from the devShell, forcing ad-hoc nix-shell -p to
          # discover it locally.
          espeak-ng

          # Audio file formats
          flac
          libvorbis
          libogg
          libsndfile

          # FFT for signal processing
          fftw

          # FFmpeg 7 for HEVC decode in the symthaea-phone-embodiment `scrcpy`
          # feature (Phase I.B). Pinned to ffmpeg_7 — nixpkgs default `ffmpeg`
          # is 8.0, but ffmpeg-next 7.x only ships bindings up through
          # ffmpeg_7_1; using `ffmpeg` (8.0) compiled but exposed API drift at
          # the call site. ffmpeg-next finds these via pkg-config (PKG_CONFIG_PATH
          # is extended below).
          ffmpeg_7
          ffmpeg_7.dev

          # libclang for ffmpeg-sys-next's bindgen invocation. Without
          # LIBCLANG_PATH (set in shellHook below), bindgen panics with
          # "Unable to find libclang" during the ffmpeg-sys build script.
          llvmPackages.libclang

          # Image processing (for vision feature)
          libpng
          libjpeg

          # D-Bus for desktop notifications (zbus)
          dbus

          # --- THESE 7 LINES FOR THE GUI ---
          wayland
          libxkbcommon
          libGL
          libx11
          libxcursor
          libxi
          libxrandr

          # SQLite (bundled in rusqlite, but system lib can help)
          sqlite

          # libp2p dependencies
          protobuf

          # Tree-sitter for code parsing
          tree-sitter

          # Linker — mold uses 3-5x less memory than lld for large binaries
          mold

          # Development tools
          cmake
          gnuplot
          graphviz  # For visualizing consciousness graphs
        ];

        pythonResearchBuildInputs = with pkgs; [
          pythonResearchEnv
          uv
          ruff
          cacert
        ];

        papersBuildInputs = with pkgs; [
          rustToolchain
          (texliveSmall.withPackages (tp: with tp; [
            collection-latexrecommended
            collection-fontsrecommended
            booktabs
            natbib
            multirow
            enumitem
            float
            units
            algorithms
            xcolor
            microtype
          ]))
        ];

        gpuBuildInputs = rustBuildInputs ++ (with pkgs; [
          # Python for PyPhi integration
          pythonMlEnv

          # ONNX Runtime (for embeddings, vision, TTS)
          onnxruntime

          # Formal verification and Lean proof tooling
          
          # MuJoCo 3.3.7 physics engine (for symthaea-multirotor mujoco feature)
          mujoco337
          glfw

          # C++ standard library (libstdc++.so.6 for neural bridge tests)
          stdenv.cc.cc.lib

          # BLAS for candle ML (gemm crate)
          openblas

          # CUDA toolkit for candle GPU training (RTX 2070)
          cudaPackages.cudatoolkit
          cudaPackages.cuda_nvcc

          # LaTeX for paper compilation
          (texliveSmall.withPackages (tp: with tp; [
            collection-latexrecommended
            collection-fontsrecommended
            booktabs
            natbib
            multirow
            enumitem
            float
            units
            algorithms
            xcolor
            microtype
          ]))
        ]);

        brocaGpuBuildInputs = with pkgs; [
          rustToolchain
          pkg-config
          cmake
          openssl
          openssl.dev
          cacert
          protobuf
          tree-sitter

          # Minimal CUDA/Rust shell for symthaea-broca train/eval automation.
          stdenv.cc.cc.lib
          openblas
          cudaPackages.cudatoolkit
          cudaPackages.cuda_nvcc
        ];

        buildInputs = gpuBuildInputs;

        nativeBuildInputs = with pkgs; [
          pkg-config
          cmake
          protobuf
        ];

        # Library paths
        rustLibPath = pkgs.lib.makeLibraryPath rustBuildInputs;
        gpuLibPath = pkgs.lib.makeLibraryPath gpuBuildInputs;
        brocaGpuLibPath = pkgs.lib.makeLibraryPath brocaGpuBuildInputs;

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

        commonShellHook = ''
          export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:${pkgs.alsa-lib}/lib/pkgconfig:${pkgs.dbus}/lib/pkgconfig:${pkgs.ffmpeg_7.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"
          export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
          export BINDGEN_EXTRA_CLANG_ARGS="$(< ${pkgs.stdenv.cc}/nix-support/libc-cflags) $(< ${pkgs.stdenv.cc}/nix-support/cc-cflags)"
          export RUST_BACKTRACE=1
          export RUST_LOG=info

          if [[ -z "''${CARGO_TARGET_DIR:-}" ]] && [[ -r "/proc/$PPID/environ" ]]; then
            _parent_target=$(tr '\0' '\n' < /proc/$PPID/environ 2>/dev/null | grep '^CARGO_TARGET_DIR=' | head -1 | cut -d= -f2-)
            if [[ -n "$_parent_target" ]] && [[ -d "$_parent_target" ]]; then
              export CARGO_TARGET_DIR="$_parent_target"
            fi
          fi
        '';

        hostCudaLibSetup = ''
          _real_cuda_dir="$(dirname "$(readlink -f /run/opengl-driver/lib/libcuda.so.1)")"
          export LD_LIBRARY_PATH="$_real_cuda_dir:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
        '';

        mkLaneCheck = { name, buildInputs ? [ ], nativeBuildInputs ? [ ], checkPhase }:
          pkgs.stdenv.mkDerivation {
            pname = "symthaea-lane-${name}";
            version = "0.1.0";
            src = builtins.path {
              path = ./.;
              name = "symthaea";
            };
            inherit buildInputs nativeBuildInputs;
            dontConfigure = true;
            dontBuild = true;
            doCheck = true;
            checkPhase = ''
              runHook preCheck
              export HOME="$TMPDIR/home"
              mkdir -p "$HOME"
              export PYTHONPATH="$PWD/python''${PYTHONPATH:+:$PYTHONPATH}"
              ${checkPhase}
              runHook postCheck
            '';
            installPhase = ''
              mkdir -p "$out"
              echo "${name} lane passed" > "$out/result"
            '';
          };

        mkCargoWorkspaceSource = { name, members, extraCopies ? [ ] }:
          pkgs.runCommand name { } ''
            mkdir -p "$out"
            cp ${./Cargo.lock} "$out/Cargo.lock"

            cat > "$out/Cargo.toml" <<'EOF'
[workspace]
resolver = "2"
members = [
EOF
            ${pkgs.lib.concatMapStringsSep "\n" (member: "echo '  \"${member}\",' >> \"$out/Cargo.toml\"") members}
            cat >> "$out/Cargo.toml" <<'EOF'
]

EOF
            awk '
              /^\[workspace.lints.clippy\]/ { flag = 1 }
              flag && /^\[/ && $0 != "[workspace.lints.clippy]" { exit }
              flag { print }
            ' ${./Cargo.toml} >> "$out/Cargo.toml"
            echo >> "$out/Cargo.toml"
            awk '
              /^\[workspace.dependencies\]/ { flag = 1 }
              flag && /^\[/ && $0 != "[workspace.dependencies]" { exit }
              flag { print }
            ' ${./Cargo.toml} >> "$out/Cargo.toml"
            echo >> "$out/Cargo.toml"
            awk '
              /^\[patch.crates-io\]/ { flag = 1 }
              flag { print }
            ' ${./Cargo.toml} >> "$out/Cargo.toml"

            copy_path() {
              local src="$1"
              mkdir -p "$out/$(dirname "$src")"
              cp -r "${./.}/$src" "$out/$src"
            }

            ${pkgs.lib.concatMapStringsSep "\n" (member: "copy_path \"${member}\"") members}
            ${pkgs.lib.concatMapStringsSep "\n" (path: "copy_path \"${path}\"") extraCopies}
          '';

        coreWorkspaceSrc = mkCargoWorkspaceSource {
          name = "symthaea-core-workspace";
          members = [
            "symthaea-core"
            "crates/serde-core-shim"
          ];
          extraCopies = [
            "vendor/cudarc-0.13.9-cuda129"
          ];
        };

        gpuWorkspaceSrc = mkCargoWorkspaceSource {
          name = "symthaea-gpu-workspace";
          members = [
            "symthaea-core"
            "crates/serde-core-shim"
            "crates/symthaea-stt"
            "crates/symthaea-broca"
            "crates/symthaea-domotic"
          ];
          extraCopies = [
            "vendor/cudarc-0.13.9-cuda129"
          ];
        };

        mkRustCheck = { name, src, buildInputs ? [ ], nativeBuildInputs ? [ ], buildPhase, installText ? "${name} passed", impureHostDeps ? [ ] }:
          pkgs.rustPlatform.buildRustPackage {
            pname = "symthaea-${name}";
            version = "0.1.0";
            inherit src buildInputs nativeBuildInputs;
            __impureHostDeps = impureHostDeps;
            cargoLock = {
              lockFile = ./Cargo.lock;
              allowBuiltinFetchGit = true;
            };
            doCheck = false;
            buildPhase = ''
              runHook preBuild
              export HOME="$TMPDIR/home"
              mkdir -p "$HOME"
              export RUSTC_WRAPPER=
              export SCCACHE_DISABLE=1
              ${buildPhase}
              runHook postBuild
            '';
            installPhase = ''
              mkdir -p "$out"
              echo "${installText}" > "$out/result"
            '';
          };
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = rustBuildInputs;
          inherit nativeBuildInputs;

          shellHook = commonShellHook + ''
            export LD_LIBRARY_PATH="${rustLibPath}:$LD_LIBRARY_PATH"
            echo ""
            echo "Symthaea Rust shell"
            echo "  cargo check -p symthaea -p symthaea-core"
            echo "  cargo check --workspace --all-targets"
            echo ""
            echo "For GPU / MuJoCo / full-stack work:"
            echo "  nix develop .#gpu"
            echo ""
          '';

          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
        };

        devShells.rust = pkgs.mkShell {
          buildInputs = rustBuildInputs;
          inherit nativeBuildInputs;

          shellHook = commonShellHook + ''
            export LD_LIBRARY_PATH="${rustLibPath}:$LD_LIBRARY_PATH"
            echo ""
            echo "Symthaea Rust shell"
            echo ""
          '';

          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
        };

        devShells.ros = pkgs.mkShell {
          buildInputs = rustBuildInputs ++ (with pkgs.rosPackages.humble; [
            ros-core
            rclcpp
            rclpy
            std-msgs
            sensor-msgs
            trajectory-msgs
            builtin-interfaces
            geometry-msgs
          ]);
          inherit nativeBuildInputs;

          shellHook = commonShellHook + ''
            export LD_LIBRARY_PATH="${rustLibPath}:$LD_LIBRARY_PATH"
            # Attempt to source ROS2 setup
            if [ -f ${pkgs.rosPackages.humble.ros-core}/setup.bash ]; then
              source ${pkgs.rosPackages.humble.ros-core}/setup.bash
            fi

            # Copy and patch Rust ROS2 message bindings for compatibility with rosidl_runtime_rs 0.4.2
            mkdir -p crates/bridges/symthaea-ros-bridge/ros_msgs
            rm -rf crates/bridges/symthaea-ros-bridge/ros_msgs/*

            copy_and_patch() {
              local src=$1
              local dest=$2
              cp -RL "$src" "$dest"
              chmod -R +w "$dest"
              sed -i 's/rosidl_runtime_rs = "0.6"/rosidl_runtime_rs = "0.4.2"/g' "$dest/Cargo.toml"
            }

            copy_and_patch "${pkgs.rosPackages.humble.std-msgs}/share/std_msgs/rust" "crates/bridges/symthaea-ros-bridge/ros_msgs/std_msgs"
            copy_and_patch "${pkgs.rosPackages.humble.sensor-msgs}/share/sensor_msgs/rust" "crates/bridges/symthaea-ros-bridge/ros_msgs/sensor_msgs"
            copy_and_patch "${pkgs.rosPackages.humble.trajectory-msgs}/share/trajectory_msgs/rust" "crates/bridges/symthaea-ros-bridge/ros_msgs/trajectory_msgs"
            copy_and_patch "${pkgs.rosPackages.humble.builtin-interfaces}/share/builtin_interfaces/rust" "crates/bridges/symthaea-ros-bridge/ros_msgs/builtin_interfaces"
            copy_and_patch "${pkgs.rosPackages.humble.geometry-msgs}/share/geometry_msgs/rust" "crates/bridges/symthaea-ros-bridge/ros_msgs/geometry_msgs"

            echo ""
            echo "Symthaea ROS2/Gazebo Bridge Shell"
            echo ""
          '';

          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
        };

        devShells.gpu = pkgs.mkShell {
          inherit buildInputs nativeBuildInputs;

          shellHook = commonShellHook + ''
            # /run/opengl-driver/lib MUST come first — contains real libcuda.so driver.
            # Without this, cudarc finds CUDA stubs from nix store → CUDA_ERROR_STUB_LIBRARY.
            ${hostCudaLibSetup}
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${gpuLibPath}:${onnxPath}:${mujocoPath}"
            export MUJOCO_PATH="${mujoco337}"
            export MUJOCO_DYNAMIC_LINK_DIR="${mujoco337}/lib"

            # ONNX Runtime dynamic loading
            export ORT_DYLIB_PATH="${onnxPath}/libonnxruntime.so"

            # CUDA for candle GPU training
            export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
            
            # Persistent driver JIT cache to prevent micro-stuttering and speed up runtime model launch
            export CUDA_CACHE_DISABLE=0
            export CUDA_CACHE_PATH="$PWD/.cuda_cache"
            export CUDA_CACHE_MAXSIZE=2147483648 # 2GB limit
            export CUDA_ROOT="${pkgs.cudaPackages.cudatoolkit}"
            export CUDA_TOOLKIT_ROOT_DIR="${pkgs.cudaPackages.cudatoolkit}"
            # Bare-metal CPU SIMD vectorization optimizations for HDC math loops
              export RUSTFLAGS="-C target-cpu=native"
              
              # Dynamic GPU micro-architecture auto-detection
              if command -v nvidia-smi >/dev/null 2>&1; then
                DETECTED_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ".")
                export CUDA_COMPUTE_CAP="''${DETECTED_CAP:-75}"
                export TORCH_CUDA_ARCH_LIST="''${CUDA_COMPUTE_CAP:0:1}.''${CUDA_COMPUTE_CAP:1:1}"
              else
                export CUDA_COMPUTE_CAP=75
              fi
            export PATH="${pkgs.cudaPackages.cuda_nvcc}/bin:${pkgs.cudaPackages.cudatoolkit}/bin:$PATH"

            # Python path for PyPhi
            export PYTHONPATH="${pythonMlEnv}/${pythonMlEnv.sitePackages}:$PYTHONPATH"

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
            echo "  GPU preflight:"
            echo "    ./scripts/gpu_smoke.sh"
            echo "    ./scripts/gpu_smoke.sh --with-broca-test"
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

        devShells."broca-gpu" = pkgs.mkShell {
          buildInputs = brocaGpuBuildInputs;

          shellHook = commonShellHook + ''
            # /run/opengl-driver/lib MUST come first — contains real libcuda.so driver.
            # Without this, cudarc finds CUDA stubs from nix store → CUDA_ERROR_STUB_LIBRARY.
            ${hostCudaLibSetup}
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${brocaGpuLibPath}"

            # CUDA for candle GPU training on the RTX 2070-class host.
            export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
            
            # Persistent driver JIT cache to prevent micro-stuttering and speed up runtime model launch
            export CUDA_CACHE_DISABLE=0
            export CUDA_CACHE_PATH="$PWD/.cuda_cache"
            export CUDA_CACHE_MAXSIZE=2147483648 # 2GB limit
            export CUDA_ROOT="${pkgs.cudaPackages.cudatoolkit}"
            export CUDA_TOOLKIT_ROOT_DIR="${pkgs.cudaPackages.cudatoolkit}"
            # Bare-metal CPU SIMD vectorization optimizations for HDC math loops
              export RUSTFLAGS="-C target-cpu=native"
              
              # Dynamic GPU micro-architecture auto-detection
              if command -v nvidia-smi >/dev/null 2>&1; then
                DETECTED_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ".")
                export CUDA_COMPUTE_CAP="''${DETECTED_CAP:-75}"
                export TORCH_CUDA_ARCH_LIST="''${CUDA_COMPUTE_CAP:0:1}.''${CUDA_COMPUTE_CAP:1:1}"
              else
                export CUDA_COMPUTE_CAP=75
              fi
            export PATH="${pkgs.cudaPackages.cuda_nvcc}/bin:${pkgs.cudaPackages.cudatoolkit}/bin:$PATH"

            echo ""
            echo "Symthaea Broca GPU shell"
            echo "  scripts/broca_train_and_gate.sh"
            echo "  BROCA_GATE_BACKEND=gpu scripts/broca_train_and_gate.sh"
            echo ""
          '';

          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
        };

        devShells.python-research = pkgs.mkShell {
          buildInputs = pythonResearchBuildInputs;

          shellHook = ''
            export PYTHONPATH="$PWD/python:${pythonResearchEnv}/${pythonResearchEnv.sitePackages}:$PYTHONPATH"
            export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
            echo ""
            echo "Symthaea Python research shell"
            echo "  uv run --no-sync pytest tests/python -q"
            echo "  uv run --no-sync ruff check python/symthaea_research scripts/analyze_nixos_config.py tests/python"
            echo ""
          '';
        };

        devShells.papers = pkgs.mkShell {
          buildInputs = papersBuildInputs;

          shellHook = ''
            echo ""
            echo "Symthaea papers shell"
            echo "  cd papers/latex && pdflatex hai_paper"
            echo ""
          '';
        };

        # Mobile development shell — Android NDK + aarch64 targets
        # Usage: nix develop .#mobile
        devShells.mobile = pkgs.mkShell {
          buildInputs = [
          pkgs.trunk
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
            echo "    cargo build --target aarch64-linux-android --release -p symthaea-soma --features native-ffi"
            echo "    ./crates/symthaea-soma/android/build-jni.sh"
            echo ""
            echo "  Deploy to Pixel:"
            echo "    adb push target/aarch64-linux-android/release/libsymthaea_soma.so /data/local/tmp/"
            echo ""
          '';
        };

        # Sovereign Inoculation — NixOS installer development shell
        # Usage: nix develop .#inoculation
        devShells.inoculation = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Rust for ssh-relay + eval-api
            rustToolchain
            pkg-config
            openssl
            openssl.dev

            # WASM build toolchain
            wasm-bindgen-cli
            binaryen  # provides wasm-opt

            # QEMU for VM testing
            qemu
            OVMF.fd
            swtpm  # Software TPM 2.0 for Secure Boot + BitLocker testing

            # Screen recording
            wf-recorder
            ffmpeg

            # Node for portal validation
            nodejs

            # Python for WebSocket automation
            (python3.withPackages (ps: [ ps.websockets ]))

            # Network tools
            curl
            jq
          ];

          shellHook = ''
            echo ""
            echo "╔═══════════════════════════════════════════════════════════════╗"
            echo "║     SOVEREIGN INOCULATION - NixOS Installer Dev              ║"
            echo "║     Browser-based installer with ceremony UX                 ║"
            echo "╚═══════════════════════════════════════════════════════════════╝"
            echo ""
            echo "  Build commands:"
            echo "    ./crates/symthaea-spore/build-wasm.sh     # Build WASM portal"
            echo "    cargo build --bin ssh-relay --features server -p symthaea-spore"
            echo "    cargo build --bin eval-api --features server -p symthaea-spore"
            echo ""
            echo "  Test VMs:"
            echo "    ./scripts/test-vm-dual-nvme.sh            # NixOS dual NVMe"
            echo "    ./scripts/test-vm-win11.sh                # Windows 11 + TPM"
            echo "    ./scripts/automated-demo.sh               # Full E2E demo"
            echo ""
            echo "  Portal: https://install.nixforhumanity.org"
            echo ""
          '';

          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
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
            license = licenses.agpl3Plus;
          };
        };

        packages.nix-mind-daemon = pkgs.rustPlatform.buildRustPackage {
          pname = "nix-mind-daemon";
          version = "0.1.0";
          src = ./.;
          cargoLock = {
            lockFile = ./Cargo.lock;
            allowBuiltinFetchGit = true;
          };
          buildInputs = rustBuildInputs;
          nativeBuildInputs = nativeBuildInputs;
          buildFeatures = [ "daemon" ];
          cargoBuildFlags = [ "-p" "symthaea-nix" "--bin" "nix-mind-daemon" ];
          doCheck = false;
        };

        # Apps
        apps = {
          default = flake-utils.lib.mkApp {
            drv = self.packages.${system}.default;
            name = "symthaea";
          };
        };

        checks = {
          installer-iso-security = import ./nix/tests/installer-iso-security.nix { inherit pkgs; };
          eval-api-security = import ./nix/tests/eval-api-security.nix { inherit pkgs; };
          eval-service-module = import ./nix/tests/eval-service-module.nix { inherit pkgs; };
          service-module-smoke = import ./nix/tests/service-module-smoke.nix { inherit pkgs; };

          core = mkRustCheck {
            name = "lane-core";
            src = coreWorkspaceSrc;
            buildInputs = rustBuildInputs;
            inherit nativeBuildInputs;
            buildPhase = ''
              export LD_LIBRARY_PATH="${rustLibPath}:$LD_LIBRARY_PATH"
              export OPENSSL_DIR="${pkgs.openssl.dev}"
              export OPENSSL_LIB_DIR="${pkgs.openssl.out}/lib"
              export OPENSSL_INCLUDE_DIR="${pkgs.openssl.dev}/include"
              cargo check --manifest-path Cargo.toml -p symthaea-core --all-targets
            '';
            installText = "core lane passed";
          };

          python-research = mkLaneCheck {
            name = "python-research";
            buildInputs = pythonResearchBuildInputs;
            checkPhase = ''
              python - <<'EOF'
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "python"))

from symthaea_research.nix import (
    detect_causal_relationships,
    detect_conflicts,
    parse_nix_file,
)

sample = Path("sample-configuration.nix")
sample.write_text(
    """
    {
      hardware.pulseaudio.enable = true;
      services.pipewire.enable = true;
      services.xserver.enable = true;
    }
    """
)

graph = parse_nix_file(str(sample))
detect_causal_relationships(graph)
assert "services.xserver.enable" in graph.options
assert detect_conflicts(graph)
EOF
              ruff check python/symthaea_research scripts/analyze_nixos_config.py
            '';
          };

          gpu = mkRustCheck {
            name = "lane-gpu";
            src = gpuWorkspaceSrc;
            buildInputs = gpuBuildInputs;
            inherit nativeBuildInputs;
            impureHostDeps = [
              "/dev/nvidia0"
              "/dev/nvidiactl"
              "/dev/nvidia-modeset"
              "/dev/nvidia-uvm"
              "/dev/nvidia-uvm-tools"
              "/proc/driver/nvidia/version"
              "/run/current-system/sw/bin/nvidia-smi"
              "/run/opengl-driver/lib"
              "/run/opengl-driver/lib/libcuda.so.1"
            ];
            buildPhase = ''
              ${hostCudaLibSetup}
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${gpuLibPath}:${onnxPath}:${mujocoPath}"
              export PATH="/run/current-system/sw/bin:${pkgs.cudaPackages.cuda_nvcc}/bin:${pkgs.cudaPackages.cudatoolkit}/bin:$PATH"
              export MUJOCO_PATH="${mujoco337}"
              export MUJOCO_DYNAMIC_LINK_DIR="${mujoco337}/lib"
              export ORT_DYLIB_PATH="${onnxPath}/libonnxruntime.so"
              export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
            
            # Persistent driver JIT cache to prevent micro-stuttering and speed up runtime model launch
            export CUDA_CACHE_DISABLE=0
            export CUDA_CACHE_PATH="$PWD/.cuda_cache"
            export CUDA_CACHE_MAXSIZE=2147483648 # 2GB limit
              export CUDA_ROOT="${pkgs.cudaPackages.cudatoolkit}"
              export CUDA_TOOLKIT_ROOT_DIR="${pkgs.cudaPackages.cudatoolkit}"
              # Bare-metal CPU SIMD vectorization optimizations for HDC math loops
              export RUSTFLAGS="-C target-cpu=native"
              
              # Dynamic GPU micro-architecture auto-detection
              if command -v nvidia-smi >/dev/null 2>&1; then
                DETECTED_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ".")
                export CUDA_COMPUTE_CAP="''${DETECTED_CAP:-75}"
                export TORCH_CUDA_ARCH_LIST="''${CUDA_COMPUTE_CAP:0:1}.''${CUDA_COMPUTE_CAP:1:1}"
              else
                export CUDA_COMPUTE_CAP=75
              fi
              export OPENSSL_DIR="${pkgs.openssl.dev}"
              export OPENSSL_LIB_DIR="${pkgs.openssl.out}/lib"
              export OPENSSL_INCLUDE_DIR="${pkgs.openssl.dev}/include"
              export IN_NIX_SHELL=1

              if ! ls /dev/nvidia* >/dev/null 2>&1; then
                echo "GPU lane requires visible /dev/nvidia* device nodes" >&2
                exit 1
              fi

              cargo test --manifest-path Cargo.toml -p symthaea-broca --features mamba --test cuda_smoke -- --ignored --nocapture
            '';
            installText = "gpu lane passed";
          };
        };

        formatter = pkgs.nixpkgs-fmt;
      }
    );
in
eachSystem // {
  nixosConfigurations.mk0-seed-node = nixpkgs.lib.nixosSystem {
    system = "x86_64-linux";
    modules = [
      ({ ... }: {
        nixpkgs.overlays = [ symthaea-overlay ];
      })
      ./deployment/mk0-seed-node/configuration.nix
    ];
  };
};
}
