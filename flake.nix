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
        nixward-daemon = self.packages.${final.system}.nixward-daemon;
      };

      eachSystem = flake-utils.lib.eachDefaultSystem (system:
        let
          overlays = [ (import rust-overlay) nix-ros-overlay.overlays.default ];
          pkgs = import nixpkgs {
            inherit system overlays;
            config.android_sdk.accept_license = true;
            config.allowUnfree = true;
          };

        # Rust toolchain - read from rust-toolchain.toml (single source of truth), NOT
        # stable.latest and NOT a second hardcoded version string. stable.latest silently
        # drifts ahead of the repo pin as rust-overlay updates; a second hardcoded string
        # here would just as silently drift *away* from rust-toolchain.toml the next time
        # only one of the two gets bumped. Either fragments sccache's cache (compiler binary
        # is part of the cache key) between devShell builds and direct-cargo builds.
        rustToolchainToml = builtins.fromTOML (builtins.readFile ./rust-toolchain.toml);
        rustChannel = rustToolchainToml.toolchain.channel;
        rustToolchain = pkgs.rust-bin.stable.${rustChannel}.default.override {
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

        # External ab initio quantum chemistry reference (pyscf) for
        # cross-verifying symthaea-quantum-chemistry's own results against
        # real reference numbers -- kept as its own devShell (not merged
        # into pythonResearchEnv above) since it's a real, heavier
        # dependency (numpy/scipy/h5py/BLAS) that the "lightweight"
        # research lane's automated check shouldn't have to pull in.
        pythonQcVerifyEnv = pkgs.python311.withPackages (ps: with ps; [
          pyscf
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

        # Rust toolchain with mobile targets - reads rust-toolchain.toml, same single source
        # of truth as rustToolchain above (see its comment for why not a second hardcoded string)
        rustToolchainMobile = pkgs.rust-bin.stable.${rustChannel}.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
          targets = [ "aarch64-linux-android" "aarch64-apple-ios" ];
        };

        commonShellHook = ''
          export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:${pkgs.alsa-lib}/lib/pkgconfig:${pkgs.dbus}/lib/pkgconfig:${pkgs.ffmpeg_7.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"
          export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
          export BINDGEN_EXTRA_CLANG_ARGS="$(< ${pkgs.stdenv.cc}/nix-support/libc-cflags) $(< ${pkgs.stdenv.cc}/nix-support/cc-cflags)"
          export RUST_BACKTRACE=1
          export RUST_LOG=info

          # .cargo/config.toml's [build] rustflags hardcode target-cpu=native
          # and -fuse-ld=mold for the native/host target — mold has no
          # wasm32 support, and wasm32-unknown-unknown's rustc invokes
          # rust-lld directly rather than via a cc driver, so the bare
          # `-fuse-ld=mold` link-arg reaches lld as a raw unrecognized
          # argument and the link step fails outright ("lld: error: unknown
          # argument: -fuse-ld=mold"). A `[target.wasm32-unknown-unknown]
          # rustflags = []` AND a `.cargo/config.toml` `[env]` table entry
          # for this same env var (with force = true) were both tried and
          # neither took effect — confirmed via `cargo build -v`, the
          # native-target flags still appeared on the wasm32 invocation.
          # This plain shell export is the one mechanism verified (via the
          # same `cargo build -v` check) to actually change the rustc
          # invocation. The value must be non-empty: Cargo appears to treat
          # an empty-string override at this tier as "unset" and falls
          # through to [build] anyway (also confirmed empirically) — so
          # this deliberately repeats -C debuginfo=2, which the dev profile
          # already sets by default, purely because it's a genuine no-op.
          # Found + fixed 2026-07-11 building symthaea-ui, the first wasm32
          # crate actually `trunk build`-ed in this workspace in a while.
          export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS="-C debuginfo=2"

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
        # Tauri v2 desktop-app build deps (Linux/GTK+WebKit), mirroring
        # mycelix-workspace/happs/lucid/ui/flake.nix's `.#tauri` shell —
        # reused verbatim rather than reinvented, since that shell already
        # proved out the exact package set this NixOS system needs.
        tauriDeps = with pkgs; [
          pkg-config
          openssl
          openssl.dev
          glib
          glib.dev
          gtk3
          gtk3.dev
          webkitgtk_4_1
          libsoup_3
          libsoup_3.dev
          cairo
          cairo.dev
          pango
          pango.dev
          gdk-pixbuf
          gdk-pixbuf.dev
          atk
          atk.dev
          harfbuzz
          harfbuzz.dev
          dbus
          librsvg
          librsvg.dev
          libappindicator-gtk3
          libayatana-appindicator
          glib-networking
        ];

        tauriPkgConfigPath = pkgs.lib.concatStringsSep ":" [
          "${pkgs.openssl.dev}/lib/pkgconfig"
          "${pkgs.gtk3.dev}/lib/pkgconfig"
          "${pkgs.glib.dev}/lib/pkgconfig"
          "${pkgs.gdk-pixbuf.dev}/lib/pkgconfig"
          "${pkgs.webkitgtk_4_1}/lib/pkgconfig"
          "${pkgs.libsoup_3.dev}/lib/pkgconfig"
          "${pkgs.cairo.dev}/lib/pkgconfig"
          "${pkgs.pango.dev}/lib/pkgconfig"
          "${pkgs.atk.dev}/lib/pkgconfig"
          "${pkgs.harfbuzz.dev}/lib/pkgconfig"
          "${pkgs.librsvg.dev}/lib/pkgconfig"
        ];
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

        # Muse audio-rendering shell: wires the two real render backends
        # `symthaea-muse` already supports but flake.nix never provided.
        # `fluid_render.rs`'s own doc comment says "the Muse Studio launcher
        # provides both [fluidsynth + a soundfont] via nix-shell" — that was
        # aspirational, flake.nix had zero fluidsynth references and
        # `muse_studio` always fell back to the harsher in-crate synth.
        # `pkgs.soundfont-fluid` provides FluidR3_GM2-2.sf2 — the exact
        # "FluidR3" soundfont the A/B listening test in fluid_render.rs's
        # module doc settled on. VCSL (`SYMTHAEA_VCSL_DIR`) is not a nix
        # dependency (it's an already-checked-out git submodule under
        # data/samples/vcsl), so it's discovered at shell-hook runtime
        # relative to $PWD rather than baked in as a Nix path — this shell
        # is meant to be entered from the `symthaea/` directory, same
        # convention every other cargo command in this repo already uses.
        devShells.muse = pkgs.mkShell {
          buildInputs = rustBuildInputs ++ [ pkgs.fluidsynth pkgs.soundfont-fluid ];
          inherit nativeBuildInputs;

          shellHook = commonShellHook + ''
            export LD_LIBRARY_PATH="${rustLibPath}:$LD_LIBRARY_PATH"
            export SYMTHAEA_FLUIDSYNTH="${pkgs.fluidsynth}/bin/fluidsynth"
            export SYMTHAEA_SOUNDFONT="${pkgs.soundfont-fluid}/share/soundfonts/FluidR3_GM2-2.sf2"
            if [[ -d "./data/samples/vcsl" ]]; then
              export SYMTHAEA_VCSL_DIR="./data/samples/vcsl"
            fi
            echo ""
            echo "Symthaea Muse shell — real render backends wired"
            echo "  FluidSynth: $SYMTHAEA_FLUIDSYNTH"
            echo "  Soundfont:  $SYMTHAEA_SOUNDFONT"
            echo "  VCSL dir:   ''${SYMTHAEA_VCSL_DIR:-not found — run from the symthaea/ directory with the vcsl submodule checked out}"
            echo "  cargo run -p symthaea-muse --bin muse_studio --features studio --release"
            echo ""
          '';

          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
        };

        # Formal-verification shell: adds the Lean 4 toolchain so the
        # symthaea-lean-bridge gates (propositional proofs + axiom-provenance
        # via `#print axioms`) can run against real Lean. Core Lean 4 only —
        # Mathlib-dependent arithmetic proofs need a separate lake project.
        devShells.formal = pkgs.mkShell {
          buildInputs = rustBuildInputs ++ [ pkgs.lean4 ];
          inherit nativeBuildInputs;

          shellHook = commonShellHook + ''
            echo ""
            echo "Symthaea formal-verification shell (Lean 4)"
            lean --version 2>/dev/null || true
            echo ""
          '';
        };

        # Tauri v2 desktop-app shell — Muse Desktop
        # (crates/domains/symthaea-muse-ui/src-tauri) and any future Tauri
        # wrapper crate in this workspace.
        # Usage: nix develop .#tauri
        devShells.tauri = pkgs.mkShell {
          buildInputs = rustBuildInputs ++ tauriDeps;
          inherit nativeBuildInputs;

          shellHook = commonShellHook + ''
            export LD_LIBRARY_PATH="${rustLibPath}:${pkgs.libayatana-appindicator}/lib:${pkgs.libappindicator-gtk3}/lib:$LD_LIBRARY_PATH"
            echo ""
            echo "Symthaea Tauri v2 desktop shell"
            echo "  cd crates/domains/symthaea-muse-ui/src-tauri && cargo run"
            echo ""
          '';

          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
          PKG_CONFIG_PATH = tauriPkgConfigPath;
          GIO_MODULE_DIR = "${pkgs.glib-networking}/lib/gio/modules";
          GIO_EXTRA_MODULES = "${pkgs.glib-networking}/lib/gio/modules";
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

        devShells.qc-verify = pkgs.mkShell {
          buildInputs = [ pythonQcVerifyEnv ];

          shellHook = ''
            echo ""
            echo "Symthaea quantum-chemistry verification shell"
            echo "  python3 -c 'import pyscf; print(pyscf.__version__)'"
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

        packages.nixward-daemon = pkgs.rustPlatform.buildRustPackage {
          pname = "nixward-daemon";
          version = "0.1.0";
          src = ./.;
          cargoLock = {
            lockFile = ./Cargo.lock;
            allowBuiltinFetchGit = true;
          };
          buildInputs = rustBuildInputs;
          nativeBuildInputs = nativeBuildInputs;
          buildFeatures = [ "daemon" ];
          cargoBuildFlags = [ "-p" "nixward" "--bin" "nixward-daemon" ];
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
