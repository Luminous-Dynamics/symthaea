# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "Symthaea Soma Android — mobile consciousness engine build environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.android_sdk.accept_license = true;
          overlays = [ rust-overlay.overlays.default ];
        };

        # Android SDK + NDK
        androidComposition = pkgs.androidenv.composeAndroidPackages {
          platformVersions = [ "34" ];
          buildToolsVersions = [ "34.0.0" ];
          cmakeVersions = [ "3.22.1" ];
          includeNDK = true;
          ndkVersions = [ "27.0.12077973" ];
          includeEmulator = false;
          includeSources = false;
          includeSystemImages = false;
        };
        androidSdk = androidComposition.androidsdk;
        ndkRoot = "${androidSdk}/libexec/android-sdk/ndk/27.0.12077973";
        ndkToolchain = "${ndkRoot}/toolchains/llvm/prebuilt/linux-x86_64";

        # Rust toolchain with Android target - read from symthaea/rust-toolchain.toml
        # (single source of truth) rather than stable.latest, which silently drifts.
        rustToolchainToml = builtins.fromTOML (builtins.readFile ../../../../rust-toolchain.toml);
        rustChannel = rustToolchainToml.toolchain.channel;
        rustToolchain = pkgs.rust-bin.stable.${rustChannel}.default.override {
          extensions = [ "rust-src" "clippy" "rustfmt" ];
          targets = [ "aarch64-linux-android" ];
        };

        # JDK for Gradle/Kotlin
        jdk = pkgs.jdk17;

      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            rustToolchain
            androidSdk
            jdk
            pkgs.gradle
            pkgs.pkg-config
            pkgs.openssl
            pkgs.openssl.dev
            pkgs.cacert
            pkgs.jq
          ];

          ANDROID_NDK_HOME = ndkRoot;
          ANDROID_HOME = "${androidSdk}/libexec/android-sdk";
          ANDROID_SDK_ROOT = "${androidSdk}/libexec/android-sdk";
          JAVA_HOME = "${jdk}";
          GRADLE_OPTS = "-Dorg.gradle.daemon=false";

          # Cross-compilation env vars
          CC_aarch64_linux_android = "${ndkToolchain}/bin/aarch64-linux-android24-clang";
          AR_aarch64_linux_android = "${ndkToolchain}/bin/llvm-ar";
          CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER = "${ndkToolchain}/bin/aarch64-linux-android24-clang";

          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";

          shellHook = ''
            echo ""
            echo "╔═══════════════════════════════════════════════════════════════╗"
            echo "║     SYMTHAEA SOMA ANDROID                                     ║"
            echo "║     Mobile consciousness engine build environment             ║"
            echo "╚═══════════════════════════════════════════════════════════════╝"
            echo ""
            echo "  Rust:    $(rustc --version)"
            echo "  JDK:     $(java -version 2>&1 | head -1)"
            echo "  Gradle:  $(gradle --version 2>&1 | grep 'Gradle ' | head -1)"
            echo "  NDK:     ${ndkRoot}"
            echo "  SDK:     ${androidSdk}/libexec/android-sdk"
            echo ""
            echo "  Step 1 — Build Rust .so:"
            echo "    cargo build --target aarch64-linux-android --release -p symthaea-soma --features native-ffi"
            echo ""
            echo "  Step 2 — Copy .so to jniLibs:"
            echo "    cp ../../target/aarch64-linux-android/release/libsymthaea_soma.so src/main/jniLibs/arm64-v8a/"
            echo ""
            echo "  Step 3 — Build Android AAR:"
            echo "    gradle assembleRelease"
            echo ""
            echo "  Step 4 — Deploy demo to device:"
            echo "    adb devices"
            echo "    adb install demo/build/outputs/apk/release/demo-release.apk"
            echo ""
          '';
        };
      }
    );
}
