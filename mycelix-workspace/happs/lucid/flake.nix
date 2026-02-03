# LUCID hApp - Living Unified Consciousness for Insight & Discovery
# Personal Knowledge Graph on Holochain
#
# Usage:
#   nix develop              # Full dev environment
#   nix develop .#ci         # CI environment (minimal)
#   nix develop .#tauri      # Tauri desktop development
{
  description = "LUCID - Personal Knowledge Graph hApp for Mycelix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # Holochain from holonix - using stable 0.6.x release (recommended)
    # Compatible with: Tryorama 0.19.0, @holochain/client 0.20.0
    holonix = {
      url = "github:holochain/holonix/main-0.6";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Rust toolchain
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, holonix, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };

        # Holochain packages from holonix
        holochainPackages = holonix.packages.${system};

        # Rust toolchain with wasm32 target
        # Use latest stable Rust for zome builds (wasmer issues affect sweettest only)
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
          targets = [ "wasm32-unknown-unknown" ];
        };

        # Common environment variables for Holochain builds
        holochainEnvVars = {
          CARGO_TARGET_DIR = "target";
          RUST_BACKTRACE = "1";
        };

        # Node.js packages for UI development
        nodeEnv = with pkgs; [
          nodejs_20
          nodePackages.pnpm
          nodePackages.typescript
          nodePackages.typescript-language-server
        ];

        # Tauri desktop app dependencies (Linux)
        tauriDeps = with pkgs; [
          # Build dependencies
          pkg-config
          openssl
          openssl.dev

          # GTK/WebKit for Tauri on Linux
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

          # Additional runtime deps
          dbus
          librsvg
          librsvg.dev
          libappindicator-gtk3
          glib-networking

          # AppImage support
          appimage-run
        ];

        # PKG_CONFIG_PATH for Tauri build
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
        devShells = {
          # Full development environment
          default = pkgs.mkShell {
            name = "lucid";

            buildInputs = [
              # Holochain tools
              holochainPackages.holochain
              holochainPackages.hc
              holochainPackages.lair-keystore
              holochainPackages.bootstrap-srv  # kitsune2-bootstrap-srv for tests

              # Rust
              rustToolchain
              pkgs.pkg-config
              pkgs.openssl
              pkgs.openssl.dev
              pkgs.clang
              pkgs.llvmPackages.libclang

              # Build tools
              pkgs.just
              pkgs.watchexec
            ] ++ nodeEnv ++ tauriDeps;

            inherit (holochainEnvVars) CARGO_TARGET_DIR RUST_BACKTRACE;

            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            OPENSSL_DIR = "${pkgs.openssl.dev}";
            OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
            OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
            PKG_CONFIG_PATH = tauriPkgConfigPath;

            # GTK environment for Tauri
            GIO_MODULE_DIR = "${pkgs.glib-networking}/lib/gio/modules";
            GIO_EXTRA_MODULES = "${pkgs.glib-networking}/lib/gio/modules";

            shellHook = ''
              echo ""
              echo "🧠 LUCID - Personal Knowledge Graph"
              echo ""
              echo "Commands:"
              echo "  just build        - Build all WASM zomes"
              echo "  just test         - Run tests"
              echo "  just pack         - Package DNA/hApp"
              echo "  just ui           - Start UI dev server"
              echo "  just tauri-dev    - Start Tauri desktop app"
              echo ""
              echo "Holochain: $(holochain --version)"
              echo "Rust: $(rustc --version)"
              echo ""
            '';
          };

          # CI environment (minimal, fast to build)
          ci = pkgs.mkShell {
            name = "lucid-ci";
            buildInputs = [
              holochainPackages.holochain
              holochainPackages.hc
              holochainPackages.lair-keystore
              holochainPackages.bootstrap-srv  # kitsune2-bootstrap-srv for tests
              rustToolchain
              pkgs.nodejs_20
              pkgs.nodePackages.pnpm
              pkgs.just
              pkgs.pkg-config
              pkgs.openssl
              pkgs.openssl.dev
              pkgs.clang
              pkgs.llvmPackages.libclang
            ];

            inherit (holochainEnvVars) CARGO_TARGET_DIR RUST_BACKTRACE;
            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            OPENSSL_DIR = "${pkgs.openssl.dev}";
            OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
            OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
          };

          # UI-only development (no Holochain)
          ui = pkgs.mkShell {
            name = "lucid-ui";
            buildInputs = nodeEnv ++ [ pkgs.just ];

            shellHook = ''
              echo "LUCID UI Development Environment"
              echo "  cd ui && npm run dev"
            '';
          };

          # Tauri desktop app development
          tauri = pkgs.mkShell {
            name = "lucid-tauri";
            buildInputs = nodeEnv ++ tauriDeps ++ [
              rustToolchain
              pkgs.cargo-tauri
              pkgs.just
              pkgs.pkg-config
              pkgs.clang
              pkgs.llvmPackages.libclang
            ];

            # GTK environment variables for Tauri on Linux
            GIO_MODULE_DIR = "${pkgs.glib-networking}/lib/gio/modules";
            GIO_EXTRA_MODULES = "${pkgs.glib-networking}/lib/gio/modules";
            WEBKIT_DISABLE_COMPOSITING_MODE = "1";

            # PKG_CONFIG_PATH for finding GTK/WebKit libraries
            PKG_CONFIG_PATH = tauriPkgConfigPath;

            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            OPENSSL_DIR = "${pkgs.openssl.dev}";
            OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
            OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";

            shellHook = ''
              echo ""
              echo "🖥️  LUCID Tauri Desktop App Development"
              echo ""
              echo "Commands:"
              echo "  cd ui && cargo tauri dev      - Start Tauri dev mode"
              echo "  cd ui && cargo tauri build    - Build release"
              echo ""
              echo "Environment:"
              echo "  PKG_CONFIG_PATH includes GTK, WebKit, and Tauri deps"
              echo ""
            '';
          };
        };

        # Packages
        packages = {
          # Build all WASM zomes
          wasm = pkgs.stdenv.mkDerivation {
            name = "lucid-zomes";
            src = ./.;

            nativeBuildInputs = [
              rustToolchain
              pkgs.pkg-config
              pkgs.clang
              pkgs.llvmPackages.libclang
            ];

            buildInputs = [ pkgs.openssl ];

            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

            buildPhase = ''
              export HOME=$TMPDIR
              cargo build --release --target wasm32-unknown-unknown
            '';

            installPhase = ''
              mkdir -p $out/lib
              find target/wasm32-unknown-unknown/release -name "*.wasm" -exec cp {} $out/lib/ \;
            '';
          };

          # Package DNA
          dna = pkgs.stdenv.mkDerivation {
            name = "lucid-dna";
            src = ./.;

            nativeBuildInputs = [
              rustToolchain
              holochainPackages.hc
              pkgs.pkg-config
              pkgs.clang
              pkgs.llvmPackages.libclang
            ];

            buildInputs = [ pkgs.openssl ];

            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

            buildPhase = ''
              export HOME=$TMPDIR
              cargo build --release --target wasm32-unknown-unknown
              hc dna pack . -o lucid.dna
            '';

            installPhase = ''
              mkdir -p $out
              cp lucid.dna $out/
            '';
          };

          # Package hApp
          happ = pkgs.stdenv.mkDerivation {
            name = "lucid-happ";
            src = ./.;

            nativeBuildInputs = [
              rustToolchain
              holochainPackages.hc
              pkgs.pkg-config
              pkgs.clang
              pkgs.llvmPackages.libclang
            ];

            buildInputs = [ pkgs.openssl ];

            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

            buildPhase = ''
              export HOME=$TMPDIR
              cargo build --release --target wasm32-unknown-unknown
              hc dna pack . -o lucid.dna
              hc app pack . -o lucid.happ
            '';

            installPhase = ''
              mkdir -p $out
              cp lucid.happ $out/
              cp lucid.dna $out/
            '';
          };
        };

        # Default package
        defaultPackage = self.packages.${system}.happ;
      }
    );
}
