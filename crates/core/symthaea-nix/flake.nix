# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "Symthaea NixOS Mind: Conscious NixOS management via HDC and active inference";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    let
      # NixOS module (system-independent)
      nixosModules.default = import ./nix/module.nix;
      nixosModules.nix-mind = nixosModules.default;
    in
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default;

        commonBuildInputs = with pkgs; [
          pkg-config
          openssl
          openssl.dev
          sqlite
          tree-sitter
          dbus
        ];

        commonNativeBuildInputs = with pkgs; [
          pkg-config
          cmake
        ];

        # Base package derivation
        mkNixMindPackage = { name, features, binName ? name }: pkgs.rustPlatform.buildRustPackage {
          pname = name;
          version = "0.1.0";
          src = ../..;  # Root of the symthaea workspace
          cargoLock = {
            lockFile = ../../Cargo.lock;
            allowBuiltinFetchGit = true;
          };

          buildInputs = commonBuildInputs;
          nativeBuildInputs = commonNativeBuildInputs;

          cargoBuildFlags = [
            "-p" "symthaea-nix"
            "--bin" binName
            "--features" features
          ];

          doCheck = false;

          # Only install the specific binary
          installPhase = ''
            mkdir -p $out/bin
            cp target/release/${binName} $out/bin/
          '';

          meta = with pkgs.lib; {
            description = "Conscious NixOS management tool (${name})";
            homepage = "https://luminousdynamics.org";
            license = licenses.mit;
          };
        };

      in {
        packages = {
          # CLI tool
          nix-mind = mkNixMindPackage {
            name = "nix-mind";
            features = "cli";
            binName = "nix-mind";
          };

          # TUI application
          nix-mind-tui = mkNixMindPackage {
            name = "nix-mind-tui";
            features = "tui";
            binName = "nix-mind-tui";
          };

          # Background daemon
          nix-mind-daemon = mkNixMindPackage {
            name = "nix-mind-daemon";
            features = "daemon";
            binName = "nix-mind-daemon";
          };

          default = self.packages.${system}.nix-mind;
        };

        apps = {
          nix-mind = flake-utils.lib.mkApp {
            drv = self.packages.${system}.nix-mind;
          };
          nix-mind-tui = flake-utils.lib.mkApp {
            drv = self.packages.${system}.nix-mind-tui;
          };
          default = self.apps.${system}.nix-mind;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = commonBuildInputs ++ [
            rustToolchain
            pkgs.cargo-watch
          ];
          nativeBuildInputs = commonNativeBuildInputs;

          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";

          shellHook = ''
            echo "nix-mind development shell"
            echo "  cargo build -p symthaea-nix --features cli   # CLI"
            echo "  cargo build -p symthaea-nix --features tui   # TUI"
            echo "  cargo test -p symthaea-nix --features tui    # Tests"
          '';
        };
      }
    ) // {
      # Top-level NixOS modules (outside eachDefaultSystem)
      inherit nixosModules;
    };
}
