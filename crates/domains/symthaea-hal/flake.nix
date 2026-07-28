# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "symthaea-hal: Hardware abstraction layer for Symthaea humanoid robotics";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;

      # Rust channel - read from symthaea/rust-toolchain.toml (single source of
      # truth) rather than stable.latest, which silently drifts over time.
      # Computed once here since it doesn't depend on `system`.
      rustChannel = (builtins.fromTOML (builtins.readFile ../../../rust-toolchain.toml)).toolchain.channel;

      mkHalPackage = { pkgs, rustPlatform, features ? [ "linux" ], name ? "symthaea-hal" }:
        rustPlatform.buildRustPackage {
          pname = name;
          version = "0.1.0";
          src = ../..;  # workspace root

          buildAndTestSubdir = "crates/symthaea-hal";
          cargoLock.lockFile = ../../Cargo.lock;

          cargoBuildFlags = [
            "--features" (builtins.concatStringsSep "," features)
          ];

          nativeBuildInputs = with pkgs; [
            pkg-config
          ];

          buildInputs = with pkgs; [
            # i2c-tools for linux feature
          ] ++ pkgs.lib.optionals (builtins.elem "linux" features) [
            # linux-embedded-hal uses libi2c
          ];

          # Run tests whenever the build platform can execute the target.
          # Cross packages remain build-only and are exercised by target CI/HIL.
          doCheck = pkgs.stdenv.buildPlatform.canExecute pkgs.stdenv.hostPlatform;
          cargoTestFlags = [
            "--features" (builtins.concatStringsSep "," features)
          ];

          meta = with pkgs.lib; {
            description = "Hardware abstraction layer for Symthaea humanoid robot";
            homepage = "https://github.com/Luminous-Dynamics/symthaea";
            license = licenses.agpl3Plus;
          };
        };
    in
    {
      packages = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          rustToolchain = pkgs.rust-bin.stable.${rustChannel}.default;
          rustPlatform = pkgs.makeRustPlatform {
            cargo = rustToolchain;
            rustc = rustToolchain;
          };
        in
        {
          default = mkHalPackage {
            inherit pkgs rustPlatform;
            features = [ "linux" ];
          };

          calibrate = mkHalPackage {
            inherit pkgs rustPlatform;
            name = "hal-calibrate";
            features = [ "linux" "calibrate" ];
          };
        } // (if system == "x86_64-linux" then {
          aarch64 = let
            pkgsCross = import nixpkgs {
              inherit system;
              crossSystem.config = "aarch64-unknown-linux-gnu";
              overlays = [ rust-overlay.overlays.default ];
            };
            crossRustPlatform = pkgsCross.makeRustPlatform {
              cargo = pkgsCross.rust-bin.stable.${rustChannel}.default;
              rustc = pkgsCross.rust-bin.stable.${rustChannel}.default;
            };
          in
            mkHalPackage {
              pkgs = pkgsCross;
              rustPlatform = crossRustPlatform;
              features = [ "linux" ];
              name = "symthaea-hal-aarch64";
            };
        } else {})
      );

      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          rustToolchain = pkgs.rust-bin.stable.${rustChannel}.default.override {
            extensions = [ "rust-src" "rust-analyzer" ];
          };
        in
        {
          default = pkgs.mkShell {
            buildInputs = with pkgs; [
              rustToolchain
              pkg-config
              i2c-tools
            ];

            shellHook = ''
              echo "symthaea-hal dev shell"
              echo "  cargo build -p symthaea-hal --features linux"
              echo "  cargo test -p symthaea-hal"
              echo "  cargo build -p symthaea-hal --features calibrate"
            '';
          };
        }
      );
    };
}
