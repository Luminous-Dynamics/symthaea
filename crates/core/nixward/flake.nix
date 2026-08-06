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
      nixosModules.nixward = nixosModules.default;
    in
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Rust toolchain - read from symthaea/rust-toolchain.toml (single source
        # of truth) rather than stable.latest, which silently drifts over time.
        rustToolchainToml = builtins.fromTOML (builtins.readFile ../../../rust-toolchain.toml);
        rustChannel = rustToolchainToml.toolchain.channel;
        rustToolchain = pkgs.rust-bin.stable.${rustChannel}.default;

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

        # nixward itself depends on nothing but symthaea-core (checked
        # directly in its Cargo.toml) -- but Cargo must still resolve every
        # *workspace member's* manifest before it can build anything at all,
        # via the crates/core/* and crates/domains/* globs in the symthaea
        # workspace root. One of those members, symthaea-infrastructure (an
        # OPTIONAL dependency of the root `symthaea` crate, not of
        # nixward), has its own optional chain
        # (symthaea-engineering -> symthaea-broca -> ...) that reaches
        # SEVERAL sibling top-level monorepo directories -- first found
        # mycelix-workspace/ (for mycelix-zkp-core), then symtropy/ (for
        # symtropy-robotics-bridge-core) once that was included. Given the
        # chain cascades rather than bottoming out at one sibling,
        # allowlisting subtrees one discovery at a time isn't worth the
        # per-attempt Nix build cost -- widened to the full monorepo root
        # instead, filtered only to exclude build-artifact/VCS noise (never
        # actually needed by any workspace member's manifest, and would
        # bloat the Nix store copy for nothing). See
        # SYMTHAEA_NIXOS_MANAGEMENT_IMPROVEMENT_PLAN_2026-07-26.md
        # "Remaining work plan" item 1.
        monorepoRoot = ../../../..;
        excludedDirNames = [ "target" "node_modules" ".git" ".jj" "dist" "result" ];
        monorepoSrc = pkgs.lib.cleanSourceWith {
          name = "luminous-dynamics-nixward-src";
          src = monorepoRoot;
          filter = path: type:
            let
              relPath = pkgs.lib.removePrefix (toString monorepoRoot) (toString path);
              excluded =
                builtins.elem (baseNameOf path) excludedDirNames
                || builtins.any (d: pkgs.lib.hasInfix "/${d}/" relPath) excludedDirNames;
            in
            !excluded;
        };

        # Base package derivation
        mkNixwardPackage = { name, features, binName ? name }: pkgs.rustPlatform.buildRustPackage {
          pname = name;
          version = "0.1.0";
          src = monorepoSrc;
          sourceRoot = "luminous-dynamics-nixward-src/symthaea";
          cargoLock = {
            lockFile = ../../../Cargo.lock;
            allowBuiltinFetchGit = true;
          };

          buildInputs = commonBuildInputs;
          nativeBuildInputs = commonNativeBuildInputs;

          # This NixOS host sets RUSTC_WRAPPER=sccache system-wide (root
          # CLAUDE.md Rule 5's dev-environment setup), and it leaks into
          # this build regardless of the invoking shell's own env -- Nix
          # builds don't inherit the calling shell's environment at all by
          # design, so this must come through nix-daemon's own (systemd
          # service) environment, itself carrying the system-wide
          # NixOS `environment.variables` setting. Neither unsetting
          # RUSTC_WRAPPER in the invoking shell nor giving the build a
          # throwaway HOME (to hide ~/.cargo/config.toml, which also sets
          # `rustc-wrapper = "sccache"`) fixed it -- both tried and
          # verified ineffective. sccache isn't on PATH inside the Nix
          # build sandbox regardless of which of these is the real leak
          # path, so explicitly clearing the wrapper at the derivation
          # level is the one fix guaranteed to override whatever inherited
          # it. Found while verifying this flake actually builds
          # end-to-end for the first time (error was: "could not execute
          # process `sccache ... cargo-auditable rustc -vV`").
          env.HOME = "$TMPDIR";
          env.RUSTC_WRAPPER = "";

          cargoBuildFlags = [
            "-p" "nixward"
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
          nixward = mkNixwardPackage {
            name = "nixward";
            features = "cli";
            binName = "nixward";
          };

          # TUI application
          nixward-tui = mkNixwardPackage {
            name = "nixward-tui";
            features = "tui";
            binName = "nixward-tui";
          };

          # Background daemon
          nixward-daemon = mkNixwardPackage {
            name = "nixward-daemon";
            features = "daemon";
            binName = "nixward-daemon";
          };

          default = self.packages.${system}.nixward;
        };

        apps = {
          nixward = flake-utils.lib.mkApp {
            drv = self.packages.${system}.nixward;
          };
          nixward-tui = flake-utils.lib.mkApp {
            drv = self.packages.${system}.nixward-tui;
          };
          default = self.apps.${system}.nixward;
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
            echo "nixward development shell"
            echo "  cargo build -p nixward --features cli   # CLI"
            echo "  cargo build -p nixward --features tui   # TUI"
            echo "  cargo test -p nixward --features tui    # Tests"
          '';
        };
      }
    ) // {
      # Top-level NixOS modules (outside eachDefaultSystem)
      inherit nixosModules;
    };
}
