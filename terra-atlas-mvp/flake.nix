{
  description = "Terra Atlas MVP - Development environment with Playwright support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Create FHS environment for Playwright
        playwrightEnv = pkgs.buildFHSEnv {
          name = "playwright-fhs";
          targetPkgs = pkgs: with pkgs; [
            # Node.js and npm
            nodejs_20

            # Playwright browser dependencies
            alsa-lib
            at-spi2-atk
            at-spi2-core
            atk
            cairo
            cups
            dbus
            expat
            fontconfig
            freetype
            gdk-pixbuf
            glib
            gtk3
            libdrm
            libxkbcommon
            mesa
            nspr
            nss
            pango
            xorg.libX11
            xorg.libXcomposite
            xorg.libXdamage
            xorg.libXext
            xorg.libXfixes
            xorg.libXrandr
            xorg.libxcb
            xorg.libxshmfence

            # Additional runtime dependencies
            libuuid
            libGL
          ];

          multiPkgs = pkgs: with pkgs; [
            # 32-bit support if needed
          ];

          runScript = "bash";

          profile = ''
            export PLAYWRIGHT_BROWSERS_PATH=$HOME/.cache/ms-playwright
            export PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=true
          '';
        };

      in
      {
        devShells = {
          default = pkgs.mkShell {
            buildInputs = with pkgs; [
              nodejs_20
              nodePackages.npm
            ];

            shellHook = ''
              echo "🌍 Terra Atlas Development Environment"
              echo "📦 Node.js: $(node --version)"
              echo ""
              echo "Commands:"
              echo "  npm run dev         - Start dev server"
              echo "  npm run build       - Build for production"
              echo "  nix develop .#playwright - Enter Playwright FHS environment"
              echo ""
            '';
          };

          # Playwright environment with FHS
          playwright = pkgs.mkShell {
            buildInputs = [ playwrightEnv ];

            shellHook = ''
              echo "🎭 Playwright FHS Environment"
              echo "📦 Node.js available via FHS wrapper"
              echo ""
              echo "Usage:"
              echo "  playwright-fhs"
              echo "  # Then inside FHS environment:"
              echo "  node screenshot-firefox.js"
              echo ""
            '';
          };

          python = pkgs.mkShell {
            buildInputs = with pkgs; [
              python311
              poetry
            ];

            shellHook = ''
              echo "🐍 Terra Atlas Poetry Environment"
              echo "Use 'poetry install' to set up the virtualenv."
              export POETRY_VIRTUALENVS_IN_PROJECT=true
            '';
          };
        };

        packages = {
          # Package for running screenshots
          screenshot = pkgs.writeShellScriptBin "take-screenshot" ''
            ${playwrightEnv}/bin/playwright-fhs -c "
              cd /srv/luminous-dynamics/terra-atlas-mvp
              node screenshot-firefox.js
            "
          '';

          telemetry-report = pkgs.writeShellScriptBin "telemetry-report" ''
            export NODE_PATH=${pkgs.nodejs_20}/lib/node_modules
            cd /srv/luminous-dynamics/terra-atlas-mvp
            ${pkgs.nodejs_20}/bin/node scripts/analyze-telemetry.mjs "$@"
          '';
        };
      });
}
