{
  description = "Mycelix Marketplace - reproducible dev shell and builds";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        nodejs = pkgs.nodejs_20;
        frontendSrc = ./frontend;
        npmDepsHash = "sha256-7FNbB1/niXZDSAyB6Q+K117tK6obQbYH9csNY4oAtcw=";
      in {
        packages = {
          frontend = pkgs.buildNpmPackage {
            pname = "mycelix-marketplace-frontend";
            version = "1.0.0";
            src = frontendSrc;
            npmDepsHash = npmDepsHash;
            npmBuildScript = "build";
            # expose the static build output
            installPhase = ''
              runHook preInstall
              mkdir -p $out
              if [ -d .vercel/output ]; then
                cp -r .vercel/output/. "$out/"
              elif [ -d build ]; then
                cp -r build/. "$out/"
              elif [ -d .svelte-kit ]; then
                cp -r .svelte-kit "$out/.svelte-kit"
              else
                echo "warning: no known build artifacts (.vercel/output, build/, .svelte-kit/) produced" >&2
              fi
              runHook postInstall
            '';
          };
          default = self.packages.${system}.frontend;
        };

        checks = {
          frontend-check = pkgs.buildNpmPackage {
            pname = "mycelix-marketplace-frontend-check";
            version = "1.0.0";
            src = frontendSrc;
            npmDepsHash = npmDepsHash;
            npmBuildScript = "check";
            installPhase = ''
              runHook preInstall
              mkdir -p $out
              touch $out/done
              runHook postInstall
            '';
          };

          frontend-lint = pkgs.buildNpmPackage {
            pname = "mycelix-marketplace-frontend-lint";
            version = "1.0.0";
            src = frontendSrc;
            npmDepsHash = npmDepsHash;
            npmBuildScript = "lint";
            installPhase = ''
              runHook preInstall
              mkdir -p $out
              touch $out/done
              runHook postInstall
            '';
          };

          frontend-test = pkgs.buildNpmPackage {
            pname = "mycelix-marketplace-frontend-test";
            version = "1.0.0";
            src = frontendSrc;
            npmDepsHash = npmDepsHash;
            npmBuildScript = "test";
            installPhase = ''
              runHook preInstall
              mkdir -p $out
              touch $out/done
              runHook postInstall
            '';
          };
        };

        apps = {
          check = (flake-utils.lib.mkApp {
            drv = pkgs.writeShellApplication {
              name = "mycelix-check";
              runtimeInputs = [ nodejs pkgs.git ];
              text = ''
                repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
                cd "$repo_root/frontend"
                if [ ! -d node_modules ]; then
                  echo "node_modules missing. Run 'nix develop' followed by 'cd frontend && npm install' first." >&2
                  exit 1
                fi
                npm run check
              '';
            };
          }) // {
            meta.description = "Run svelte-check via nix (requires node_modules provisioned in frontend/)";
          };

          dev = (flake-utils.lib.mkApp {
            drv = pkgs.writeShellApplication {
              name = "mycelix-dev";
              runtimeInputs = [ nodejs pkgs.git ];
              text = ''
                repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
                cd "$repo_root/frontend"
                if [ ! -d node_modules ]; then
                  echo "node_modules missing. Run 'nix develop' followed by 'cd frontend && npm install' first." >&2
                  exit 1
                fi
                npm run dev -- --host
              '';
            };
          }) // {
            meta.description = "Launch `npm run dev` (Vite) via nix; assumes dependencies already installed";
          };

          lint = (flake-utils.lib.mkApp {
            drv = pkgs.writeShellApplication {
              name = "mycelix-lint";
              runtimeInputs = [ nodejs pkgs.git ];
              text = ''
                repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
                cd "$repo_root/frontend"
                if [ ! -d node_modules ]; then
                  echo "node_modules missing. Run 'nix develop' followed by 'cd frontend && npm install' first." >&2
                  exit 1
                fi
                npm run lint
              '';
            };
          }) // {
            meta.description = "Run eslint (with Svelte sync) via nix; requires node_modules";
          };

          test = (flake-utils.lib.mkApp {
            drv = pkgs.writeShellApplication {
              name = "mycelix-test";
              runtimeInputs = [ nodejs pkgs.git ];
              text = ''
                repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
                cd "$repo_root/frontend"
                if [ ! -d node_modules ]; then
                  echo "node_modules missing. Run 'nix develop' followed by 'cd frontend && npm install' first." >&2
                  exit 1
                fi
                npm run test
              '';
            };
          }) // {
            meta.description = "Execute vitest suite via nix; requires node_modules";
          };
        };

        devShells.default = pkgs.mkShell {
          name = "mycelix-marketplace";
          packages = [
            nodejs
            pkgs.nodePackages.pnpm
            pkgs.nodePackages.typescript
            pkgs.nodePackages."svelte-language-server"
          ];
          shellHook = ''
            export NPM_CONFIG_PREFIX="$HOME/.npm-global"
            export PATH="$NPM_CONFIG_PREFIX/bin:$PATH"
            echo "🔧 Entered Mycelix dev shell (frontend/). Run 'cd frontend && npm install' on first use."
          '';
        };

        formatter = pkgs.nixpkgs-fmt;
      });
}
