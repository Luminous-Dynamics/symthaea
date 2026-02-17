{
  description = "Mycelix Music - Development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Node.js and package managers
            nodejs_20
            nodePackages.npm

            # Development tools
            git
            jq
            curl

            # TypeScript development
            nodePackages.typescript
            nodePackages.typescript-language-server
          ];

          shellHook = ''
            echo "🎵 Mycelix Music Development Environment"
            echo "======================================="
            echo ""
            echo "Available tools:"
            echo "  - Node.js: $(node --version)"
            echo "  - npm: $(npm --version)"
            echo ""
            echo "Quick start (UI Demo):"
            echo "  1. cd apps/web && npm run dev  # Start frontend"
            echo ""
            echo "Note: Smart contract deployment requires Foundry"
            echo "      (not included in this environment)"
            echo ""
          '';
        };
      }
    );
}
