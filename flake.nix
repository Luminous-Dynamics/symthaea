{
  description = "Luminous Dynamics Monorepo - Common Development Environment";

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
          name = "luminous-dynamics-dev";

          buildInputs = with pkgs; [
            # Common tools across all projects
            nodejs_20
            rustc
            cargo

            # Database (PostgreSQL already in your environment)
            postgresql_15

            # Common utilities
            git
            curl
            jq

            # Nix tooling
            nixfmt
            nil  # Nix LSP

            # LaTeX (papers — HAI, psych-bench, stewardship)
            (texlive.combine {
              inherit (texlive)
                scheme-medium
                changepage
                marvosym
                cm-super
                ;
            })
          ];

          shellHook = ''
            echo "🌟 Luminous-Dynamics Development Environment"
            echo "📦 Node.js: $(node --version)"
            echo "🦀 Rust: $(rustc --version)"
            echo "❄️  Nix: $(nix --version)"
            echo ""
            echo "Commands:"
            echo "  lum-start  - Start all services"
            echo "  lum-status - Connect to overmind"
            echo "  lum-stop   - Stop all services"
            echo ""
            echo "🌊 We flow with Nix!"

            # Project-specific setup
            export LUMINOUS_DEV=true
            export LUMINOUS_ROOT="$PWD"
            export NODE_ENV=development

            # PostgreSQL setup
            export PGDATA="$PWD/.postgres"
            export PGHOST="localhost"
            export PGUSER="$USER"
            export PGDATABASE="luminous"
          '';
        };
      });
}
