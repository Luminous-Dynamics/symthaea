{
  description = "Mycelix Sovereign — Secure Sovereign Operations Suite";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Suite-level tooling — component builds happen in component flakes
            nixfmt-rfc-style
            shellcheck
          ];

          shellHook = ''
            echo "Mycelix Sovereign suite devShell"
            echo ""
            echo "This is the meta-repo. Component builds happen in their own flakes:"
            echo "  - xenia-wire, xenia-peer:    /srv/luminous-dynamics/xenia-*/"
            echo "  - mycelix-pulse:             /srv/luminous-dynamics/mycelix-pulse/"
            echo "  - mycelix-identity:          /srv/luminous-dynamics/mycelix-identity/"
            echo "  - Athena L1 (Symthaea core): /srv/luminous-dynamics/symthaea/"
            echo ""
            echo "Primary deliverable of this flake: nixosModules.default"
          '';
        };

        formatter = pkgs.nixfmt-rfc-style;

        checks = {
          # Placeholder — component-level checks happen upstream
          meta-smoke = pkgs.runCommand "mycelix-sovereign-meta-smoke" { } ''
            echo "meta-repo smoke: OK" > $out
          '';
        };
      }
    )
    // {
      # Primary deployment artifact: NixOS module bundling the four suite components.
      nixosModules.default = import ./nixos-modules;
      nixosModules.mycelix-sovereign = self.nixosModules.default;
    };
}
