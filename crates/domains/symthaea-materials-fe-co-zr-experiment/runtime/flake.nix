{
  description = "Fe-Co-Zr retrospective experiment runtime tools";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/d233902339c02a9c334e7e593de68855ad26c4cb";

  outputs = { nixpkgs, ... }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; }));
    in {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            curl
            cacert
            gzip
            jq
            git
            coreutils
            gnugrep
            gnused
            gawk
            percona-server_8_4
          ];

          shellHook = ''
            export LANG=C.UTF-8
            export LC_ALL=C.UTF-8
            export TZ=UTC
            export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            echo "Fe-Co-Zr runtime shell: exact nixpkgs d233902339c02a9c334e7e593de68855ad26c4cb"
            echo "Percona 8.4 here is the integration-fixture MySQL-compatible engine; qualification must record the exact chosen engine."
            echo "qmpy 1.4 runtime is intentionally NOT synthesized by this shell; qualify it separately before extraction."
          '';
        };
      });
    };
}
