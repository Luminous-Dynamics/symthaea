{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    foundry
    solc
  ];
  shellHook = ''
    echo "Foundry dev shell ready"
  '';
}
