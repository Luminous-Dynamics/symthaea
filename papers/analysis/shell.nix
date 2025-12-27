{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python311
    python311Packages.numpy
    python311Packages.scipy
    python311Packages.matplotlib
  ];

  shellHook = ''
    echo "Python environment for Consciousness Framework v3.0"
    echo "Run: python consciousness_framework_v3.py"
    echo "Test: python test_consciousness_framework.py"
  '';
}
