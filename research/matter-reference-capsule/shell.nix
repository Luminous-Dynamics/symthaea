# Matter Reference Capsule execution shell.
#
# This shell intentionally reuses the root repository's exact nixpkgs lock
# instead of introducing a second moving package source. The Python/AiiDA graph
# is frozen separately by uv.lock; bootstrap.sh refuses execution without it.
#
# Some locked PyPI wheels (notably pyzmq) contain native C++ extensions. On a
# Nix-built Python interpreter those wheels must not fall back to an ambient
# host libstdc++. Bind their C++ runtime explicitly from the same locked nixpkgs
# universe and expose that store identity for qualification.
let
  lock = builtins.fromJSON (builtins.readFile ../../flake.lock);
  nixpkgsNodeName = lock.nodes.root.inputs.nixpkgs;
  nixpkgsLocked = lock.nodes.${nixpkgsNodeName}.locked;
  nixpkgsSource = builtins.fetchTree {
    inherit (nixpkgsLocked) type owner repo rev narHash;
  };
  pkgs = import nixpkgsSource { system = builtins.currentSystem; };
  cppRuntime = pkgs.stdenv.cc.cc.lib;
  cppLibraryPath = pkgs.lib.makeLibraryPath [ cppRuntime ];
in
pkgs.mkShell {
  packages = [
    pkgs.python313
    pkgs.uv
    pkgs.quantum-espresso
    pkgs.git
    pkgs.cacert
    cppRuntime
  ];

  shellHook = ''
    export SYMTHAEA_MATTER_CAPSULE_SHELL=1
    export SYMTHAEA_QE_STORE="${pkgs.quantum-espresso}"
    export SYMTHAEA_PYTHON="${pkgs.python313}/bin/python3.13"
    export SYMTHAEA_CPP_RUNTIME_STORE="${cppRuntime}"

    # Deliberately replace, rather than extend, an ambient loader path. Native
    # Python wheels may use only the C++ runtime declared by this capsule shell.
    export LD_LIBRARY_PATH="${cppLibraryPath}"

    echo ""
    echo "Symthaea Matter Reference Capsule shell"
    echo "  nixpkgs node: ${nixpkgsNodeName}"
    echo "  nixpkgs rev:  ${nixpkgsLocked.rev}"
    echo "  QE store:     $SYMTHAEA_QE_STORE"
    echo "  Python:       $SYMTHAEA_PYTHON"
    echo "  C++ runtime:  $SYMTHAEA_CPP_RUNTIME_STORE"
    echo ""
    echo "A committed uv.lock is required before bootstrap/execution."
    echo ""
  '';
}
