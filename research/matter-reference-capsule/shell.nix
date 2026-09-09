# Matter Reference Capsule execution shell.
#
# This shell intentionally reuses the root repository's exact nixpkgs lock
# instead of introducing a second moving package source. The Python/AiiDA graph
# is frozen separately by uv.lock; bootstrap.sh refuses execution without it.
#
# Locked PyPI wheels can contain native extensions whose ELF dependencies are
# not Nix-patched. Supply the minimum runtime libraries discovered by the
# qualification lane from the same locked nixpkgs universe. The verifier audits
# the complete frozen site-packages ELF closure and fails on unresolved or
# undeclared host dependencies.
let
  lock = builtins.fromJSON (builtins.readFile ../../flake.lock);
  nixpkgsNodeName = lock.nodes.root.inputs.nixpkgs;
  nixpkgsLocked = lock.nodes.${nixpkgsNodeName}.locked;
  nixpkgsSource = builtins.fetchTree {
    inherit (nixpkgsLocked) type owner repo rev narHash;
  };
  pkgs = import nixpkgsSource { system = builtins.currentSystem; };
  cppRuntime = pkgs.stdenv.cc.cc.lib;
  libgccRuntime = pkgs.stdenv.cc.cc.libgcc;
  zlibRuntime = pkgs.zlib;
  glibcRuntime = pkgs.glibc;
  nativeLoaderPath = pkgs.lib.makeLibraryPath [ cppRuntime libgccRuntime zlibRuntime ];
  nativeRuntimeStores = "${cppRuntime}:${libgccRuntime}:${zlibRuntime}:${glibcRuntime}";
in
pkgs.mkShell {
  packages = [
    pkgs.python313
    pkgs.uv
    pkgs.quantum-espresso
    pkgs.git
    pkgs.cacert
    cppRuntime
    libgccRuntime
    zlibRuntime
  ];

  shellHook = ''
    export SYMTHAEA_MATTER_CAPSULE_SHELL=1
    export SYMTHAEA_QE_STORE="${pkgs.quantum-espresso}"
    export SYMTHAEA_PYTHON="${pkgs.python313}/bin/python3.13"
    export SYMTHAEA_CPP_RUNTIME_STORE="${cppRuntime}"
    export SYMTHAEA_LIBGCC_RUNTIME_STORE="${libgccRuntime}"
    export SYMTHAEA_ZLIB_RUNTIME_STORE="${zlibRuntime}"
    export SYMTHAEA_GLIBC_RUNTIME_STORE="${glibcRuntime}"
    export SYMTHAEA_NATIVE_RUNTIME_STORES="${nativeRuntimeStores}"

    # Deliberately replace, rather than extend, any ambient loader path. Native
    # wheels may use only the explicitly declared support libraries plus their
    # own lock-bound vendored libraries and the Nix interpreter's glibc closure.
    export LD_LIBRARY_PATH="${nativeLoaderPath}"

    echo ""
    echo "Symthaea Matter Reference Capsule shell"
    echo "  nixpkgs node:   ${nixpkgsNodeName}"
    echo "  nixpkgs rev:    ${nixpkgsLocked.rev}"
    echo "  QE store:       $SYMTHAEA_QE_STORE"
    echo "  Python:         $SYMTHAEA_PYTHON"
    echo "  C++ runtime:    $SYMTHAEA_CPP_RUNTIME_STORE"
    echo "  libgcc runtime: $SYMTHAEA_LIBGCC_RUNTIME_STORE"
    echo "  zlib runtime:   $SYMTHAEA_ZLIB_RUNTIME_STORE"
    echo "  glibc runtime:  $SYMTHAEA_GLIBC_RUNTIME_STORE"
    echo ""
    echo "A committed uv.lock is required before bootstrap/execution."
    echo ""
  '';
}
