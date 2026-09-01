{ pkgs, src ? ../.. }:

pkgs.rustPlatform.buildRustPackage {
  pname = "spore-boot-tools";
  version = "0.3.0";
  inherit src;

  cargoLock = {
    lockFile = src + "/Cargo.lock";
    allowBuiltinFetchGit = true;
  };

  # Build only the two boot-path binaries and their small dependency closures;
  # do not pull the full Symthaea application into the host boot package.
  cargoBuildFlags = [
    "-p"
    "symthaea-quicken-fb"
    "--bin"
    "quicken-fb"
    "-p"
    "symthaea-boot-state"
    "--bin"
    "spore-boot-state"
  ];

  # Package tests are exercised by repository CI. Keeping the install
  # derivation build-only avoids running unrelated workspace tests here.
  doCheck = false;

  installPhase = ''
    runHook preInstall
    mkdir -p "$out/bin"

    quicken="$(${pkgs.findutils}/bin/find target -type f -name quicken-fb -perm -0100 -print -quit)"
    state_tool="$(${pkgs.findutils}/bin/find target -type f -name spore-boot-state -perm -0100 -print -quit)"

    test -n "$quicken"
    test -n "$state_tool"
    install -Dm755 "$quicken" "$out/bin/quicken-fb"
    install -Dm755 "$state_tool" "$out/bin/spore-boot-state"
    runHook postInstall
  '';

  meta = with pkgs.lib; {
    description = "Fail-open state-aware Spore boot renderer and lifecycle state tool";
    license = licenses.agpl3Plus;
    platforms = platforms.linux;
    mainProgram = "quicken-fb";
  };
}
