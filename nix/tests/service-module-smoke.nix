# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# NixOS test: boot the packaged Symthaea service module and confirm the
# socket-activated daemon receives the expected arguments and environment.
#
# Run (from symthaea/):
#   nix build .#checks.x86_64-linux.service-module-smoke

{ pkgs }:

let
  nixosTest =
    if pkgs ? testers && pkgs.testers ? nixosTest
    then pkgs.testers.nixosTest
    else throw "pkgs.testers.nixosTest is missing (nixpkgs too old?)";

  stubPackage = pkgs.writeShellScriptBin "symthaea" ''
    set -eu
    state_dir="$SYMTHAEA_DATA_DIR/state"
    mkdir -p "$state_dir"
    printf '%s\n' "$@" > "$state_dir/args.txt"
    env | sort > "$state_dir/env.txt"
    touch "$state_dir/started"
    sleep 30
  '';
in
nixosTest {
  name = "service-module-smoke";

  nodes.machine = { pkgs, ... }: {
    imports = [ ../module.nix ];

    networking.hostName = "machine";

    services.symthaea = {
      enable = true;
      package = stubPackage;
    };

    environment.systemPackages = [
      pkgs.coreutils
      pkgs.netcat-openbsd
    ];
  };

  testScript = ''
    start_all()

    machine.wait_for_unit("symthaea.socket")
    machine.wait_until_succeeds("test -S /run/symthaea/symthaea.sock")

    machine.succeed("timeout 2 nc -U /run/symthaea/symthaea.sock || true")
    machine.wait_for_unit("symthaea.service")
    machine.wait_until_succeeds("test -f /var/lib/symthaea/state/started")

    machine.succeed("grep -Fx -- '--socket' /var/lib/symthaea/state/args.txt")
    machine.succeed("grep -Fx -- '/run/symthaea/symthaea.sock' /var/lib/symthaea/state/args.txt")
    machine.succeed("grep -Fx -- '--loop-interval' /var/lib/symthaea/state/args.txt")
    machine.succeed("grep -Fx -- '500' /var/lib/symthaea/state/args.txt")
    machine.succeed("grep -Fx -- '--state-file' /var/lib/symthaea/state/args.txt")
    machine.succeed("grep -Fx -- '/var/lib/symthaea/state/symthaea-state.bin' /var/lib/symthaea/state/args.txt")
    machine.succeed("grep -Fx 'SYMTHAEA_DATA_DIR=/var/lib/symthaea' /var/lib/symthaea/state/env.txt")
    machine.succeed("grep -Fx 'SYMTHAEA_SERVICE_AUDIT_LOG_PATH=/var/lib/symthaea/logs/service-audit.jsonl' /var/lib/symthaea/state/env.txt")
    machine.succeed("systemctl show symthaea.service -p RestrictAddressFamilies | grep -Fx 'RestrictAddressFamilies=AF_UNIX'")
  '';
}
