# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# NixOS test: validate the hardened installer ISO security posture.
#
# Run (from symthaea/):
#   nix build .#checks.x86_64-linux.installer-iso-security

{ pkgs }:

let
  modulesPath = "${pkgs.path}/nixos/modules";
  installerIsoModule = import ../modules/installer-iso-module.nix { inherit modulesPath; };
in
pkgs.nixosTest {
  name = "installer-iso-security";

  nodes = {
    installer = { ... }: {
      imports = [ installerIsoModule ];
    };

    attacker = { pkgs, ... }: {
      environment.systemPackages = [ pkgs.netcat-openbsd pkgs.iproute2 pkgs.ripgrep ];
    };
  };

  testScript = ''
    installer.wait_for_unit("network-online.target")
    installer.wait_for_unit("sshd.service")
    installer.wait_for_unit("sovereign-one-time-root-password.service")

    # One-time password material exists and is not world-readable.
    installer.succeed("test -f /run/sovereign-inoculation/root-password")
    installer.succeed("stat -c '%a %U %G' /run/sovereign-inoculation/root-password | rg -q '^600 root root$'")

    # Avahi should not be active (no mDNS broadcast).
    installer.fail("systemctl is-active --quiet avahi-daemon.service")

    # Firewall should only allow SSH from the network.
    installer_ip = installer.succeed("ip -4 -o addr show dev eth1 | awk '{print $4}' | cut -d/ -f1").strip()
    attacker.wait_for_unit("network-online.target")
    attacker.wait_until_succeeds(f"nc -zvw2 {installer_ip} 22")

    for port in [8091, 8090, 8094, 8080, 3000]:
      attacker.fail(f"nc -zvw2 {installer_ip} {port}")

    # Password should rotate on reboot.
    pw1 = installer.succeed("cat /run/sovereign-inoculation/root-password").strip()
    installer.reboot()
    installer.wait_for_unit("sshd.service")
    installer.wait_for_unit("sovereign-one-time-root-password.service")
    pw2 = installer.succeed("cat /run/sovereign-inoculation/root-password").strip()
    assert pw1 != pw2, "expected one-time root password to change across reboot"
  '';
}

