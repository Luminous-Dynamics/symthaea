# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Custom NixOS minimal ISO with SSH pre-enabled for testing
# Build: nix-build '<nixpkgs/nixos>' -A config.system.build.isoImage -I nixos-config=./test-iso.nix
# Or: nix build .#test-iso (if added to flake)
{ config, pkgs, lib, ... }:
{
  imports = [
    <nixpkgs/nixos/modules/installer/cd-dvd/installation-cd-minimal.nix>
  ];

  # SSH enabled with empty root password (for testing only)
  services.openssh = {
    enable = true;
    settings = {
      PermitRootLogin = "yes";
      PermitEmptyPasswords = "yes";
    };
  };

  # Root with password "test" for SSH testing
  users.users.root.hashedPasswordFile = null;
  users.users.root.password = "test";
  # Also inject host SSH pubkey for key-based auth
  users.users.root.openssh.authorizedKeys.keyFiles = [
    /home/tstoltz/.ssh/id_ed25519.pub
  ];

  # Auto-login on serial console
  systemd.services."serial-getty@ttyS0" = {
    overrideStrategy = "asDropin";
    serviceConfig.ExecStart = [
      ""
      "${pkgs.util-linux}/sbin/agetty --autologin root --noclear ttyS0 115200 vt100"
    ];
  };

  # Enable serial console
  boot.kernelParams = [ "console=ttyS0,115200" "console=tty0" ];

  # Enable flakes (required for disko and nixos-install --flake)
  nix.settings.experimental-features = [ "nix-command" "flakes" ];

  # Useful packages for testing
  environment.systemPackages = with pkgs; [ vim curl wget util-linux parted btrfs-progs gptfdisk dosfstools e2fsprogs git ];
}
