# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Sovereign Inoculation — Custom NixOS Installer ISO module (hardened)
#
# This module is shared by the ISO builder and NixOS VM tests.

{ modulesPath }:
{ config, lib, pkgs, ... }:
{
  imports = [
    (modulesPath + "/installer/cd-dvd/installation-cd-minimal.nix")
  ];

  # ISO metadata
  isoImage.isoName = "sovereign-inoculation-${config.system.nixos.label}-x86_64.iso";
  isoImage.volumeID = "SOVEREIGN";
  isoImage.squashfsCompression = "zstd -Xcompression-level 15";

  # ── SSH ──
  services.openssh = {
    enable = true;
    settings = {
      PermitRootLogin = "yes";
      PasswordAuthentication = true;
    };
  };

  # Lock down exposed ports on the live environment (SSH only).
  networking.firewall.enable = true;
  networking.firewall.allowedTCPPorts = [ 22 ];

  # One-time SSH password on boot (printed to console).
  #
  # This replaces the previous static root password and prevents opportunistic
  # takeover on shared networks during install.
  systemd.services.sovereign-one-time-root-password = {
    description = "Generate one-time root password for SSH (Sovereign Inoculation)";
    wantedBy = [ "multi-user.target" ];
    before = [ "sshd.service" ];
    requiredBy = [ "sshd.service" ];
    serviceConfig = {
      Type = "oneshot";
      RemainAfterExit = true;
      ExecStart = pkgs.writeShellScript "sovereign-one-time-root-password" ''
        set -euo pipefail
        DIR="/run/sovereign-inoculation"
        install -d -m 0700 "$DIR"

        # 20 chars from a friendly alphabet (no ambiguous 0/O/1/l).
        PASS="$(${pkgs.coreutils}/bin/tr -dc 'A-HJ-NP-Za-km-z2-9' </dev/urandom | ${pkgs.coreutils}/bin/head -c 20 || true)"
        ${pkgs.coreutils}/bin/echo -n "$PASS" > "$DIR/root-password"
        ${pkgs.coreutils}/bin/chmod 0600 "$DIR/root-password"

        ${pkgs.coreutils}/bin/echo "root:$PASS" | ${pkgs.shadow}/bin/chpasswd
      '';
    };
  };

  # Set hostname (no mDNS broadcast by default).
  networking.hostName = "sovereign-inoculation";

  # ── Installation tools ──
  environment.systemPackages = with pkgs; [
    # Filesystem tools
    btrfs-progs
    dosfstools
    e2fsprogs
    parted
    util-linux  # for lsblk, sgdisk
    gptfdisk    # sgdisk

    # Encryption
    cryptsetup

    # Windows partition support
    ntfs3g

    # Network
    networkmanager
    iw
    wirelesstools

    # System tools
    htop
    vim
    git
    curl
    jq
  ];

  # Enable NetworkManager for WiFi
  networking.networkmanager.enable = true;
  networking.wireless.enable = lib.mkForce false;

  # ── Console greeting ──
  services.getty.helpLine = lib.mkForce ''

    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║   SOVEREIGN INOCULATION — NixOS Installer                     ║
    ║                                                                ║
    ║   SSH is enabled with a ONE-TIME root password.               ║
    ║   The password is printed below during boot.                  ║
    ║                                                                ║
    ║   Recommended flow (secure):                                  ║
    ║     - Run ssh-relay locally on your operator machine           ║
    ║     - SSH into this installer environment using the password  ║
    ║                                                                ║
    ║   SECURITY NOTE: avoid untrusted Wi-Fi during install.         ║
    ║                                                                ║
    ╚══════════════════════════════════════════════════════════════════╝

  '';

  services.getty.greetingLine = ''
    Sovereign Inoculation — NixOS Installer
    Network: \4{eth0}  \4{wlan0}
  '';

  # Print connection instructions (including the one-time password) on tty1.
  systemd.services.show-sovereign-ssh-info = {
    description = "Display Sovereign SSH connection info on console";
    after = [
      "network-online.target"
      "sshd.service"
      "sovereign-one-time-root-password.service"
      "getty@tty1.service"
    ];
    wants = [ "network-online.target" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "oneshot";
      StandardOutput = "tty";
      StandardError = "tty";
      TTYPath = "/dev/tty1";
      TTYReset = true;
      TTYVHangup = true;
      ExecStart = pkgs.writeShellScript "show-sovereign-ssh-info" ''
        set -euo pipefail
        sleep 2  # wait for network
        IP="$(${pkgs.iproute2}/bin/ip -4 route get 1 2>/dev/null | ${pkgs.gawk}/bin/awk '{print $7}' | ${pkgs.coreutils}/bin/head -1)"
        PASS="$(${pkgs.coreutils}/bin/cat /run/sovereign-inoculation/root-password 2>/dev/null || true)"
        echo ""
        echo "════════════════════════════════════════════════════"
        echo "  Sovereign Inoculation ready (locked down)"
        echo ""
        echo "  SSH:   ssh root@$IP"
        echo "  PASS:  $PASS"
        echo ""
        echo "  NOTE: Password regenerates on every reboot."
        echo "════════════════════════════════════════════════════"
        echo ""
      '';
    };
  };

  system.stateVersion = "25.05";
}
