# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Sovereign Inoculation — Custom NixOS Installer ISO module (hardened)
#
# This module is shared by the ISO builder and NixOS VM tests.

{ modulesPath }:
{ config, lib, pkgs, ... }:
let
  isoFileName = "sovereign-inoculation-${config.system.nixos.label}-x86_64.iso";
in
{
  imports = [
    (modulesPath + "/installer/cd-dvd/installation-cd-minimal.nix")
  ];

  # ISO metadata
  # nixpkgs 25.05+ renamed `isoImage.isoName` -> `image.fileName`.
  image.fileName = isoFileName;
  isoImage.volumeID = "SOVEREIGN";
  isoImage.squashfsCompression = "zstd -Xcompression-level 15";

  # No SSH needed — relay executes commands locally.
  # Firewall: only the relay WebSocket port is open.
  networking.firewall.enable = true;
  networking.firewall.allowedTCPPorts = [ 8094 ];

  # ── Generate random token + password at boot (HIGH-1/HIGH-5: no hardcoded secrets) ──
  systemd.services.sovereign-token = {
    description = "Generate random relay token and root password";
    wantedBy = [ "multi-user.target" ];
    before = [ "sovereign-relay.service" "show-relay-url.service" ];
    serviceConfig = {
      Type = "oneshot";
      RemainAfterExit = true;
      ExecStart = pkgs.writeShellScript "gen-token" ''
        ${pkgs.coreutils}/bin/mkdir -p /run/sovereign
        ${pkgs.coreutils}/bin/head -c 16 /dev/urandom | ${pkgs.coreutils}/bin/od -An -tx1 | ${pkgs.coreutils}/bin/tr -d ' \n' > /run/sovereign/token
        ${pkgs.coreutils}/bin/chmod 600 /run/sovereign/token
        PASS=$(${pkgs.coreutils}/bin/head -c 8 /dev/urandom | ${pkgs.coreutils}/bin/od -An -tx1 | ${pkgs.coreutils}/bin/tr -d ' \n')
        echo "root:$PASS" | ${pkgs.shadow}/bin/chpasswd
        echo "$PASS" > /run/sovereign/root-password
        ${pkgs.coreutils}/bin/chmod 600 /run/sovereign/root-password
      '';
    };
  };

  # ── Sovereign Relay (WebSocket SSH bridge on port 8094) ──
  systemd.services.sovereign-relay = let
    relayBin = ../../target/release/ssh-relay;
    relay = pkgs.runCommand "sovereign-relay-1.2.0" {} ''
      mkdir -p $out/bin
      cp ${relayBin} $out/bin/sovereign-relay
      chmod +x $out/bin/sovereign-relay
    '';
  in {
    description = "NixForHumanity WebSocket Relay (local execution)";
    after = [ "network.target" "sovereign-token.service" ];
    wants = [ "network.target" ];
    requires = [ "sovereign-token.service" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      ExecStart = pkgs.writeShellScript "start-relay" ''
        TOKEN=$(${pkgs.coreutils}/bin/cat /run/sovereign/token)
        ${relay}/bin/sovereign-relay --port 8094 --bind 0.0.0.0 --token "$TOKEN" --tls
      '';
      Restart = "always";
      RestartSec = 3;
    };
  };

  # Root password set at boot by sovereign-token service (random each boot).
  # A fallback initialPassword is needed for the NixOS build, but the
  # sovereign-token service overwrites it at boot with a random password.
  users.users.root.initialPassword = "changeme-at-boot";

  # ── mDNS / Avahi (auto-discovery) ──
  services.avahi = {
    enable = true;
    nssmdns4 = true;
    publish = {
      enable = true;
      addresses = true;
      workstation = true;
    };
  };

  networking.hostName = lib.mkDefault "sovereign-inoculation";

  # ZFS support (for ZFS install layout)
  boot.supportedFilesystems = [ "zfs" ];

  # ── Installation tools ──
  environment.systemPackages = with pkgs; [
    # Filesystem tools
    btrfs-progs
    dosfstools
    e2fsprogs
    parted
    util-linux  # for lsblk, sgdisk
    gptfdisk    # sgdisk
    zfs         # ZFS userspace tools

    # Encryption
    cryptsetup
    libfido2

    # Windows partition support
    ntfs3g
    ntfsprogs     # ntfsresize for shrinking Windows partitions

    # RAID
    mdadm

    # Disk health
    smartmontools

    # Network
    networkmanager
    iw
    wirelesstools

    # QR code display on console
    qrencode

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
    ║   NixForHumanity — NixOS Installer                             ║
    ║                                                                ║
    ║   Scan the QR code below with your phone, or open:             ║
    ║     https://install.nixforhumanity.org                        ║
    ║                                                                ║
    ║   Password: see console output (random each boot)              ║
    ║   No other tools needed.                                       ║
    ║                                                                ║
    ╚══════════════════════════════════════════════════════════════════╝

  '';

  services.getty.greetingLine = ''
    NixForHumanity — NixOS Installer
    Network: \4{eth0}  \4{wlan0}
    Relay:   ws://\4{eth0}:8094
  '';

  # Print connection instructions on tty1.
  systemd.services.show-relay-url = {
    description = "Display NixForHumanity relay URL on console";
    after = [ "network-online.target" "sshd.service" "sovereign-relay.service" "sovereign-token.service" ];
    wants = [ "network-online.target" ];
    requires = [ "sovereign-token.service" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "oneshot";
      ExecStart = pkgs.writeShellScript "show-relay-url" ''
        sleep 3
        IP="$(${pkgs.iproute2}/bin/ip -4 route get 1 2>/dev/null | ${pkgs.gawk}/bin/awk '{print $7}' | ${pkgs.coreutils}/bin/head -1)"
        TOKEN="$(${pkgs.coreutils}/bin/cat /run/sovereign/token 2>/dev/null || echo 'TOKEN_UNAVAILABLE')"
        ROOT_PASS="$(${pkgs.coreutils}/bin/cat /run/sovereign/root-password 2>/dev/null || echo 'PASS_UNAVAILABLE')"
        URL="https://install.nixforhumanity.org/?target=''${IP}&token=''${TOKEN}"

        # Generate a short pairing code from the IP (deterministic, memorable)
        CODE=$(echo "$IP" | ${pkgs.coreutils}/bin/md5sum | ${pkgs.coreutils}/bin/tr -dc 'A-Z0-9' | ${pkgs.coreutils}/bin/head -c 6)

        echo ""
        echo "════════════════════════════════════════════════════"
        echo "  NixForHumanity ready!"
        echo ""
        echo "  Scan this QR code with your phone:"
        echo ""
        ${pkgs.qrencode}/bin/qrencode -t ANSIUTF8 "$URL" 2>/dev/null || echo "  (qrencode not available)"
        echo ""
        echo "  Or open: https://install.nixforhumanity.org"
        echo "  Target:  $IP"
        echo "  Token:   $TOKEN"
        echo "  Code:    $CODE"
        echo "  Root pw: $ROOT_PASS"
        echo "  mDNS:    sovereign-inoculation.local"
        echo "════════════════════════════════════════════════════"
        echo ""
      '';
    };
  };

  system.stateVersion = "25.05";
}
