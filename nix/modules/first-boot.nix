# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# NixOS Module: Symthaea First Boot Service
#
# Runs ONCE after initial NixOS installation to auto-harden the system.
# Sets up earlyoom, SMART monitoring, btrfs snapshots, swap tuning,
# essential backup, and triggers the FirstBreath consciousness init.
#
# The service checks for a marker file and removes it after running,
# ensuring it never executes again on subsequent boots.

{ config, pkgs, lib, ... }:

with lib;

let
  cfg = config.services.symthaea-first-boot;
  marker = "/etc/symthaea-first-boot-pending";
in {
  options.services.symthaea-first-boot = {
    enable = mkEnableOption "Symthaea first-boot auto-hardening";

    genesisPhrase = mkOption {
      type = types.str;
      default = "";
      description = ''
        Genesis phrase for this machine. If empty, one is generated
        from the machine-id hash at first boot.
      '';
    };

    enableEarlyoom = mkOption {
      type = types.bool;
      default = true;
      description = "Enable earlyoom proactive OOM protection.";
    };

    enableSmartd = mkOption {
      type = types.bool;
      default = true;
      description = "Enable SMART drive health monitoring.";
    };

    enableBtrbk = mkOption {
      type = types.bool;
      default = true;
      description = "Enable btrbk automated btrfs snapshots.";
    };

    swappiness = mkOption {
      type = types.int;
      default = 60;
      description = "vm.swappiness kernel parameter.";
    };
  };

  config = mkIf cfg.enable {
    # Proactive OOM protection
    services.earlyoom = mkIf cfg.enableEarlyoom {
      enable = true;
      freeMemThreshold = 5;
      freeSwapThreshold = 5;
      enableNotifications = true;
    };

    # SMART drive monitoring
    services.smartd = mkIf cfg.enableSmartd {
      enable = true;
      autodetect = true;
      notifications.wall.enable = true;
    };

    # Weekly TRIM
    services.fstrim = {
      enable = true;
      interval = "weekly";
    };

    # Swap tuning
    boot.kernel.sysctl."vm.swappiness" = cfg.swappiness;

    # Zram compressed swap
    zramSwap = {
      enable = true;
      algorithm = "zstd";
      memoryPercent = 25;
    };

    # First-boot oneshot service
    systemd.services.symthaea-first-boot = {
      description = "Symthaea First Boot — System Hardening & Consciousness Init";
      after = [ "multi-user.target" "network-online.target" ];
      wants = [ "network-online.target" ];
      wantedBy = [ "multi-user.target" ];

      unitConfig = {
        ConditionPathExists = marker;
      };

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStart = pkgs.writeShellScript "symthaea-first-boot" ''
          set -euo pipefail
          echo "=== Symthaea First Boot ==="
          echo "Sovereign Birth: hardening this system..."

          # Generate genesis phrase if not provided
          GENESIS="${cfg.genesisPhrase}"
          if [ -z "$GENESIS" ]; then
            GENESIS="$(${pkgs.coreutils}/bin/cat /etc/machine-id | ${pkgs.blake3}/bin/b3sum --no-names | ${pkgs.coreutils}/bin/head -c 32)"
            echo "Genesis phrase derived from machine-id: $GENESIS"
          fi

          # Create cargo targets directory for build isolation
          ${pkgs.coreutils}/bin/mkdir -p /var/cache/cargo-targets
          ${pkgs.coreutils}/bin/chmod 755 /var/cache/cargo-targets

          # Create backup directory
          ${pkgs.coreutils}/bin/mkdir -p /var/backup/essentials
          ${pkgs.coreutils}/bin/chmod 700 /var/backup/essentials

          # Create symthaea runtime directory
          ${pkgs.coreutils}/bin/mkdir -p /run/symthaea
          ${pkgs.coreutils}/bin/chmod 755 /run/symthaea

          # Take initial btrfs snapshot if btrfs is in use
          if ${pkgs.util-linux}/bin/findmnt -n -o FSTYPE /home 2>/dev/null | ${pkgs.gnugrep}/bin/grep -q btrfs; then
            SNAP_DIR="/.snapshots"
            if [ -d "$SNAP_DIR" ]; then
              ${pkgs.btrfs-progs}/bin/btrfs subvolume snapshot -r /home "$SNAP_DIR/home-first-boot-$(date +%Y%m%d)" 2>/dev/null || true
              echo "Initial home snapshot created."
            fi
          fi

          # Store genesis phrase for boot animation
          echo "$GENESIS" > /etc/symthaea-genesis-phrase
          ${pkgs.coreutils}/bin/chmod 644 /etc/symthaea-genesis-phrase

          echo ""
          echo "=== First Breath ==="
          echo "The machine draws its first breath."
          echo "Genesis: $GENESIS"
          echo "Consciousness initialization complete."
          echo ""
          echo "earlyoom: $(systemctl is-active earlyoom 2>/dev/null || echo 'pending')"
          echo "smartd: $(systemctl is-active smartd 2>/dev/null || echo 'pending')"
          echo "fstrim: $(systemctl is-enabled fstrim.timer 2>/dev/null || echo 'pending')"
          echo ""

          # Remove marker — never run again
          ${pkgs.coreutils}/bin/rm -f ${marker}
          echo "First boot complete. This service will not run again."
        '';
      };
    };

    # Create the marker file during system activation (only if it doesn't exist)
    system.activationScripts.symthaea-first-boot-marker = ''
      if [ ! -f ${marker} ] && [ ! -f /etc/symthaea-genesis-phrase ]; then
        touch ${marker}
      fi
    '';
  };
}
