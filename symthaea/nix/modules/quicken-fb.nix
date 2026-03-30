# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# NixOS Module: Symthaea Consciousness Boot Animation
#
# Runs the mycelial colonization animation (symthaea-quicken-fb) on bare-metal
# DRM/KMS framebuffer during early boot, before the display manager starts.
# The animation is seeded by the installation's genesis phrase and dissolves
# automatically when Wayland/X11 takes over.
#
# Usage:
# {
#   services.symthaea-boot = {
#     enable = true;
#     package = symthaea-quicken-fb;
#     genesisPhrase = "your sovereign phrase here";
#   };
# }

{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.symthaea-boot;
in {
  options.services.symthaea-boot = {
    enable = mkEnableOption "Symthaea consciousness boot animation";

    package = mkOption {
      type = types.package;
      description = "The symthaea-quicken-fb package containing the quicken-fb binary.";
    };

    genesisPhrase = mkOption {
      type = types.str;
      default = "consciousness awakens";
      description = ''
        Genesis phrase for deterministic boot animation seeding.
        Each unique phrase produces a distinct mycelial growth pattern
        via BLAKE3 hashing — the machine's visual fingerprint.
      '';
    };

    progressPipe = mkOption {
      type = types.str;
      default = "/run/symthaea/boot-progress";
      description = ''
        Named pipe (FIFO) for receiving installation progress events.
        When a NixOS installation is in progress, the installer writes
        derivation completion events here to trigger network pulses.
      '';
    };

    device = mkOption {
      type = types.str;
      default = "/dev/dri/card0";
      description = "DRM device for bare-metal framebuffer rendering.";
    };
  };

  config = mkIf cfg.enable {
    # Run the boot animation as an early systemd service.
    # Starts after DRM devices are available but before the display manager.
    # StopWhenUnneeded ensures automatic dissolve when Wayland/X11 compositor
    # pulls graphical.target, replacing the framebuffer output.
    systemd.services.symthaea-boot-animation = {
      description = "Symthaea Consciousness Boot Animation";
      documentation = [ "https://github.com/Luminous-Dynamics/symthaea-hlb" ];

      # DRM devices must be ready; filesystems must be mounted for the pipe
      after = [ "systemd-udev-settle.service" "local-fs.target" ];

      # Must finish (or be stopped) before display manager claims the GPU
      before = [ "display-manager.service" "graphical.target" ];

      # Pull into the graphical boot path
      wantedBy = [ "graphical.target" ];

      unitConfig = {
        # Only start if the DRM device actually exists (headless servers skip)
        ConditionPathExists = cfg.device;

        # Auto-stop when graphical.target no longer needs us —
        # i.e., when the display manager / compositor is running
        StopWhenUnneeded = true;

        # Give the animation time to run its fade-to-black exit sequence
        # (CONTRACTION_DURATION + FLASH_HOLD + FADE_DURATION = 5s)
        TimeoutStopSec = 8;
      };

      serviceConfig = {
        Type = "simple";

        ExecStart = concatStringsSep " " [
          "${cfg.package}/bin/quicken-fb"
          "--genesis-phrase '${cfg.genesisPhrase}'"
          "--progress-pipe ${cfg.progressPipe}"
          "--device ${cfg.device}"
        ];

        # DRM access requires video and render groups
        SupplementaryGroups = [ "video" "render" ];

        # Run as root for early-boot DRM access (pre-login)
        User = "root";

        # Clean shutdown via SIGTERM triggers the fade-to-black sequence
        KillSignal = "SIGTERM";

        # Security: restrict what the animation can do
        NoNewPrivileges = true;
        ProtectHome = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictNamespaces = true;
        LockPersonality = true;
        RestrictRealtime = true;
        RestrictSUIDSGID = true;

        # Allow access to /dev/dri/* but nothing else sensitive
        PrivateTmp = true;
      };
    };

    # Create the runtime directory and progress pipe
    systemd.tmpfiles.rules = [
      "d /run/symthaea 0755 root root -"
      "p ${cfg.progressPipe} 0644 root root -"
    ];
  };
}
