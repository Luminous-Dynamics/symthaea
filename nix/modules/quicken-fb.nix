# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# NixOS Module: Symthaea / Spore Boot Animation
#
# Runs the mycelial colonization animation (symthaea-quicken-fb) on bare-metal
# DRM/KMS framebuffer during early graphical boot. Optional typed telemetry is
# presentation-only: loss or corruption of that channel must never block boot.

{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.symthaea-boot;
  telemetryArgs = optionals cfg.telemetry.enable [
    "--boot-events-socket ${escapeShellArg cfg.telemetry.eventSocket}"
    "--boot-state-path ${escapeShellArg cfg.telemetry.statePath}"
  ];
in {
  options.services.symthaea-boot = {
    enable = mkEnableOption "Symthaea/Spore boot animation";

    package = mkOption {
      type = types.package;
      description = "The symthaea-quicken-fb package containing the quicken-fb binary.";
    };

    genesisPhrase = mkOption {
      type = types.str;
      default = "consciousness awakens";
      description = ''
        Genesis phrase for deterministic boot animation seeding. Each unique
        phrase produces a distinct mycelial growth pattern via BLAKE3 hashing.
      '';
    };

    progressPipe = mkOption {
      type = types.str;
      default = "/run/symthaea/boot-progress";
      description = ''
        Named pipe (FIFO) for receiving installation progress events. This path
        remains independent from typed system-boot telemetry.
      '';
    };

    telemetry = {
      enable = mkOption {
        type = types.bool;
        default = false;
        description = "Consume normalized, lineage-bound boot telemetry in quicken-fb.";
      };

      eventSocket = mkOption {
        type = types.str;
        default = "/run/symthaea/boot-events.sock";
        description = "Root-owned/group-writable Unix datagram endpoint bound by quicken-fb.";
      };

      statePath = mkOption {
        type = types.str;
        default = "/run/symthaea-boot/state-v1.json";
        description = "Read-only-from-renderer authoritative snapshot side channel.";
      };
    };

    device = mkOption {
      type = types.str;
      default = "/dev/dri/card0";
      description = "DRM device for bare-metal framebuffer rendering.";
    };
  };

  config = mkIf cfg.enable (mkMerge [
    {
      assertions = [
        {
          assertion = !cfg.telemetry.enable || hasPrefix "/run/symthaea/" cfg.telemetry.eventSocket;
          message = "services.symthaea-boot.telemetry.eventSocket must stay beneath /run/symthaea";
        }
        {
          assertion = !cfg.telemetry.enable || hasPrefix "/run/symthaea-boot/" cfg.telemetry.statePath;
          message = "services.symthaea-boot.telemetry.statePath must stay beneath /run/symthaea-boot";
        }
      ];

      systemd.services.symthaea-boot-animation = {
        description = "Symthaea Spore Boot Animation";
        documentation = [
          "https://github.com/Luminous-Dynamics/symthaea/blob/main/docs/architecture/BOOT_PROTOCOL_V1.md"
        ];

        after = [ "systemd-udev-settle.service" "local-fs.target" ];
        before = [ "display-manager.service" "graphical.target" ];
        wantedBy = [ "graphical.target" ];

        unitConfig = {
          ConditionPathExists = cfg.device;
          StopWhenUnneeded = true;
          TimeoutStopSec = 8;
        };

        serviceConfig = {
          Type = "simple";

          ExecStart = concatStringsSep " " ([
            "${cfg.package}/bin/quicken-fb"
            "--genesis-phrase ${escapeShellArg cfg.genesisPhrase}"
            "--progress-pipe ${escapeShellArg cfg.progressPipe}"
            "--device ${escapeShellArg cfg.device}"
          ] ++ telemetryArgs);

          SupplementaryGroups = [ "video" "render" ]
            ++ optional cfg.telemetry.enable "symthaea-boot";

          User = "root";
          KillSignal = "SIGTERM";

          # When telemetry is enabled, Unix sockets created by the renderer are
          # owner/group accessible but not writable by unrelated local users.
          UMask = if cfg.telemetry.enable then "0007" else "0022";

          NoNewPrivileges = true;
          ProtectHome = true;
          ProtectKernelTunables = true;
          ProtectKernelModules = true;
          ProtectControlGroups = true;
          RestrictNamespaces = true;
          LockPersonality = true;
          RestrictRealtime = true;
          RestrictSUIDSGID = true;
          PrivateTmp = true;
        };
      };

      systemd.tmpfiles.rules = [
        (if cfg.telemetry.enable
          then "d /run/symthaea 0770 root symthaea-boot -"
          else "d /run/symthaea 0755 root root -")
        "p ${cfg.progressPipe} 0644 root root -"
      ];
    }

    (mkIf cfg.telemetry.enable {
      users.groups.symthaea-boot = {};
    })
  ]);
}
