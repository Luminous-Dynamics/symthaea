# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# NixOS Module: Symthaea / Spore Boot Animation
#
# Runs the mycelial colonization animation (symthaea-quicken-fb) on bare-metal
# DRM/KMS framebuffer during early graphical boot. Optional typed telemetry,
# handoff receipts, and performance receipts are presentation-only.

{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.symthaea-boot;
  telemetryArgs = optionals cfg.telemetry.enable [
    "--boot-events-socket ${escapeShellArg cfg.telemetry.eventSocket}"
    "--boot-state-path ${escapeShellArg cfg.telemetry.statePath}"
  ];
  handoffArgs = optionals cfg.handoff.enable [
    "--handoff-receipt ${escapeShellArg cfg.handoff.receiptPath}"
  ];
  performanceArgs = optionals cfg.performance.enable [
    "--performance-receipt ${escapeShellArg cfg.performance.receiptPath}"
  ];
  seedArgs = if cfg.genesisPhrase == null then [
    "--visual-seed-file ${escapeShellArg cfg.visualSeedFile}"
  ] else [
    # Compatibility only. This value is presentation input and MUST NOT contain
    # credentials, recovery phrases, key material, or private identity data.
    "--genesis-phrase ${escapeShellArg cfg.genesisPhrase}"
  ];
  visualSeedInit = pkgs.writeShellScript "symthaea-boot-visual-seed-init" ''
    set -eu
    seed_file=${escapeShellArg cfg.visualSeedFile}
    seed_dir="$(${pkgs.coreutils}/bin/dirname -- "$seed_file")"
    ${pkgs.coreutils}/bin/mkdir -p -- "$seed_dir"

    if [ ! -s "$seed_file" ]; then
      tmp="$seed_file.tmp.$$"
      trap '${pkgs.coreutils}/bin/rm -f -- "$tmp"' EXIT
      ${pkgs.coreutils}/bin/head -c 32 /dev/urandom | ${pkgs.coreutils}/bin/base64 > "$tmp"
      ${pkgs.coreutils}/bin/chmod 0644 "$tmp"
      ${pkgs.coreutils}/bin/mv -f -- "$tmp" "$seed_file"
      trap - EXIT
    fi
  '';
in {
  options.services.symthaea-boot = {
    enable = mkEnableOption "Symthaea/Spore boot animation";

    package = mkOption {
      type = types.package;
      description = "The symthaea-quicken-fb package containing the quicken-fb binary.";
    };

    visualSeedFile = mkOption {
      type = types.str;
      default = "/var/lib/symthaea/boot-visual-seed";
      description = ''
        Persistent presentation-only seed file for deterministic boot artwork.
        It must be a direct child of /var/lib/symthaea so the renderer's writable
        state remains confined to its systemd-managed StateDirectory. The module
        creates the file from random bytes when absent. This file is not a
        credential, recovery secret, key-derivation input, or authority-bearing
        machine identity and should not be reused for any security purpose.
      '';
    };

    genesisPhrase = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = ''
        DEPRECATED compatibility input for historical quicken-fb deployments.
        When non-null it is passed through the deprecated --genesis-phrase flag.
        It is presentation-only and MUST NOT contain credentials, recovery
        phrases, key material, or private identity data. Prefer visualSeedFile.
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

    handoff = {
      enable = mkOption {
        type = types.bool;
        default = false;
        description = ''
          Emit a diagnostic acknowledgement only after quicken-fb has dropped
          its DRM framebuffer and restored the saved CRTC. This does not by
          itself authorize display-manager startup.
        '';
      };

      receiptPath = mkOption {
        type = types.str;
        default = "/run/symthaea/boot-display-released-v1.json";
        description = "Ephemeral post-DRM-release acknowledgement path.";
      };

      stopTimeoutMs = mkOption {
        type = types.ints.between 100 5000;
        default = 1000;
        description = ''
          Hard systemd stop bound for the decorative renderer. Expiry permits
          systemd to terminate the renderer rather than allowing presentation
          to hold login or recovery indefinitely.
        '';
      };
    };

    performance = {
      enable = mkOption {
        type = types.bool;
        default = false;
        description = ''
          Collect in-memory renderer timings and write one receipt on exit.
          Disabled by default so normal boots do not allocate measurement vectors.
        '';
      };

      receiptPath = mkOption {
        type = types.str;
        default = "/run/symthaea/boot-performance-v1.json";
        description = "Ephemeral renderer performance receipt path.";
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
          assertion = cfg.genesisPhrase != null || builtins.dirOf cfg.visualSeedFile == "/var/lib/symthaea";
          message = "services.symthaea-boot.visualSeedFile must be a direct child of /var/lib/symthaea";
        }
        {
          assertion = !cfg.telemetry.enable || hasPrefix "/run/symthaea/" cfg.telemetry.eventSocket;
          message = "services.symthaea-boot.telemetry.eventSocket must stay beneath /run/symthaea";
        }
        {
          assertion = !cfg.telemetry.enable || hasPrefix "/run/symthaea-boot/" cfg.telemetry.statePath;
          message = "services.symthaea-boot.telemetry.statePath must stay beneath /run/symthaea-boot";
        }
        {
          assertion = !cfg.handoff.enable || hasPrefix "/run/symthaea/" cfg.handoff.receiptPath;
          message = "services.symthaea-boot.handoff.receiptPath must stay beneath /run/symthaea";
        }
        {
          assertion = !cfg.performance.enable || hasPrefix "/run/symthaea/" cfg.performance.receiptPath;
          message = "services.symthaea-boot.performance.receiptPath must stay beneath /run/symthaea";
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
          # Kept for compatibility until the separately qualified display-manager
          # handoff trigger replaces the historical lifecycle wiring.
          StopWhenUnneeded = true;
        };

        serviceConfig = {
          Type = "simple";

          ExecStart = concatStringsSep " " ([
            "${cfg.package}/bin/quicken-fb"
          ] ++ seedArgs ++ [
            "--progress-pipe ${escapeShellArg cfg.progressPipe}"
            "--device ${escapeShellArg cfg.device}"
          ] ++ telemetryArgs ++ handoffArgs ++ performanceArgs);

          ExecStartPre =
            optional (cfg.genesisPhrase == null) "${visualSeedInit}"
            ++ optionals cfg.handoff.enable [
              "${pkgs.coreutils}/bin/rm -f -- ${escapeShellArg cfg.handoff.receiptPath}"
            ]
            ++ optionals cfg.performance.enable [
              "${pkgs.coreutils}/bin/rm -f -- ${escapeShellArg cfg.performance.receiptPath}"
            ];

          # Unix-domain socket ownership follows the process primary GID, not
          # supplementary groups. Use the telemetry group as the primary group
          # so the observer DynamicUser can actually write boot-events.sock.
          User = "root";
          Group = if cfg.telemetry.enable then "symthaea-boot" else "root";
          SupplementaryGroups = [ "video" "render" ];
          KillSignal = "SIGTERM";
          TimeoutStopSec = "${toString cfg.handoff.stopTimeoutMs}ms";

          # StateDirectory is the only persistent write surface. /run/symthaea
          # remains writable for the socket, FIFO and bounded diagnostic receipts.
          StateDirectory = "symthaea";
          StateDirectoryMode = "0755";
          ReadWritePaths = [ "/run/symthaea" ];

          # When telemetry is enabled, the renderer socket is root:symthaea-boot
          # and owner/group accessible, but unrelated local users cannot write it.
          UMask = if cfg.telemetry.enable then "0007" else "0022";

          NoNewPrivileges = true;
          ProtectSystem = "strict";
          ProtectHome = true;
          ProtectKernelTunables = true;
          ProtectKernelModules = true;
          ProtectKernelLogs = true;
          ProtectControlGroups = true;
          ProtectClock = true;
          ProtectHostname = true;
          RestrictNamespaces = true;
          LockPersonality = true;
          MemoryDenyWriteExecute = true;
          RestrictRealtime = true;
          RestrictSUIDSGID = true;
          RestrictAddressFamilies = [ "AF_UNIX" ];
          SystemCallArchitectures = "native";
          PrivateTmp = true;

          # Presentation needs exactly one DRM primary node. DevicePolicy=closed
          # retains systemd's standard pseudo-devices (including /dev/urandom)
          # while denying access to unrelated hardware.
          DevicePolicy = "closed";
          DeviceAllow = [ "${cfg.device} rw" ];
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
