# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Structured systemd -> Symthaea boot-event observer.
#
# The service is deliberately weakly attached to the boot transaction: it is
# WantedBy basic.target, never RequiredBy, is Type=simple, and performs all D-Bus
# observation after systemd considers the process started. Its failure therefore
# cannot determine target success or machine boot health.

{ config, lib, ... }:

let
  cfg = config.services.symthaea-boot.observer;
  telemetry = config.services.symthaea-boot.telemetry;
  confinedRuntimePath = prefix: path:
    lib.hasPrefix prefix path
    && lib.removePrefix prefix path != ""
    && !(builtins.elem ".." (lib.splitString "/" path));
  inherit (lib) mkDefault mkEnableOption mkIf mkOption types;
in {
  options.services.symthaea-boot.observer = {
    enable = mkEnableOption "structured Symthaea boot observer";

    package = mkOption {
      type = types.package;
      description = ''
        Package containing bin/symthaea-boot-observer. Kept explicit until the
        observer receives a first-class package output in the Symthaea flake.
      '';
    };

    outputSocket = mkOption {
      type = types.str;
      default = "/run/symthaea/boot-events.sock";
      description = "Unix datagram destination used for normalized boot events.";
    };

    statePath = mkOption {
      type = types.str;
      default = "/run/symthaea-boot/state-v1.json";
      description = "Ephemeral lineage-bound snapshot used by presentation consumers.";
    };
  };

  config = mkIf (config.services.symthaea-boot.enable && cfg.enable) {
    # Enabling the observer opts the renderer into the typed channel by default.
    # Explicit user configuration still wins over mkDefault.
    services.symthaea-boot.telemetry.enable = mkDefault true;

    assertions = [
      {
        assertion = confinedRuntimePath "/run/symthaea/" cfg.outputSocket;
        message = "services.symthaea-boot.observer.outputSocket must stay beneath /run/symthaea without '..' traversal";
      }
      {
        assertion = confinedRuntimePath "/run/symthaea-boot/" cfg.statePath;
        message = ''
          services.symthaea-boot.observer.statePath must stay beneath
          /run/symthaea-boot without '..' traversal so the observer remains
          ephemeral and writable only through its DynamicUser runtime directory.
        '';
      }
      {
        assertion = !telemetry.enable || cfg.outputSocket == telemetry.eventSocket;
        message = "observer.outputSocket and telemetry.eventSocket must match when typed boot telemetry is enabled";
      }
      {
        assertion = !telemetry.enable || cfg.statePath == telemetry.statePath;
        message = "observer.statePath and telemetry.statePath must match when typed boot telemetry is enabled";
      }
    ];

    systemd.services.symthaea-boot-observer = {
      description = "Symthaea Structured Boot Observer";
      documentation = [
        "https://github.com/Luminous-Dynamics/symthaea/blob/main/docs/architecture/BOOT_PROTOCOL_V1.md"
      ];

      after = [ "dbus.service" ];
      wantedBy = [ "basic.target" ];

      unitConfig.IgnoreOnIsolate = true;

      serviceConfig = {
        Type = "simple";
        ExecStart = ''
          ${cfg.package}/bin/symthaea-boot-observer \
            --config /run/symthaea-boot/observer-v1.json
        '';

        Restart = "on-failure";
        RestartSec = "500ms";

        DynamicUser = true;
        RuntimeDirectory = "symthaea-boot";
        RuntimeDirectoryMode = "0755";
        SupplementaryGroups = [ "symthaea-boot" ];
        UMask = "0022";

        NoNewPrivileges = true;
        PrivateTmp = true;
        PrivateDevices = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectKernelLogs = true;
        ProtectControlGroups = true;
        ProtectClock = true;
        ProtectHostname = true;
        LockPersonality = true;
        MemoryDenyWriteExecute = true;
        RestrictRealtime = true;
        RestrictSUIDSGID = true;
        RestrictNamespaces = true;
        RestrictAddressFamilies = [ "AF_UNIX" ];
        SystemCallArchitectures = "native";
        ReadWritePaths = [ "/run/symthaea-boot" ];
      };
    };

    systemd.services.symthaea-boot-observer.preStart = ''
      cat > /run/symthaea-boot/observer-v1.json <<'JSON'
      ${builtins.toJSON {
        output_socket = cfg.outputSocket;
        state_path = cfg.statePath;
        watched_units = [
          {
            unit = "local-fs-pre.target";
            domain = "storage";
            phase = "storage";
            criticality = "critical";
            boot_ready = false;
          }
          {
            unit = "local-fs.target";
            domain = "filesystems";
            phase = "filesystems";
            criticality = "critical";
            boot_ready = false;
          }
          {
            unit = "network.target";
            domain = "network";
            phase = "network";
            criticality = "non-critical";
            boot_ready = false;
          }
          {
            unit = "multi-user.target";
            domain = "services";
            phase = "services";
            criticality = "critical";
            boot_ready = false;
          }
          {
            unit = "display-manager.service";
            domain = "graphics";
            phase = "graphics";
            criticality = "critical";
            boot_ready = false;
          }
          {
            unit = "graphical.target";
            domain = "session";
            phase = "ready";
            criticality = "critical";
            boot_ready = true;
          }
        ];
      }}
      JSON
    '';
  };
}
