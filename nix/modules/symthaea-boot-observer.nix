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
  inherit (lib) mkEnableOption mkIf mkOption types;
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
      description = "Ephemeral normalized snapshot used by late presentation consumers.";
    };
  };

  config = mkIf (config.services.symthaea-boot.enable && cfg.enable) {
    assertions = [
      {
        assertion = lib.hasPrefix "/" cfg.outputSocket;
        message = "services.symthaea-boot.observer.outputSocket must be absolute";
      }
      {
        assertion = lib.hasPrefix "/run/symthaea-boot/" cfg.statePath;
        message = ''
          services.symthaea-boot.observer.statePath must stay beneath
          /run/symthaea-boot so the observer remains ephemeral and writable by
          its DynamicUser runtime directory.
        '';
      }
    ];

    systemd.services.symthaea-boot-observer = {
      description = "Symthaea Structured Boot Observer";
      documentation = [
        "https://github.com/Luminous-Dynamics/symthaea/blob/main/docs/architecture/BOOT_PROTOCOL_V1.md"
      ];

      # D-Bus is the only structured authority this service needs. Starting at
      # basic.target is early enough to observe network/services/graphics while
      # initial systemd monotonic timestamps reconstruct earlier unit readiness.
      after = [ "dbus.service" ];
      wantedBy = [ "basic.target" ];

      unitConfig = {
        # Never let a presentation observer hold shutdown/recovery transactions.
        IgnoreOnIsolate = true;
      };

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

        # State persistence is the only writable path. Sending to the renderer's
        # Unix datagram socket does not require filesystem write permission.
        ReadWritePaths = [ "/run/symthaea-boot" ];
      };
    };

    # RuntimeDirectory is created by systemd before ExecStart. Generate the
    # observer config there from declarative Nix settings without making config
    # evaluation or the renderer dependent on the observer.
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
            domain = "graphics";
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
