# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Symthaea boot observability policy.
#
# This module controls presentation only. It must never become a boot target
# dependency and deliberately keeps systemd/journald authoritative.

{ config, lib, ... }:

let
  cfg = config.services.symthaea-boot.observability;
  inherit (lib) mkEnableOption mkIf mkOption types;
in {
  options.services.symthaea-boot.observability = {
    enable = mkEnableOption "Symthaea boot observability policy";

    defaultMode = mkOption {
      type = types.enum [ "ambient" "diagnostics" ];
      default = "ambient";
      description = ''
        Default boot presentation. Ambient keeps routine console chatter quiet;
        diagnostics retains native initrd/systemd status visibility. Raw
        diagnostics remain an independent Linux console concern.
      '';
    };

    autoEscalate = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Keep systemd status in auto mode so failures and unusually slow boots may
        surface even when the normal presentation is quiet.
      '';
    };

    kernelConsoleLogLevel = mkOption {
      type = types.ints.between 1 7;
      default = 3;
      description = "Kernel console log level used during the ambient boot path.";
    };

    rawLogVT = mkOption {
      type = types.ints.between 2 63;
      default = 2;
      description = ''
        Reserved virtual terminal for the future raw journal/systemd viewer.
        This v1 policy records the contract but does not yet install the viewer.
      '';
    };
  };

  config = mkIf (config.services.symthaea-boot.enable && cfg.enable) {
    # Presentation policy is explicitly opt-in. Enabling the renderer alone must
    # not silently change the host's native console/debugging behavior.
    boot.consoleLogLevel = lib.mkDefault cfg.kernelConsoleLogLevel;
    boot.initrd.verbose = lib.mkDefault (cfg.defaultMode == "diagnostics");

    boot.kernelParams =
      lib.optionals (cfg.defaultMode == "ambient") [
        "rd.udev.log_level=3"
        "quiet"
        "systemd.show_status=${if cfg.autoEscalate then "auto" else "false"}"
        "rd.systemd.show_status=${if cfg.autoEscalate then "auto" else "false"}"
      ]
      ++ lib.optionals (cfg.defaultMode == "diagnostics") [
        "systemd.show_status=true"
        "rd.systemd.show_status=true"
      ];

    assertions = [
      {
        assertion = cfg.rawLogVT != 1;
        message = "services.symthaea-boot.observability.rawLogVT must not use tty1; tty1 is reserved for the graphical boot path.";
      }
    ];
  };
}
