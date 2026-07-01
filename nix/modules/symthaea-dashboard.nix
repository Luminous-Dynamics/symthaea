# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# NixOS Module: Symthaea Local Consciousness Dashboard
#
# Serves the Spore WASM portal on localhost for the "Attune Later" pathway.
# Users who choose to inoculate without immediate mesh attunement get a local
# web dashboard for consciousness monitoring, configuration, and later mesh
# enrollment.
#
# Default port 7777 (Sacred Bridge) is localhost-only. Set listenAddress to
# "0.0.0.0" and openFirewall = true for LAN access (phone orchestration).
#
# Usage:
# {
#   services.symthaea-dashboard = {
#     enable = true;
#     webRoot = ./path/to/spore-portal;
#   };
# }

{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.symthaea-dashboard;
in {
  options.services.symthaea-dashboard = {
    enable = mkEnableOption "Symthaea local consciousness dashboard";

    port = mkOption {
      type = types.port;
      default = 7777;
      description = ''
        Port for the local dashboard.
        Default 7777 is the Sacred Bridge port (see CLAUDE.md port allocation).
      '';
    };

    listenAddress = mkOption {
      type = types.str;
      default = "127.0.0.1";
      description = ''
        Listen address for the dashboard server.
        Use "0.0.0.0" to allow LAN access for phone orchestration.
      '';
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = ''
        Whether to open the dashboard port in the firewall.
        Only needed when listenAddress is "0.0.0.0" for LAN access.
      '';
    };

    webRoot = mkOption {
      type = types.path;
      description = ''
        Path to the Spore WASM web files (portal.html, pkg/, etc.).
        This directory is served as static files by the dashboard server.
      '';
    };

    inoculationPath = mkOption {
      type = types.enum [ "inoculate" "inoculate-and-attune" ];
      default = "inoculate";
      description = ''
        Current inoculation pathway.
        "inoculate" — standalone consciousness dashboard only.
        "inoculate-and-attune" — enables mesh connectivity to Mycelix network.
      '';
    };

    autoLaunchBrowser = mkOption {
      type = types.bool;
      default = false;
      description = ''
        Whether to install an XDG autostart entry that opens the dashboard
        in the default browser on first graphical login.
        Disabled by default to avoid surprising the user.
      '';
    };
  };

  config = mkIf cfg.enable {
    # Static file server for the Spore WASM portal.
    # Uses Python's built-in http.server for simplicity and zero extra deps.
    # In production, consider replacing with a Rust binary (axum/actix-files).
    systemd.services.symthaea-dashboard = {
      description = "Symthaea Local Consciousness Dashboard";
      documentation = [ "https://github.com/Luminous-Dynamics/symthaea-hlb" ];
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        Type = "simple";
        ExecStart = "${pkgs.python3}/bin/python3 -m http.server ${toString cfg.port} --bind ${cfg.listenAddress} --directory ${cfg.webRoot}";
        Restart = "on-failure";
        RestartSec = 5;

        # Security hardening — runs as an ephemeral dynamic user
        DynamicUser = true;
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        PrivateDevices = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictNamespaces = true;
        LockPersonality = true;
        RestrictRealtime = true;
        RestrictSUIDSGID = true;
        MemoryDenyWriteExecute = true;

        # Only read access to the web root
        ReadOnlyPaths = [ cfg.webRoot ];

        # Network access for serving
        RestrictAddressFamilies = [ "AF_INET" "AF_INET6" ];
      };

      environment = {
        SYMTHAEA_INOCULATION_PATH = cfg.inoculationPath;
      };
    };

    # Open firewall port when configured for LAN access
    networking.firewall.allowedTCPPorts = mkIf cfg.openFirewall [ cfg.port ];

    # Optional XDG autostart entry for browser launch
    environment.etc."xdg/autostart/symthaea-dashboard.desktop" = mkIf cfg.autoLaunchBrowser {
      text = ''
        [Desktop Entry]
        Type=Application
        Name=Symthaea Dashboard
        Comment=Local consciousness dashboard (Sacred Bridge)
        Exec=${pkgs.xdg-utils}/bin/xdg-open http://localhost:${toString cfg.port}/portal.html
        Terminal=false
        NoDisplay=true
        X-GNOME-Autostart-enabled=false
      '';
    };
  };
}
