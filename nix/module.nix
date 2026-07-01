# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# NixOS Module for Symthaea
#
# Provides consciousness-aware NixOS configuration management.
#
# Usage:
# {
#   services.symthaea = {
#     enable = true;
#     package = symthaea;
#   };
# }

{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.symthaea;
  settingsFormat = pkgs.formats.toml { };
in {
  options.services.symthaea = {
    enable = mkEnableOption "Symthaea consciousness-aware configuration service";

    package = mkOption {
      type = types.package;
      default = pkgs.symthaea or (throw "symthaea package not found in pkgs");
      defaultText = literalExpression "pkgs.symthaea";
      description = "The Symthaea package to use.";
    };

    socketPath = mkOption {
      type = types.path;
      default = "/run/symthaea/symthaea.sock";
      description = "Path to the Unix socket for IPC.";
    };

    dataDir = mkOption {
      type = types.path;
      default = "/var/lib/symthaea";
      description = "Directory for state and audit logs.";
    };

    metricsInterval = mkOption {
      type = types.int;
      default = 500;
      description = "Main service loop interval in milliseconds.";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = "Whether to open firewall ports (not needed for local Unix socket).";
    };

    user = mkOption {
      type = types.str;
      default = "symthaea";
      description = "User account under which Symthaea runs.";
    };

    group = mkOption {
      type = types.str;
      default = "symthaea";
      description = "Group account under which Symthaea runs.";
    };

    shellCompletions = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Whether to install shell completions.";
      };
    };

    gui = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Whether to install the GUI application.";
      };
    };

    settings = mkOption {
      type = settingsFormat.type;
      default = { };
      example = literalExpression ''
        {
          consciousness = {
            phi_threshold = 0.5;
            auto_center = true;
          };
          safety = {
            dry_run_destructive = true;
            confirm_threshold = "high";
          };
        }
      '';
      description = "Configuration settings for Symthaea.";
    };
  };

  config = mkIf cfg.enable {
    # Create user and group
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      home = cfg.dataDir;
      description = "Symthaea service user";
    };

    users.groups.${cfg.group} = { };

    # Create data directory
    systemd.tmpfiles.rules = [
      "d '${cfg.dataDir}' 0750 ${cfg.user} ${cfg.group} -"
      "d '${cfg.dataDir}/logs' 0750 ${cfg.user} ${cfg.group} -"
      "d '${cfg.dataDir}/state' 0750 ${cfg.user} ${cfg.group} -"
      "d '/run/symthaea' 0755 ${cfg.user} ${cfg.group} -"
    ];

    services.logrotate.settings."${cfg.dataDir}/logs/service-audit.jsonl" = {
      frequency = "weekly";
      rotate = 8;
      compress = true;
      missingok = true;
      notifempty = true;
      copytruncate = true;
      su = "${cfg.user} ${cfg.group}";
    };

    # Systemd socket activation
    systemd.sockets.symthaea = {
      description = "Symthaea IPC Socket";
      wantedBy = [ "sockets.target" ];

      socketConfig = {
        ListenStream = cfg.socketPath;
        SocketMode = "0666";
        SocketUser = cfg.user;
        SocketGroup = cfg.group;
        Accept = false;
        MaxConnections = 64;
        KeepAlive = true;
        ReceiveBuffer = "64K";
      };
    };

    # Systemd service
    systemd.services.symthaea = {
      description = "Symthaea Consciousness-Aware Configuration Service";
      documentation = [ "https://github.com/Luminous-Dynamics/symthaea-hlb" ];
      after = [ "network.target" "symthaea.socket" ];
      requires = [ "symthaea.socket" ];

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = "${cfg.package}/bin/symthaea --socket ${cfg.socketPath} --loop-interval ${toString cfg.metricsInterval} --state-file ${cfg.dataDir}/state/symthaea-state.bin";
        Restart = "on-failure";
        RestartSec = 5;

        # Security hardening
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        PrivateDevices = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictAddressFamilies = [ "AF_UNIX" ];
        RestrictNamespaces = true;
        LockPersonality = true;
        MemoryDenyWriteExecute = true;
        RestrictRealtime = true;
        RestrictSUIDSGID = true;

        # Directories
        RuntimeDirectory = "symthaea";
        StateDirectory = "symthaea";
        LogsDirectory = "symthaea";

        # Environment
        Environment = [
          "SYMTHAEA_DATA_DIR=${cfg.dataDir}"
          "SYMTHAEA_SERVICE_AUDIT_LOG_PATH=${cfg.dataDir}/logs/service-audit.jsonl"
          "RUST_LOG=symthaea=info"
        ];
      };
    };

    # Add to system packages
    environment.systemPackages = with cfg.package; [
      cfg.package
    ] ++ optional cfg.gui.enable cfg.package;

    # Shell completions
    programs.bash.interactiveShellInit = mkIf cfg.shellCompletions.enable ''
      source ${cfg.package}/share/bash-completion/completions/symthaea.bash 2>/dev/null || true
    '';

    programs.zsh.interactiveShellInit = mkIf cfg.shellCompletions.enable ''
      fpath+=(${cfg.package}/share/zsh/site-functions)
    '';

    programs.fish.interactiveShellInit = mkIf cfg.shellCompletions.enable ''
      source ${cfg.package}/share/fish/vendor_completions.d/symthaea.fish 2>/dev/null || true
    '';

    # XDG desktop entry for GUI
    environment.etc."xdg/autostart/symthaea-gui.desktop" = mkIf cfg.gui.enable {
      text = ''
        [Desktop Entry]
        Type=Application
        Name=Symthaea
        Comment=Consciousness-Aware NixOS Configuration
        Exec=${cfg.package}/bin/symthaea-gui
        Icon=system-software-install
        Categories=System;Settings;
        Terminal=false
      '';
    };
  };
}
