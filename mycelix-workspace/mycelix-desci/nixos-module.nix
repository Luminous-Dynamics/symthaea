# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.mycelix-desci;
in {
  options.services.mycelix-desci = {
    enable = mkEnableOption "Mycelix-DeSci API server";

    package = mkOption {
      type = types.package;
      default = pkgs.mycelix-api or (import ./. { inherit pkgs; }).packages.${pkgs.system}.mycelix-api;
      description = ''
        The mycelix-api package to use.
        Defaults to the package from this flake.
      '';
    };

    port = mkOption {
      type = types.port;
      default = 8080;
      description = "Port for the API server to listen on";
      example = 3000;
    };

    host = mkOption {
      type = types.str;
      default = "0.0.0.0";
      description = "Host address to bind to";
      example = "127.0.0.1";
    };

    logLevel = mkOption {
      type = types.enum [ "trace" "debug" "info" "warn" "error" ];
      default = "info";
      description = "Logging level for the application";
    };

    corsOrigins = mkOption {
      type = types.str;
      default = "*";
      description = ''
        CORS allowed origins.
        Use "*" to allow all origins, or specify comma-separated origins.
      '';
      example = "https://example.com,https://app.example.com";
    };

    user = mkOption {
      type = types.str;
      default = "mycelix";
      description = "User account under which mycelix-api runs";
    };

    group = mkOption {
      type = types.str;
      default = "mycelix";
      description = "Group under which mycelix-api runs";
    };

    dataDir = mkOption {
      type = types.path;
      default = "/var/lib/mycelix";
      description = "Directory where mycelix-api stores data";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = "Whether to open the firewall port for the API server";
    };

    extraEnvironment = mkOption {
      type = types.attrsOf types.str;
      default = {};
      description = "Additional environment variables to set for the service";
      example = {
        CUSTOM_VAR = "value";
      };
    };
  };

  config = mkIf cfg.enable {
    # Create system user and group
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      home = cfg.dataDir;
      createHome = true;
      description = "Mycelix-DeSci API server user";
    };

    users.groups.${cfg.group} = {};

    # Systemd service configuration
    systemd.services.mycelix-api = {
      description = "Mycelix-DeSci API Server";
      documentation = [ "https://github.com/Luminous-Dynamics/mycelix-desci" ];

      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];

      # Environment variables
      environment = {
        PORT = toString cfg.port;
        RUST_LOG = "mycelix_api=${cfg.logLevel}";
        CORS_ORIGINS = cfg.corsOrigins;
      } // cfg.extraEnvironment;

      serviceConfig = {
        Type = "simple";
        ExecStart = "${cfg.package}/bin/mycelix-api";

        # Restart configuration
        Restart = "always";
        RestartSec = "10s";

        # User and group
        User = cfg.user;
        Group = cfg.group;
        WorkingDirectory = cfg.dataDir;

        # Security hardening
        # Process sandboxing
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ cfg.dataDir ];
        NoNewPrivileges = true;

        # Network
        PrivateDevices = true;
        RestrictAddressFamilies = [ "AF_INET" "AF_INET6" "AF_UNIX" ];

        # System calls
        SystemCallFilter = [ "@system-service" "~@privileged" ];
        SystemCallErrorNumber = "EPERM";

        # Kernel
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectKernelLogs = true;
        ProtectControlGroups = true;

        # Misc hardening
        RestrictNamespaces = true;
        LockPersonality = true;
        RestrictRealtime = true;
        RestrictSUIDSGID = true;
        RemoveIPC = true;
        PrivateMounts = true;
        ProtectClock = true;
        ProtectProc = "invisible";
        ProcSubset = "pid";
        ProtectHostname = true;

        # Capabilities
        CapabilityBoundingSet = [ "" ];
        AmbientCapabilities = [ "" ];

        # Resource limits
        LimitNOFILE = "65536";

        # Umask
        UMask = "0077";
      };

      # Pre-start checks
      preStart = ''
        # Ensure data directory exists and has correct permissions
        mkdir -p ${cfg.dataDir}
        chown ${cfg.user}:${cfg.group} ${cfg.dataDir}
        chmod 0750 ${cfg.dataDir}
      '';
    };

    # Firewall configuration
    networking.firewall = mkIf cfg.openFirewall {
      allowedTCPPorts = [ cfg.port ];
    };

    # Assertions for configuration validation
    assertions = [
      {
        assertion = cfg.port > 0 && cfg.port < 65536;
        message = "services.mycelix-desci.port must be between 1 and 65535";
      }
      {
        assertion = cfg.user != "root";
        message = "services.mycelix-desci.user must not be root for security";
      }
    ];
  };

  meta = {
    maintainers = with lib.maintainers; [ ];
    doc = ./docs/NIX.md;
  };
}
