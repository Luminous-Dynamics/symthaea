# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Mycelix Bootstrap Node — NixOS Module
#
# Runs an Iroh relay/bootstrap node for Mycelix WAN peer discovery.
# Bootstrap nodes help new peers find the network without centralized DNS.
#
# Usage:
#   services.mycelix-bootstrap = {
#     enable = true;
#     listenPort = 4433;
#     region = "us-east";
#     # dataDir defaults to /var/lib/mycelix-bootstrap
#   };
#
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
{ config, lib, pkgs, ... }:

let
  cfg = config.services.mycelix-bootstrap;
in
{
  options.services.mycelix-bootstrap = {
    enable = lib.mkEnableOption "Mycelix bootstrap node for WAN peer discovery";

    package = lib.mkOption {
      type = lib.types.package;
      description = "The Iroh relay binary package";
      # Default assumes iroh is built from the workspace or available in overlay
      default = pkgs.iroh or pkgs.hello;  # placeholder until iroh is packaged
    };

    listenPort = lib.mkOption {
      type = lib.types.port;
      default = 4433;
      description = "QUIC listen port for Iroh relay";
    };

    metricsPort = lib.mkOption {
      type = lib.types.port;
      default = 9091;
      description = "Prometheus metrics endpoint port";
    };

    dataDir = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/mycelix-bootstrap";
      description = "Persistent data directory (node identity, peer cache)";
    };

    region = lib.mkOption {
      type = lib.types.str;
      default = "unknown";
      description = "Geographic region label for this bootstrap node (e.g., us-east, eu-west, ap-south)";
    };

    alpn = lib.mkOption {
      type = lib.types.str;
      default = "symthaea/1";
      description = "ALPN protocol identifier for Iroh connections";
    };

    maxPeers = lib.mkOption {
      type = lib.types.int;
      default = 500;
      description = "Maximum concurrent peer connections";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Whether to open the listen port in the firewall";
    };

    logLevel = lib.mkOption {
      type = lib.types.str;
      default = "info";
      description = "Rust log level (trace, debug, info, warn, error)";
    };
  };

  config = lib.mkIf cfg.enable {
    # System user and group
    users.users.mycelix-bootstrap = {
      isSystemUser = true;
      group = "mycelix-bootstrap";
      home = cfg.dataDir;
      description = "Mycelix bootstrap node service user";
    };
    users.groups.mycelix-bootstrap = { };

    # Persistent directories
    systemd.tmpfiles.rules = [
      "d ${cfg.dataDir} 0750 mycelix-bootstrap mycelix-bootstrap -"
      "d ${cfg.dataDir}/keys 0700 mycelix-bootstrap mycelix-bootstrap -"
      "d ${cfg.dataDir}/peers 0750 mycelix-bootstrap mycelix-bootstrap -"
    ];

    # Main service
    systemd.services.mycelix-bootstrap = {
      description = "Mycelix Bootstrap Node (Iroh relay for WAN peer discovery)";
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      wantedBy = [ "multi-user.target" ];

      environment = {
        RUST_LOG = "iroh=${cfg.logLevel},mycelix=${cfg.logLevel}";
        IROH_DATA_DIR = cfg.dataDir;
        MYCELIX_REGION = cfg.region;
        MYCELIX_MAX_PEERS = toString cfg.maxPeers;
        MYCELIX_ALPN = cfg.alpn;
      };

      serviceConfig = {
        Type = "simple";
        User = "mycelix-bootstrap";
        Group = "mycelix-bootstrap";
        ExecStart = "${cfg.package}/bin/iroh relay --port ${toString cfg.listenPort} --metrics-port ${toString cfg.metricsPort}";
        Restart = "on-failure";
        RestartSec = 10;
        WatchdogSec = 120;

        # Resource limits
        LimitNOFILE = 65536;
        MemoryMax = "256M";

        # Security hardening (matches symthaea.service pattern)
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
        RestrictAddressFamilies = [ "AF_INET" "AF_INET6" "AF_UNIX" ];

        # Writable paths
        ReadWritePaths = [ cfg.dataDir ];
        StateDirectory = "mycelix-bootstrap";
      };
    };

    # Firewall
    networking.firewall.allowedUDPPorts = lib.mkIf cfg.openFirewall [ cfg.listenPort ];
    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.metricsPort ];
  };
}
