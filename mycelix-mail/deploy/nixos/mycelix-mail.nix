# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Mycelix Mail NixOS Module
# Declarative configuration for self-hosting on NixOS

{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.mycelix-mail;
in {
  options.services.mycelix-mail = {
    enable = mkEnableOption "Mycelix Mail email server";

    domain = mkOption {
      type = types.str;
      description = "Primary domain for Mycelix Mail";
      example = "mail.example.com";
    };

    package = mkOption {
      type = types.package;
      default = pkgs.mycelix-mail;
      description = "Mycelix Mail package to use";
    };

    dataDir = mkOption {
      type = types.path;
      default = "/var/lib/mycelix-mail";
      description = "Directory for Mycelix Mail data";
    };

    user = mkOption {
      type = types.str;
      default = "mycelix";
      description = "User account under which Mycelix Mail runs";
    };

    group = mkOption {
      type = types.str;
      default = "mycelix";
      description = "Group under which Mycelix Mail runs";
    };

    api = {
      port = mkOption {
        type = types.port;
        default = 8080;
        description = "Port for the API server";
      };

      address = mkOption {
        type = types.str;
        default = "127.0.0.1";
        description = "Address to bind the API server";
      };
    };

    database = {
      host = mkOption {
        type = types.str;
        default = "/run/postgresql";
        description = "PostgreSQL host (socket path or hostname)";
      };

      name = mkOption {
        type = types.str;
        default = "mycelix";
        description = "Database name";
      };

      user = mkOption {
        type = types.str;
        default = "mycelix";
        description = "Database user";
      };

      createLocally = mkOption {
        type = types.bool;
        default = true;
        description = "Create database locally";
      };
    };

    redis = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Enable Redis for caching and queues";
      };

      createLocally = mkOption {
        type = types.bool;
        default = true;
        description = "Create Redis instance locally";
      };
    };

    meilisearch = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Enable Meilisearch for full-text search";
      };

      createLocally = mkOption {
        type = types.bool;
        default = true;
        description = "Create Meilisearch instance locally";
      };
    };

    ollama = {
      enable = mkOption {
        type = types.bool;
        default = false;
        description = "Enable Ollama for AI features";
      };

      models = mkOption {
        type = types.listOf types.str;
        default = [ "llama3.2" "nomic-embed-text" ];
        description = "Ollama models to pull on startup";
      };
    };

    ssl = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Enable SSL/TLS";
      };

      acme = mkOption {
        type = types.bool;
        default = true;
        description = "Use ACME (Let's Encrypt) for certificates";
      };

      email = mkOption {
        type = types.str;
        default = "";
        description = "Email for ACME notifications";
      };
    };

    nginx = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Enable nginx reverse proxy";
      };
    };

    backup = {
      enable = mkOption {
        type = types.bool;
        default = false;
        description = "Enable automated backups";
      };

      schedule = mkOption {
        type = types.str;
        default = "daily";
        description = "Backup schedule (systemd calendar format)";
      };

      retention = mkOption {
        type = types.int;
        default = 30;
        description = "Number of days to retain backups";
      };

      location = mkOption {
        type = types.path;
        default = "/var/backup/mycelix-mail";
        description = "Backup storage location";
      };
    };

    secretsFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = "Path to secrets file (contains JWT_SECRET, etc.)";
    };
  };

  config = mkIf cfg.enable {
    # User and group
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      home = cfg.dataDir;
      createHome = true;
    };

    users.groups.${cfg.group} = {};

    # PostgreSQL
    services.postgresql = mkIf cfg.database.createLocally {
      enable = true;
      ensureDatabases = [ cfg.database.name ];
      ensureUsers = [
        {
          name = cfg.database.user;
          ensureDBOwnership = true;
        }
      ];
    };

    # Redis
    services.redis.servers.mycelix = mkIf (cfg.redis.enable && cfg.redis.createLocally) {
      enable = true;
      port = 6379;
    };

    # Meilisearch
    services.meilisearch = mkIf (cfg.meilisearch.enable && cfg.meilisearch.createLocally) {
      enable = true;
      listenAddress = "127.0.0.1";
      listenPort = 7700;
    };

    # Ollama
    services.ollama = mkIf cfg.ollama.enable {
      enable = true;
      acceleration = "cuda";  # or "rocm" for AMD
    };

    # Main Mycelix Mail service
    systemd.services.mycelix-mail-api = {
      description = "Mycelix Mail API Server";
      after = [
        "network.target"
        "postgresql.service"
      ] ++ optional cfg.redis.createLocally "redis-mycelix.service"
        ++ optional cfg.meilisearch.createLocally "meilisearch.service";

      wants = [
        "postgresql.service"
      ] ++ optional cfg.redis.createLocally "redis-mycelix.service"
        ++ optional cfg.meilisearch.createLocally "meilisearch.service";

      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        WorkingDirectory = cfg.dataDir;
        ExecStart = "${cfg.package}/bin/mycelix-api";
        Restart = "always";
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
        ReadWritePaths = [ cfg.dataDir ];

        # Environment
        EnvironmentFile = mkIf (cfg.secretsFile != null) cfg.secretsFile;
      };

      environment = {
        DATABASE_URL = "postgres://${cfg.database.user}@${cfg.database.host}/${cfg.database.name}";
        REDIS_URL = mkIf cfg.redis.enable "redis://127.0.0.1:6379";
        MEILISEARCH_URL = mkIf cfg.meilisearch.enable "http://127.0.0.1:7700";
        BIND_ADDRESS = "${cfg.api.address}:${toString cfg.api.port}";
        DATA_DIR = cfg.dataDir;
        RUST_LOG = "info";
      };
    };

    # Sync worker
    systemd.services.mycelix-mail-sync = {
      description = "Mycelix Mail Sync Worker";
      after = [ "mycelix-mail-api.service" ];
      wants = [ "mycelix-mail-api.service" ];
      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        WorkingDirectory = cfg.dataDir;
        ExecStart = "${cfg.package}/bin/mycelix-worker sync";
        Restart = "always";
        RestartSec = 5;
        EnvironmentFile = mkIf (cfg.secretsFile != null) cfg.secretsFile;
      };

      environment = {
        DATABASE_URL = "postgres://${cfg.database.user}@${cfg.database.host}/${cfg.database.name}";
        REDIS_URL = mkIf cfg.redis.enable "redis://127.0.0.1:6379";
        RUST_LOG = "info";
      };
    };

    # Nginx reverse proxy
    services.nginx = mkIf cfg.nginx.enable {
      enable = true;

      virtualHosts.${cfg.domain} = {
        forceSSL = cfg.ssl.enable;
        enableACME = cfg.ssl.enable && cfg.ssl.acme;

        locations."/" = {
          proxyPass = "http://127.0.0.1:3000";
          proxyWebsockets = true;
        };

        locations."/api" = {
          proxyPass = "http://127.0.0.1:${toString cfg.api.port}";
          proxyWebsockets = true;
          extraConfig = ''
            proxy_read_timeout 300s;
            proxy_send_timeout 300s;
            client_max_body_size 50M;
          '';
        };

        locations."/health" = {
          proxyPass = "http://127.0.0.1:${toString cfg.api.port}";
        };
      };
    };

    # ACME
    security.acme = mkIf (cfg.ssl.enable && cfg.ssl.acme) {
      acceptTerms = true;
      defaults.email = cfg.ssl.email;
    };

    # Backup service
    systemd.services.mycelix-mail-backup = mkIf cfg.backup.enable {
      description = "Mycelix Mail Backup";
      serviceConfig = {
        Type = "oneshot";
        User = cfg.user;
        ExecStart = "${cfg.package}/bin/mycelix-backup --output ${cfg.backup.location}";
      };
    };

    systemd.timers.mycelix-mail-backup = mkIf cfg.backup.enable {
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = cfg.backup.schedule;
        Persistent = true;
      };
    };

    # Firewall
    networking.firewall.allowedTCPPorts = mkIf cfg.nginx.enable [
      80
      443
    ];
  };
}
