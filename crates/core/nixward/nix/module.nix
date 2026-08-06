# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{ config, lib, pkgs, ... }:

let
  cfg = config.services.nixward;
in {
  options.services.nixward = {
    enable = lib.mkEnableOption "nixward daemon for continuous NixOS awareness";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.nixward-daemon or (throw "nixward-daemon package not found; add the symthaea overlay");
      description = "The nixward-daemon package to use.";
    };

    snapshotInterval = lib.mkOption {
      type = lib.types.int;
      default = 60;
      description = "Seconds between full system state snapshots.";
    };

    pollInterval = lib.mkOption {
      type = lib.types.int;
      default = 5;
      description = "Seconds between journal poll cycles.";
    };

    surpriseThreshold = lib.mkOption {
      type = lib.types.float;
      default = 0.3;
      description = "Prediction error threshold for episodic storage (0.0-1.0).";
    };

    stateDir = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/nixward";
      description = "Directory for persistent daemon state (world model, causal graph, episodes).";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "nixward";
      description = "User account under which the daemon runs.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "nixward";
      description = "Group under which the daemon runs.";
    };

    support = {
      watchdog = {
        enable = lib.mkOption {
          type = lib.types.bool;
          default = true;
          description = "Enable post-rebuild watchdog monitoring.";
        };

        timeout = lib.mkOption {
          type = lib.types.str;
          default = "5m";
          description = "Maximum monitoring duration before declaring stable.";
        };

        consecutiveFailures = lib.mkOption {
          type = lib.types.int;
          default = 3;
          description = "Number of consecutive degraded checks before reverting.";
        };
      };

      predictive = {
        enable = lib.mkOption {
          type = lib.types.bool;
          default = true;
          description = "Enable LTC-based predictive failure monitoring.";
        };

        horizons = lib.mkOption {
          type = lib.types.listOf lib.types.str;
          default = [ "1h" "6h" "24h" "7d" ];
          description = "Time horizons for predictions.";
        };
      };

      autonomyLevel = lib.mkOption {
        type = lib.types.enum [ "advisory" "semi-autonomous" "full-autonomous" ];
        default = "advisory";
        description = ''
          advisory: suggest only, never execute.
          semi-autonomous: auto-execute safe operations (GC, optimise).
          full-autonomous: auto-execute all except switch (still requires watchdog approval).
        '';
      };
    };

    ollama = {
      endpoint = lib.mkOption {
        type = lib.types.str;
        default = "http://localhost:11434";
        description = "Ollama API endpoint for LLM fallback queries.";
      };

      model = lib.mkOption {
        type = lib.types.str;
        default = "gemma3:1b";
        description = "Primary Ollama model (approved: gemma3:1b, qwen3:1.7b, gemma4:e2b, mistral:7b).";
      };

      timeout = lib.mkOption {
        type = lib.types.int;
        default = 30;
        description = "Ollama request timeout in seconds.";
      };
    };

    knowledgeLearning = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable dynamic knowledge learning from resolved incidents.";
    };
  };

  config = lib.mkIf cfg.enable {
    # Create dedicated user/group
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      home = cfg.stateDir;
      description = "nixward daemon user";
    };
    users.groups.${cfg.group} = {};

    # Ensure state directory exists
    systemd.tmpfiles.rules = [
      "d ${cfg.stateDir} 0750 ${cfg.user} ${cfg.group} -"
    ];

    # Write config file
    environment.etc."nixward/config.json".text = builtins.toJSON {
      snapshot_interval = cfg.snapshotInterval;
      poll_interval = cfg.pollInterval;
      surprise_threshold = cfg.surpriseThreshold;
      state_dir = cfg.stateDir;
      ollama_endpoint = cfg.ollama.endpoint;
      ollama_model = cfg.ollama.model;
      ollama_timeout = cfg.ollama.timeout;
      enable_knowledge_learning = cfg.knowledgeLearning;
      support = {
        watchdog = {
          enable = cfg.support.watchdog.enable;
          timeout = cfg.support.watchdog.timeout;
          consecutive_failures = cfg.support.watchdog.consecutiveFailures;
        };
        predictive = {
          enable = cfg.support.predictive.enable;
          horizons = cfg.support.predictive.horizons;
        };
        autonomy_level = cfg.support.autonomyLevel;
      };
    };

    # Systemd service
    systemd.services.nixward = {
      description = "nixward: Continuous NixOS Awareness Daemon";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" "systemd-journald.service" ];

      serviceConfig = {
        Type = "simple";
        ExecStart = "${cfg.package}/bin/nixward-daemon";
        User = cfg.user;
        Group = cfg.group;
        Restart = "on-failure";
        RestartSec = 10;

        # State directory
        StateDirectory = "nixward";
        StateDirectoryMode = "0750";

        # Read-only access to system state
        ReadOnlyPaths = [
          "/nix/store"
          "/nix/var/nix/profiles"
          "/etc/nixos"
          "/run/systemd"
        ];

        # Hardening
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        NoNewPrivileges = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictNamespaces = true;
        LockPersonality = true;
        RestrictRealtime = true;
        MemoryDenyWriteExecute = true;

        # Allow systemctl and journalctl queries
        CapabilityBoundingSet = "";
        SystemCallFilter = [ "@system-service" "~@privileged" ];
      };

      environment = {
        XDG_DATA_HOME = cfg.stateDir;
        NIXWARD_CONFIG = "/etc/nixward/config.json";
      };
    };
  };
}
