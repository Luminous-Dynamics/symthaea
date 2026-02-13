{ config, lib, pkgs, ... }:

let
  cfg = config.services.nix-mind;
in {
  options.services.nix-mind = {
    enable = lib.mkEnableOption "nix-mind daemon for continuous NixOS awareness";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.nix-mind-daemon or (throw "nix-mind-daemon package not found; add the symthaea overlay");
      description = "The nix-mind-daemon package to use.";
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
      default = "/var/lib/nix-mind";
      description = "Directory for persistent daemon state (world model, causal graph, episodes).";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "nix-mind";
      description = "User account under which the daemon runs.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "nix-mind";
      description = "Group under which the daemon runs.";
    };
  };

  config = lib.mkIf cfg.enable {
    # Create dedicated user/group
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      home = cfg.stateDir;
      description = "nix-mind daemon user";
    };
    users.groups.${cfg.group} = {};

    # Ensure state directory exists
    systemd.tmpfiles.rules = [
      "d ${cfg.stateDir} 0750 ${cfg.user} ${cfg.group} -"
    ];

    # Write config file
    environment.etc."nix-mind/config.json".text = builtins.toJSON {
      snapshot_interval = cfg.snapshotInterval;
      poll_interval = cfg.pollInterval;
      surprise_threshold = cfg.surpriseThreshold;
      state_dir = cfg.stateDir;
    };

    # Systemd service
    systemd.services.nix-mind = {
      description = "nix-mind: Continuous NixOS Awareness Daemon";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" "systemd-journald.service" ];

      serviceConfig = {
        Type = "simple";
        ExecStart = "${cfg.package}/bin/nix-mind-daemon";
        User = cfg.user;
        Group = cfg.group;
        Restart = "on-failure";
        RestartSec = 10;

        # State directory
        StateDirectory = "nix-mind";
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
        NIX_MIND_CONFIG = "/etc/nix-mind/config.json";
      };
    };
  };
}
