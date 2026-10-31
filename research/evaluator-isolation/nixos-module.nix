# Research-only NixOS module for the Symthaea temporal evaluator.
#
# Import this module explicitly. It is not wired into the repository's default
# NixOS configuration and does not change production Symthaea behavior.
#
# The three executables are supplied by the operator. The generator receives
# only its own seed credential; the evaluator receives the signing credential.
# The candidate process does not receive either credential or any oracle path.
{ config, lib, ... }:

let
  inherit (lib) mkEnableOption mkIf mkOption types;
  cfg = config.services.symthaea-temporal-isolation;
  state = cfg.stateDirectory;

  commonSandbox = {
    Type = "oneshot";
    UMask = "0027";

    ProtectSystem = "strict";
    ProtectHome = true;
    PrivateTmp = true;
    PrivateDevices = true; # CPU-only qualification profile; GPU requires a separately reviewed device allow-list.
    PrivateNetwork = true;
    PrivateIPC = true;
    ProtectHostname = true;
    RestrictAddressFamilies = [ "AF_UNIX" ];

    NoNewPrivileges = true;
    CapabilityBoundingSet = "";
    AmbientCapabilities = "";
    RestrictSUIDSGID = true;
    LockPersonality = true;
    MemoryDenyWriteExecute = true;
    RestrictNamespaces = true;

    ProtectKernelTunables = true;
    ProtectKernelModules = true;
    ProtectControlGroups = true;
    ProtectClock = true;
    ProtectProc = "invisible";
    ProcSubset = "pid";

    SystemCallArchitectures = "native";
    # Prevent a service from undoing the mount namespace protections.
    SystemCallFilter = [ "@system-service" "~@mount" ];
    SystemCallErrorNumber = "EPERM";
  };

  commonInvisiblePaths = [
    # PrivateNetwork does not by itself hide host AF_UNIX sockets.
    "-/run/dbus/system_bus_socket"
    "-/run/systemd/private"
    "-/run/systemd/journal/socket"
    "-/run/systemd/journal/stdout"
    "-/run/user"
    # Particularly important on NixOS: do not let an untrusted candidate
    # contact the privileged Nix daemon or common container/host daemons.
    "-/nix/var/nix/daemon-socket/socket"
    "-/run/nix/daemon-socket/socket"
    "-/run/docker.sock"
    "-/var/run/docker.sock"
    "-/run/containerd/containerd.sock"
    "-/run/udev/control"
    "-/run/NetworkManager/private"
    "-/run/avahi-daemon/socket"
    "-/run/podman/podman.sock"
    "-/etc/credstore"
    "-/etc/credstore.encrypted"
    "-/run/credstore"
    "-/run/credstore.encrypted"
  ];
in
{
  options.services.symthaea-temporal-isolation = {
    enable = mkEnableOption "research-only isolated Symthaea temporal benchmark services";

    stateDirectory = mkOption {
      type = types.str;
      default = "/var/lib/symthaea-temporal-eval";
      description = "Dedicated state root for public tasks, oracle, candidate exchange, evaluator-private data and receipts.";
    };

    taskgenExecutable = mkOption {
      type = types.str;
      default = "";
      description = "Absolute path to the trusted task generator executable.";
    };

    candidateExecutable = mkOption {
      type = types.str;
      default = "";
      description = "Absolute path to the untrusted candidate runner executable.";
    };

    evaluatorExecutable = mkOption {
      type = types.str;
      default = "";
      description = "Absolute path to the trusted evaluator executable.";
    };

    generatorSeedCredential = mkOption {
      type = types.str;
      default = "/etc/credstore.encrypted/symthaea-temporal-generator-seed";
      description = "Encrypted credential path delivered only to the task generator service.";
    };

    evaluatorSigningKeyCredential = mkOption {
      type = types.str;
      default = "/etc/credstore.encrypted/symthaea-temporal-evaluator-signing-key";
      description = "Encrypted evaluator signing credential delivered only to the evaluator service.";
    };
  };

  config = mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.taskgenExecutable != "" && lib.hasPrefix "/" cfg.taskgenExecutable;
        message = "services.symthaea-temporal-isolation.taskgenExecutable must be an absolute executable path.";
      }
      {
        assertion = cfg.candidateExecutable != "" && lib.hasPrefix "/" cfg.candidateExecutable;
        message = "services.symthaea-temporal-isolation.candidateExecutable must be an absolute executable path.";
      }
      {
        assertion = cfg.evaluatorExecutable != "" && lib.hasPrefix "/" cfg.evaluatorExecutable;
        message = "services.symthaea-temporal-isolation.evaluatorExecutable must be an absolute executable path.";
      }
      {
        assertion = lib.hasPrefix "/" cfg.stateDirectory;
        message = "services.symthaea-temporal-isolation.stateDirectory must be absolute.";
      }
    ];

    users.groups.symthaea-temporal-public = {};
    users.groups.symthaea-temporal-oracle = {};
    users.groups.symthaea-temporal-exchange = {};
    users.groups.symthaea-temporal-generator = {};
    users.groups.symthaea-temporal-candidate = {};
    users.groups.symthaea-temporal-evaluator = {};

    users.users.symthaea-temporal-generator = {
      isSystemUser = true;
      group = "symthaea-temporal-generator";
      extraGroups = [ "symthaea-temporal-public" "symthaea-temporal-oracle" ];
    };

    users.users.symthaea-temporal-candidate = {
      isSystemUser = true;
      group = "symthaea-temporal-candidate";
      extraGroups = [ "symthaea-temporal-public" "symthaea-temporal-exchange" ];
    };

    users.users.symthaea-temporal-evaluator = {
      isSystemUser = true;
      group = "symthaea-temporal-evaluator";
      extraGroups = [
        "symthaea-temporal-public"
        "symthaea-temporal-oracle"
        "symthaea-temporal-exchange"
      ];
    };

    systemd.tmpfiles.rules = [
      "d ${state} 0755 root root -"
      # Task generator owns writes; candidate and evaluator can read public task files.
      "d ${state}/public 2750 symthaea-temporal-generator symthaea-temporal-public -"
      # Candidate has no oracle group membership; evaluator can read oracle files.
      "d ${state}/oracle 2750 symthaea-temporal-generator symthaea-temporal-oracle -"
      # Candidate owns exchange writes. Evaluator group gets search-only directory access
      # and reads the single run-bound filename, not arbitrary directory listings.
      "d ${state}/exchange 2710 symthaea-temporal-candidate symthaea-temporal-exchange -"
      "d ${state}/candidate-work 0700 symthaea-temporal-candidate symthaea-temporal-candidate -"
      "d ${state}/evaluator-private 0700 symthaea-temporal-evaluator symthaea-temporal-evaluator -"
      "d ${state}/evaluator-private/replay-ledger 0700 symthaea-temporal-evaluator symthaea-temporal-evaluator -"
      "d ${state}/receipts 0700 symthaea-temporal-evaluator symthaea-temporal-evaluator -"
    ];

    systemd.services.symthaea-temporal-taskgen = {
      description = "Symthaea temporal benchmark task generator (trusted)";
      after = [ "local-fs.target" ];
      serviceConfig = commonSandbox // {
        User = "symthaea-temporal-generator";
        Group = "symthaea-temporal-generator";
        ExecStart = cfg.taskgenExecutable;
        ReadWritePaths = [ "${state}/public" "${state}/oracle" ];
        InaccessiblePaths = [
          "${state}/exchange"
          "${state}/candidate-work"
          "${state}/evaluator-private"
          "${state}/receipts"
        ] ++ commonInvisiblePaths;
        LoadCredentialEncrypted = "taskgen-seed:${cfg.generatorSeedCredential}";
        StandardInput = "null";
        StandardOutput = "journal";
        StandardError = "journal";
        TimeoutStartSec = "10min";
        KillMode = "control-group";
      };
    };

    systemd.services.symthaea-temporal-candidate = {
      description = "Symthaea temporal benchmark candidate (untrusted)";
      requires = [ "symthaea-temporal-taskgen.service" ];
      after = [ "symthaea-temporal-taskgen.service" ];
      serviceConfig = commonSandbox // {
        User = "symthaea-temporal-candidate";
        Group = "symthaea-temporal-candidate";
        ExecStart = cfg.candidateExecutable;
        ReadOnlyPaths = [ "${state}/public" ];
        ReadWritePaths = [ "${state}/exchange" "${state}/candidate-work" ];
        InaccessiblePaths = [
          "${state}/oracle"
          "${state}/evaluator-private"
          "${state}/receipts"
        ] ++ commonInvisiblePaths;
        # No evaluator credential, task seed, inherited socket activation or
        # network socket is configured for this service.
        StandardInput = "null";
        StandardOutput = "journal";
        StandardError = "journal";
        TimeoutStartSec = "30min";
        KillMode = "control-group";
      };
    };

    systemd.services.symthaea-temporal-evaluator = {
      description = "Symthaea temporal benchmark evaluator (trusted)";
      requires = [ "symthaea-temporal-candidate.service" ];
      after = [ "symthaea-temporal-candidate.service" ];
      serviceConfig = commonSandbox // {
        User = "symthaea-temporal-evaluator";
        Group = "symthaea-temporal-evaluator";
        ExecStart = cfg.evaluatorExecutable;
        ReadOnlyPaths = [
          "${state}/public"
          "${state}/oracle"
          "${state}/exchange"
        ];
        ReadWritePaths = [ "${state}/evaluator-private" "${state}/receipts" ];
        InaccessiblePaths = [ "${state}/candidate-work" ] ++ commonInvisiblePaths;
        LoadCredentialEncrypted = "receipt-signing-key:${cfg.evaluatorSigningKeyCredential}";
        StandardInput = "null";
        StandardOutput = "journal";
        StandardError = "journal";
        TimeoutStartSec = "10min";
        KillMode = "control-group";
      };
    };
  };
}
