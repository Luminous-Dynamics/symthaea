# NixOS module for the Mycelix Mesh Bridge service.
#
# Runs the mesh-bridge binary alongside the Holochain conductor,
# relaying TEND exchanges, food harvests, and emergency messages
# over LoRa (SX1276) and/or B.A.T.M.A.N. mesh when internet is down.
#
# Usage in configuration.nix:
#   imports = [ ./mesh-bridge.nix ];
#   services.mycelix-mesh-bridge.enable = true;

{ config, lib, pkgs, ... }:

let
  cfg = config.services.mycelix-mesh-bridge;
in {
  options.services.mycelix-mesh-bridge = {
    enable = lib.mkEnableOption "Mycelix Mesh Bridge (LoRa/BATMAN relay)";

    package = lib.mkOption {
      type = lib.types.package;
      description = "The mesh-bridge binary package.";
    };

    conductorUrl = lib.mkOption {
      type = lib.types.str;
      default = "ws://127.0.0.1:8888";
      description = "AppWebsocket URL for the local Holochain conductor.";
    };

    pollIntervalSecs = lib.mkOption {
      type = lib.types.int;
      default = 10;
      description = "Seconds between conductor poll cycles.";
    };

    meshTransport = lib.mkOption {
      type = lib.types.enum [ "loopback" "lora" "batman" ];
      default = "lora";
      description = "Mesh transport backend.";
    };

    loraDevice = lib.mkOption {
      type = lib.types.str;
      default = "/dev/spidev0.0";
      description = "SPI device path for SX1276 LoRa HAT.";
    };

    loraFrequencyMhz = lib.mkOption {
      type = lib.types.int;
      default = 868;
      description = "LoRa frequency in MHz (868 for EU/ZA, 915 for US).";
    };

    batmanInterface = lib.mkOption {
      type = lib.types.str;
      default = "bat0";
      description = "B.A.T.M.A.N. mesh interface name.";
    };

    appTokenFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Path to file containing the MESH_APP_TOKEN for conductor auth.";
    };

    encryptionKeyFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Path to file containing the 64-char hex MESH_ENCRYPTION_KEY for PSK mesh encryption.";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "mycelix";
      description = "User to run the mesh-bridge service as.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "mycelix";
      description = "Group to run the mesh-bridge service as.";
    };
  };

  config = lib.mkIf cfg.enable {
    # Ensure the user/group exist
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      extraGroups = [ "spi" "gpio" ] ++ lib.optional (cfg.meshTransport == "lora") "dialout";
    };
    users.groups.${cfg.group} = {};

    # SPI access for LoRa HAT
    services.udev.extraRules = lib.mkIf (cfg.meshTransport == "lora") ''
      SUBSYSTEM=="spidev", GROUP="spi", MODE="0660"
    '';

    systemd.services.mycelix-mesh-bridge = {
      description = "Mycelix Mesh Bridge — LoRa/BATMAN relay for offline resilience";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" "holochain-conductor.service" ];
      wants = [ "holochain-conductor.service" ];

      environment = {
        RUST_LOG = "mesh_bridge=info";
        CONDUCTOR_URL = cfg.conductorUrl;
        POLL_INTERVAL_SECS = toString cfg.pollIntervalSecs;
        MESH_TRANSPORT = cfg.meshTransport;
        LORA_DEVICE = cfg.loraDevice;
        LORA_FREQUENCY_MHZ = toString cfg.loraFrequencyMhz;
        BATMAN_INTERFACE = cfg.batmanInterface;
      };

      serviceConfig = {
        ExecStart = "${cfg.package}/bin/mesh-bridge";
        Restart = "always";
        RestartSec = 10;
        Type = "notify";
        WatchdogSec = 90;
        StateDirectory = "mycelix-mesh-bridge";

        User = cfg.user;
        Group = cfg.group;

        # Read secrets from file(s) if provided
        EnvironmentFile =
          lib.optional (cfg.appTokenFile != null) cfg.appTokenFile
          ++ lib.optional (cfg.encryptionKeyFile != null) cfg.encryptionKeyFile;

        # Hardening
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectControlGroups = true;
        RestrictNamespaces = true;
        RestrictRealtime = true;
        MemoryDenyWriteExecute = true;

        # Allow SPI device access for LoRa
        DeviceAllow = lib.mkIf (cfg.meshTransport == "lora") [ cfg.loraDevice ];

        # Log rotation via journal
        StandardOutput = "journal";
        StandardError = "journal";
        SyslogIdentifier = "mycelix-mesh-bridge";
      };
    };
  };
}
