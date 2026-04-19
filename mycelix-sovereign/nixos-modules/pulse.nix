# Pulse — PQC-encrypted operations email (NixOS skeleton)
#
# Not yet wired. Pending W1 Stream B (worktree merge to main) and W2 Stream B
# (PQC envelope sealing + epoch ratchet + real SMTP bridge).

{ config, lib, ... }:

let
  inherit (lib) mkEnableOption mkOption types;
  suiteCfg = config.services.mycelix-sovereign;
in
{
  options.services.mycelix-sovereign.pulse = {
    enable = mkEnableOption "Pulse PQC email service";

    smtpListenPort = mkOption {
      type = types.port;
      default = 25;
      description = "Port on which the Pulse SMTP bridge listens for outbound mail.";
    };

    imapListenPort = mkOption {
      type = types.port;
      default = 143;
      description = "Port on which the Pulse IMAP bridge listens.";
    };

    conductorAdminPort = mkOption {
      type = types.port;
      default = 33800;
      description = "Holochain conductor admin WebSocket port (shared with other Mycelix hApps).";
    };

    conductorAppPort = mkOption {
      type = types.port;
      default = 8888;
      description = "Holochain conductor app WebSocket port.";
    };

    dataPath = mkOption {
      type = types.path;
      default = "${suiteCfg.dataDir}/tenants/${suiteCfg.tenantId}/pulse";
      defaultText = "<dataDir>/tenants/<tenantId>/pulse";
      description = "Directory for Pulse persistent state (envelopes, keystore, ratchet state).";
    };
  };

  # config block intentionally omitted — see MYCELIX_SOVEREIGN_PLAN.md §6 W1+W2.
}
