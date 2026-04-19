# Xenia — PAM / verifiable-consent remote support (NixOS skeleton)
#
# Not yet wired: this declares the options surface that the admin console and
# xenia-ledger will consume. `config` fills in once the xenia-capture crate,
# ML-KEM handshake wiring, consent state machine, and ledger crate land in W0.

{ config, lib, ... }:

let
  inherit (lib) mkEnableOption mkOption types;
  suiteCfg = config.services.mycelix-sovereign;
in
{
  options.services.mycelix-sovereign.xenia = {
    enable = mkEnableOption "Xenia PAM service";

    listenPort = mkOption {
      type = types.port;
      default = 7420;
      description = "Port on which the Xenia peer daemon listens.";
    };

    adminConsolePort = mkOption {
      type = types.port;
      default = 8420;
      description = "Port on which the Xenia admin console (Leptos) is served.";
    };

    ledgerPath = mkOption {
      type = types.path;
      default = "${suiteCfg.dataDir}/tenants/${suiteCfg.tenantId}/xenia/ledger";
      defaultText = "<dataDir>/tenants/<tenantId>/xenia/ledger";
      description = ''
        Directory where xenia-ledger stores its append-only, hash-chained,
        signed consent log. This directory MUST be backed up — it is the
        third-party-verifiable record of every privileged session.
      '';
    };

    captureBackend = mkOption {
      type = types.enum [ "scap" "xcap" ];
      default = "scap";
      description = ''
        Screen-capture backend. See docs/adr/0001-screen-capture-backend.md
        for the rationale. "scap" is the default for cross-platform Wayland
        support; "xcap" is the fallback if scap's beta API proves unstable.
      '';
    };
  };

  # config block intentionally omitted at this commit — Xenia services are
  # pending implementation of xenia-capture, xenia-ledger, and the admin
  # console. See MYCELIX_SOVEREIGN_PLAN.md §6 W0.
}
