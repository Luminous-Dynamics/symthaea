# Identity — DID + MFA + verifiable credentials (NixOS skeleton)
#
# Not yet wired. Pending W1 Stream A (DID login, cross-cluster auth context,
# admin console integration).

{ config, lib, ... }:

let
  inherit (lib) mkEnableOption mkOption types;
  suiteCfg = config.services.mycelix-sovereign;
in
{
  options.services.mycelix-sovereign.identity = {
    enable = mkEnableOption "Mycelix Identity service (DID + MFA + VC)";

    didMethod = mkOption {
      type = types.str;
      default = "did:mycelix";
      description = ''
        DID method prefix. Default "did:mycelix" uses our own method spec
        for Holochain-anchored DIDs. Other methods (did:web, did:key) may be
        accepted on import but not minted.
      '';
    };

    dataPath = mkOption {
      type = types.path;
      default = "${suiteCfg.dataDir}/tenants/${suiteCfg.tenantId}/identity";
      defaultText = "<dataDir>/tenants/<tenantId>/identity";
      description = "Directory for Identity persistent state (keystore, VC store, MFA secrets).";
    };

    mfaRequired = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Whether MFA is required for all DID logins. Default true — required
        for NIS2 Article 21 compliance. Operators may disable for development
        or bootstrapping only.
      '';
    };
  };

  # config block intentionally omitted — see MYCELIX_SOVEREIGN_PLAN.md §6 W1.
}
