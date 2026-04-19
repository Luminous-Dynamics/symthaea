# Mycelix Sovereign — NixOS module (top-level)
#
# Bundles the four suite components behind a single enable flag with shared
# configuration for data root, domain, and cross-component auth.
#
# Component modules are intentionally skeleton-level at this commit — they will
# be filled in per the W0→W3 plan in MYCELIX_SOVEREIGN_PLAN.md.

{ config, lib, pkgs, ... }:

let
  inherit (lib)
    mkEnableOption
    mkOption
    mkIf
    mkDefault
    types;
  cfg = config.services.mycelix-sovereign;
in
{
  imports = [
    ./xenia.nix
    ./pulse.nix
    ./athena.nix
    ./identity.nix
  ];

  options.services.mycelix-sovereign = {
    enable = mkEnableOption "Mycelix Sovereign — Secure Sovereign Operations Suite";

    domain = mkOption {
      type = types.str;
      example = "sovereign.example.org";
      description = ''
        Public domain at which the Suite is served.

        Used as the suffix for per-component service hostnames
        (e.g. admin.<domain>, mail.<domain>, xenia.<domain>).
      '';
    };

    dataDir = mkOption {
      type = types.path;
      default = "/var/lib/mycelix-sovereign";
      description = ''
        Root directory for Suite state. Per-tenant data lives in
        <dataDir>/tenants/<tenant-id>/<component>/.
      '';
    };

    tenantId = mkOption {
      type = types.str;
      default = "default";
      description = ''
        Tenant identifier. In year-1 self-hosted deployments this is typically
        "default". Multi-tenant deployments (year-2 managed tier) use one NixOS
        module per tenant — never shared control plane.
      '';
    };

    user = mkOption {
      type = types.str;
      default = "mycelix-sovereign";
      description = "System user that owns Suite services.";
    };

    group = mkOption {
      type = types.str;
      default = "mycelix-sovereign";
      description = "System group that owns Suite services.";
    };
  };

  config = mkIf cfg.enable {
    # When the Suite is enabled, enable each component by default. Operators
    # can disable individual components via services.mycelix-sovereign.<name>.enable = false.
    services.mycelix-sovereign.xenia.enable    = mkDefault true;
    services.mycelix-sovereign.pulse.enable    = mkDefault true;
    services.mycelix-sovereign.athena.enable   = mkDefault true;
    services.mycelix-sovereign.identity.enable = mkDefault true;

    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      home = cfg.dataDir;
      createHome = true;
      description = "Mycelix Sovereign service account";
    };
    users.groups.${cfg.group} = { };

    # Ensure per-tenant data directory exists with the correct ownership.
    systemd.tmpfiles.rules = [
      "d ${cfg.dataDir}                             0750 ${cfg.user} ${cfg.group} - -"
      "d ${cfg.dataDir}/tenants                     0750 ${cfg.user} ${cfg.group} - -"
      "d ${cfg.dataDir}/tenants/${cfg.tenantId}     0750 ${cfg.user} ${cfg.group} - -"
    ];
  };
}
