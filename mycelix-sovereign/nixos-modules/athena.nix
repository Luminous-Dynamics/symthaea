# Athena L1 — AI triage agent with sandboxed tool-use (NixOS skeleton)
#
# Not yet wired. Pending W2 Stream A (KB connector, ticket API, escalation
# handoff, per-tenant sandbox hardening).

{ config, lib, ... }:

let
  inherit (lib) mkEnableOption mkOption types;
  suiteCfg = config.services.mycelix-sovereign;
in
{
  options.services.mycelix-sovereign.athena = {
    enable = mkEnableOption "Athena L1 AI triage agent";

    ticketApiPort = mkOption {
      type = types.port;
      default = 8430;
      description = "Port on which the Athena ticket ingestion API listens.";
    };

    sandboxRoot = mkOption {
      type = types.path;
      default = "${suiteCfg.dataDir}/tenants/${suiteCfg.tenantId}/athena/sandbox";
      defaultText = "<dataDir>/tenants/<tenantId>/athena/sandbox";
      description = ''
        Per-tenant sandbox root for Athena's tool-use actions (read_file,
        list_dir, future actions). This directory is the only filesystem
        location Athena may read from or write to. Outside paths are rejected
        by the consciousness-gated action executor.
      '';
    };

    ollamaModel = mkOption {
      type = types.str;
      default = "gemma4:e2b";
      description = ''
        Default Ollama model for Athena's LLM backend. See approved model list
        in the parent CLAUDE.md. Alternatives: broca (Symthaea native) and
        liquid-mamba — both gated on feature flags.
      '';
    };

    knowledgeBase = mkOption {
      type = types.listOf types.str;
      default = [ ];
      example = [ "nist-sp-800-53" "iso-27001-annex-a" "customer-sops" ];
      description = ''
        Knowledge bases wired into Athena's system prompt via
        mycelix-knowledge (claims, graph, query, factcheck zomes). Empty by
        default — operators must explicitly opt in to KB seeding.
      '';
    };
  };

  # config block intentionally omitted — see MYCELIX_SOVEREIGN_PLAN.md §6 W2.
}
