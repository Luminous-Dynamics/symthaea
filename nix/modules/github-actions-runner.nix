# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Credential-safe NixOS module for a repository-scoped GitHub Actions runner.
#
# This module intentionally exposes a very small API. Trust-boundary properties
# are fixed rather than configurable: repository scope, ephemeral lifecycle,
# access-token registration, default-label suppression, and the unique CPU
# capability label cannot be weakened by a host configuration typo.

{ config, lib, ... }:

let
  cfg = config.services.symthaea-ci-runner;
  runnerKey = "symthaea-validation";
  repositoryUrl = "https://github.com/Luminous-Dynamics/symthaea";
  capabilityLabel = "symthaea-trusted-cpu-v1";
in
{
  options.services.symthaea-ci-runner = {
    enable = lib.mkEnableOption "ephemeral trusted-CPU Symthaea GitHub Actions runner";

    name = lib.mkOption {
      type = lib.types.str;
      default = "symthaea-nixos-validation";
      description = "GitHub-visible runner registration name; use a unique value per host.";
    };

    tokenFile = lib.mkOption {
      # externalPath accepts only absolute context-free strings outside the Nix
      # store. In particular, a true Nix path value cannot be copied into the
      # store accidentally through this API.
      type = lib.types.nullOr lib.types.externalPath;
      default = null;
      example = "/run/secrets/github-runner/symthaea-pat";
      description = ''
        External runtime path containing a repository-scoped GitHub access token
        used to obtain short-lived runner registration tokens. For the v1
        static-secret deployment, use a fine-grained PAT restricted to
        Luminous-Dynamics/symthaea with only repository Administration: write
        permission. Keep the credential in a root-owned runtime secret file;
        sops-nix or agenix are recommended. The file should not be group- or
        world-readable.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.tokenFile != null;
        message = "services.symthaea-ci-runner.tokenFile must point to an external runtime secret";
      }
      {
        assertion = cfg.name != "";
        message = "services.symthaea-ci-runner.name must be non-empty";
      }
    ];

    services.github-runners.${runnerKey} = {
      enable = true;
      url = repositoryUrl;
      name = cfg.name;
      tokenFile = cfg.tokenFile;

      # Fixed trust boundary. Do not make these host-configurable here.
      tokenType = "access";
      ephemeral = true;
      replace = true;
      noDefaultLabels = true;
      extraLabels = [ capabilityLabel ];

      # The trusted smoke uses no JavaScript actions. Keep only the current
      # pinned nixpkgs runtime available for future explicitly-reviewed trusted
      # workflows.
      nodeRuntimes = [ "node24" ];

      # Intentionally no Symthaea-specific ambient packages. The upstream
      # runner supplies bash/coreutils/git/tar/gzip/Nix; all build dependencies
      # live in a pinned per-job Nix shell.
      extraPackages = [ ];
    };
  };
}
