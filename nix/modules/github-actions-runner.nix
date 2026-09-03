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

{ config, lib, pkgs, ... }:

let
  cfg = config.services.symthaea-ci-runner;
  runnerKey = "symthaea-validation";
  repositoryUrl = "https://github.com/Luminous-Dynamics/symthaea";
  repositoryName = "Luminous-Dynamics/symthaea";
  repositoryId = "1136141775";
  capabilityLabel = "symthaea-trusted-cpu-v1";
  tokenPath = if cfg.tokenFile == null then "/dev/null" else toString cfg.tokenFile;
  trustedHarnessCommit =
    if cfg.trustedHarnessCommit == null then "" else cfg.trustedHarnessCommit;

  credentialPreflight = pkgs.writeShellScript "symthaea-ci-runner-credential-preflight" ''
    set -euo pipefail

    token=${lib.escapeShellArg tokenPath}

    if [ ! -f "$token" ]; then
      echo 'Symthaea CI runner credential must be a regular file' >&2
      exit 1
    fi

    owner_uid="$(${pkgs.coreutils}/bin/stat -Lc '%u' -- "$token")"
    if [ "$owner_uid" != '0' ]; then
      echo 'Symthaea CI runner credential must be owned by root' >&2
      exit 1
    fi

    mode="$(${pkgs.coreutils}/bin/stat -Lc '%a' -- "$token")"
    case "$mode" in
      400|600) ;;
      *)
        echo 'Symthaea CI runner credential mode must be exactly 0400 or 0600' >&2
        exit 1
        ;;
    esac

    size="$(${pkgs.coreutils}/bin/stat -Lc '%s' -- "$token")"
    if [ "$size" -le 0 ] || [ "$size" -gt 4096 ]; then
      echo 'Symthaea CI runner credential must be non-empty and at most 4096 bytes' >&2
      exit 1
    fi

    # GitHub access tokens are single printable, non-whitespace strings. Reject
    # newlines and all other whitespace/control bytes without ever echoing,
    # hashing, or otherwise reproducing the credential value.
    if [ "$(${pkgs.coreutils}/bin/wc -l < "$token")" -ne 0 ]; then
      echo 'Symthaea CI runner credential must not contain a newline' >&2
      exit 1
    fi
    if ! LC_ALL=C ${pkgs.gnugrep}/bin/grep -Eq '^[[:graph:]]+$' "$token"; then
      echo 'Symthaea CI runner credential contains whitespace or control bytes' >&2
      exit 1
    fi
  '';

  # Defense in depth for the privileged self-hosted capability. GitHub executes
  # this hook synchronously after job assignment and before workflow steps. The
  # hook pins the reviewed harness commit deployed by the operator and rejects
  # every other repository/ref/event/workflow identity. This does not replace
  # server-side branch/ruleset protection; trusted main must still be protected.
  jobAdmissionHook = pkgs.writeShellScript "symthaea-ci-runner-job-admission" ''
    set -euo pipefail

    reject() {
      echo "Symthaea trusted-runner admission rejected: $1" >&2
      exit 1
    }

    expected_harness=${lib.escapeShellArg trustedHarnessCommit}

    [ "${GITHUB_REPOSITORY:-}" = ${lib.escapeShellArg repositoryName} ] \
      || reject 'wrong repository'
    [ "${GITHUB_REPOSITORY_ID:-}" = ${lib.escapeShellArg repositoryId} ] \
      || reject 'wrong repository id'
    [ "${GITHUB_SERVER_URL:-}" = 'https://github.com' ] \
      || reject 'wrong GitHub server'
    [ "${GITHUB_EVENT_NAME:-}" = 'workflow_dispatch' ] \
      || reject 'workflow is not manually dispatched'
    [ "${GITHUB_REF:-}" = 'refs/heads/main' ] \
      || reject 'workflow ref is not main'
    [ "${GITHUB_REF_TYPE:-}" = 'branch' ] \
      || reject 'workflow ref is not a branch'
    [ "${GITHUB_REF_PROTECTED:-}" = 'true' ] \
      || reject 'main is not protected by GitHub policy'
    [ -n "$expected_harness" ] \
      || reject 'trusted harness commit is not configured'
    [ "${GITHUB_SHA:-}" = "$expected_harness" ] \
      || reject 'job commit does not match deployed trusted harness'
    [ "${GITHUB_WORKFLOW_SHA:-}" = "$expected_harness" ] \
      || reject 'workflow commit does not match deployed trusted harness'

    case "${GITHUB_WORKFLOW_REF:-}" in
      '${repositoryName}/.github/workflows/self-hosted-runner-smoke.yml@refs/heads/main'|\
      '${repositoryName}/.github/workflows/self-hosted-ai-assurance-foundation-recovery.yml@refs/heads/main'|\
      '${repositoryName}/.github/workflows/self-hosted-ai-assurance-budget-recovery.yml@refs/heads/main'|\
      '${repositoryName}/.github/workflows/self-hosted-sym-arch-002a-core-recovery.yml@refs/heads/main')
        ;;
      *)
        reject 'workflow path is not in the trusted capability allowlist'
        ;;
    esac
  '';
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
        used to obtain short-lived runner registration tokens. For the v2
        static-secret deployment, use a fine-grained PAT restricted to
        Luminous-Dynamics/symthaea with only repository Administration: write
        permission. Keep the credential in a root-owned runtime secret file with
        mode exactly 0400 or 0600. The service rejects empty credentials and any
        credential containing whitespace or control bytes before registration.
      '';
    };

    trustedHarnessCommit = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "0123456789abcdef0123456789abcdef01234567";
      description = ''
        Exact reviewed main commit allowed to schedule onto the trusted CPU
        capability. Any later main commit fails the host-side pre-job admission
        hook until this pin is deliberately updated and the host is rebuilt.
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
      {
        assertion =
          cfg.trustedHarnessCommit != null
          && builtins.match "[0-9a-f]{40}" cfg.trustedHarnessCommit != null;
        message = "services.symthaea-ci-runner.trustedHarnessCommit must be an exact lowercase 40-hex Git commit";
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

    systemd.services."github-runner-${runnerKey}" = {
      # Host-owned, Nix-store-pinned gate executed by the GitHub runner before
      # any workflow-defined step. A non-zero exit rejects the job.
      environment.ACTIONS_RUNNER_HOOK_JOB_STARTED = toString jobAdmissionHook;

      # Enforce runtime credential invariants immediately before the pinned
      # upstream root bootstrap copies the access token into private runner state.
      # The leading '+' keeps this check in the same privileged ExecStartPre phase
      # as the upstream credential-copy lifecycle; the job process never receives
      # this privilege or access to the original token path.
      serviceConfig.ExecStartPre = lib.mkBefore [ "+${credentialPreflight}" ];
    };
  };
}
