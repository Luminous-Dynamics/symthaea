# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Credential-safe NixOS module for a repository-scoped GitHub Actions runner.
#
# This module intentionally exposes a very small API. Trust-boundary properties
# are fixed rather than configurable: repository scope, ephemeral lifecycle,
# access-token registration, default-label suppression, the unique CPU
# capability label, and the Stage-F one-time authorization consumer cannot be
# weakened by a host configuration typo.

{ config, lib, ... }:

let
  cfg = config.services.symthaea-ci-runner;
  runnerKey = "symthaea-validation";
  runnerService = "github-runner-${runnerKey}";
  repositoryUrl = "https://github.com/Luminous-Dynamics/symthaea";
  capabilityLabel = "symthaea-trusted-cpu-v1";
  authorizationGroup = "symthaea-stage-f-authorization";
  authorizationLedgerDir = "/var/lib/symthaea-stage-f-authorizations";
  authorizationSocket = "/run/symthaea-stage-f-authorization.sock";
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

    users.groups.${authorizationGroup} = { };

    # The ledger is deliberately outside the upstream runner StateDirectory,
    # because ephemeral runner startup wipes that state before re-registration.
    # Only the root-owned socket consumer can mutate ledger entries.
    systemd.tmpfiles.rules = [
      "d ${authorizationLedgerDir} 0700 root root - -"
    ];

    systemd.sockets.symthaea-stage-f-authorization = {
      description = "Symthaea Stage-F one-time authorization socket";
      wantedBy = [ "sockets.target" ];
      socketConfig = {
        ListenStream = authorizationSocket;
        SocketUser = "root";
        SocketGroup = authorizationGroup;
        SocketMode = "0660";
        RemoveOnStop = true;
        Accept = true;
      };
    };

    # One root-owned process per local socket connection. The protocol is tiny:
    #   BOOT_ID
    #   CONSUME <64-hex nonce> <64-hex authorization-sha256>
    #   STATUS  <64-hex nonce> <64-hex authorization-sha256>
    # A successful CONSUME atomically reserves the nonce directory, then writes
    # an atomic immutable record. STATUS is read-only: it can prove UNUSED,
    # CONSUMED_STATUS, CONSUMED_DIFFERENT, or INCOMPLETE without reopening
    # authority. Reusing the nonce therefore fails closed even across runner
    # re-registration, while a lost CONSUME response remains auditable.
    systemd.services."symthaea-stage-f-authorization@" = {
      description = "Consume or inspect one Symthaea Stage-F authorization";
      script = ''
        set -euo pipefail
        umask 077
        IFS= read -r request
        read -r operation nonce authorization_sha extra <<< "$request"
        boot_id="$(< /proc/sys/kernel/random/boot_id)"

        case "$operation" in
          BOOT_ID)
            [[ -z "$nonce" && -z "$authorization_sha" && -z "$extra" ]]
            [[ "$boot_id" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]
            printf 'BOOT_ID %s\n' "$boot_id"
            ;;
          CONSUME)
            [[ "$nonce" =~ ^[0-9a-f]{64}$ ]]
            [[ "$authorization_sha" =~ ^[0-9a-f]{64}$ ]]
            [[ -z "$extra" ]]
            marker="${authorizationLedgerDir}/$nonce"
            record="$marker/consumption-record"
            if mkdir --mode=0700 -- "$marker" 2>/dev/null; then
              tmp="$marker/.consumption-record"
              {
                printf 'authorization_sha256=%s\n' "$authorization_sha"
                printf 'boot_id=%s\n' "$boot_id"
              } > "$tmp"
              chmod 0400 "$tmp"
              mv -T -- "$tmp" "$record"
              printf 'CONSUMED %s %s %s\n' "$nonce" "$authorization_sha" "$boot_id"
            else
              printf 'ALREADY_CONSUMED %s %s\n' "$nonce" "$boot_id"
              exit 73
            fi
            ;;
          STATUS)
            [[ "$nonce" =~ ^[0-9a-f]{64}$ ]]
            [[ "$authorization_sha" =~ ^[0-9a-f]{64}$ ]]
            [[ -z "$extra" ]]
            marker="${authorizationLedgerDir}/$nonce"
            record="$marker/consumption-record"
            if [[ ! -d "$marker" ]]; then
              printf 'UNUSED %s %s\n' "$nonce" "$boot_id"
              exit 0
            fi
            if [[ ! -f "$record" ]]; then
              printf 'INCOMPLETE %s %s\n' "$nonce" "$boot_id"
              exit 0
            fi

            line1=''
            line2=''
            line3=''
            {
              IFS= read -r line1 || true
              IFS= read -r line2 || true
              IFS= read -r line3 || true
            } < "$record"
            key1=''
            stored_authorization_sha=''
            extra1=''
            key2=''
            stored_boot_id=''
            extra2=''
            IFS='=' read -r key1 stored_authorization_sha extra1 <<< "$line1"
            IFS='=' read -r key2 stored_boot_id extra2 <<< "$line2"

            if [[ "$key1" != 'authorization_sha256' || ! "$stored_authorization_sha" =~ ^[0-9a-f]{64}$ || -n "$extra1" || "$key2" != 'boot_id' || ! "$stored_boot_id" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ || -n "$extra2" || -n "$line3" ]]; then
              printf 'INCOMPLETE %s %s\n' "$nonce" "$boot_id"
              exit 0
            fi

            if [[ "$stored_authorization_sha" == "$authorization_sha" ]]; then
              printf 'CONSUMED_STATUS %s %s %s\n' "$nonce" "$stored_authorization_sha" "$stored_boot_id"
            else
              printf 'CONSUMED_DIFFERENT %s %s\n' "$nonce" "$stored_boot_id"
            fi
            ;;
          *)
            echo 'INVALID_REQUEST' >&2
            exit 64
            ;;
        esac
      '';
      serviceConfig = {
        Type = "simple";
        User = "root";
        Group = "root";
        UMask = "0077";
        StandardInput = "socket";
        StandardOutput = "socket";
        StandardError = "journal";
        NoNewPrivileges = true;
        PrivateDevices = true;
        PrivateTmp = true;
        ProtectHome = true;
        ProtectSystem = "strict";
        ProtectKernelLogs = true;
        ProtectKernelModules = true;
        ProtectKernelTunables = true;
        ProtectControlGroups = true;
        RestrictRealtime = true;
        RestrictSUIDSGID = true;
        LockPersonality = true;
        MemoryDenyWriteExecute = true;
        RestrictAddressFamilies = [ "AF_UNIX" ];
        ReadWritePaths = [ authorizationLedgerDir ];
      };
    };

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

    # The DynamicUser runner can reach only the local authorization socket via a
    # fixed supplementary group. It has no direct filesystem access to the
    # root-owned persistent ledger.
    systemd.services.${runnerService}.serviceConfig.SupplementaryGroups =
      lib.mkAfter [ authorizationGroup ];
  };
}
