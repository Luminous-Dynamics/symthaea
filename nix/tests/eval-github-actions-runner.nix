# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Eval-only test for the Symthaea GitHub Actions runner module. This never
# contacts GitHub and never requires a real token.

{ pkgs }:

let
  lib = pkgs.lib;
  module = ../modules/github-actions-runner.nix;
  fakeToken = "/run/secrets/github-runner/test-token";
  storeToken = builtins.toFile "symthaea-runner-test-token" "not-a-real-secret";
  storeTokenString = toString storeToken;
  upstreamService = "${pkgs.path}/nixos/modules/services/continuous-integration/github-runner/service.nix";

  evalWith = runnerConfig:
    import "${pkgs.path}/nixos/lib/eval-config.nix" {
      system = pkgs.stdenv.hostPlatform.system;
      modules = [
        module
        ({ ... }: {
          services.symthaea-ci-runner = runnerConfig;
        })
      ];
    };

  evaluated = evalWith {
    enable = true;
    tokenFile = fakeToken;
  };

  blankNameEval = evalWith {
    enable = true;
    tokenFile = fakeToken;
    name = "";
  };

  runner = evaluated.config.services.github-runners."symthaea-validation";
  service = evaluated.config.systemd.services."github-runner-symthaea-validation";
  publicOptions = evaluated.options.services.symthaea-ci-runner;
  tokenFileType = publicOptions.tokenFile.type;
  firstExecStartPre = builtins.elemAt service.serviceConfig.ExecStartPre 0;

  hasFailedAssertion = needle: assertions:
    lib.any (
      entry: (!entry.assertion) && lib.hasInfix needle entry.message
    ) assertions;
in
pkgs.runCommand "eval-github-actions-runner" { } ''
  # Registration and routing contract.
  test '${runner.url}' = 'https://github.com/Luminous-Dynamics/symthaea'
  test '${runner.name}' = 'symthaea-nixos-validation'
  test '${runner.tokenType}' = 'access'
  test '${if runner.ephemeral then "true" else "false"}' = 'true'
  test '${if runner.replace then "true" else "false"}' = 'true'
  test '${if runner.noDefaultLabels then "true" else "false"}' = 'true'
  test '${if runner.runnerGroup == null then "null" else runner.runnerGroup}' = 'null'
  test '${if runner.workDir == null then "null" else runner.workDir}' = 'null'
  test '${lib.concatStringsSep "," runner.extraLabels}' = 'symthaea-trusted-cpu-v1'
  test '${lib.concatStringsSep "," runner.nodeRuntimes}' = 'node24'
  test '${toString (builtins.length runner.extraPackages)}' = '0'

  # The Symthaea-specific API must expose only the minimal host knobs. Routing,
  # lifecycle, token mode, labels, packages, and runner groups are fixed.
  test '${lib.concatStringsSep "," (builtins.attrNames publicOptions)}' = 'enable,name,tokenFile'

  # Secret-path safety is a type-system invariant, not only a later assertion:
  # accept an external absolute string, reject both a true Nix path and a string
  # naming a Nix-store path. `externalPath` also rejects relative paths.
  test '${if tokenFileType.check fakeToken then "true" else "false"}' = 'true'
  test '${if tokenFileType.check storeToken then "true" else "false"}' = 'false'
  test '${if tokenFileType.check storeTokenString then "true" else "false"}' = 'false'
  test '${if tokenFileType.check "relative/token" then "true" else "false"}' = 'false'

  # Non-path policy still fails closed through explicit assertions.
  test '${if hasFailedAssertion "name must be non-empty" blankNameEval.config.assertions then "true" else "false"}' = 'true'

  # Ephemeral runner registration is not enough by itself. Freeze the pinned
  # nixpkgs lifecycle that backs Symthaea's current persistent-appliance recovery
  # profile: state/runtime locations are service-private, the work directory is
  # runtime-backed, and successful one-job deregistration causes a clean
  # re-registration cycle on the same host.
  test '${service.serviceConfig.Restart}' = 'on-success'
  test '${lib.concatStringsSep "," service.serviceConfig.RuntimeDirectory}' = 'github-runner/symthaea-validation'
  test '${lib.concatStringsSep "," service.serviceConfig.StateDirectory}' = 'github-runner/symthaea-validation'
  test '${service.serviceConfig.WorkingDirectory}' = '%t/github-runner/symthaea-validation'

  # Pinned nixpkgs systemd hardening contract.
  test '${if service.serviceConfig.DynamicUser then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.PrivateDevices then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.PrivateMounts then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.PrivateUsers then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.PrivateTmp then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.ProtectClock then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.ProtectControlGroups then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.ProtectHome then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.ProtectHostname then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.ProtectKernelLogs then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.ProtectKernelModules then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.NoNewPrivileges then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.RemoveIPC then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.RestrictNamespaces then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.RestrictRealtime then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.RestrictSUIDSGID then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.ProtectKernelTunables then "true" else "false"}' = 'true'
  test '${service.serviceConfig.ProtectSystem}' = 'strict'
  test '${service.serviceConfig.ProtectProc}' = 'invisible'
  test '${service.serviceConfig.UMask}' = '0066'
  test '${service.serviceConfig.StateDirectoryMode}' = '0700'

  syscall_filter='${lib.concatStringsSep "," service.serviceConfig.SystemCallFilter}'
  echo "$syscall_filter" | grep -F '~@mount'
  echo "$syscall_filter" | grep -F '~@module'
  echo "$syscall_filter" | grep -F '~@raw-io'
  echo "$syscall_filter" | grep -F '~@reboot'

  # A root-owned 0400/0600 token file is intentionally supported. The pinned
  # nixpkgs service performs its first pre-start stage as root, copies the
  # credential into the private state directory, then configures the DynamicUser
  # runner from that temporary copy.
  test '${if lib.hasPrefix "+" firstExecStartPre then "true" else "false"}' = 'true'

  # Network is intentionally available for GitHub/Nix/Cargo access, but the
  # allowed address-family set must not grow to raw packet sockets.
  test '${if service.serviceConfig.PrivateNetwork then "true" else "false"}' = 'false'
  families='${lib.concatStringsSep "," service.serviceConfig.RestrictAddressFamilies}'
  echo "$families" | grep -F 'AF_INET'
  echo "$families" | grep -F 'AF_INET6'
  echo "$families" | grep -F 'AF_UNIX'
  if echo "$families" | grep -F 'AF_PACKET'; then
    echo 'trusted CPU runner unexpectedly permits AF_PACKET' >&2
    exit 1
  fi

  # Both the original external token and the persistent comparison copy must be
  # inaccessible to the job process after bootstrap.
  inaccessible='${lib.concatStringsSep "," service.serviceConfig.InaccessiblePaths}'
  echo "$inaccessible" | grep -F -- '-${fakeToken}'
  echo "$inaccessible" | grep -F -- '.current-token'

  # Protect the pinned upstream bootstrap and cleanup lifecycle itself. The root
  # pre-start copies the original token, keeps a private comparison copy, removes
  # the temporary registration copy after configure, clears runner state in
  # ephemeral mode, clears the runtime-backed work directory on every start,
  # restarts after successful one-job deregistration, and disables self-update.
  grep -F -- 'install --mode=666' '${upstreamService}'
  grep -F -- 'install --mode=600' '${upstreamService}'
  grep -F 'rm "' '${upstreamService}' | grep -F 'newConfigTokenPath'
  grep -F -- 'In ephemeral mode, we always want to start with a clean state' '${upstreamService}'
  grep -F -- 'find -H "$WORK_DIRECTORY" -mindepth 1 -delete' '${upstreamService}'
  grep -F -- 'Restart = if cfg.ephemeral then "on-success" else "no";' '${upstreamService}'
  grep -F -- '--disableupdate' '${upstreamService}'

  touch "$out"
''
