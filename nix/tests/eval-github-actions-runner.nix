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
  fakeHarnessCommit = "0123456789abcdef0123456789abcdef01234567";
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
    trustedHarnessCommit = fakeHarnessCommit;
  };

  blankNameEval = evalWith {
    enable = true;
    tokenFile = fakeToken;
    trustedHarnessCommit = fakeHarnessCommit;
    name = "";
  };

  missingHarnessEval = evalWith {
    enable = true;
    tokenFile = fakeToken;
  };

  malformedHarnessEval = evalWith {
    enable = true;
    tokenFile = fakeToken;
    trustedHarnessCommit = "not-a-git-commit";
  };

  runner = evaluated.config.services.github-runners."symthaea-validation";
  service = evaluated.config.systemd.services."github-runner-symthaea-validation";
  publicOptions = evaluated.options.services.symthaea-ci-runner;
  tokenFileType = publicOptions.tokenFile.type;
  firstExecStartPre = builtins.elemAt service.serviceConfig.ExecStartPre 0;
  secondExecStartPre = builtins.elemAt service.serviceConfig.ExecStartPre 1;
  credentialPreflight = lib.removePrefix "+" firstExecStartPre;
  jobAdmissionHook = service.environment.ACTIONS_RUNNER_HOOK_JOB_STARTED;

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
  # lifecycle, token mode, labels, packages, and runner groups are fixed. The
  # harness commit is deliberately explicit because any main advance must fail
  # closed until the trusted host is consciously re-pinned and rebuilt.
  test '${lib.concatStringsSep "," (builtins.attrNames publicOptions)}' = 'enable,name,tokenFile,trustedHarnessCommit'

  # Secret-path safety is a type-system invariant, not only a later assertion:
  # accept an external absolute string, reject both a true Nix path and a string
  # naming a Nix-store path. `externalPath` also rejects relative paths.
  test '${if tokenFileType.check fakeToken then "true" else "false"}' = 'true'
  test '${if tokenFileType.check storeToken then "true" else "false"}' = 'false'
  test '${if tokenFileType.check storeTokenString then "true" else "false"}' = 'false'
  test '${if tokenFileType.check "relative/token" then "true" else "false"}' = 'false'

  # Non-path policy fails closed through explicit assertions.
  test '${if hasFailedAssertion "name must be non-empty" blankNameEval.config.assertions then "true" else "false"}' = 'true'
  test '${if hasFailedAssertion "trustedHarnessCommit" missingHarnessEval.config.assertions then "true" else "false"}' = 'true'
  test '${if hasFailedAssertion "trustedHarnessCommit" malformedHarnessEval.config.assertions then "true" else "false"}' = 'true'

  # Pinned nixpkgs systemd hardening contract.
  test '${if service.serviceConfig.DynamicUser then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.PrivateDevices then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.PrivateMounts then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.PrivateUsers then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.PrivateTmp then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.ProtectHome then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.NoNewPrivileges then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.RestrictNamespaces then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.RestrictRealtime then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.RestrictSUIDSGID then "true" else "false"}' = 'true'
  test '${if service.serviceConfig.ProtectKernelTunables then "true" else "false"}' = 'true'
  test '${service.serviceConfig.ProtectSystem}' = 'strict'
  test '${service.serviceConfig.ProtectProc}' = 'invisible'
  test '${service.serviceConfig.UMask}' = '0066'
  test '${service.serviceConfig.StateDirectoryMode}' = '0700'

  # Symthaea prepends a root-only credential-policy check before the upstream
  # root bootstrap. Pin both the ordering and the checks themselves so a future
  # module/nixpkgs edit cannot silently turn operator guidance back into a soft
  # convention.
  test '${if lib.hasPrefix "+" firstExecStartPre then "true" else "false"}' = 'true'
  echo '${firstExecStartPre}' | grep -F 'symthaea-ci-runner-credential-preflight'
  test -x '${credentialPreflight}'
  grep -F -- "stat -Lc '%u'" '${credentialPreflight}'
  grep -F -- "stat -Lc '%a'" '${credentialPreflight}'
  grep -F -- "stat -Lc '%s'" '${credentialPreflight}'
  grep -F -- 'wc -l' '${credentialPreflight}'
  grep -F -- "grep -Eq '^[[:graph:]]+$'" '${credentialPreflight}'
  grep -F -- 'must be owned by root' '${credentialPreflight}'
  grep -F -- 'mode must be exactly 0400 or 0600' '${credentialPreflight}'

  # The policy checker may inspect metadata/content predicates, but it must never
  # reproduce the credential value through common output/hash utilities.
  if grep -Eq '(cat|head|tail|printf|echo|md5sum|sha[0-9]*sum).*\$token' '${credentialPreflight}'; then
    echo 'credential preflight can reproduce token contents' >&2
    exit 1
  fi

  # The pinned nixpkgs service's next pre-start stage must still be privileged;
  # the exact generated store-path name is intentionally not asserted here.
  test '${if lib.hasPrefix "+" secondExecStartPre then "true" else "false"}' = 'true'

  # Host-side job admission is defense in depth below GitHub routing/rulesets.
  # It executes before workflow steps and pins both the server-provided workflow
  # identity and the exact reviewed harness commit deployed on the host.
  test -x '${jobAdmissionHook}'
  grep -F -- 'Luminous-Dynamics/symthaea' '${jobAdmissionHook}'
  grep -F -- '1136141775' '${jobAdmissionHook}'
  grep -F -- 'workflow_dispatch' '${jobAdmissionHook}'
  grep -F -- 'refs/heads/main' '${jobAdmissionHook}'
  grep -F -- 'GITHUB_REF_PROTECTED' '${jobAdmissionHook}'
  grep -F -- 'GITHUB_SHA' '${jobAdmissionHook}'
  grep -F -- 'GITHUB_WORKFLOW_SHA' '${jobAdmissionHook}'
  grep -F -- '${fakeHarnessCommit}' '${jobAdmissionHook}'
  grep -F -- 'self-hosted-runner-smoke.yml@refs/heads/main' '${jobAdmissionHook}'
  grep -F -- 'self-hosted-ai-assurance-foundation-recovery.yml@refs/heads/main' '${jobAdmissionHook}'
  grep -F -- 'self-hosted-ai-assurance-budget-recovery.yml@refs/heads/main' '${jobAdmissionHook}'
  grep -F -- 'self-hosted-sym-arch-002a-core-recovery.yml@refs/heads/main' '${jobAdmissionHook}'
  grep -F -- "reject 'workflow path is not in the trusted capability allowlist'" '${jobAdmissionHook}'

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

  # Protect the pinned upstream bootstrap lifecycle itself. The root pre-start
  # copies the original token, keeps a private comparison copy, removes the
  # temporary registration copy after configure, and disables runner self-update.
  grep -F -- 'install --mode=666' '${upstreamService}'
  grep -F -- 'install --mode=600' '${upstreamService}'
  grep -F 'rm "' '${upstreamService}' | grep -F 'newConfigTokenPath'
  grep -F -- '--disableupdate' '${upstreamService}'

  touch "$out"
''
