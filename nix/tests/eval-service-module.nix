# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Eval-only test: validate the packaged Symthaea NixOS module defaults for
# service command wiring, state paths, and audit log persistence.
#
# Run (from symthaea/):
#   nix build .#checks.x86_64-linux.eval-service-module

{ pkgs }:

let
  stubPackage = pkgs.writeShellScriptBin "symthaea" ''
    exit 0
  '';

  evaluated = import "${pkgs.path}/nixos/lib/eval-config.nix" {
    system = pkgs.stdenv.hostPlatform.system;
    modules = [
      ../module.nix
      ({ ... }: {
        services.symthaea = {
          enable = true;
          package = stubPackage;
        };
      })
    ];
  };

  service = evaluated.config.systemd.services.symthaea;
  socket = evaluated.config.systemd.sockets.symthaea;
  logrotate = evaluated.config.services.logrotate.settings."${evaluated.config.services.symthaea.dataDir}/logs/service-audit.jsonl";
in
pkgs.runCommand "eval-service-module" { } ''
  case '${service.serviceConfig.ExecStart}' in
    *"/bin/symthaea --socket /run/symthaea/symthaea.sock --loop-interval 500 --state-file /var/lib/symthaea/state/symthaea-state.bin"*) ;;
    *)
      echo "unexpected ExecStart: ${service.serviceConfig.ExecStart}" >&2
      exit 1
      ;;
  esac

  echo '${pkgs.lib.concatStringsSep "\n" service.serviceConfig.Environment}' | grep -Fx 'SYMTHAEA_SERVICE_AUDIT_LOG_PATH=/var/lib/symthaea/logs/service-audit.jsonl'
  echo '${socket.socketConfig.ListenStream}' | grep -Fx '/run/symthaea/symthaea.sock'
  echo '${pkgs.lib.concatStringsSep " " service.serviceConfig.RestrictAddressFamilies}' | grep -Fx 'AF_UNIX'
  echo '${if service.serviceConfig.NoNewPrivileges then "true" else "false"}' | grep -Fx 'true'
  echo '${if service.serviceConfig.PrivateTmp then "true" else "false"}' | grep -Fx 'true'
  echo '${service.serviceConfig.StateDirectory}' | grep -Fx 'symthaea'
  echo '${toString logrotate.rotate}' | grep -Fx '8'
  echo '${logrotate.frequency}' | grep -Fx 'weekly'
  echo '${if logrotate.compress then "true" else "false"}' | grep -Fx 'true'
  echo '${if logrotate.copytruncate then "true" else "false"}' | grep -Fx 'true'

  touch "$out"
''
