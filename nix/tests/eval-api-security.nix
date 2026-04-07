# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# NixOS test: validate eval-api security posture (bind defaults + CORS allowlist).
#
# Run (from symthaea/):
#   nix build .#checks.x86_64-linux.eval-api-security

{ pkgs }:

let
  nixosTest =
    # nixpkgs 2025-10+ renamed pkgs.nixosTest -> pkgs.testers.nixosTest
    if pkgs ? testers && pkgs.testers ? nixosTest
    then pkgs.testers.nixosTest
    else throw "pkgs.testers.nixosTest is missing (nixpkgs too old?)";

  evalApi = pkgs.rustPlatform.buildRustPackage {
    pname = "symthaea-eval-api";
    version = "0.1.0";
    # Symthaea lives in a monorepo and has optional path deps that point outside
    # `symthaea/`. For Nix builds we include only the minimal sibling paths
    # needed for Cargo to resolve manifests.
    src = pkgs.lib.cleanSourceWith {
      src = ../../../.;
      filter =
        path: _type:
        let
          p = toString path;
          root = toString ../../../.;
        in
          pkgs.lib.hasPrefix "${root}/symthaea" p
          || pkgs.lib.hasPrefix "${root}/crates" p
          || pkgs.lib.hasPrefix "${root}/mycelix-identity" p
          || pkgs.lib.hasPrefix "${root}/mycelix-finance" p
          || pkgs.lib.hasPrefix "${root}/mycelix-workspace" p
          || pkgs.lib.hasPrefix "${root}/mycelix-position" p;
    };
    sourceRoot = "source/symthaea";
    cargoLock = {
      lockFile = ../../Cargo.lock;
      allowBuiltinFetchGit = true;
    };
    nativeBuildInputs = with pkgs; [
      pkg-config
      mold
    ];
    buildInputs = with pkgs; [
      openssl
      libssh2
    ];
    cargoBuildFlags = [
      "-p"
      "symthaea-spore"
      "--bin"
      "eval-api"
      "--features"
      "server"
    ];
    doCheck = false;
  };
in
nixosTest {
  name = "eval-api-security";

  nodes = {
    api = { pkgs, ... }: {
      networking.hostName = "api";
      environment.systemPackages = [
        evalApi
        pkgs.curl
        pkgs.jq
        pkgs.iproute2
        pkgs.netcat-openbsd
      ];

      systemd.services.eval-api = {
        description = "Symthaea eval-api (security test)";
        wantedBy = [ "multi-user.target" ];
        after = [ "network.target" ];
        serviceConfig = {
          ExecStart = "${evalApi}/bin/eval-api --port 8090";
          Restart = "always";
          RestartSec = 1;
        };
      };
    };

    attacker = { pkgs, ... }: {
      networking.hostName = "attacker";
      environment.systemPackages = [
        pkgs.curl
        pkgs.iproute2
        pkgs.netcat-openbsd
      ];
    };
  };

  testScript = ''
    start_all()

    api.wait_for_unit("eval-api.service")
    api.wait_until_succeeds("curl -sSf http://127.0.0.1:8090/api/v1/health | jq -e '.ok == true'")

    # Ensure eval-api is loopback-only by default.
    api.succeed("ss -ltnp | grep -E '127\\.0\\.0\\.1:8090\\s'")

    api_ip = api.succeed("ip -4 -o addr show dev eth1 | awk '{print $4}' | cut -d/ -f1").strip()
    attacker.wait_for_unit("multi-user.target")
    attacker.fail(f"nc -zvw2 {api_ip} 8090")

    # CORS allowlist should reflect allowed localhost origins, not wildcard.
    api.succeed("curl -s -D- -o /dev/null -H 'Origin: http://localhost:7777' http://127.0.0.1:8090/api/v1/health | grep -i '^access-control-allow-origin: http://localhost:7777'")
    api.succeed("! curl -s -D- -o /dev/null -H 'Origin: http://evil.example' http://127.0.0.1:8090/api/v1/health | grep -i '^access-control-allow-origin:'")
  '';
}
