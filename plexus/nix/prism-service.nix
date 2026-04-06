# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Symthaea Prism — NixOS production services
#
# Deployment:
#   1. Build: cd /srv/luminous-dynamics/plexus && cargo build -p plexus-serve -p plexus-proxy --release
#   2. Build UI: cd plexus-ui && trunk build --release
#   3. Add to imports in /etc/nixos/configuration.nix:
#        imports = [ /srv/luminous-dynamics/plexus/nix/prism-service.nix ];
#   4. sudo nixos-rebuild switch
#
# Services:
#   - symthaea-prism: static file server for the WASM app (port 8130)
#   - symthaea-prism-proxy: CORS proxy for URL fetching (port 8131)

{ config, pkgs, ... }:

let
  prismDir = "/srv/luminous-dynamics/plexus";
  distDir = "${prismDir}/plexus-ui/dist";
  serveBin = "${prismDir}/target/release/plexus-serve";
  proxyBin = "${prismDir}/target/release/plexus-proxy";
in
{
  # ── Static file server (serves the WASM app) ──
  systemd.services.symthaea-prism = {
    description = "Symthaea Prism — Epistemic Search Browser";
    wantedBy = [ "multi-user.target" ];
    after = [ "network.target" ];

    environment = {
      PRISM_DIST = distDir;
      PRISM_ADDR = "0.0.0.0:8130";
      RUST_LOG = "info";
    };

    serviceConfig = {
      Type = "simple";
      User = "tstoltz";
      ExecStart = serveBin;
      Restart = "always";
      RestartSec = "3s";
      StartLimitBurst = 5;
      StartLimitIntervalSec = 60;

      # Security hardening
      NoNewPrivileges = true;
      ProtectSystem = "strict";
      ProtectHome = "read-only";
      ReadOnlyPaths = [ distDir ];
      PrivateTmp = true;
    };
  };

  # ── CORS proxy (enables URL fetching from WASM) ──
  systemd.services.symthaea-prism-proxy = {
    description = "Symthaea Prism CORS Proxy";
    wantedBy = [ "multi-user.target" ];
    after = [ "network.target" ];

    environment = {
      RUST_LOG = "info";
    };

    serviceConfig = {
      Type = "simple";
      User = "tstoltz";
      ExecStart = proxyBin;
      Restart = "always";
      RestartSec = "3s";
      StartLimitBurst = 5;
      StartLimitIntervalSec = 60;

      NoNewPrivileges = true;
      ProtectSystem = "strict";
      ProtectHome = "read-only";
      PrivateTmp = true;
    };
  };
}
