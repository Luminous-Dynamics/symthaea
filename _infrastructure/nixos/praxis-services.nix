# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Praxis services — Cloudflare Tunnel (SPA server served by Caddy)
# Imported from configuration.nix
#
# Reconciled 2026-07-02 to match the live deployed config
# (/etc/nixos/modules/system/praxis-services.nix), which had already
# diverged from this repo copy: the old python SPA-server systemd service
# was removed once Caddy took over serving the static build (Caddy's own
# config already points at mycelix-workspace/mycelix-praxis/apps/leptos/dist,
# the post-migration path — see mycelix-workspace/scripts/sync-to-standalone.sh
# and task #38). This repo copy previously still had the old SPA-server
# block referencing the pre-migration top-level mycelix-praxis path — kept
# in sync with live reality rather than blindly path-fixed.

{ config, pkgs, ... }:

{
  # Praxis Cloudflare Tunnel (tunnel name: "edunet" until migration to "praxis")
  systemd.services.praxis-tunnel = {
    description = "Praxis Cloudflare Tunnel";
    after = [ "network-online.target" ];
    wants = [ "network-online.target" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "simple";
      User = "tstoltz";
      ExecStart = "${pkgs.cloudflared}/bin/cloudflared tunnel --config /home/tstoltz/.cloudflared/config.yml run edunet";
      Restart = "always";
      RestartSec = 10;
    };
  };

  # Open port 8107 in firewall (Caddy serves the SPA here now, not the
  # removed python SPA-server service — kept since Caddy still needs it).
  networking.firewall.allowedTCPPorts = [ 8107 ];
}
