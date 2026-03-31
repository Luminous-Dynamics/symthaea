# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# EduNet services — SPA server + Cloudflare Tunnel
# Imported from configuration.nix

{ config, pkgs, ... }:

let
  edunetSpaScript = pkgs.writeScript "edunet-spa" ''
    #!${pkgs.python3}/bin/python3
    import http.server, os

    class H(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            p = self.translate_path(self.path)
            if os.path.isfile(p):
                super().do_GET()
            else:
                self.path = "/index.html"
                super().do_GET()
        def log_message(self, format, *args):
            pass

    http.server.HTTPServer(("", 8107), H).serve_forever()
  '';
in
{
  # EduNet SPA server on port 8107 (canonical port per PORTS.md)
  systemd.services.edunet-spa = {
    description = "EduNet SPA Server (port 8107)";
    after = [ "network.target" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "simple";
      User = "tstoltz";
      WorkingDirectory = "/srv/luminous-dynamics/mycelix-edunet/apps/leptos/dist";
      ExecStart = "${edunetSpaScript}";
      Restart = "always";
      RestartSec = 5;
    };
  };

  # EduNet Cloudflare Tunnel
  systemd.services.edunet-tunnel = {
    description = "EduNet Cloudflare Tunnel";
    after = [ "network-online.target" "edunet-spa.service" ];
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

  # Open port 8107 in firewall
  networking.firewall.allowedTCPPorts = [ 8107 ];
}
