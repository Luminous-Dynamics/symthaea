# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portal services — SPA server on port 8124
# Serves the Mycelix Portal orbital UI at portal.mycelix.net
# Imported from configuration.nix

{ config, pkgs, ... }:

let
  portalSpaScript = pkgs.writeScript "portal-spa" ''
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

    http.server.HTTPServer(("", 8124), H).serve_forever()
  '';
in
{
  # Portal SPA server on port 8124 (per PORTS.md)
  systemd.services.portal-spa = {
    description = "Mycelix Portal SPA Server (port 8124)";
    after = [ "network.target" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "simple";
      User = "tstoltz";
      WorkingDirectory = "/srv/luminous-dynamics/mycelix-portal/dist";
      ExecStart = "${portalSpaScript}";
      Restart = "always";
      RestartSec = 5;
    };
  };

  # Open port 8124 in firewall
  networking.firewall.allowedTCPPorts = [ 8124 ];
}
