# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# DeSci services — SPA server + API server
# Port 8106 (SPA), 8080 (API, internal)
# Cloudflare Tunnel already routes desci.mycelix.net → :8106
# Imported from configuration.nix

{ config, pkgs, ... }:

let
  desciSpaScript = pkgs.writeScript "desci-spa" ''
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

    http.server.HTTPServer(("", 8106), H).serve_forever()
  '';
in
{
  # DeSci SPA server on port 8106 (canonical port per PORTS.md)
  systemd.services.desci-spa = {
    description = "DeSci SPA Server (port 8106)";
    after = [ "network.target" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "simple";
      User = "tstoltz";
      WorkingDirectory = "/srv/luminous-dynamics/mycelix-desci/apps/leptos/dist";
      ExecStart = "${desciSpaScript}";
      Restart = "always";
      RestartSec = 5;
    };
  };

  # Open port 8106 in firewall
  networking.firewall.allowedTCPPorts = [ 8106 ];
}
