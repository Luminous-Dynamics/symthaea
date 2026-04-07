# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Knowledge services — SPA server on port 8114
# Serves the Knowledge Graph frontend at knowledge.mycelix.net
# Imported from configuration.nix

{ config, pkgs, ... }:

let
  knowledgeSpaScript = pkgs.writeScript "knowledge-spa" ''
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

    http.server.HTTPServer(("", 8114), H).serve_forever()
  '';
in
{
  systemd.services.knowledge-spa = {
    description = "Knowledge Graph SPA Server (port 8114)";
    after = [ "network.target" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "simple";
      User = "tstoltz";
      WorkingDirectory = "/srv/luminous-dynamics/mycelix-knowledge/apps/leptos/dist";
      ExecStart = "${knowledgeSpaScript}";
      Restart = "always";
      RestartSec = 5;
    };
  };

  networking.firewall.allowedTCPPorts = [ 8114 ];
}
