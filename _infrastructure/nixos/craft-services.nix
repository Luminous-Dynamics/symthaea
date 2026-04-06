# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Craft services — SPA server on port 8129
# Imported from configuration.nix

{ config, pkgs, ... }:

let
  craftSpaScript = pkgs.writeScript "craft-spa" ''
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

    http.server.HTTPServer(("", 8129), H).serve_forever()
  '';
in
{
  # Craft SPA server on port 8129 (canonical port per PORTS.md)
  systemd.services.craft-spa = {
    description = "Craft SPA Server (port 8129)";
    after = [ "network.target" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "simple";
      User = "tstoltz";
      WorkingDirectory = "/srv/luminous-dynamics/mycelix-craft/apps/leptos/dist";
      ExecStart = "${craftSpaScript}";
      Restart = "always";
      RestartSec = 5;
    };
  };

  # Open port 8129 in firewall
  networking.firewall.allowedTCPPorts = [ 8129 ];
}
