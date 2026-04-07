# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Energy SPA on port 8108
{ config, pkgs, ... }:
let
  spaScript = pkgs.writeScript "energy-spa" ''
    #!${pkgs.python3}/bin/python3
    import http.server, os
    class H(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            p = self.translate_path(self.path)
            if os.path.isfile(p): super().do_GET()
            else: self.path = "/index.html"; super().do_GET()
        def log_message(self, f, *a): pass
    http.server.HTTPServer(("", 8108), H).serve_forever()
  '';
in {
  systemd.services.energy-spa = {
    description = "Energy SPA Server (port 8108)";
    after = [ "network.target" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "simple"; User = "tstoltz";
      WorkingDirectory = "/srv/luminous-dynamics/mycelix-energy/apps/leptos/dist";
      ExecStart = "${spaScript}";
      Restart = "always"; RestartSec = 5;
    };
  };
  networking.firewall.allowedTCPPorts = [ 8108 ];
}
