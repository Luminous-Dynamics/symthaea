# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Climate services — SPA server on port 8103
# Serves the Leptos CSR WASM bundle at climate.mycelix.net
# Imported from configuration.nix

{ config, pkgs, ... }:

let
  climateSpaScript = pkgs.writeScript "climate-spa" ''
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

    http.server.HTTPServer(("", 8103), H).serve_forever()
  '';
in
{
  # Climate SPA server on port 8103 (per PORTS.md)
  systemd.services.climate-spa = {
    description = "Climate SPA Server (port 8103)";
    after = [ "network.target" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "simple";
      User = "tstoltz";
      WorkingDirectory = "/srv/luminous-dynamics/mycelix-climate/apps/leptos/dist";
      ExecStart = "${climateSpaScript}";
      Restart = "always";
      RestartSec = 5;
    };
  };

  # Open port 8103 in firewall
  networking.firewall.allowedTCPPorts = [ 8103 ];
}
