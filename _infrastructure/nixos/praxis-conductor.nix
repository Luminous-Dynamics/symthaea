# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Praxis Holochain Conductor
#
# Uses the holonix-pinned toolchain from the Praxis flake.
# Start with: nix develop /srv/luminous-dynamics/mycelix-workspace/mycelix-praxis -c hc sandbox --piped generate -a 8207 /srv/luminous-dynamics/mycelix-workspace/mycelix-praxis/happ/mycelix-praxis.happ --run=8307
#
# The hc sandbox command:
# 1. Generates conductor config with correct lair-keystore setup
# 2. Installs the hApp into the conductor
# 3. Starts serving on the specified ports
#
# Ports per PORTS.md: admin=8207, app=8307

{ config, pkgs, ... }:

{
  # Create persistent data directory
  systemd.tmpfiles.rules = [
    "d /var/lib/mycelix-praxis 0750 tstoltz users -"
  ];

  # Note: The conductor is started via `nix develop` + `hc sandbox`
  # rather than a raw binary, because holonix pins the exact versions
  # of holochain, hc, and lair-keystore that match the DNA.
  #
  # To start manually:
  #   cd /srv/luminous-dynamics/mycelix-workspace/mycelix-praxis
  #   echo "" | nix develop -c hc sandbox --piped generate -a 8207 happ/mycelix-praxis.happ --run=8307
  #
  # To make persistent, uncomment the service below after testing:
  #
  # systemd.services.praxis-conductor = {
  #   description = "Praxis Holochain Conductor";
  #   after = [ "network-online.target" ];
  #   wants = [ "network-online.target" ];
  #   serviceConfig = {
  #     Type = "simple";
  #     User = "tstoltz";
  #     WorkingDirectory = "/srv/luminous-dynamics/mycelix-workspace/mycelix-praxis";
  #     ExecStart = "${pkgs.nix}/bin/nix develop -c sh -c 'echo \"\" | hc sandbox --piped generate -a 8207 happ/mycelix-praxis.happ --run=8307'";
  #     Restart = "on-failure";
  #     RestartSec = 15;
  #   };
  # };

  networking.firewall.allowedTCPPorts = [ 8207 8307 ];
}
