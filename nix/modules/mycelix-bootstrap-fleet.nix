# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Mycelix Bootstrap Fleet — Multi-Region Deployment Configuration
#
# Example configurations for the 3 bootstrap nodes referenced in
# symthaea/src/swarm/config.rs MYCELIX_BOOTSTRAP_NODES.
#
# Deploy to 3 geographically distributed VPS instances:
#   1. bootstrap-1.mycelix.luminousdynamics.org (US-East)
#   2. bootstrap-2.mycelix.luminousdynamics.org (EU-West)
#   3. bootstrap.mycelix.community             (AP-South)
#
# Each node gets a unique identity key in /var/lib/mycelix-bootstrap/keys/
# that persists across reboots. The Iroh ticket for each node (containing
# its EndpointAddr) should be compiled into MYCELIX_BOOTSTRAP_NODES after
# first startup.
#
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
{ ... }:

{
  # ── Node 1: US-East (Primary) ─────────────────────────────────────
  #
  # Import this in the NixOS configuration for bootstrap-1:
  #   imports = [ ./mycelix-bootstrap.nix ./mycelix-bootstrap-fleet.nix ];
  #   services.mycelix-bootstrap = fleet.us-east;

  fleet = {
    us-east = {
      enable = true;
      listenPort = 4433;
      metricsPort = 9091;
      region = "us-east";
      maxPeers = 500;
      logLevel = "info";
    };

    eu-west = {
      enable = true;
      listenPort = 4433;
      metricsPort = 9091;
      region = "eu-west";
      maxPeers = 500;
      logLevel = "info";
    };

    ap-south = {
      enable = true;
      listenPort = 4433;
      metricsPort = 9091;
      region = "ap-south";
      maxPeers = 300;  # Community node, smaller capacity
      logLevel = "info";
    };
  };

  # ── Monitoring ────────────────────────────────────────────────────
  #
  # Each bootstrap node exports Prometheus metrics on metricsPort.
  # Add these targets to your Prometheus scrape config:
  #
  #   scrape_configs:
  #     - job_name: 'mycelix-bootstrap'
  #       static_configs:
  #         - targets:
  #           - 'bootstrap-1.mycelix.luminousdynamics.org:9091'
  #           - 'bootstrap-2.mycelix.luminousdynamics.org:9091'
  #           - 'bootstrap.mycelix.community:9091'

  # ── DNS Records ───────────────────────────────────────────────────
  #
  # Required DNS records (add to mycelix.luminousdynamics.org zone):
  #
  #   bootstrap-1.mycelix  A      <us-east-ip>
  #   bootstrap-1.mycelix  AAAA   <us-east-ipv6>
  #   bootstrap-2.mycelix  A      <eu-west-ip>
  #   bootstrap-2.mycelix  AAAA   <eu-west-ipv6>
  #
  # Required DNS records (add to mycelix.community zone):
  #
  #   bootstrap  A      <ap-south-ip>
  #   bootstrap  AAAA   <ap-south-ipv6>
  #
  # Enable DNSSEC on both zones for tamper resistance.

  # ── First Boot Procedure ──────────────────────────────────────────
  #
  # 1. Deploy NixOS config to each VPS
  # 2. Start service: systemctl start mycelix-bootstrap
  # 3. Service generates Ed25519 identity key in /var/lib/mycelix-bootstrap/keys/
  # 4. Read Iroh ticket from logs: journalctl -u mycelix-bootstrap | grep "ticket"
  # 5. Update MYCELIX_BOOTSTRAP_NODES in symthaea/src/swarm/config.rs with real tickets
  # 6. Or set MYCELIX_BOOTSTRAP_NODES env var at runtime for testing
}
