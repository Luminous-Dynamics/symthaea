# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# NixOS module for a Pulse SMTP gateway host.
#
# Phase 5 of PULSE_READINESS_PLAN.md. Same module is used for:
#   - Phase 5A: inside a `nixosTest` VM for integration testing
#   - Phase 5B: on a Hetzner CX22 production host (when funded)
#
# Deploy to Hetzner (Phase 5B) via `nixos-rebuild switch --target-host
# mail.mycelix.net --flake .#pulse-gateway`. Test locally (Phase 5A)
# via `nix build .#checks.x86_64-linux.pulse-gateway-e2e`.

{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.pulse-smtp-gateway;

  # The gateway binary. Default is the stock-rustPlatform inline build
  # (works in nixos-25.05+ where cargo is recent enough). Hosts on older
  # nixpkgs (24.05 and below) must override `cfg.package` with a build
  # using rust-overlay's stable.latest — see the flake.nix at this
  # repository's root for the canonical approach.
  defaultGatewayPkg = pkgs.rustPlatform.buildRustPackage {
    pname = "pulse-smtp-gateway";
    version = "0.1.0-alpha.1";
    src = ../../crates/pulse-smtp-gateway;
    cargoLock.lockFile = ../../crates/pulse-smtp-gateway/Cargo.lock;
    nativeBuildInputs = with pkgs; [ pkg-config ];
    buildInputs = [ ];
    doCheck = false;
  };

  gatewayPkg = cfg.package;

  configTOML = pkgs.writeText "pulse-gateway-config.toml" ''
    [domain]
    name = "${cfg.domain}"
    hostname = "${cfg.hostname}"
    postmaster_did = "${cfg.postmasterDid}"
    abuse_did = "${cfg.abuseDid}"

    [listener]
    port = ${toString cfg.port}
    bind = "${cfg.bind}"
    max_size = ${toString cfg.maxMessageSize}
    ${optionalString (cfg.tlsCertPath != null) ''tls_cert = "${cfg.tlsCertPath}"''}
    ${optionalString (cfg.tlsKeyPath != null) ''tls_key = "${cfg.tlsKeyPath}"''}

    [dns]
    upstream = "${cfg.dnsUpstream}"

    [dkim]
    rsa_key_path = "/run/credentials/pulse-smtp-gateway.service/dkim-rsa"
    ed25519_key_path = "/run/credentials/pulse-smtp-gateway.service/dkim-ed25519"

    [rspamd]
    endpoint = "${cfg.rspamdEndpoint}"
    reject_actions = ["reject"]

    [zome]
    conductor_url = "${cfg.conductorUrl}"
    app_id = "${cfg.appId}"
    gateway_did = "${cfg.gatewayDid}"
    gateway_signing_key_path = "/run/credentials/pulse-smtp-gateway.service/dilithium"

    [rate_limit]
    per_ip_per_minute = ${toString cfg.rateLimitPerMinute}
    per_ip_per_hour = ${toString cfg.rateLimitPerHour}

    [verp]
    hmac_secret_path = "/run/credentials/pulse-smtp-gateway.service/verp-hmac"
    prefix = "bounce"

    ${optionalString (cfg.testAliases != {}) ''
      [test_aliases]
      ${concatStringsSep "\n" (mapAttrsToList (alias: did: ''"${alias}" = "${did}"'') cfg.testAliases)}
    ''}
  '';

in {
  options.services.pulse-smtp-gateway = {
    enable = mkEnableOption "Pulse SMTP gateway daemon";

    package = mkOption {
      type = types.package;
      default = defaultGatewayPkg;
      defaultText = lib.literalExpression ''
        rustPlatform.buildRustPackage { pname = "pulse-smtp-gateway"; ... }
      '';
      description = ''
        The pulse-smtp-gateway package to use. Override on hosts running
        nixpkgs older than 25.05 to supply a build with a recent rustc
        (1.85+ for edition2024 deps). The flake.nix at the repository root
        wires rust-overlay's stable.latest for this purpose.
      '';
    };

    domain = mkOption {
      type = types.str;
      example = "mycelix.net";
      description = "Mail domain we receive for (the part after @).";
    };

    hostname = mkOption {
      type = types.str;
      example = "mail.mycelix.net";
      description = "FQDN of this host. Must match forward + reverse DNS.";
    };

    port = mkOption {
      type = types.port;
      default = 25;
      description = ''
        SMTP listen port. Use 25 in production, 2525 in nixosTest VMs
        (unprivileged).
      '';
    };

    bind = mkOption {
      type = types.str;
      default = "0.0.0.0";
      description = "Bind address. Typically 0.0.0.0 in production.";
    };

    maxMessageSize = mkOption {
      type = types.int;
      default = 26214400; # 25 MB, matches Gmail's limit
      description = "SMTP SIZE limit in bytes.";
    };

    tlsCertPath = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = "TLS cert path (from security.acme in production).";
    };

    tlsKeyPath = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = "TLS key path.";
    };

    dnsUpstream = mkOption {
      type = types.str;
      default = "1.1.1.1:853";
      description = "DNS-over-TLS resolver for SPF/DKIM/DMARC checks.";
    };

    rspamdEndpoint = mkOption {
      type = types.str;
      default = "http://127.0.0.1:11333";
      description = "Local rspamd sidecar URL.";
    };

    conductorUrl = mkOption {
      type = types.str;
      example = "ws://localhost:8888";
      description = "Holochain conductor app-port WebSocket URL.";
    };

    appId = mkOption {
      type = types.str;
      default = "mycelix_mail";
      description = "Installed hApp ID.";
    };

    gatewayDid = mkOption {
      type = types.str;
      example = "did:mycelix:gateway-mail-001";
      description = "Gateway's own DID. Must be registered in mycelix-identity.";
    };

    postmasterDid = mkOption {
      type = types.str;
      example = "did:mycelix:tristan";
      description = "DID that owns postmaster@\${domain}.";
    };

    abuseDid = mkOption {
      type = types.str;
      example = "did:mycelix:tristan";
      description = "DID that owns abuse@\${domain}. Can be same as postmaster.";
    };

    rateLimitPerMinute = mkOption {
      type = types.int;
      default = 30;
      description = "Max inbound messages per source IP per minute.";
    };

    rateLimitPerHour = mkOption {
      type = types.int;
      default = 300;
      description = "Max inbound messages per source IP per hour.";
    };

    secrets = mkOption {
      type = types.attrsOf types.path;
      description = ''
        Paths to secret files. Keys: "dkim-rsa", "dkim-ed25519",
        "dilithium", "verp-hmac". Loaded via systemd LoadCredential=.
      '';
    };

    testAliases = mkOption {
      type = types.attrsOf types.str;
      default = {};
      example = { "alice@mycelix.test" = "did:mycelix:alice"; };
      description = ''
        Pre-populated alias→DID map for the StubZomeBridge (Phase 5A
        nixosTest only). Phase 5B with real holochain_client ignores
        this entirely. Used to seed the smoke test without needing a
        running identity cluster.
      '';
    };

    enableRspamd = mkOption {
      type = types.bool;
      default = true;
      description = "Run rspamd as a sidecar on this host.";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = true;
      description = "Open SMTP port + 465 + 587 + 80 + 443 (MTA-STS).";
    };
  };

  config = mkIf cfg.enable {

    # rspamd sidecar. Default modules handle SPF/DKIM/DMARC in addition to
    # spam scoring; we rely on mail-auth in our daemon for the
    # authoritative auth decision and let rspamd only do spam scoring.
    services.rspamd = mkIf cfg.enableRspamd {
      enable = true;
      # Local binding only; our Rust daemon talks over loopback.
      locals = {
        "dkim_signing.conf".text = "enabled = false;";
      };
    };

    # The gateway systemd service.
    systemd.services.pulse-smtp-gateway = {
      description = "Pulse SMTP gateway";
      after = [ "network-online.target" ] ++ optional cfg.enableRspamd "rspamd.service";
      wants = [ "network-online.target" ] ++ optional cfg.enableRspamd "rspamd.service";
      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        ExecStart = "${gatewayPkg}/bin/pulse-smtp-gateway --config ${configTOML}";
        Restart = "on-failure";
        RestartSec = "5s";

        # systemd-managed secrets. Each entry mounts readable-by-service
        # under /run/credentials/pulse-smtp-gateway.service/<name>.
        LoadCredential = mapAttrsToList (name: path: "${name}:${path}") cfg.secrets;

        # Hardening — NixOS patterns for a network-facing daemon.
        DynamicUser = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        PrivateDevices = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        NoNewPrivileges = true;
        LockPersonality = true;
        RestrictNamespaces = true;
        MemoryDenyWriteExecute = true;
        RestrictRealtime = true;
        SystemCallFilter = [ "@system-service" "~@privileged" ];

        # Port 25 needs CAP_NET_BIND_SERVICE if running unprivileged.
        AmbientCapabilities = optional (cfg.port < 1024) "CAP_NET_BIND_SERVICE";
        CapabilityBoundingSet = optional (cfg.port < 1024) "CAP_NET_BIND_SERVICE";
      };
    };

    # MTA-STS policy served over HTTPS at /.well-known/mta-sts.txt.
    # Phase 5B only — in Phase 5A VM tests we skip this.
    services.nginx = mkIf (cfg.tlsCertPath != null) {
      enable = true;
      virtualHosts."mta-sts.${cfg.domain}" = {
        forceSSL = true;
        enableACME = true;
        locations."/.well-known/mta-sts.txt".extraConfig = ''
          default_type text/plain;
          return 200 "version: STSv1\nmode: testing\nmx: ${cfg.hostname}\nmax_age: 604800\n";
        '';
      };
    };

    # Firewall.
    networking.firewall.allowedTCPPorts = mkIf cfg.openFirewall [
      25 # SMTP
      465 # SMTPS (legacy)
      587 # Submission
      80 # ACME http-01
      443 # ACME tls-alpn + MTA-STS policy
    ];

    # Helpful sanity: reject mail if DNS isn't configured. This is a boot-
    # time check that prevents a silently-misconfigured host from
    # accepting mail it can't verify.
    assertions = [
      {
        assertion = cfg.hostname != "" && (builtins.stringLength cfg.hostname) > (builtins.stringLength cfg.domain);
        message = "services.pulse-smtp-gateway.hostname must be a subdomain of .domain";
      }
    ];
  };
}
