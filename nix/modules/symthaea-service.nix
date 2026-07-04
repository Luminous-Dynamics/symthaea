# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# NixOS Module: Symthaea Service Daemon
#
# Deploys `src/bin/symthaea.rs` ("Symthaea Service Daemon") as a persistent,
# hardened systemd service listening on a Unix socket. This is the piece
# that was missing for real curriculum growth: `CurriculumExtender`'s
# autonomous-learning trigger only fires from inside `Symthaea::process()`
# (the facade), and before this module existed nothing on this host called
# that facade with real traffic — curriculum.json sat unchanged for over
# four months. See Part 2 of
# /home/tstoltz/.claude/plans/fuzzy-beaming-brook.md for the full design.
#
# Verified 2026-07-04 (manual, pre-systemd rehearsal): a real query sent over
# this exact wire protocol produced real curriculum growth end-to-end
# (13 -> 21 objectives), after fixing two reliability bugs this same session
# uncovered (see commit b057e0f435): a too-short Ollama response timeout and
# a too-small LLM max-token budget for curriculum-synthesis JSON.
#
# Transport is Unix-socket-only, no TCP bind — sidesteps the daemon's own
# forced-refusal-to-start-without-auth check (which only applies to
# non-loopback TCP binds) while staying filesystem-permission-gated. A
# bearer token is still configured (sops secret
# `symthaea_service_bearer_token`) as defense in depth, read from the
# decrypted secret file and exported by a small wrapper script before exec —
# sops-nix decrypts secrets to their raw value with no `KEY=value` framing,
# so `EnvironmentFile=` can't point at it directly.
#
# Deliberately does NOT set SYMTHAEA_LLM_PROVIDER / OPENAI_API_KEY /
# ANTHROPIC_API_KEY: leaving these unset keeps `create_backend_from_env()`
# on its Ollama default, which is the only backend `CurriculumExtender`'s
# `broca_lite`/`symthaea-spore` path talks to — this daemon has zero code
# path to the Broca production checkpoint the fine-tuning bridge promotes
# into (checked, not just assumed — see the plan doc's "Key facts" section).

{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.symthaea-service;
  repoRoot = "/srv/luminous-dynamics/symthaea";
  binPath = "${repoRoot}/target/release/symthaea";

  launcher = pkgs.writeShellScript "symthaea-service-launcher" ''
    set -euo pipefail
    export SYMTHAEA_SERVICE_BEARER_TOKEN="$(cat "$CREDENTIALS_DIRECTORY/bearer_token")"
    exec "${binPath}" \
      --socket "${cfg.socketPath}" \
      ${optionalString cfg.verbose "-v"}
  '';
in {
  options.services.symthaea-service = {
    enable = mkEnableOption "the Symthaea Service Daemon (persistent consciousness facade, Unix socket)";

    socketPath = mkOption {
      type = types.str;
      default = "/run/symthaea-service/symthaea.sock";
      description = ''
        Unix socket path. Lives under systemd's RuntimeDirectory, which is
        auto-created with correct ownership and auto-cleaned on stop — no
        manual ReadWritePaths entry needed for it.
      '';
    };

    curriculumPath = mkOption {
      type = types.str;
      default = "/srv/luminous-dynamics/.symthaea/curriculum.json";
      description = ''
        SYMTHAEA_CURRICULUM_PATH. Without this, CurriculumExtender falls
        back to its own XDG-based default (~/.local/share/symthaea/
        curriculum.json), which is NOT where real curriculum data lives on
        this host — confirmed 2026-07-04, same gotcha already documented in
        broca-curriculum-cycle-timer.nix.
      '';
    };

    llmModel = mkOption {
      type = types.str;
      default = "gemma4:e2b";
      description = "SYMTHAEA_LLM_MODEL — must be one of the approved models in root CLAUDE.md.";
    };

    llmTimeoutSecs = mkOption {
      type = types.int;
      default = 600;
      description = ''
        SYMTHAEA_LLM_TIMEOUT_SECS. The Ollama backend's own default (180s)
        reliably timed out on real curriculum-synthesis prompts under this
        host's realistic concurrent-session load (measured 2026-07-04: load
        average ~43 on 12 cores). 600s gave enough headroom in that same
        test; raise further if synthesis still times out under load.
      '';
    };

    verbose = mkOption {
      type = types.bool;
      default = false;
      description = "Pass -v to the daemon for debug-level tracing.";
    };

    user = mkOption {
      type = types.str;
      default = "tstoltz";
      description = "User to run the daemon as.";
    };
  };

  config = mkIf cfg.enable {
    sops.secrets.symthaea_service_bearer_token.owner = config.users.users.${cfg.user}.name;

    systemd.services.symthaea-service = {
      description = "Symthaea Service Daemon (persistent consciousness facade)";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        RuntimeDirectory = "symthaea-service";
        LoadCredential = [ "bearer_token:${config.sops.secrets.symthaea_service_bearer_token.path}" ];
        # Uses serviceConfig.Environment (raw systemd string list), not the
        # higher-level `environment = {...}` attrset — that option conflicts
        # with a default PATH NixOS's own systemd.nix module sets (option
        # merge error, hit and fixed 2026-07-04). No custom PATH needed here
        # anyway: binPath is a fully-built static release binary with no
        # runtime cargo/sccache dependency (unlike broca-curriculum-cycle,
        # which does `cargo run` and genuinely needs one).
        Environment = [
          "SYMTHAEA_CURRICULUM_PATH=${cfg.curriculumPath}"
          "SYMTHAEA_LLM_MODEL=${cfg.llmModel}"
          "SYMTHAEA_LLM_TIMEOUT_SECS=${toString cfg.llmTimeoutSecs}"
        ];
        ExecStart = "${launcher}";
        Restart = "always";
        RestartSec = "5s";
        StartLimitBurst = 5;
        StartLimitIntervalSec = 60;

        # Hardening — mirrors mycelix-prism/nix/prism-service.nix, plus
        # ReadWritePaths for the curriculum store this daemon writes to.
        # ProtectHome=true (not read-only) since this daemon has no
        # legitimate reason to touch /home at all, unlike Prism which needs
        # read access to its own repo tree under /srv.
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        ReadWritePaths = [ "/srv/luminous-dynamics/.symthaea" ];
      };
    };
  };
}
