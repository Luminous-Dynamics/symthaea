# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# NixOS Module: Broca Curriculum Fine-Tuning Cycle Timer (Phase 4)
#
# Runs scripts/broca_curriculum_cycle.sh on a schedule: checks for new
# curriculum objectives, synthesizes training pairs, fine-tunes a candidate
# checkpoint, and gates it against production. Safe to run unattended —
# the cycle script never touches the production checkpoint itself (enforced
# by its own EXIT-trap invariant check); it only ever writes a
# PROMOTION_READY.json marker for a human to act on via
# scripts/broca_promote_candidate.sh. This timer does NOT promote anything.
#
# The script's own lockfile (target/broca-curriculum-cycle.lock via flock)
# makes it safe for this timer to coexist with manual invocations — a
# manual run and a scheduled run will never race.
#
# See /home/tstoltz/.claude/plans/fuzzy-beaming-brook.md for the full
# bridge design (Phases 0-3) this timer schedules.
#
# Usage:
# {
#   services.broca-curriculum-cycle = {
#     enable = true;
#     # trainBackend = "cpu";  # or "gpu" / "auto" — see BROCA_TRAIN_BACKEND
#     # onCalendar = "weekly"; # default
#   };
# }

{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.broca-curriculum-cycle;
  repoRoot = "/srv/luminous-dynamics/symthaea";
in {
  options.services.broca-curriculum-cycle = {
    enable = mkEnableOption "scheduled Broca curriculum-conditioned fine-tuning cycle";

    onCalendar = mkOption {
      type = types.str;
      default = "weekly";
      description = ''
        systemd OnCalendar schedule for the cycle. Cheap early-exit inside
        the script itself (fewer than BROCA_CURRICULUM_MIN_NEW_OBJECTIVES
        pending) means running this more often than objectives actually
        accumulate costs little beyond the cost of that check.
      '';
    };

    trainBackend = mkOption {
      type = types.enum [ "cpu" "gpu" "auto" ];
      default = "cpu";
      description = ''
        BROCA_TRAIN_BACKEND passed to the cycle script. "gpu"/"auto" require
        the nix develop .#broca-gpu shell (handled automatically by the
        script) and a working CUDA setup. Defaults to "cpu" to match the
        script's own safe default — flip deliberately, not as a side effect
        of enabling this timer.
      '';
    };

    minNewObjectives = mkOption {
      type = types.int;
      default = 20;
      description = "BROCA_CURRICULUM_MIN_NEW_OBJECTIVES passed to the cycle script.";
    };

    user = mkOption {
      type = types.str;
      default = "tstoltz";
      description = "User to run the cycle as — needs cargo/nix/Ollama access.";
    };
  };

  config = mkIf cfg.enable {
    systemd.timers.broca-curriculum-cycle = {
      description = "Scheduled Broca curriculum fine-tuning cycle";
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = cfg.onCalendar;
        Persistent = true;
        # Jitter so this doesn't compete for CPU/GPU at the exact same
        # instant as other scheduled jobs on this machine.
        RandomizedDelaySec = "30m";
      };
    };

    systemd.services.broca-curriculum-cycle = {
      description = "Broca curriculum fine-tuning cycle (never promotes; writes PROMOTION_READY.json for human review)";
      path = with pkgs; [ cargo git flock coreutils gnugrep gawk python3 nix ];
      serviceConfig = {
        Type = "oneshot";
        User = cfg.user;
        WorkingDirectory = repoRoot;
        ExecStart = "${repoRoot}/scripts/broca_curriculum_cycle.sh";
        Environment = [
          "BROCA_TRAIN_BACKEND=${cfg.trainBackend}"
          "BROCA_CURRICULUM_MIN_NEW_OBJECTIVES=${toString cfg.minNewObjectives}"
        ];
        # No Restart= — a failed cycle should surface as a failed unit, not
        # silently retry and potentially mask a real problem (fail-closed,
        # matching the cycle script's own philosophy). Training duration is
        # unpredictable (depends on accumulated corpus size and epochs), so
        # don't impose a tight timeout that could kill a legitimate long run.
        TimeoutStartSec = "12h";
        NoNewPrivileges = true;
      };
    };
  };
}
