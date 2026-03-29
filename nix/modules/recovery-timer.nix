# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# NixOS Module: Symthaea Recovery Partition Timer
#
# Implements the Ahimsa 30-day recovery window (Task D). After installation,
# the recovery partition (~32GB) is preserved for 30 days so the user can
# revert to their previous OS. After the grace period expires, a notification
# informs the user that the space can be reclaimed.
#
# The timer runs daily, checking the install timestamp stored in the recovery
# partition. Notifications appear in the last 7 days as a countdown, and after
# expiry as a reclaim prompt.
#
# Usage:
# {
#   services.symthaea-recovery = {
#     enable = true;
#     gracePeriodDays = 30;  # default
#   };
# }

{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.symthaea-recovery;

  recoveryScript = pkgs.writeShellScript "check-symthaea-recovery" ''
    set -euo pipefail

    RECOVERY_MARKER="/recovery/.symthaea-recovery-date"

    # If no recovery partition or no marker, nothing to do
    if [ ! -f "$RECOVERY_MARKER" ]; then
      exit 0
    fi

    INSTALL_DATE=$(cat "$RECOVERY_MARKER")
    CURRENT_DATE=$(date +%s)
    DAYS_ELAPSED=$(( (CURRENT_DATE - INSTALL_DATE) / 86400 ))
    DAYS_REMAINING=$(( ${toString cfg.gracePeriodDays} - DAYS_ELAPSED ))

    if [ "$DAYS_REMAINING" -le 0 ]; then
      # Grace period expired — notify that space can be reclaimed
      ${pkgs.libnotify}/bin/notify-send \
        --urgency=normal \
        --icon=dialog-information \
        "Symthaea Recovery" \
        "Your ${toString cfg.gracePeriodDays}-day recovery window has expired. The recovery partition (32GB) can be reclaimed. Run: sudo symthaea reclaim-recovery"
    elif [ "$DAYS_REMAINING" -le 7 ]; then
      # Final week countdown
      ${pkgs.libnotify}/bin/notify-send \
        --urgency=low \
        --icon=dialog-information \
        "Symthaea Recovery" \
        "Your recovery partition expires in $DAYS_REMAINING day(s). After expiry, 32GB will be reclaimable."
    fi
  '';
in {
  options.services.symthaea-recovery = {
    enable = mkEnableOption "Symthaea recovery partition management (Ahimsa window)";

    gracePeriodDays = mkOption {
      type = types.int;
      default = 30;
      description = ''
        Number of days to preserve the recovery partition before allowing
        reclamation. This is the Ahimsa window — ensuring the user can
        always revert to their previous operating system.
      '';
    };
  };

  config = mkIf cfg.enable {
    # Daily timer with randomized delay to avoid thundering herd on fleet
    systemd.timers.symthaea-recovery-check = {
      description = "Daily check of Symthaea recovery partition status";
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = "daily";
        Persistent = true;
        RandomizedDelaySec = "1h";
      };
    };

    # Oneshot service triggered by the timer
    systemd.services.symthaea-recovery-check = {
      description = "Check Symthaea recovery partition status";
      serviceConfig = {
        Type = "oneshot";
        ExecStart = recoveryScript;

        # Security: minimal privileges — only reads the marker file
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        PrivateDevices = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;

        # Need read access to /recovery partition
        ReadOnlyPaths = [ "/recovery" ];
      };
    };
  };
}
