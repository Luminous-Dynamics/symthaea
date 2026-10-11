# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Symthaea NixOS Modules
#
# Composable modules for the Spore/NixOS boot and system experience.
# Each module is independently enableable:
#
#   services.symthaea-boot.enable = true;               # Spore boot renderer (quicken-fb)
#   services.symthaea-boot.observability.enable = true; # Explicit quiet/diagnostic presentation policy
#   services.symthaea-dashboard.enable = true;          # Local web dashboard (Sacred Bridge)
#   services.symthaea-recovery.enable = true;           # Ahimsa recovery timer
#   services.symthaea-first-boot.enable = true;         # Auto-hardening on first boot
#   services.broca-curriculum-cycle.enable = true;      # Scheduled Broca fine-tuning cycle
#
# Import this file to get all modules, or import individual files.

{
  imports = [
    ./quicken-fb.nix
    ./symthaea-boot-observability.nix
    ./symthaea-dashboard.nix
    ./recovery-timer.nix
    ./first-boot.nix
    ./broca-curriculum-cycle-timer.nix
  ];
}
