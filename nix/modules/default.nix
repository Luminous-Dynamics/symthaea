# Symthaea NixOS Modules
#
# Composable modules for the God-Tier NixOS boot sequence.
# Each module is independently enableable:
#
#   services.symthaea-boot.enable = true;       # Boot animation (quicken-fb)
#   services.symthaea-dashboard.enable = true;   # Local web dashboard (Sacred Bridge)
#   services.symthaea-recovery.enable = true;    # Ahimsa recovery timer
#
# Import this file to get all modules, or import individual files.

{
  imports = [
    ./quicken-fb.nix
    ./symthaea-dashboard.nix
    ./recovery-timer.nix
  ];
}
