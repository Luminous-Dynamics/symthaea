# NixOS Troubleshooting

*Quick solutions to common problems, without the jargon.*

---

## The Golden Rule

If your system is broken after a rebuild: **reboot, pick the previous generation from the boot menu.** You are now back to a working system. Take a breath. Then fix the config and try again.

---

## Build Errors

### "error: attribute 'X' not found"
**What happened:** You typed a package name that doesn't exist in nixpkgs.
**Fix:** Search for it at [search.nixos.org/packages](https://search.nixos.org/packages). Package names in Nix are sometimes different from what you'd expect (`google-chrome` not `chrome`, `vscode` not `code`).

### "error: undefined variable 'pkgs'"
**What happened:** You used `pkgs.something` outside of a context where `pkgs` is defined.
**Fix:** Make sure your function starts with `{ config, pkgs, ... }:` — the `pkgs` argument must be listed.

### "infinite recursion encountered"
**What happened:** Two parts of your config depend on each other in a circle.
**Fix:** This usually happens with `imports` or `mkIf`. Simplify: remove the last thing you added and add it back piece by piece.

### "collision between X and Y"
**What happened:** Two packages provide the same file.
**Fix:** Remove one, or use `lib.mkForce` to explicitly choose which wins:
```nix
environment.systemPackages = with pkgs; [
  (lib.hiPrio package-you-prefer)
  other-package
];
```

---

## Hardware Issues

### WiFi Not Working
```nix
# Try NetworkManager (recommended)
networking.networkmanager.enable = true;

# Some WiFi chips need firmware
hardware.enableRedistributableFirmware = true;

# Broadcom chips specifically
boot.kernelModules = [ "wl" ];
hardware.enableAllFirmware = true;
```

### NVIDIA: Black Screen After Login
```nix
# Make sure modesetting is enabled
hardware.nvidia.modesetting.enable = true;

# If using Wayland (GNOME/KDE), this helps:
hardware.nvidia.open = true;  # For Turing+ GPUs (RTX 20xx and newer)

# If still broken, try X11 instead of Wayland:
services.xserver.displayManager.gdm.wayland = false;
```

### No Sound
```nix
# Modern NixOS uses PipeWire (not PulseAudio):
services.pulseaudio.enable = false;  # must be false
services.pipewire = {
  enable = true;
  alsa.enable = true;
  pulse.enable = true;  # PulseAudio compatibility
};
security.rtkit.enable = true;
```

### Bluetooth Not Connecting
```nix
hardware.bluetooth = {
  enable = true;
  powerOnBoot = true;
  settings.General.Experimental = true;  # enables some newer features
};
services.blueman.enable = true;  # GUI for Bluetooth
```

---

## Package Management

### "I Installed a Package But Can't Find It"
- Rebuild first: `sudo nixos-rebuild switch`
- Log out and back in (some PATH changes need a new session)
- For GUI apps, check your application menu or run it from terminal
- Verify it's installed: `which packagename` or `nix-locate bin/packagename`

### "I Need an Older Version of a Package"
```nix
# Pin a specific nixpkgs commit for that package
let
  oldPkgs = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/COMMIT_HASH.tar.gz";
    sha256 = "...";
  }) {};
in {
  environment.systemPackages = [
    oldPkgs.specific-package
  ];
}
```

Or with flakes, add an older nixpkgs as a second input.

### "This Package Needs allowUnfree"
```nix
nixpkgs.config.allowUnfree = true;
# Or for specific packages only:
nixpkgs.config.allowUnfreePredicate = pkg:
  builtins.elem (lib.getName pkg) [
    "spotify" "discord" "vscode" "nvidia-x11"
  ];
```

---

## System Administration

### "I Want to Free Disk Space"
```bash
# Remove old generations (keeps current)
sudo nix-collect-garbage --delete-older-than 30d

# Remove ALL old generations (keeps only current)
sudo nix-collect-garbage -d

# See how much space each generation uses
nix-env --list-generations --profile /nix/var/nix/profiles/system

# Optimize the store (hard-link identical files)
nix store optimise
```

### "How Do I Update Everything?"
```bash
# With flakes:
cd /etc/nixos
sudo nix flake update
sudo nixos-rebuild switch --flake .

# Without flakes:
sudo nix-channel --update
sudo nixos-rebuild switch
```

### "A Service Isn't Starting"
```bash
# Check its status
systemctl status servicename

# See its logs
journalctl -u servicename -f

# Restart it
sudo systemctl restart servicename
```

---

## Getting Help

1. **Search options:** [search.nixos.org/options](https://search.nixos.org/options)
2. **Search packages:** [search.nixos.org/packages](https://search.nixos.org/packages)
3. **NixOS Wiki:** [wiki.nixos.org](https://wiki.nixos.org)
4. **NixOS Discourse:** [discourse.nixos.org](https://discourse.nixos.org) — friendly, searchable
5. **Matrix chat:** `#nixos:nixos.org` — real-time help

When asking for help, share:
- Your `configuration.nix` (or relevant section)
- The exact error message
- What you were trying to do

---

*Remember: you can always roll back. Experiment boldly.*
