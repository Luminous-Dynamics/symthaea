# Sovereign Inoculation — Deep Research Findings

Compiled 2026-03-30 from web research on installer UX, innovation, and implementation details.

## Key Strategic Insights

1. **NixOS Calamares was archived Aug 2025.** No maintained graphical NixOS installer exists. We fill this vacuum.
2. **Fedora Anaconda is moving to Web UI** (Cockpit-based, Fedora 42+). Validates our browser-based approach.
3. **App migration scanning is unique** — no installer does this.
4. **Local LLM config generation during install** — nobody does this.
5. **Git-initialized config from minute zero** — nobody does this.
6. **Docker Compose → NixOS container conversion** — nobody does this.
7. **Dotfile-to-Nix translation** (not just copying) — nobody does this.

## Implementation Details Reference

### WiFi: Use `nmcli` (NixOS ISO ships NetworkManager)
- `nmcli device wifi list` for scanning
- `nmcli device wifi connect SSID password PASS` for connecting
- WPA3: `nmcli connection add type wifi wifi-sec.key-mgmt sae`
- Enterprise 802.1x: `nmcli connection add ... 802-1x.eap peap`
- Captive portal: `curl -s http://connectivitycheck.gstatic.com/generate_204`
- Persist WiFi config in installed system via `networking.networkmanager.ensureProfiles`

### GPU Detection: Parse `lspci -nn` for vendor IDs
- 10de = NVIDIA, 1002 = AMD, 8086 = Intel
- Hybrid: count > 1 VGA/3D devices
- NVIDIA open kernel module: Ampere+ (PCI ID >= 0x2200)
- NixOS NVIDIA: `hardware.nvidia.open`, `hardware.nvidia.modesetting.enable`
- NixOS AMD: `services.xserver.videoDrivers = ["amdgpu"]` (usually auto)
- Optimus: `hardware.nvidia.prime.offload.enable` with bus IDs

### Desktop Environments: Exact NixOS modules
- GNOME: `services.xserver.displayManager.gdm.enable` + `services.desktopManager.gnome.enable`
- KDE Plasma 6: `services.displayManager.sddm.enable` + `services.desktopManager.plasma6.enable`
- Hyprland: `programs.hyprland.enable` + `services.displayManager.sddm.wayland.enable`
- Sway: `programs.sway.enable` + greetd
- XFCE: `services.xserver.desktopManager.xfce.enable` + lightdm

### Flake Generation
- Template with: nixpkgs, home-manager, disko, lanzaboote, nixos-hardware
- Use Tera/Handlebars templating in Rust
- Include user's package choices from app migration results
- Generate disko-config.nix for LUKS+btrfs declaratively

### Pre-Install Disk Snapshot: Tiered approach
- Tier 1 (always, instant): `sfdisk -d` partition table + first 1MB
- Tier 2 (opt-in, fast): `partclone`/`btrfs send`/`ntfsclone` per partition
- Tier 3 (explicit, slow): full `dd` disk image
- Store on external USB, network share, or second disk

### ARM64: SSH relay works unchanged
- Pi 4: U-Boot (generic-extlinux-compatible), NOT systemd-boot
- Pi 5: UEFI (can use systemd-boot)
- Detect platform: `/proc/device-tree/model`
- Detect UEFI: `[ -d /sys/firmware/efi ]`

### Accessibility: Browser-native advantages
- ARIA roles: `role="progressbar"`, `aria-live="polite"` for status
- Focus management: move focus to step heading on navigation
- Skip navigation link for keyboard users
- `prefers-reduced-motion` for ceremony animations
- Test with axe-core, Lighthouse accessibility audit

## Innovation Ideas (Prioritized)

### Build Now
1. Git-initialized config (post-install systemd oneshot)
2. Pre-install disk snapshot (Tier 1 always, Tier 2 opt-in)
3. Desktop preview gallery (pre-rendered screenshots)
4. Post-install Day 1 checklist
5. Hardware probe submission (opt-in compatibility DB)

### Build Next
6. Dotfile-to-Nix translation (home-manager config generation)
7. Package migration scanner (dpkg/rpm/pacman → nixpkgs mapping)
8. Recovery partition (2GB, minimal NixOS + installer)
9. Air-gapped install mode (bundled nix cache on ISO)
10. Config health check timer

### Build Later
11. Local LLM config assistant (gemma3:4b on live ISO)
12. Docker Compose → NixOS container conversion
13. "Fork this config" sharing gallery
14. Auto-push config to Codeberg/GitHub
15. Gamified NixOS skill tree
