# Sovereign Inoculation Launch Posts (v1.4)

## 1. NixOS Discourse

**Title:** Sovereign Inoculation: Browser-based NixOS installer with LUKS, Secure Boot, app migration, and a consciousness ceremony

---

Hey all,

I built a browser-based NixOS installer. Boot a live ISO, open https://install.nixforhumanity.org on any device (laptop, phone, tablet), and the portal walks you through the entire install — hardware detection, app compatibility scanning, desktop environment selection, disk encryption, and `nixos-install` — all streamed back to the browser in real time.

**How it works:**

1. Boot target from a NixOS ISO (or our custom Sovereign ISO with auto-starting relay)
2. Open the portal on any device on the same network
3. The portal auto-discovers the relay via mDNS, or you enter the target's IP
4. It probes hardware (GPU, WiFi, TPM, existing OS, BitLocker, RAID), scans your existing apps, and shows a compatibility report
5. Pick your desktop environment (GNOME, KDE, Hyprland, Sway, XFCE), encryption, disk layout
6. Progress streams back with time estimates and a constellation visualization

**9 disk layouts:**
- Single disk (btrfs, 6 subvolumes, zstd)
- Single disk encrypted (LUKS2 + btrfs)
- Alongside Windows/Linux (reuses existing ESP, detects BitLocker)
- Dual NVMe (OS + data)
- RAID1 btrfs (mirrored across 2 disks)
- RAID1 mdadm (software mirror + btrfs)
- SATA, VPS, auto-detect

**What makes it different:**

- **App migration scanner**: Mounts your existing OS read-only, scans installed apps (Windows Program Files, macOS /Applications, Linux packages), and maps them to nixpkgs equivalents. Shows you what has a direct match, what has alternatives, and what doesn't exist yet.
- **Deep scan**: Reads your git config, shell aliases, SSH keys, editor setup, Docker projects — and composes a personalized welcome message when the install completes.
- **Server safety detection**: Risk-scores the target machine (Docker containers, databases, web servers, uptime, SSH keys). Blocks install on production servers unless you explicitly override.
- **System config from the portal**: Desktop environment, GPU driver (NVIDIA/AMD/Intel auto-detected), timezone (IP geolocation), keyboard layout, PipeWire audio, flakes enabled — all configured from the browser and written to a separate `sovereign-config.nix` module.
- **WebSocket reconnect**: If the connection drops during install (nixos-install can take 10+ minutes), the portal auto-reconnects and checks install status.
- **Input sanitization**: Disk paths validated against injection. Passphrases never logged, never written to the install script.
- **Git-initialized config**: `/etc/nixos/.git` exists from minute zero.
- **The ceremony**: Optional. C Lydian harmony tones, Phi-rising animation, TTS narration. Touches nothing on the machine. I find the 10-minute wait a good moment for intentionality.

**Technical stack:**
- SSH relay: Rust (axum + async-ssh2-tokio), ~2,500 LOC
- Portal: Static HTML + 1.1MB WASM consciousness kernel, no framework
- Partitioning: Direct sgdisk + mkfs.btrfs (no disko dependency on the live ISO)
- Custom ISO config with relay + avahi auto-start (`nix/installer-iso.nix`)

**Testing:**
- 4 layouts verified E2E in QEMU (single, LUKS, Win11 alongside, GNOME desktop)
- 13 persona-based tests (student, developer, gamer, musician, sysadmin, privacy advocate, teacher, DevOps, Pi hobbyist, ARM64, Chromebook, accessibility, data scientist)
- 9/9 E2E relay tests passing (including command injection blocking)
- Win11 dual-boot verified (both OSes boot, ESP shared)

**Links:**
- Portal: https://install.nixforhumanity.org
- Source: https://github.com/Luminous-Dynamics/symthaea
- Relay code: `crates/symthaea-spore/src/bin/ssh_relay.rs`
- Guides: `docs/guides/` (Day 1, Week 1, Troubleshooting, Mastering Nix, Secrets, Contributing)

**What I'd love feedback on:**
- Would you use this? Or is the terminal workflow fine for you?
- Has anyone hit issues with shared ESP on Windows dual-boot?
- The btrfs subvolume layout (`@`, `@home`, `@nix`, `@log`, `@snapshots`, `@swap`) — opinions?
- What hardware should I test on next? (Only QEMU so far)
- Is the app migration scanner useful, or is it a gimmick?

---

## 2. Hacker News

**Title:** Show HN: Install NixOS from your phone

---

Sovereign Inoculation is a browser-based NixOS installer. Boot a live ISO, open the portal on any device (including your phone), and it handles hardware detection, app migration scanning, LUKS encryption, and nixos-install — all via a Rust WebSocket SSH relay. It detects your existing apps and shows what has NixOS equivalents before you commit. 9 disk layouts, 13 persona-tested, 4 QEMU-verified installs including Windows dual-boot.

https://install.nixforhumanity.org

https://github.com/Luminous-Dynamics/symthaea

---

## 3. Reddit r/NixOS

**Title:** I built a browser-based NixOS installer you can run from your phone

---

**What it does:** Boot any NixOS ISO, open https://install.nixforhumanity.org on your phone/laptop/tablet, and it installs NixOS for you — including hardware detection, desktop environment selection (GNOME/KDE/Hyprland/Sway/XFCE), LUKS encryption, GPU driver auto-configuration, and app migration scanning.

**The app scanner** mounts your existing Windows/macOS/Linux partition read-only and maps your installed apps to nixpkgs equivalents. So you know before you install what you're keeping and what's changing.

**9 layouts:** single disk, encrypted (LUKS2), alongside Windows, dual NVMe, RAID1 (btrfs + mdadm), SATA, VPS, auto-detect.

**Safety:** Server detection blocks installs on production machines (Docker, databases, web servers). Input sanitization prevents command injection. Passphrases never logged. WebSocket auto-reconnects if the connection drops.

**The weird part:** There's an optional consciousness ceremony during the install — harmony tones, a constellation visualization of packages downloading, a personalized welcome message when it completes. It reads your git config, SSH keys, and editor setup and says "Welcome home, [name]. I brought your tools." Completely optional. Completely unnecessary. I couldn't help myself.

**Tested:** 4 QEMU installs verified (including Win11 dual-boot), 13 personas, 9/9 E2E tests, command injection blocked.

**Not yet tested on real hardware.** Would love volunteers, especially with NVIDIA GPUs, WiFi-only setups, or existing dual-boot machines.

Portal: https://install.nixforhumanity.org
Source: https://github.com/Luminous-Dynamics/symthaea
