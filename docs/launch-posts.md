# Sovereign Inoculation Launch Posts

## 1. NixOS Discourse

**Title:** Sovereign Inoculation: Browser-based NixOS installer with automated btrfs and SSH relay

---

Hey all,

I built a browser-based NixOS installer called **Sovereign Inoculation**. You boot a live ISO, open a URL, and the portal handles the rest — hardware probe, disk selection, partitioning, btrfs subvolumes, and `nixos-install` — all streamed back to the browser in real time.

**How it works:**

1. Boot target from any NixOS live ISO (SSH must be reachable)
2. Open the portal in a browser — can be on the same machine or any device on the network
3. Enter the target's IP — the portal probes hardware via a WebSocket SSH relay and shows a disk selector from live `lsblk` output
4. Pick a disk layout, confirm, and the install runs automatically
5. Progress streams back with per-stage updates and a completion signal

**Technical details:**
- **SSH relay**: Rust (axum + async-ssh2-tokio), WebSocket bridge, ~1200 LOC. Runs locally on the target or on a trusted machine. Streams `STAGE:` markers for progress tracking.
- **Partitioning**: Direct `sgdisk` + `mkfs.btrfs` (no disko dependency on the live ISO). Six btrfs subvolumes: `@`, `@home`, `@nix`, `@log`, `@snapshots`, `@swap` with zstd compression and noatime.
- **First-boot hardening**: Automatically configures earlyoom, smartd, fstrim, zram, btrfs scrub. One-shot NixOS module.
- **Frontend**: Static HTML + 1.1MB WASM consciousness kernel. No framework, no build step, self-hostable.
- **Eval API**: Optional Nix flake evaluator — validates your config before install.

**Disk layouts:**
- **Single disk**: Full wipe, EFI + btrfs, 6 subvolumes
- **Dual NVMe**: OS on standard drive, data on fast drive
- **Alongside Windows**: Finds free space, creates btrfs in it, preserves Windows bootloader
- **SATA**: Same as single, adjusted partition naming
- **VPS**: Minimal ext4 + zram for cloud instances

**What's intentionally weird:** There's an optional ceremony during the install — C Lydian harmony tones, a Phi-rising animation, TTS narration. It runs purely in-browser, touches nothing on the machine, and you can ignore it completely. I find the 5-minute wait during `nixos-install` a good moment for intentionality. Your mileage will vary.

**What's NOT supported yet (and I want to fix):**
- Secure Boot (lanzaboote integration planned)
- LUKS + TPM2 auto-unlock
- Software RAID (mdadm/btrfs multi-device)
- Full hardware compatibility testing beyond QEMU

**Testing:** Full E2E verified in QEMU with dual NVMe VMs. Automated test suite drives the entire flow via WebSocket. Real hardware testing is the next step.

**Links:**
- Portal: https://luminous-dynamics.github.io/symthaea/ (click "Inoculate" tab)
- Source: https://github.com/Luminous-Dynamics/symthaea
- Install scripts: `crates/symthaea-spore/src/bin/ssh_relay.rs`

**What I'd love feedback on:**
- Has anyone tried dual-booting alongside Windows with btrfs? Does resizing the Windows partition trigger BitLocker recovery?
- Btrfs subvolume layout — is `@swap` on btrfs controversial? Should I use a separate partition?
- Is the `lsblk` JSON probe missing any hardware edge cases you've hit?
- Would you actually use a GUI NixOS installer, or is the terminal workflow fine?

---

## 2. Hacker News

**Title:** Show HN: Install NixOS from a browser tab

---

Sovereign Inoculation is a browser-based NixOS installer. A Rust WebSocket relay bridges the browser to the target machine over SSH — it probes hardware, shows a disk selector, and runs an automated btrfs install with streaming progress. No disko dependency on the live ISO, just sgdisk and mkfs.btrfs. Tested E2E in QEMU.

https://github.com/Luminous-Dynamics/symthaea

---

## 3. Reddit r/NixOS

**Title:** I built a browser-based NixOS installer with automated btrfs and an SSH relay

---

**What it does:** Boot any NixOS live ISO, open a portal URL in your browser, enter the target IP, pick a disk, and it handles the rest — `sgdisk` partitioning, btrfs with 6 subvolumes (`@`, `@home`, `@nix`, `@log`, `@snapshots`, `@swap`), zstd compression, `nixos-install`, and a first-boot module that wires up earlyoom, zram, btrbk, fstrim. Five layouts: single disk, dual NVMe, alongside Windows, SATA, VPS.

**How:** A Rust SSH relay (axum WebSocket) bridges the browser to the target machine. Progress streams back in real time. No disko download needed on the live ISO — it uses sgdisk + mkfs directly.

**What's weird:** There's an optional consciousness ceremony during the install — ambient tones, an animation, TTS narration. It does absolutely nothing to the machine and you can ignore it. I added it because staring at a progress bar for 5 minutes felt like a missed opportunity. Fair warning.

**Not yet supported:** Secure Boot, RAID, LUKS+TPM2. These are planned.

Would love feedback on the btrfs layout and any real hardware edge cases. Tested in QEMU only so far.

Portal: https://luminous-dynamics.github.io/symthaea/ (Inoculate tab)
Source: https://github.com/Luminous-Dynamics/symthaea
