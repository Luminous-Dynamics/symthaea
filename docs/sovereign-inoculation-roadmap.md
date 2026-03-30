# Sovereign Inoculation — Full Feature Roadmap

## v0.1 (Current — DONE)
- [x] Single disk btrfs install (6 subvolumes, zstd, earlyoom/smartd/fstrim/zram)
- [x] Dual NVMe layout
- [x] Alongside Windows layout (free space, untested against real Windows)
- [x] SATA layout
- [x] VPS layout
- [x] Ceremony UX (C Lydian, Phi rising, TTS narration)
- [x] SSH relay (Rust/axum WebSocket bridge)
- [x] Eval API (Nix flake evaluation)
- [x] Portal deployed to GitHub Pages
- [x] E2E automated testing (QEMU dual NVMe)
- [x] App migration database (120+ app mappings)

## v0.2 — Safety & Detection (Next)

### Pre-Install Intelligence
- [ ] **App migration scanner**: Mount existing OS partition read-only, scan installed apps, show compatibility report before install
  - Windows: scan Program Files + Start Menu shortcuts
  - macOS: scan /Applications + Homebrew Cellar
  - Linux: scan /usr/bin + flatpak + snap
  - Map against 120+ nixpkgs equivalents with match quality (exact/alternative/compatibility/web/none)
  - Portal UI: categorized report with "X of Y apps have direct replacements"
- [ ] **Comprehensive hardware probe** (`probe_hardware` relay action):
  - Block devices with model/serial/transport/size
  - EFI state (available, Secure Boot on/off, Setup Mode)
  - TPM 2.0 availability
  - Existing RAID (mdadm, ZFS pools, btrfs multi-device)
  - Existing encryption (LUKS, BitLocker)
  - LVM volume groups
  - EFI boot entries (existing OSes)
  - NVMe namespace enumeration
- [ ] **Existing OS detection**: Show detected operating systems before disk selection
  - Windows: NTFS + Windows markers + BitLocker status
  - macOS: APFS detection
  - Linux: ext4/btrfs/xfs with /etc/os-release
  - EFI boot entries from efibootmgr
- [ ] **BitLocker warning**: Detect BitLocker, warn before any partition changes
- [ ] **Data loss confirmation**: Explicit "this will destroy data on /dev/X" with disk model/serial

### Testing
- [ ] **Win11 QEMU VM**: Install Win11 Enterprise eval (free ISO), test alongside layout
  - Script: `scripts/test-vm-win11.sh`
  - Test BitLocker detection
  - Test ESP reuse (don't create new ESP)
  - Test that Windows still boots after NixOS install
- [ ] **Automated E2E with WebSocket keepalive fix**: Fix timeout during long nixos-install

## v0.3 — Encryption & Secure Boot

### LUKS Full Disk Encryption
- [ ] **`single-luks` layout**: LUKS2 + btrfs subvolumes
  - `cryptsetup luksFormat --type luks2` → `cryptsetup open` → btrfs on /dev/mapper/cryptroot
  - Portal UI: passphrase input field (strength meter)
  - NixOS config: `boot.initrd.luks.devices."cryptroot"`
- [ ] **`dual-luks` layout**: Encrypted dual NVMe
- [ ] **`alongside-luks` layout**: Encrypted NixOS alongside unencrypted Windows

### Secure Boot (lanzaboote)
- [ ] **Detection**: Read `/sys/firmware/efi/efivars/SecureBoot-*`, `bootctl status`
- [ ] **Key enrollment**: `sbctl create-keys && sbctl enroll-keys --microsoft` when Setup Mode active
- [ ] **NixOS config generation**: lanzaboote flake input, `boot.lanzaboote.enable`, `pkiBundle`
- [ ] **Portal UI**: Secure Boot toggle with firmware mode display
  - Setup Mode → auto-enroll
  - User Mode → instructions to enter BIOS and clear keys

### Windows Dual-Boot Improvements
- [ ] **ESP reuse**: Detect existing ESP, mount it as /boot instead of creating new one
- [ ] **Partition safety**: Never resize NTFS if BitLocker detected
- [ ] **systemd-boot coexistence**: Install alongside Windows Boot Manager on shared ESP

### Testing
- [ ] Win11 + BitLocker + NixOS alongside (QEMU)
- [ ] Win11 + Secure Boot + NixOS with lanzaboote (QEMU)
- [ ] Verify Windows boots after NixOS install

## v0.4 — RAID & Advanced Storage

### Software RAID
- [ ] **btrfs RAID1**: Multi-device btrfs with raid1 data + raid1 metadata
  - Portal UI: multi-disk selector, RAID level picker
  - `mkfs.btrfs -d raid1 -m raid1 /dev/nvme0n1p2 /dev/nvme1n1p1`
  - NixOS config: `boot.initrd.supportedFilesystems = [ "btrfs" ]`
- [ ] **mdadm RAID1**: Mirror for systems that prefer it
  - `mdadm --create /dev/md0 --level=1 --raid-devices=2`
  - NixOS config: `boot.swraid.enable = true`
  - ESP mirroring via `grub.mirroredBoots`
- [ ] **Existing RAID detection**: Warn before touching RAID member disks
- [ ] **RAID level guidance**: Explain trade-offs in portal (mirror vs stripe vs raid10)

### ZFS Support
- [ ] **ZFS pool creation**: raidz1/2/3, mirror
  - Auto-generate `networking.hostId`
  - Legacy mountpoints for NixOS compatibility
  - Separate /boot partition (ZFS can't boot UEFI)
- [ ] **ZFS detection**: `zpool import` for existing pools

### TPM2 Auto-Unlock
- [ ] **systemd-cryptenroll**: `--tpm2-device=auto --tpm2-pcrs=0+7`
- [ ] **Requires**: LUKS (v0.3) + Secure Boot (v0.3) + `boot.initrd.systemd.enable = true`
- [ ] **Portal UI**: TPM2 toggle (only shown if TPM detected)
- [ ] **Passphrase fallback**: Always keep passphrase slot

### YubiKey / FIDO2
- [ ] **FIDO2 enrollment**: `systemd-cryptenroll --fido2-device=auto`
- [ ] **Portal UI**: "Touch your YubiKey" prompt during enrollment
- [ ] **Coexistence**: FIDO2 + TPM2 + passphrase as separate LUKS slots

## v0.5 — Platform-Specific Support

### macOS / Apple Silicon
- [ ] **Detection**: `uname -m` + device tree check for Apple hardware
- [ ] **Apple Silicon → Asahi redirect**: "Use Asahi Linux first, then run Sovereign Inoculation"
- [ ] **Intel Mac**: UEFI dual-boot support (similar to Windows alongside)
- [ ] **App scanner**: macOS /Applications + Homebrew scan

### Chromebook / eMMC
- [ ] **eMMC detection**: `/dev/mmcblk*` handling
- [ ] **Performance warning**: Slow storage advisory
- [ ] **Partition naming**: `mmcblk0p1` not `mmcblk01`

### USB-Only Install
- [ ] **Removable media mode**: `canTouchEfiVariables = false`, `grub.removable = true`
- [ ] **UUID-based fstab**: No `/dev/sdX` paths

### Network / Remote
- [ ] **PXE boot**: Custom NixOS ISO with embedded web server + mDNS
- [ ] **nixos-anywhere integration**: Generate command or drive it from portal
- [ ] **Serial console**: For headless servers

## v0.6 — Polish & Community

### Portal UX
- [ ] **Guided wizard**: Step-by-step flow instead of single page
- [ ] **Undo/rollback**: Snapshot before install, offer revert
- [ ] **i18n**: Multi-language support
- [ ] **Accessibility**: Screen reader support, keyboard navigation
- [ ] **Mobile portal**: Install NixOS from your phone (connect to target on LAN)

### NixOS Config Generator
- [ ] **Flake generator**: Build a complete flake.nix from user choices
- [ ] **Desktop environment picker**: GNOME, KDE, Hyprland, Sway, XFCE
- [ ] **Package pre-selection**: Based on app migration results
- [ ] **User/locale/timezone**: From browser detection or manual input
- [ ] **Driver selection**: NVIDIA, AMD, Intel GPU detection + config

### Community
- [ ] **Hardware compatibility database**: User-reported results
- [ ] **Plugin system**: Community-contributed disk layouts
- [ ] **flake templates**: Starter configs for common setups

---

## QEMU Test Matrix

| VM | Script | Purpose | Status |
|----|--------|---------|--------|
| Dual NVMe (NixOS) | `test-vm-dual-nvme.sh` | Single/dual disk install | **Passing** |
| Single disk (NixOS) | `test-vm-single.sh` | Single disk install | Exists, untested |
| SATA (NixOS) | `test-vm-sata.sh` | SATA layout | Exists, untested |
| Win11 UEFI+TPM | `test-vm-win11.sh` | Dual-boot alongside | **Script ready** |
| Win11+NixOS dual | `test-vm-win11.sh --dual-boot` | Full dual-boot | Needs Win11 ISO |

### To add:
- [ ] RAID1 btrfs (2-disk NixOS VM)
- [ ] LUKS encrypted (single disk)
- [ ] Secure Boot (OVMF + sbctl)
- [ ] eMMC (emulated mmcblk device)

### What we CAN'T test in QEMU:
- Real hardware quirks (firmware bugs, GPU OPROM, NVMe namespace variations)
- macOS (license violation, poor fidelity)
- Real TPM hardware (swtpm is software emulation)
- Actual boot timing and performance
