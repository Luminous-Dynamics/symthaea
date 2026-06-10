# Sovereign Inoculation: Edge Case Support Plan

Based on research into NixOS installer best practices (2025-2026).

## Priority Tiers

### Tier 1: Must-Have for Launch (v0.2)

#### 1.1 Enhanced Hardware Discovery
**Current**: `lsblk -J` for disk detection.
**Needed**: Comprehensive probe that detects RAID, LUKS, LVM, BitLocker, existing OSes, NVMe namespaces.

```bash
# Add to discovery script:
blkid -o full                          # Filesystem signatures, BitLocker detection
mdadm --examine --scan                 # Software RAID arrays
btrfs device scan && btrfs fi show     # Multi-device btrfs
zpool import 2>/dev/null               # ZFS pools
pvs/vgs/lvs                           # LVM volumes
efibootmgr -v                         # Existing UEFI boot entries
os-prober                             # Existing OS installations
nvme list                             # NVMe namespaces
```

**Implementation**: Add a `probe_hardware` action to ssh-relay that runs all detection and returns JSON. Portal displays detected systems with warnings before any partitioning.

#### 1.2 BitLocker Warning
**Issue**: Resizing Windows partition (or even modifying GPT) triggers BitLocker recovery.
**Fix**: Detect BitLocker via `blkid | grep BitLocker`. Show prominent warning: "BitLocker recovery key required after partition changes."
**Alongside layout**: Use unallocated space only, reuse existing ESP. Don't resize NTFS.

#### 1.3 Existing OS Detection
Show detected operating systems before disk selection:
- Windows: NTFS with Windows markers + BitLocker status
- Linux: ext4/btrfs/xfs with `/etc/os-release`
- macOS: APFS detection
- EFI boot entries from `efibootmgr`

### Tier 2: Next Release (v0.3)

#### 2.1 LUKS Full Disk Encryption
Add `single-luks` and `dual-luks` layouts:

```bash
cryptsetup luksFormat --type luks2 /dev/nvme0n1p2
cryptsetup open /dev/nvme0n1p2 cryptroot
mkfs.btrfs -f -L nixos /dev/mapper/cryptroot
# Then same btrfs subvolume layout on /dev/mapper/cryptroot
```

NixOS config addition:
```nix
boot.initrd.luks.devices."cryptroot" = {
  device = "/dev/disk/by-uuid/XXXX";
};
```

**Portal UX**: Passphrase input field (entered in browser, sent over SSH for `cryptsetup luksFormat`). Warn about passphrase strength. Option for no-passphrase (TPM2-only, Tier 3).

#### 2.2 Secure Boot (lanzaboote)
**Prerequisites**: Firmware must be in Setup Mode (Secure Boot disabled).

Detection:
```bash
bootctl status | grep "Secure Boot"  # enabled/disabled
bootctl status | grep "Setup Mode"   # setup/user
```

If Setup Mode active:
```bash
sbctl create-keys
sbctl enroll-keys --microsoft  # Include MS certs for dual-boot + GPU OPROMs
```

NixOS config:
```nix
boot.loader.systemd-boot.enable = lib.mkForce false;
boot.lanzaboote = {
  enable = true;
  pkiBundle = "/etc/secureboot";
};
environment.systemPackages = [ pkgs.sbctl ];
```

**Portal UX**: Checkbox "Enable Secure Boot". If firmware is in User Mode, show instructions to enter BIOS and clear keys first.

#### 2.3 eMMC Support
Partition naming: `/dev/mmcblk0p1` (not `/dev/mmcblk01`).
Already handled by the partition name detection in current install script:
```bash
if [ -b "{disk}p1" ]; then BOOT="{disk}p1"; else BOOT="{disk}1"; fi
```

Performance warning in portal: "eMMC storage detected. Expect slower install and boot times."

### Tier 3: Future (v0.4+)

#### 3.1 TPM2 Auto-Unlock
After LUKS setup + Secure Boot:
```bash
systemd-cryptenroll /dev/nvme0n1p2 --tpm2-device=auto --tpm2-pcrs=0+7
```

Requires:
- `boot.initrd.systemd.enable = true` (mandatory for TPM2)
- `cryptTabExtraOpts = [ "tpm2-device=auto" ]`

**Important**: Always keep passphrase slot as fallback. TPM2 fails on firmware update, key rotation, etc.

#### 3.2 Software RAID
**mdadm RAID1** (mirrored boot):
```bash
mdadm --create /dev/md0 --level=1 --raid-devices=2 /dev/sda2 /dev/sdb2
```

NixOS: `boot.swraid.enable = true;`

**btrfs RAID1** (simpler, our default FS):
```bash
mkfs.btrfs -d raid1 -m raid1 /dev/nvme0n1p2 /dev/nvme1n1p1
```

NixOS: `boot.initrd.supportedFilesystems = [ "btrfs" ];`

**Portal UX**: Multi-disk selection UI. RAID level picker (mirror/stripe/raid10). Warn about btrfs raid5/6 instability.

**Note**: ESP cannot be RAID. Use separate ESPs or `grub.mirroredBoots`.

#### 3.3 ZFS Support
```bash
zpool create -f -o ashift=12 -O compression=zstd -O atime=off \
  rpool mirror /dev/disk/by-id/X /dev/disk/by-id/Y
zfs create -o mountpoint=legacy rpool/root
zfs create -o mountpoint=legacy rpool/home
zfs create -o mountpoint=legacy rpool/nix
```

NixOS:
```nix
boot.supportedFilesystems = [ "zfs" ];
networking.hostId = "GENERATED";  # REQUIRED
```

**Gotchas**: License (CDDL, can't be in main kernel), kernel version pinning, `hostId` required.

#### 3.4 YubiKey / FIDO2 Unlock
```bash
systemd-cryptenroll /dev/nvme0n1p2 --fido2-device=auto
```

Requires systemd initrd + physical key presence at every boot.

#### 3.5 USB-Only Install
```nix
boot.loader.systemd-boot.enable = true;
boot.loader.efi.canTouchEfiVariables = false;  # Don't write NVRAM
```

Or with GRUB:
```nix
boot.loader.grub.removable = true;  # Install to fallback EFI path
```

#### 3.6 PXE Boot
Generate netboot images, serve via TFTP+HTTP, browser installer auto-starts on boot. Would require custom NixOS ISO with embedded web server.

#### 3.7 nixos-anywhere Integration
Offer "Remote Install" mode that generates a nixos-anywhere command or drives it from the browser. Complementary to local install flow.

## Architecture Changes Needed

### ssh-relay Additions
1. **`probe_hardware` action**: Run comprehensive hardware discovery, return JSON
2. **`setup_encryption` action**: `cryptsetup luksFormat` + `cryptsetup open` with user-provided passphrase
3. **`setup_secureboot` action**: `sbctl create-keys` + `sbctl enroll-keys --microsoft`
4. **`setup_raid` action**: Create mdadm or btrfs RAID from selected devices
5. **`enroll_tpm2` action**: `systemd-cryptenroll` with TPM2

### Portal UI Additions
1. **Hardware summary panel**: Show detected RAID, LUKS, LVM, BitLocker, existing OSes
2. **Encryption toggle**: Passphrase input + TPM2 checkbox
3. **Secure Boot toggle**: With firmware mode detection
4. **Multi-disk selector**: For RAID configurations
5. **Warning system**: BitLocker, data loss, existing OS detection

### Install Script Changes
1. Support LUKS wrapping around btrfs subvolumes
2. Generate NixOS config with LUKS/ZFS/RAID modules
3. Post-install hooks for sbctl, systemd-cryptenroll
4. Flake-based config with lanzaboote input
