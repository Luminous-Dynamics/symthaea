# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Symthaea Disko Profile: NixOS Alongside Windows (Dual-Boot)
#
# SAFE dual-boot: NixOS installs in pre-created free space on the same disk.
# Windows partition is NEVER touched. Shares the existing EFI System Partition.
#
# Prerequisites:
#   1. User MUST shrink their Windows C: drive from within Windows first
#      (Disk Management → Right-click C: → Shrink Volume → at least 100 GB)
#   2. The freed space must be unallocated (not formatted)
#
# Layout after install:
#   Partition 1: EFI System Partition (existing, shared with Windows)
#   Partition 2: Microsoft Reserved (existing, untouched)
#   Partition 3: Windows NTFS (existing, shrunk by user)
#   Partition 4: NixOS root (btrfs, created by this config)
#
# systemd-boot auto-detects Windows Boot Manager on the shared ESP.
# Boot menu shows: [NixOS] [Windows Boot Manager]
#
# NOTE: This config uses a DIFFERENT approach than standard disko.
# It does NOT wipe the disk. It creates partitions ONLY in unallocated space.
# This requires the user to have already freed space from Windows.
#
# Usage:
#   # After user shrinks Windows and free space is available:
#   # Find the disk and free space:
#   lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT
#   # Create the NixOS partition in the free space:
#   sgdisk -n 0:0:0 -t 0:8300 -c 0:nixos-root /dev/nvme0n1
#   # Format and mount:
#   mkfs.btrfs -L symthaea /dev/nvme0n1p4
#   mount /dev/nvme0n1p4 /mnt
#   btrfs subvolume create /mnt/@
#   btrfs subvolume create /mnt/@home
#   btrfs subvolume create /mnt/@nix
#   btrfs subvolume create /mnt/@log
#   btrfs subvolume create /mnt/@swap
#   umount /mnt
#   mount -o subvol=@,compress=zstd:3,noatime /dev/nvme0n1p4 /mnt
#   mkdir -p /mnt/{home,nix,var/log,swap,boot}
#   mount -o subvol=@home,compress=zstd:3,noatime /dev/nvme0n1p4 /mnt/home
#   mount -o subvol=@nix,compress=zstd:3,noatime /dev/nvme0n1p4 /mnt/nix
#   mount -o subvol=@log,compress=zstd:3,noatime /dev/nvme0n1p4 /mnt/var/log
#   mount -o subvol=@swap,noatime /dev/nvme0n1p4 /mnt/swap
#   mount /dev/nvme0n1p1 /mnt/boot  # Existing EFI partition
#   nixos-install --flake /mnt/etc/nixos#guardian
#
# The generated NixOS configuration (not a disko config — disko would wipe):

{ ... }:
let
  # These must match the user's actual partition layout
  # The installer detects these via lsblk after SSH connect
  nixos_partition = "/dev/nvme0n1p4";  # Created in free space
  efi_partition = "/dev/nvme0n1p1";    # Existing Windows ESP
in
{
  # Root: btrfs with subvolumes in the NixOS partition
  fileSystems."/" = {
    device = nixos_partition;
    fsType = "btrfs";
    options = [ "subvol=@" "compress=zstd:3" "noatime" ];
  };

  fileSystems."/home" = {
    device = nixos_partition;
    fsType = "btrfs";
    options = [ "subvol=@home" "compress=zstd:3" "noatime" ];
  };

  fileSystems."/nix" = {
    device = nixos_partition;
    fsType = "btrfs";
    options = [ "subvol=@nix" "compress=zstd:3" "noatime" ];
  };

  fileSystems."/var/log" = {
    device = nixos_partition;
    fsType = "btrfs";
    options = [ "subvol=@log" "compress=zstd:3" "noatime" ];
  };

  fileSystems."/swap" = {
    device = nixos_partition;
    fsType = "btrfs";
    options = [ "subvol=@swap" "noatime" ];
  };

  # Boot: SHARED EFI System Partition (Windows + NixOS coexist)
  fileSystems."/boot" = {
    device = efi_partition;
    fsType = "vfat";
    options = [ "fmask=0077" "dmask=0077" ];
  };

  # systemd-boot: auto-detects Windows Boot Manager on the same ESP
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;

  # Fix clock drift between Windows (localtime) and NixOS (UTC)
  time.hardwareClockInLocalTime = true;

  # Swap file on btrfs
  swapDevices = [{
    device = "/swap/swapfile";
    size = 16 * 1024; # 16 GB
  }];
}
