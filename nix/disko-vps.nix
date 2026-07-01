# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Symthaea Disko Profile: Cloud VPS / Virtual Machine
#
# Minimal layout for VPS providers (Hetzner, DigitalOcean, Vultr):
#   EFI boot + ext4 root, no LUKS (provider handles encryption),
#   no swap file (use cloud provider's swap or zram only)
#
# Designed for:
#   - Cloud VPS with /dev/vda (virtio)
#   - QEMU/VirtualBox VMs for testing
#   - Small disks (20-80G typical)
#
# Usage with disko:
#   sudo nix run github:nix-community/disko -- --mode destroy ./disko-vps.nix

{ ... }:
let
  device = "/dev/vda";  # virtio disk (standard for cloud/QEMU)
in
{
  disko.devices.disk.main = {
    type = "disk";
    inherit device;
    content = {
      type = "gpt";
      partitions = {
        boot = {
          name = "boot";
          size = "512M";
          type = "EF00";
          content = {
            type = "filesystem";
            format = "vfat";
            mountpoint = "/boot";
          };
        };
        root = {
          name = "root";
          size = "100%";
          content = {
            type = "filesystem";
            format = "ext4";
            mountpoint = "/";
            mountOptions = [ "relatime" ];
          };
        };
      };
    };
  };

  # No swap file — use zram only (cloud VPS typically has limited disk)
  zramSwap = {
    enable = true;
    algorithm = "zstd";
    memoryPercent = 50;
  };
}
