# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Symthaea Disko Profile: Single SATA Drive
#
# Conservative layout for SATA HDDs/SSDs:
#   EFI boot + LUKS-encrypted ext4 root + swap partition
#   No btrfs (SATA HDDs don't benefit much from compression)
#   LUKS encryption for security
#
# Usage with disko:
#   sudo nix run github:nix-community/disko -- --mode destroy ./disko-sata.nix

{ ... }:
let
  device = "/dev/sda";  # Standard SATA device
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
            mountOptions = [ "fmask=0077" "dmask=0077" ];
          };
        };
        swap = {
          name = "swap";
          size = "8G";
          content = {
            type = "swap";
            resumeDevice = true;
          };
        };
        root = {
          name = "cryptroot";
          size = "100%";
          content = {
            type = "luks";
            name = "symthaea-root";
            settings.allowDiscards = true;
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
  };
}
