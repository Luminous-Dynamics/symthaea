# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Symthaea Disko Profile: Single NVMe with Btrfs
#
# Standard layout for single-drive systems:
#   EFI boot partition + LUKS-encrypted btrfs root with subvolumes
#
# Subvolumes: @root, @home, @nix, @log, @snapshots, @swap
# Features: zstd compression, noatime, LUKS encryption, 32G recovery partition
#
# Usage with disko:
#   sudo nix run github:nix-community/disko -- --mode destroy ./disko-single-nvme.nix

{ ... }:
let
  device = "/dev/nvme0n1";
  include_recovery = true;  # Ahimsa: preserve old OS for 30 days
in
{
  disko.devices = {
    disk.main = {
      type = "disk";
      inherit device;
      content = {
        type = "gpt";
        partitions = {
          boot = {
            name = "boot";
            size = "1G";
            type = "EF00";
            content = {
              type = "filesystem";
              format = "vfat";
              mountpoint = "/boot";
              mountOptions = [ "fmask=0077" "dmask=0077" ];
            };
          };

          recovery = {
            name = "recovery";
            size = if include_recovery then "32G" else "0";
            content = {
              type = "filesystem";
              format = "ext4";
              mountpoint = "/recovery";
              mountOptions = [ "noatime" "nofail" ];
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
                type = "btrfs";
                extraArgs = [ "-L" "symthaea" "-f" ];
                subvolumes = {
                  "@" = {
                    mountpoint = "/";
                    mountOptions = [ "compress=zstd:3" "noatime" ];
                  };
                  "@home" = {
                    mountpoint = "/home";
                    mountOptions = [ "compress=zstd:3" "noatime" ];
                  };
                  "@nix" = {
                    mountpoint = "/nix";
                    mountOptions = [ "compress=zstd:3" "noatime" ];
                  };
                  "@log" = {
                    mountpoint = "/var/log";
                    mountOptions = [ "compress=zstd:3" "noatime" ];
                  };
                  "@snapshots" = {
                    mountpoint = "/.snapshots";
                    mountOptions = [ "compress=zstd:3" "noatime" ];
                  };
                  "@swap" = {
                    mountpoint = "/swap";
                    mountOptions = [ "noatime" ];
                  };
                };
              };
            };
          };
        };
      };
    };
  };

  swapDevices = [{
    device = "/swap/swapfile";
    size = 16 * 1024; # 16 GB default, scaled by hardware probe
  }];
}
