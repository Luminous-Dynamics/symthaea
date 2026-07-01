# Symthaea Disko Profile: Dual NVMe Developer Workstation
#
# Reference layout from the Luminous Dynamics workstation:
#   Fast drive (Samsung TLC): btrfs with @home, @srv, @swap, @snapshots
#   Standard drive (Intel QLC): ext4 root (OS + nix store + ephemeral caches)
#
# This profile optimizes for:
#   - Source code on the faster drive (read-heavy, compressible)
#   - Build artifacts on the standard drive (write-heavy, ephemeral)
#   - Transparent zstd compression on btrfs
#   - Hourly automated snapshots via btrbk
#   - nofail mounts for boot resilience
#
# Usage with disko:
#   sudo nix run github:nix-community/disko -- --mode destroy ./disko-dual-nvme.nix
#
# Requires: two NVMe drives. Set fast_device and standard_device below.

{ ... }:
let
  # Configure these to match your hardware
  fast_device = "/dev/nvme0n1";      # Faster drive (TLC preferred) → data
  standard_device = "/dev/nvme1n1";  # Standard drive (QLC ok) → OS
in
{
  disko.devices = {
    disk = {
      # ── Standard Drive: OS + Nix Store ──
      standard = {
        type = "disk";
        device = standard_device;
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
            root = {
              name = "nixos-root";
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

      # ── Fast Drive: Data (btrfs with subvolumes) ──
      fast = {
        type = "disk";
        device = fast_device;
        content = {
          type = "gpt";
          partitions = {
            data = {
              name = "samsung-data";
              size = "100%";
              content = {
                type = "btrfs";
                extraArgs = [ "-L" "samsung" "-f" ];
                subvolumes = {
                  "@home" = {
                    mountpoint = "/home";
                    mountOptions = [ "compress=zstd:3" "noatime" "discard=async" "space_cache=v2" "nofail" ];
                  };
                  "@srv" = {
                    mountpoint = "/srv";
                    mountOptions = [ "compress=zstd:3" "noatime" "discard=async" "space_cache=v2" "nofail" ];
                  };
                  "@swap" = {
                    mountpoint = "/swap";
                    mountOptions = [ "noatime" "nofail" ];
                  };
                  "@snapshots" = {
                    mountpoint = "/.snapshots";
                    mountOptions = [ "compress=zstd:3" "noatime" "nofail" ];
                  };
                };
              };
            };
          };
        };
      };
    };
  };

  # Swap: 64G swapfile on fast drive + zram
  swapDevices = [{
    device = "/swap/swapfile";
    size = 64 * 1024; # 64 GB in MB
  }];
}
