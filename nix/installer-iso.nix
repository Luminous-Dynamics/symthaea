# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Sovereign Inoculation — Custom NixOS Installer ISO
#
# This ISO includes:
#   - SSH server (auto-started, one-time root password shown on console)
#   - Console greeting with IP address and instructions
#   - All tools needed for installation (btrfs, cryptsetup, ntfs, parted)
#
# Build:
#   nix-build nix/installer-iso.nix
#
# Or with flake:
#   nix build .#installer-iso
#
# The resulting ISO boots, enables SSH, generates a one-time root password,
# and prints connection instructions on the local console.

{ pkgs ? import <nixpkgs> { }, modulesPath ? "${pkgs.path}/nixos/modules" }:

let
  iso-config = import ./modules/installer-iso-module.nix { inherit modulesPath; };

in
  (pkgs.nixos iso-config).config.system.build.isoImage
