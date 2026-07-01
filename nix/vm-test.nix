# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Minimal NixOS VM for testing the symthaea-quicken-fb boot animation.
#
# Build and run:
#   cd symthaea && nix build .#vm-test
#   ./result/bin/run-symthaea-vm
#
# The VM boots with virtio-gpu and runs the mycelial colonization animation
# on DRM/KMS framebuffer during early boot.  Press Ctrl+A, X to stop QEMU.
#
# NOTE: Building a single workspace crate as a standalone Nix derivation
# requires pointing cargoLock at the workspace-level Cargo.lock and copying
# the entire workspace source tree.  The configuration below does this
# correctly.  If workspace structure changes, the `postUnpack` filter may
# need updating.

{ pkgs, lib, ... }:

let
  # Build the framebuffer binary from the workspace.
  # We supply the whole workspace source (Cargo.lock lives at root) but
  # tell cargo to build only the quicken-fb binary.
  quicken-fb = pkgs.rustPlatform.buildRustPackage {
    pname = "symthaea-quicken-fb";
    version = "0.1.0";

    # Workspace root — required so Cargo.lock and [workspace] are visible
    src = ./..;

    cargoLock = {
      lockFile = ../Cargo.lock;
      allowBuiltinFetchGit = true;
    };

    # Only build the quicken-fb binary
    cargoBuildFlags = [ "-p" "symthaea-quicken-fb" ];

    # Skip tests — they may require features / deps we don't pull in
    doCheck = false;

    nativeBuildInputs = with pkgs; [ pkg-config ];
    buildInputs = with pkgs; [ openssl ];

    meta = {
      description = "Mycelial colonization boot animation — bare-metal DRM/KMS";
      license = lib.licenses.agpl3Plus;
    };
  };
in {
  # ── NixOS module import ──────────────────────────────────────────────
  imports = [ ./modules/quicken-fb.nix ];

  # ── Boot animation ───────────────────────────────────────────────────
  services.symthaea-boot = {
    enable = true;
    genesisPhrase = "consciousness awakens in the machine";
    package = quicken-fb;
    device = "/dev/dri/card0";
  };

  # ── Minimal VM plumbing ──────────────────────────────────────────────
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;

  # virtio-gpu gives the VM a real DRM/KMS device at /dev/dri/card0
  boot.kernelModules = [ "virtio-gpu" "virtio-pci" ];

  # Auto-login root so the VM is usable without a password
  services.getty.autologinUser = "root";

  # No display manager — the animation IS the early-boot display
  services.xserver.enable = false;

  # Bare minimum networking (QEMU user-mode)
  networking.hostName = "symthaea-vm";
  networking.useDHCP = lib.mkDefault true;

  # Allow passwordless root for testing
  users.users.root.initialPassword = "";

  system.stateVersion = "25.11";
}
