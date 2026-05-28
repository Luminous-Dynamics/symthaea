# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{ config, pkgs, ... }:

{
  imports = [
    ./hardware-configuration.nix
    ../../crates/symthaea-nix/nix/module.nix
  ];

  # Networking
  networking.hostName = "mk0-seed-node";
  networking.networkmanager.enable = true;

  # Bootloader
  boot.loader.grub.enable = true;
  boot.loader.grub.device = "nodev"; # For virtual/generic deployment

  # Symthaea Sovereignty Layer
  services.nix-mind = {
    enable = true;
    support.autonomyLevel = "semi-autonomous";
    ollama = {
      endpoint = "http://localhost:11434";
      model = "gemma3:1b";
    };
  };

  # Mk0 Bootstrapper Station Packages
  environment.systemPackages = with pkgs; [
    # Core Runtime
    # symthaea-runtime (will be added via overlay)
    
    # Discovery & P2P
    iroh-relay
    
    # Utilities
    git
    vim
    htop
    powertop
  ];

  # Iroh P2P Discovery Ports
  networking.firewall.allowedUDPPorts = [ 4433 33000 ];
  networking.firewall.allowedTCPPorts = [ 4433 ];

  # System Hardening
  security.sudo.enable = false;
  users.mutableUsers = false;
  
  # Deploy the default user for the sovereign pilot
  users.users.pilot = {
    isNormalUser = true;
    extraGroups = [ "wheel" "networkmanager" "video" ];
    openssh.authorizedKeys.keys = [
      "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... sovereign-pilot" # Placeholder
    ];
  };

  # Enable SSH for remote maintenance
  services.openssh = {
    enable = true;
    settings.PasswordAuthentication = false;
    settings.KbdInteractiveAuthentication = false;
  };

  system.stateVersion = "24.11";
}
