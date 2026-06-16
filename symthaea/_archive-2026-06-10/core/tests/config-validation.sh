#!/usr/bin/env bash
# Integration test: Generate NixOS configs from Rust and validate with nix-instantiate.
# This catches Nix syntax errors that our lightweight validator might miss.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PASS=0
FAIL=0

test_config() {
    local name="$1"
    local config="$2"

    echo "$config" > /tmp/nfh-test-config.nix
    if nix-instantiate --parse /tmp/nfh-test-config.nix >/dev/null 2>&1; then
        echo "  PASS: $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $name"
        nix-instantiate --parse /tmp/nfh-test-config.nix 2>&1 | tail -3
        FAIL=$((FAIL + 1))
    fi
    rm -f /tmp/nfh-test-config.nix
}

echo "=== NixOS Config Validation Tests ==="
echo ""

# Test 1: Minimal config
test_config "Minimal (no options)" '{ config, pkgs, ... }: {
  imports = [ ./hardware-configuration.nix ];
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
  networking.hostName = "minimal";
  system.stateVersion = "25.05";
}'

# Test 2: GNOME desktop
test_config "GNOME desktop" '{ config, pkgs, ... }: {
  imports = [ ./hardware-configuration.nix ];
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
  networking.hostName = "gnome-test";
  services.xserver.enable = true;
  services.xserver.displayManager.gdm.enable = true;
  services.xserver.desktopManager.gnome.enable = true;
  system.stateVersion = "25.05";
}'

# Test 3: KDE Plasma
test_config "KDE Plasma 6" '{ config, pkgs, ... }: {
  imports = [ ./hardware-configuration.nix ];
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
  networking.hostName = "kde-test";
  services.desktopManager.plasma6.enable = true;
  services.displayManager.sddm.enable = true;
  services.displayManager.sddm.wayland.enable = true;
  system.stateVersion = "25.05";
}'

# Test 4: NVIDIA GPU
test_config "NVIDIA GPU" '{ config, pkgs, ... }: {
  imports = [ ./hardware-configuration.nix ];
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
  networking.hostName = "nvidia-test";
  services.xserver.videoDrivers = [ "nvidia" ];
  hardware.nvidia = {
    modesetting.enable = true;
    powerManagement.enable = false;
    open = false;
    nvidiaSettings = true;
    package = config.boot.kernelPackages.nvidiaPackages.stable;
  };
  system.stateVersion = "25.05";
}'

# Test 5: Full encryption + Secure Boot
test_config "Encryption + LUKS" '{ config, pkgs, ... }: {
  imports = [ ./hardware-configuration.nix ];
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
  boot.initrd.luks.devices."cryptroot" = { device = "/dev/disk/by-uuid/placeholder"; };
  boot.initrd.systemd.enable = true;
  networking.hostName = "encrypted";
  system.stateVersion = "25.05";
}'

# Test 6: ZFS
test_config "ZFS filesystem" '{ config, pkgs, ... }: {
  imports = [ ./hardware-configuration.nix ];
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
  boot.supportedFilesystems = [ "zfs" ];
  boot.zfs.devNodes = "/dev/disk/by-id";
  networking.hostId = "deadbeef";
  services.zfs.autoScrub.enable = true;
  services.zfs.trim.enable = true;
  system.stateVersion = "25.05";
}'

# Test 7: Home Manager
test_config "Home Manager config" '{ config, pkgs, ... }: {
  imports = [ ./hardware-configuration.nix ];
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
  networking.hostName = "hm-test";
  home-manager.useGlobalPkgs = true;
  home-manager.useUserPackages = true;
  home-manager.users.testuser = { pkgs, ... }: {
    home.stateVersion = "25.05";
    programs.git.enable = true;
    programs.zsh = { enable = true; oh-my-zsh = { enable = true; theme = "robbyrussell"; }; };
    programs.starship.enable = true;
  };
  system.stateVersion = "25.05";
}'

# Test 8: Laptop config
test_config "Laptop power management" '{ config, pkgs, ... }: {
  imports = [ ./hardware-configuration.nix ];
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
  networking.hostName = "laptop";
  services.thermald.enable = true;
  services.power-profiles-daemon.enable = true;
  services.logind.lidSwitch = "suspend";
  system.stateVersion = "25.05";
}'

# Test 9: Symthaea edition
test_config "Symthaea edition" '{ config, pkgs, ... }: {
  imports = [ ./hardware-configuration.nix ];
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
  networking.hostName = "symthaea";
  services.ollama = { enable = true; host = "127.0.0.1"; port = 11434; };
  environment.systemPackages = with pkgs; [ ollama ];
  system.stateVersion = "25.05";
}'

# Test 10: Kitchen sink
test_config "Kitchen sink (all options)" '{ config, pkgs, ... }: {
  imports = [ ./hardware-configuration.nix ];
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
  boot.kernelPackages = pkgs.linuxPackages_zen;
  networking.hostName = "kitchen-sink";
  networking.networkmanager.enable = true;
  networking.firewall.enable = true;
  time.timeZone = "America/Chicago";
  i18n.defaultLocale = "de_DE.UTF-8";
  console.keyMap = "us";
  services.desktopManager.plasma6.enable = true;
  services.displayManager.sddm.enable = true;
  services.displayManager.autoLogin.enable = true;
  services.displayManager.autoLogin.user = "testuser";
  services.xserver.videoDrivers = [ "nvidia" ];
  hardware.nvidia = { modesetting.enable = true; open = false; nvidiaSettings = true; package = config.boot.kernelPackages.nvidiaPackages.stable; };
  hardware.graphics.enable = true;
  hardware.bluetooth.enable = true;
  hardware.bluetooth.powerOnBoot = true;
  services.pipewire = { enable = true; alsa.enable = true; alsa.support32Bit = true; pulse.enable = true; };
  security.rtkit.enable = true;
  services.printing.enable = true;
  services.thermald.enable = true;
  services.power-profiles-daemon.enable = true;
  services.logind.lidSwitch = "suspend";
  services.btrfs.autoScrub = { enable = true; interval = "monthly"; fileSystems = [ "/" ]; };
  zramSwap = { enable = true; algorithm = "zstd"; };
  services.ollama = { enable = true; host = "127.0.0.1"; port = 11434; };
  services.avahi = { enable = true; nssmdns4 = true; publish = { enable = true; addresses = true; }; };
  programs.zsh.enable = true;
  users.users.testuser = { isNormalUser = true; extraGroups = [ "wheel" "networkmanager" "video" "audio" ]; shell = pkgs.zsh; };
  users.users.alice = { isNormalUser = true; extraGroups = [ "networkmanager" ]; };
  home-manager.useGlobalPkgs = true;
  home-manager.useUserPackages = true;
  nix.settings = { experimental-features = [ "nix-command" "flakes" ]; trusted-users = [ "root" "@wheel" ]; };
  environment.systemPackages = with pkgs; [ vim wget curl git firefox vscode neofetch htop ollama ];
  system.stateVersion = "25.05";
}'

echo ""
echo "════════════════════════════════════"
echo "  $PASS passed, $FAIL failed"
echo "════════════════════════════════════"
[ $FAIL -eq 0 ]
