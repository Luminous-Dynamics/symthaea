{ config, pkgs, ... }:
{
  imports = [ ];

  system.stateVersion = "24.11";

  boot.loader.grub.device = "/dev/vda";

  networking.hostName = "osworld";
  networking.useDHCP = true;

  services.openssh.enable = true;

  # Basic tools expected by OSWorld tasks
  environment.systemPackages = with pkgs; [
    bash
    coreutils
    curl
    git
    jq
    python3
  ];

  # Minimal desktop (adjust based on OSWorld task requirements)
  services.xserver.enable = true;
  services.xserver.displayManager.lightdm.enable = true;
  services.xserver.displayManager.autoLogin.enable = true;
  services.xserver.displayManager.autoLogin.user = "symthaea";
  services.xserver.desktopManager.xfce.enable = true;

  users.users.symthaea = {
    isNormalUser = true;
    extraGroups = [ "wheel" ];
    # Empty password for CI/automated VM usage (not internet-facing).
    initialPassword = "";
  };

  security.sudo.wheelNeedsPassword = false;
}
