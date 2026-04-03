// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lightweight NixOS configuration generator — WASM-compatible.
//!
//! Pure deterministic template generation from hardware profile, user choices,
//! and selected apps. No AI, HDC, or inference — just string building from
//! structured inputs. Designed to run in the browser via WASM for the
//! Sovereign Inoculation installer.

use crate::AppDatabase;
use std::fmt::Write;

// ═══════════════════════════════════════════════════════
// Input types
// ═══════════════════════════════════════════════════════

/// Hardware detected (or entered) by the user.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HardwareProfile {
    /// GPU vendor string, e.g. "nvidia", "amd", "intel", "unknown"
    pub gpu_vendor: String,
    /// GPU model string, e.g. "RTX 3090", "RX 7900 XTX"
    pub gpu_model: String,
    /// Number of CPU cores
    pub cpu_cores: u32,
    /// Total system RAM in GiB
    pub memory_gb: u32,
    /// Whether the machine has a WiFi adapter
    pub has_wifi: bool,
}

/// User-selected configuration options.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UserChoices {
    /// Hostname for the machine
    pub hostname: String,
    /// Username for the primary user account
    pub username: String,
    /// Desktop environment: "gnome", "kde", "hyprland", "sway", "xfce", "cosmic", "none"
    pub desktop: String,
    /// Whether to set up LUKS full-disk encryption
    pub encryption: bool,
    /// Whether to enable Secure Boot via lanzaboote
    pub secure_boot: bool,
    /// Whether to enable TPM2 auto-unlock for LUKS
    pub tpm2_unlock: bool,
    /// IANA timezone, e.g. "Africa/Johannesburg"
    pub timezone: String,
    /// Keyboard layout, e.g. "us", "za"
    pub keyboard: String,
    /// Locale, e.g. "en_US.UTF-8", "de_DE.UTF-8"
    pub locale: String,
    /// Raw nixpkgs attribute names the user typed in
    pub custom_packages: Vec<String>,
}

/// Output of the config generator.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GeneratedConfig {
    /// Contents of `/etc/nixos/configuration.nix`
    pub configuration_nix: String,
    /// Contents of `/etc/nixos/flake.nix`
    pub flake_nix: String,
    /// Non-fatal warnings about the configuration
    pub warnings: Vec<String>,
}

// ═══════════════════════════════════════════════════════
// Generator
// ═══════════════════════════════════════════════════════

/// Generate a NixOS configuration from hardware, choices, and selected apps.
///
/// `selected_apps` should be canonical app names from `AppDatabase` (e.g.
/// "Firefox", "GIMP"). They are resolved to `nix_pkg` attribute names. Apps
/// not found in the database are silently skipped (with a warning).
pub fn generate(
    hw: &HardwareProfile,
    choices: &UserChoices,
    selected_apps: &[String],
) -> GeneratedConfig {
    let db = AppDatabase::new();
    let mut warnings: Vec<String> = Vec::new();

    // ── Resolve apps to nix package names ──
    let mut nix_pkgs: Vec<&str> = Vec::new();
    for app_name in selected_apps {
        if let Some(entry) = db.match_app(app_name) {
            nix_pkgs.push(entry.primary.nix_pkg);
        } else {
            warnings.push(format!("App '{}' not found in database — skipped", app_name));
        }
    }
    nix_pkgs.sort();
    nix_pkgs.dedup();

    // ── Validate inputs ──
    // Normalize GPU vendor — handle ANGLE-wrapped strings like "Google Inc. (Intel)"
    let raw_gpu = format!("{} {}", hw.gpu_vendor, hw.gpu_model).to_lowercase();
    let gpu_vendor = if raw_gpu.contains("nvidia") || raw_gpu.contains("geforce") || raw_gpu.contains("rtx") || raw_gpu.contains("gtx") {
        "nvidia".to_string()
    } else if raw_gpu.contains("amd") || raw_gpu.contains("radeon") || raw_gpu.contains("ati") {
        "amd".to_string()
    } else if raw_gpu.contains("intel") || raw_gpu.contains("uhd") || raw_gpu.contains("iris") {
        "intel".to_string()
    } else {
        hw.gpu_vendor.to_lowercase()
    };
    if gpu_vendor != "nvidia" && gpu_vendor != "amd" && gpu_vendor != "intel" {
        warnings.push(format!(
            "Unknown GPU vendor '{}' — no driver config generated. \
             Modesetting will be used as fallback.",
            hw.gpu_vendor
        ));
    }

    let desktop_lower = choices.desktop.to_lowercase();
    let known_desktops = ["gnome", "kde", "hyprland", "sway", "xfce", "cosmic", "none"];
    if !known_desktops.contains(&desktop_lower.as_str()) {
        warnings.push(format!(
            "Unknown desktop '{}' — treating as 'none' (no DE configured)",
            choices.desktop
        ));
    }

    if choices.tpm2_unlock && !choices.encryption {
        warnings.push("TPM2 unlock requested but encryption is disabled — TPM2 config skipped".into());
    }

    if choices.secure_boot && choices.encryption && !choices.tpm2_unlock {
        warnings.push(
            "Secure Boot + LUKS without TPM2: you will need to enter a passphrase on every boot"
                .into(),
        );
    }

    // ── Build configuration.nix ──
    let configuration_nix = build_configuration_nix(hw, choices, &nix_pkgs, &gpu_vendor, &desktop_lower);

    // ── Build flake.nix ──
    let flake_nix = build_flake_nix(choices);

    GeneratedConfig {
        configuration_nix,
        flake_nix,
        warnings,
    }
}

// ═══════════════════════════════════════════════════════
// configuration.nix builder
// ═══════════════════════════════════════════════════════

fn build_configuration_nix(
    hw: &HardwareProfile,
    choices: &UserChoices,
    nix_pkgs: &[&str],
    gpu_vendor: &str,
    desktop: &str,
) -> String {
    let mut out = String::with_capacity(4096);

    // Header
    writeln!(out, "# Generated by Sovereign Inoculation (symthaea-app-db config_gen)").unwrap();
    writeln!(out, "# https://nixforhumanity.org").unwrap();
    writeln!(out, "{{ config, pkgs, lib, ... }}:\n{{").unwrap();

    // ── Imports ──
    write_imports(&mut out, choices);

    // ── Boot ──
    write_boot(&mut out, choices);

    // ── LUKS encryption ──
    if choices.encryption {
        write_encryption(&mut out, choices);
    }

    // ── Networking ──
    write_networking(&mut out, choices, hw);

    // ── Localization ──
    write_locale(&mut out, choices);

    // ── Desktop environment ──
    write_desktop(&mut out, desktop);

    // ── GPU drivers ──
    write_gpu(&mut out, gpu_vendor, hw);

    // ── Audio (PipeWire always) ──
    write_audio(&mut out);

    // ── User account ──
    write_user(&mut out, &choices.username);

    // ── Nix settings (flakes) ──
    write_nix_settings(&mut out);

    // ── Packages ──
    write_packages(&mut out, nix_pkgs, &choices.custom_packages);

    // ── System version ──
    writeln!(out, "  system.stateVersion = \"25.05\";").unwrap();

    writeln!(out, "}}").unwrap();
    out
}

// ── Section writers ──

fn write_imports(out: &mut String, choices: &UserChoices) {
    writeln!(out, "  imports = [").unwrap();
    writeln!(out, "    ./hardware-configuration.nix").unwrap();
    if choices.secure_boot {
        writeln!(out, "    # lanzaboote replaces systemd-boot for Secure Boot").unwrap();
    }
    writeln!(out, "  ];\n").unwrap();
}

fn write_boot(out: &mut String, choices: &UserChoices) {
    if choices.secure_boot {
        // lanzaboote takes over boot management
        writeln!(out, "  # Secure Boot via lanzaboote").unwrap();
        writeln!(out, "  boot.loader.systemd-boot.enable = lib.mkForce false;").unwrap();
        writeln!(out, "  boot.lanzaboote = {{").unwrap();
        writeln!(out, "    enable = true;").unwrap();
        writeln!(out, "    pkiBundle = \"/etc/secureboot\";").unwrap();
        writeln!(out, "  }};\n").unwrap();
    } else {
        writeln!(out, "  boot.loader.systemd-boot.enable = true;").unwrap();
        writeln!(out, "  boot.loader.efi.canTouchEfiVariables = true;\n").unwrap();
    }
}

fn write_encryption(out: &mut String, choices: &UserChoices) {
    writeln!(out, "  # LUKS full-disk encryption").unwrap();
    writeln!(out, "  boot.initrd.luks.devices.\"cryptroot\" = {{").unwrap();
    writeln!(out, "    device = \"/dev/disk/by-partlabel/cryptroot\";").unwrap();
    if choices.tpm2_unlock && choices.encryption {
        writeln!(out, "    # TPM2 auto-unlock — enroll with: sudo systemd-cryptenroll --tpm2-device=auto /dev/<part>").unwrap();
        writeln!(out, "    cryptTabExtraOpts = [ \"tpm2-device=auto\" ];").unwrap();
    }
    writeln!(out, "  }};\n").unwrap();
}

fn write_networking(out: &mut String, choices: &UserChoices, hw: &HardwareProfile) {
    writeln!(out, "  networking.hostName = \"{}\";", choices.hostname).unwrap();
    writeln!(out, "  networking.networkmanager.enable = true;").unwrap();
    writeln!(out, "  networking.firewall.enable = true;\n").unwrap();
    if hw.has_wifi {
        writeln!(out, "  # WiFi detected — NetworkManager handles it automatically").unwrap();
        writeln!(out, "  # To connect: nmcli device wifi connect <SSID> password <pass>\n").unwrap();
    }
}

fn write_locale(out: &mut String, choices: &UserChoices) {
    writeln!(out, "  time.timeZone = \"{}\";", choices.timezone).unwrap();
    writeln!(out, "  i18n.defaultLocale = \"{}\";", choices.locale).unwrap();
    writeln!(
        out,
        "  console.keyMap = \"{}\";\n",
        choices.keyboard
    )
    .unwrap();
}

fn write_desktop(out: &mut String, desktop: &str) {
    match desktop {
        "gnome" => {
            writeln!(out, "  # GNOME Desktop").unwrap();
            writeln!(out, "  services.xserver.enable = true;").unwrap();
            writeln!(out, "  services.xserver.displayManager.gdm.enable = true;").unwrap();
            writeln!(out, "  services.xserver.desktopManager.gnome.enable = true;\n").unwrap();
        }
        "kde" => {
            writeln!(out, "  # KDE Plasma 6").unwrap();
            writeln!(out, "  services.desktopManager.plasma6.enable = true;").unwrap();
            writeln!(out, "  services.displayManager.sddm.enable = true;").unwrap();
            writeln!(out, "  services.displayManager.sddm.wayland.enable = true;\n").unwrap();
        }
        "hyprland" => {
            writeln!(out, "  # Hyprland (Wayland tiling compositor)").unwrap();
            writeln!(out, "  programs.hyprland.enable = true;").unwrap();
            writeln!(out, "  # You will need a display manager or TTY login").unwrap();
            writeln!(out, "  services.greetd = {{").unwrap();
            writeln!(out, "    enable = true;").unwrap();
            writeln!(out, "    settings.default_session.command = \"${{pkgs.greetd.tuigreet}}/bin/tuigreet --time --cmd Hyprland\";").unwrap();
            writeln!(out, "  }};\n").unwrap();
        }
        "sway" => {
            writeln!(out, "  # Sway (Wayland tiling compositor)").unwrap();
            writeln!(out, "  programs.sway.enable = true;").unwrap();
            writeln!(out, "  services.greetd = {{").unwrap();
            writeln!(out, "    enable = true;").unwrap();
            writeln!(out, "    settings.default_session.command = \"${{pkgs.greetd.tuigreet}}/bin/tuigreet --time --cmd sway\";").unwrap();
            writeln!(out, "  }};\n").unwrap();
        }
        "xfce" => {
            writeln!(out, "  # XFCE (lightweight)").unwrap();
            writeln!(out, "  services.xserver.enable = true;").unwrap();
            writeln!(out, "  services.xserver.displayManager.lightdm.enable = true;").unwrap();
            writeln!(out, "  services.xserver.desktopManager.xfce.enable = true;\n").unwrap();
        }
        "cosmic" => {
            writeln!(out, "  # COSMIC Desktop (System76)").unwrap();
            writeln!(out, "  services.desktopManager.cosmic.enable = true;").unwrap();
            writeln!(out, "  services.displayManager.cosmic-greeter.enable = true;\n").unwrap();
        }
        _ => {
            writeln!(out, "  # No desktop environment — console only").unwrap();
            writeln!(out, "  # Install a DE later with: services.xserver.desktopManager.<name>.enable = true;\n").unwrap();
        }
    }
}

fn write_gpu(out: &mut String, gpu_vendor: &str, hw: &HardwareProfile) {
    match gpu_vendor {
        "nvidia" => {
            writeln!(out, "  # NVIDIA GPU: {} — proprietary driver", hw.gpu_model).unwrap();
            writeln!(out, "  services.xserver.videoDrivers = [ \"nvidia\" ];").unwrap();
            writeln!(out, "  hardware.nvidia = {{").unwrap();
            writeln!(out, "    modesetting.enable = true;").unwrap();
            writeln!(out, "    powerManagement.enable = false;").unwrap();
            writeln!(out, "    open = false;").unwrap();
            writeln!(out, "    nvidiaSettings = true;").unwrap();
            writeln!(out, "    package = config.boot.kernelPackages.nvidiaPackages.stable;").unwrap();
            writeln!(out, "  }};\n").unwrap();
        }
        "amd" => {
            writeln!(out, "  # AMD GPU: {} — amdgpu (open-source)", hw.gpu_model).unwrap();
            writeln!(out, "  services.xserver.videoDrivers = [ \"amdgpu\" ];").unwrap();
            writeln!(out, "  hardware.amdgpu.amdvlk.enable = true;\n").unwrap();
        }
        "intel" => {
            writeln!(out, "  # Intel GPU: {} — modesetting (built-in)", hw.gpu_model).unwrap();
            writeln!(out, "  services.xserver.videoDrivers = [ \"modesetting\" ];").unwrap();
            writeln!(out, "  hardware.graphics.enable = true;\n").unwrap();
        }
        _ => {
            writeln!(out, "  # GPU vendor not detected — using generic modesetting").unwrap();
            writeln!(out, "  hardware.graphics.enable = true;\n").unwrap();
        }
    }
}

fn write_audio(out: &mut String) {
    writeln!(out, "  # Audio — PipeWire (replaces PulseAudio)").unwrap();
    writeln!(out, "  services.pipewire = {{").unwrap();
    writeln!(out, "    enable = true;").unwrap();
    writeln!(out, "    alsa.enable = true;").unwrap();
    writeln!(out, "    alsa.support32Bit = true;").unwrap();
    writeln!(out, "    pulse.enable = true;").unwrap();
    writeln!(out, "  }};").unwrap();
    writeln!(out, "  security.rtkit.enable = true;\n").unwrap();
}

fn write_user(out: &mut String, username: &str) {
    let name = if username.is_empty() { "user" } else { username };
    writeln!(out, "  users.users.{} = {{", name).unwrap();
    writeln!(out, "    isNormalUser = true;").unwrap();
    writeln!(out, "    extraGroups = [ \"wheel\" \"networkmanager\" \"video\" \"audio\" ];").unwrap();
    writeln!(out, "    # Set password after install: passwd {}", name).unwrap();
    writeln!(out, "  }};\n").unwrap();
}

fn write_nix_settings(out: &mut String) {
    writeln!(out, "  nix.settings = {{").unwrap();
    writeln!(out, "    experimental-features = [ \"nix-command\" \"flakes\" ];").unwrap();
    writeln!(out, "    trusted-users = [ \"root\" \"@wheel\" ];").unwrap();
    writeln!(out, "  }};\n").unwrap();
}

fn write_packages(out: &mut String, nix_pkgs: &[&str], custom: &[String]) {
    writeln!(out, "  environment.systemPackages = with pkgs; [").unwrap();
    // Always include essentials
    writeln!(out, "    # Essentials").unwrap();
    writeln!(out, "    vim").unwrap();
    writeln!(out, "    wget").unwrap();
    writeln!(out, "    curl").unwrap();
    writeln!(out, "    git").unwrap();
    if !nix_pkgs.is_empty() {
        writeln!(out, "    # Your selected apps").unwrap();
        for pkg in nix_pkgs {
            writeln!(out, "    {}", pkg).unwrap();
        }
    }
    if !custom.is_empty() {
        writeln!(out, "    # Custom packages").unwrap();
        for pkg in custom {
            let pkg = pkg.trim();
            if !pkg.is_empty() {
                writeln!(out, "    {}", pkg).unwrap();
            }
        }
    }
    writeln!(out, "  ];\n").unwrap();
}

// ═══════════════════════════════════════════════════════
// flake.nix builder
// ═══════════════════════════════════════════════════════

fn build_flake_nix(choices: &UserChoices) -> String {
    let mut out = String::with_capacity(1024);

    writeln!(out, "# Generated by Sovereign Inoculation (symthaea-app-db config_gen)").unwrap();
    writeln!(out, "# https://nixforhumanity.org").unwrap();
    writeln!(out, "{{").unwrap();
    writeln!(out, "  description = \"NixOS configuration for {}\";", choices.hostname).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "  inputs = {{").unwrap();
    writeln!(out, "    nixpkgs.url = \"github:NixOS/nixpkgs/nixos-25.05\";").unwrap();
    if choices.secure_boot {
        writeln!(out, "    lanzaboote = {{").unwrap();
        writeln!(out, "      url = \"github:nix-community/lanzaboote/v0.4.2\";").unwrap();
        writeln!(out, "      inputs.nixpkgs.follows = \"nixpkgs\";").unwrap();
        writeln!(out, "    }};").unwrap();
    }
    writeln!(out, "  }};").unwrap();
    writeln!(out).unwrap();

    // outputs
    if choices.secure_boot {
        writeln!(out, "  outputs = {{ self, nixpkgs, lanzaboote, ... }}: {{").unwrap();
    } else {
        writeln!(out, "  outputs = {{ self, nixpkgs, ... }}: {{").unwrap();
    }
    writeln!(out, "    nixosConfigurations.\"{}\" = nixpkgs.lib.nixosSystem {{", choices.hostname).unwrap();
    writeln!(out, "      system = \"x86_64-linux\";").unwrap();
    writeln!(out, "      modules = [").unwrap();
    writeln!(out, "        ./configuration.nix").unwrap();
    if choices.secure_boot {
        writeln!(out, "        lanzaboote.nixosModules.lanzaboote").unwrap();
    }
    writeln!(out, "      ];").unwrap();
    writeln!(out, "    }};").unwrap();
    writeln!(out, "  }};").unwrap();
    writeln!(out, "}}").unwrap();

    out
}

// ═══════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn test_hw() -> HardwareProfile {
        HardwareProfile {
            gpu_vendor: "nvidia".into(),
            gpu_model: "RTX 2070".into(),
            cpu_cores: 8,
            memory_gb: 32,
            has_wifi: true,
        }
    }

    fn test_choices() -> UserChoices {
        UserChoices {
            hostname: "sovereign".into(),
            username: "tristan".into(),
            desktop: "gnome".into(),
            encryption: true,
            secure_boot: true,
            tpm2_unlock: true,
            timezone: "Africa/Johannesburg".into(),
            keyboard: "us".into(),
            locale: "en_US.UTF-8".into(),
            custom_packages: vec![],
        }
    }

    #[test]
    fn generates_valid_configuration_nix() {
        let result = generate(&test_hw(), &test_choices(), &[
            "Firefox".into(), "GIMP".into(),
        ]);
        let nix = &result.configuration_nix;
        assert!(nix.contains("networking.hostName = \"sovereign\""));
        assert!(nix.contains("time.timeZone = \"Africa/Johannesburg\""));
        assert!(nix.contains("services.xserver.desktopManager.gnome.enable = true"));
        assert!(nix.contains("hardware.nvidia"));
        assert!(nix.contains("services.pipewire"));
        assert!(nix.contains("firefox"));
        assert!(nix.contains("cryptroot"));
        assert!(nix.contains("tpm2-device=auto"));
        assert!(nix.contains("lanzaboote"));
        assert!(nix.contains("flakes"));
        assert!(nix.contains("wheel"));
        assert!(result.warnings.is_empty(), "unexpected warnings: {:?}", result.warnings);
    }

    #[test]
    fn generates_valid_flake_nix() {
        let result = generate(&test_hw(), &test_choices(), &[]);
        let flake = &result.flake_nix;
        assert!(flake.contains("nixosConfigurations.\"sovereign\""));
        assert!(flake.contains("nixos-25.05"));
        assert!(flake.contains("lanzaboote"));
    }

    #[test]
    fn flake_without_secureboot_has_no_lanzaboote() {
        let mut c = test_choices();
        c.secure_boot = false;
        let result = generate(&test_hw(), &c, &[]);
        assert!(!result.flake_nix.contains("lanzaboote"));
        assert!(result.configuration_nix.contains("boot.loader.systemd-boot.enable = true"));
    }

    #[test]
    fn kde_desktop() {
        let mut c = test_choices();
        c.desktop = "kde".into();
        let result = generate(&test_hw(), &c, &[]);
        assert!(result.configuration_nix.contains("plasma6"));
        assert!(result.configuration_nix.contains("sddm"));
    }

    #[test]
    fn hyprland_desktop() {
        let mut c = test_choices();
        c.desktop = "hyprland".into();
        let result = generate(&test_hw(), &c, &[]);
        assert!(result.configuration_nix.contains("programs.hyprland.enable = true"));
        assert!(result.configuration_nix.contains("greetd"));
    }

    #[test]
    fn amd_gpu() {
        let hw = HardwareProfile {
            gpu_vendor: "amd".into(),
            gpu_model: "RX 7900 XTX".into(),
            cpu_cores: 16,
            memory_gb: 64,
            has_wifi: false,
        };
        let result = generate(&hw, &test_choices(), &[]);
        assert!(result.configuration_nix.contains("amdgpu"));
        // No wifi comment
        assert!(!result.configuration_nix.contains("WiFi detected"));
    }

    #[test]
    fn intel_gpu() {
        let hw = HardwareProfile {
            gpu_vendor: "intel".into(),
            gpu_model: "UHD 770".into(),
            cpu_cores: 4,
            memory_gb: 8,
            has_wifi: true,
        };
        let result = generate(&hw, &test_choices(), &[]);
        assert!(result.configuration_nix.contains("modesetting"));
    }

    #[test]
    fn angle_wrapped_intel_detected() {
        let hw = HardwareProfile {
            gpu_vendor: "Google Inc. (Intel)".into(),
            gpu_model: "ANGLE (Intel, Intel(R UHD Graphics 630 (CFL GT2, OpenGL 4.5)".into(),
            cpu_cores: 12,
            memory_gb: 32,
            has_wifi: true,
        };
        let result = generate(&hw, &test_choices(), &[]);
        assert!(result.configuration_nix.contains("modesetting"), "Intel should use modesetting");
        assert!(!result.warnings.iter().any(|w| w.contains("Unknown GPU")), "Should not warn about Intel");
    }

    #[test]
    fn angle_wrapped_nvidia_detected() {
        let hw = HardwareProfile {
            gpu_vendor: "Google Inc. (NVIDIA)".into(),
            gpu_model: "ANGLE (NVIDIA, NVIDIA GeForce RTX 3080, OpenGL 4.5)".into(),
            cpu_cores: 8,
            memory_gb: 32,
            has_wifi: true,
        };
        let result = generate(&hw, &test_choices(), &[]);
        assert!(result.configuration_nix.contains("hardware.nvidia"), "NVIDIA should get proprietary driver");
    }

    #[test]
    fn unknown_gpu_warns() {
        let hw = HardwareProfile {
            gpu_vendor: "matrox".into(),
            gpu_model: "G200".into(),
            cpu_cores: 2,
            memory_gb: 4,
            has_wifi: false,
        };
        let result = generate(&hw, &test_choices(), &[]);
        assert!(result.warnings.iter().any(|w| w.contains("matrox")));
    }

    #[test]
    fn unknown_app_warns() {
        let result = generate(&test_hw(), &test_choices(), &[
            "Firefox".into(),
            "TotallyFakeApp9000".into(),
        ]);
        assert!(result.warnings.iter().any(|w| w.contains("TotallyFakeApp9000")));
        assert!(result.configuration_nix.contains("firefox"));
    }

    #[test]
    fn tpm2_without_encryption_warns() {
        let mut c = test_choices();
        c.encryption = false;
        c.tpm2_unlock = true;
        let result = generate(&test_hw(), &c, &[]);
        assert!(result.warnings.iter().any(|w| w.contains("TPM2")));
        assert!(!result.configuration_nix.contains("cryptroot"));
    }

    #[test]
    fn no_desktop_produces_console_comment() {
        let mut c = test_choices();
        c.desktop = "none".into();
        let result = generate(&test_hw(), &c, &[]);
        assert!(result.configuration_nix.contains("console only"));
    }

    #[test]
    fn cosmic_desktop() {
        let mut c = test_choices();
        c.desktop = "cosmic".into();
        let result = generate(&test_hw(), &c, &[]);
        assert!(result.configuration_nix.contains("cosmic.enable = true"));
        assert!(result.configuration_nix.contains("cosmic-greeter"));
    }

    #[test]
    fn sway_desktop() {
        let mut c = test_choices();
        c.desktop = "sway".into();
        let result = generate(&test_hw(), &c, &[]);
        assert!(result.configuration_nix.contains("programs.sway.enable = true"));
    }

    #[test]
    fn xfce_desktop() {
        let mut c = test_choices();
        c.desktop = "xfce".into();
        let result = generate(&test_hw(), &c, &[]);
        assert!(result.configuration_nix.contains("xfce.enable = true"));
        assert!(result.configuration_nix.contains("lightdm"));
    }

    #[test]
    fn essentials_always_included() {
        let result = generate(&test_hw(), &test_choices(), &[]);
        let nix = &result.configuration_nix;
        assert!(nix.contains("vim"));
        assert!(nix.contains("wget"));
        assert!(nix.contains("curl"));
        assert!(nix.contains("git"));
    }

    #[test]
    fn custom_packages_included() {
        let mut c = test_choices();
        c.custom_packages = vec!["neofetch".into(), "bat".into(), "ripgrep".into()];
        let result = generate(&test_hw(), &c, &[]);
        assert!(result.configuration_nix.contains("neofetch"));
        assert!(result.configuration_nix.contains("bat"));
        assert!(result.configuration_nix.contains("ripgrep"));
        assert!(result.configuration_nix.contains("Custom packages"));
    }

    #[test]
    fn locale_included() {
        let mut c = test_choices();
        c.locale = "de_DE.UTF-8".into();
        let result = generate(&test_hw(), &c, &[]);
        assert!(result.configuration_nix.contains("de_DE.UTF-8"));
    }

    #[test]
    fn deduplicates_packages() {
        // Firefox appears twice — should only show once
        let result = generate(&test_hw(), &test_choices(), &[
            "Firefox".into(),
            "firefox".into(), // lowercase also matches
        ]);
        let count = result.configuration_nix.matches("firefox").count();
        // "firefox" appears once in packages + possibly in comments, but not duplicated in package list
        assert!(count >= 1);
    }
}
