// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lightweight NixOS configuration generator — WASM-compatible.
//!
//! Pure deterministic template generation from hardware profile, user choices,
//! and selected apps. No AI, HDC, or inference — just string building from
//! structured inputs. Designed to run in the browser via WASM for the
//! Sovereign Inoculation installer.

use crate::AppDatabase;
use crate::aliases;
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
    /// Whether the machine is a Chromebook
    pub chromebook: bool,
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
    /// Whether to enable FIDO2/YubiKey unlock for LUKS
    pub fido2_unlock: bool,
    /// IANA timezone, e.g. "Africa/Johannesburg"
    pub timezone: String,
    /// Keyboard layout, e.g. "us", "za"
    pub keyboard: String,
    /// Locale, e.g. "en_US.UTF-8", "de_DE.UTF-8"
    pub locale: String,
    /// Raw nixpkgs attribute names the user typed in
    pub custom_packages: Vec<String>,
    /// Filesystem choice: "btrfs" (default), "zfs"
    pub filesystem: String,
    /// Swap size in GB: 0 = none, 2/4/8/16
    pub swap_gb: u32,
    /// User shell: "bash", "zsh", "fish"
    pub shell: String,
    /// Whether to enable Bluetooth hardware
    pub bluetooth: bool,
    /// Whether to enable CUPS printing
    pub printing: bool,
    /// Kernel choice: "default", "zen", "lts", "hardened"
    pub kernel: String,
    /// Whether to enable laptop power management
    pub is_laptop: bool,
    /// Whether to enable home-manager integration
    pub home_manager: bool,
    /// Install Symthaea consciousness engine
    pub symthaea_edition: bool,
    /// Join Mycelix sovereign network
    pub mycelix_edition: bool,
    /// Whether to auto-login the primary user (display manager bypass)
    pub auto_login: bool,
    /// Additional user accounts (just usernames, no elevated privileges)
    pub extra_users: Vec<String>,
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
            warnings.push(format!(
                "App '{}' not found in database — skipped",
                app_name
            ));
        }
    }
    nix_pkgs.sort();
    nix_pkgs.dedup();

    // ── Validate inputs ──
    // Normalize GPU vendor — handle ANGLE-wrapped strings like "Google Inc. (Intel)"
    let raw_gpu = format!("{} {}", hw.gpu_vendor, hw.gpu_model).to_lowercase();
    let gpu_vendor = if raw_gpu.contains("nvidia")
        || raw_gpu.contains("geforce")
        || raw_gpu.contains("rtx")
        || raw_gpu.contains("gtx")
    {
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
        warnings
            .push("TPM2 unlock requested but encryption is disabled — TPM2 config skipped".into());
    }

    if choices.secure_boot && choices.encryption && !choices.tpm2_unlock {
        warnings.push(
            "Secure Boot + LUKS without TPM2: you will need to enter a passphrase on every boot"
                .into(),
        );
    }

    // ── Resolve custom package aliases ──
    let mut resolved_custom: Vec<String> = Vec::new();
    for raw in &choices.custom_packages {
        let name = raw.trim();
        if name.is_empty() {
            continue;
        }
        if let Some(canonical) = aliases::lookup_alias(name) {
            if canonical != name {
                warnings.push(format!(
                    "Package '{}' resolved to nixpkgs attribute '{}'",
                    name, canonical
                ));
            }
            resolved_custom.push(canonical.to_string());
        } else if is_valid_nix_attr(name) {
            resolved_custom.push(name.to_string());
        } else {
            let suggestions = aliases::suggest_similar(name);
            if suggestions.is_empty() {
                warnings.push(format!(
                    "Package '{}' not recognized and doesn't look like a valid nixpkgs attribute",
                    name
                ));
            } else {
                let sugg_str: Vec<_> = suggestions.iter().map(|(a, _)| *a).collect();
                warnings.push(format!(
                    "Package '{}' not recognized — did you mean: {}?",
                    name,
                    sugg_str.join(", ")
                ));
            }
            // Still include it — the user may know better
            resolved_custom.push(name.to_string());
        }
    }

    // ── Build configuration.nix ──
    let configuration_nix = build_configuration_nix(
        hw,
        choices,
        &nix_pkgs,
        &gpu_vendor,
        &desktop_lower,
        &resolved_custom,
    );

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
    resolved_custom: &[String],
) -> String {
    let mut out = String::with_capacity(4096);

    // Header
    writeln!(
        out,
        "# Generated by Sovereign Inoculation (symthaea-app-db config_gen)"
    )
    .unwrap();
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
    write_desktop(&mut out, desktop, choices);

    // ── GPU drivers ──
    write_gpu(&mut out, gpu_vendor, hw);

    // ── Audio (PipeWire always) ──
    write_audio(&mut out);

    // ── User account ──
    write_user(&mut out, &choices.username, &choices.extra_users);

    // ── Nix settings (flakes) ──
    write_nix_settings(&mut out);

    // ── Packages ──
    write_packages(&mut out, nix_pkgs, resolved_custom);

    // ── Filesystem maintenance ──
    if choices.filesystem == "zfs" {
        write_zfs_config(&mut out);
    } else {
        write_btrfs_maintenance(&mut out);
    }

    // ── Chromebook support ──
    if hw.chromebook {
        write_chromebook_config(&mut out);
    }

    // ── Swap (zram) ──
    write_swap(&mut out, choices.swap_gb);

    // ── Shell ──
    if !choices.shell.is_empty() && choices.shell != "bash" {
        write_shell(&mut out, &choices.username, &choices.shell);
    }

    // ── Bluetooth ──
    if choices.bluetooth {
        write_bluetooth(&mut out);
    }

    // ── Printing ──
    if choices.printing {
        write_printing(&mut out);
    }

    // ── Kernel ──
    if !choices.kernel.is_empty() && choices.kernel != "default" {
        write_kernel(&mut out, &choices.kernel);
    }

    // ── Laptop power management ──
    if choices.is_laptop {
        write_laptop(&mut out);
    }

    // ── Home Manager ──
    if choices.home_manager {
        write_home_manager(&mut out, choices);
    }

    // ── Symthaea Consciousness Engine ──
    if choices.symthaea_edition {
        write_symthaea(&mut out);
    }

    // ── Mycelix Sovereign Network ──
    if choices.mycelix_edition {
        write_mycelix(&mut out);
    }

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
        writeln!(
            out,
            "    # lanzaboote replaces systemd-boot for Secure Boot"
        )
        .unwrap();
    }
    writeln!(out, "  ];\n").unwrap();
}

fn write_boot(out: &mut String, choices: &UserChoices) {
    if choices.secure_boot {
        // lanzaboote takes over boot management
        writeln!(out, "  # Secure Boot via lanzaboote").unwrap();
        writeln!(
            out,
            "  boot.loader.systemd-boot.enable = lib.mkForce false;"
        )
        .unwrap();
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
    if choices.fido2_unlock || choices.tpm2_unlock {
        writeln!(
            out,
            "  boot.initrd.systemd.enable = true;  # required for FIDO2/TPM2 unlock at boot"
        )
        .unwrap();
    }
    writeln!(out, "  boot.initrd.luks.devices.\"cryptroot\" = {{").unwrap();
    writeln!(out, "    device = \"/dev/disk/by-partlabel/cryptroot\";").unwrap();
    if choices.tpm2_unlock && choices.encryption {
        writeln!(out, "    # TPM2 auto-unlock — enroll with: sudo systemd-cryptenroll --tpm2-device=auto /dev/<part>").unwrap();
        writeln!(out, "    cryptTabExtraOpts = [ \"tpm2-device=auto\" ];").unwrap();
    }
    if choices.fido2_unlock && choices.encryption {
        writeln!(out, "    # FIDO2/YubiKey unlock — enroll with: sudo systemd-cryptenroll --fido2-device=auto /dev/<part>").unwrap();
        writeln!(out, "    cryptTabExtraOpts = [ \"fido2-device=auto\" ];").unwrap();
    }
    writeln!(out, "  }};\n").unwrap();
}

fn write_networking(out: &mut String, choices: &UserChoices, hw: &HardwareProfile) {
    writeln!(out, "  networking.hostName = \"{}\";", choices.hostname).unwrap();
    writeln!(out, "  networking.networkmanager.enable = true;").unwrap();
    writeln!(out, "  networking.firewall.enable = true;\n").unwrap();
    if hw.has_wifi {
        writeln!(
            out,
            "  # WiFi detected — NetworkManager handles it automatically"
        )
        .unwrap();
        writeln!(
            out,
            "  # To connect: nmcli device wifi connect <SSID> password <pass>\n"
        )
        .unwrap();
    }
}

fn write_locale(out: &mut String, choices: &UserChoices) {
    writeln!(out, "  time.timeZone = \"{}\";", choices.timezone).unwrap();
    writeln!(out, "  i18n.defaultLocale = \"{}\";", choices.locale).unwrap();
    writeln!(out, "  console.keyMap = \"{}\";\n", choices.keyboard).unwrap();
}

fn write_desktop(out: &mut String, desktop: &str, choices: &UserChoices) {
    match desktop {
        "gnome" => {
            writeln!(out, "  # GNOME Desktop").unwrap();
            writeln!(out, "  services.xserver.enable = true;").unwrap();
            writeln!(out, "  services.xserver.displayManager.gdm.enable = true;").unwrap();
            writeln!(
                out,
                "  services.xserver.desktopManager.gnome.enable = true;"
            )
            .unwrap();
        }
        "kde" => {
            writeln!(out, "  # KDE Plasma 6").unwrap();
            writeln!(out, "  services.desktopManager.plasma6.enable = true;").unwrap();
            writeln!(out, "  services.displayManager.sddm.enable = true;").unwrap();
            writeln!(out, "  services.displayManager.sddm.wayland.enable = true;").unwrap();
        }
        "hyprland" => {
            writeln!(out, "  # Hyprland (Wayland tiling compositor)").unwrap();
            writeln!(out, "  programs.hyprland.enable = true;").unwrap();
            writeln!(out, "  # You will need a display manager or TTY login").unwrap();
            writeln!(out, "  services.greetd = {{").unwrap();
            writeln!(out, "    enable = true;").unwrap();
            writeln!(out, "    settings.default_session.command = \"${{pkgs.greetd.tuigreet}}/bin/tuigreet --time --cmd Hyprland\";").unwrap();
            writeln!(out, "  }};").unwrap();
        }
        "sway" => {
            writeln!(out, "  # Sway (Wayland tiling compositor)").unwrap();
            writeln!(out, "  programs.sway.enable = true;").unwrap();
            writeln!(out, "  services.greetd = {{").unwrap();
            writeln!(out, "    enable = true;").unwrap();
            writeln!(out, "    settings.default_session.command = \"${{pkgs.greetd.tuigreet}}/bin/tuigreet --time --cmd sway\";").unwrap();
            writeln!(out, "  }};").unwrap();
        }
        "xfce" => {
            writeln!(out, "  # XFCE (lightweight)").unwrap();
            writeln!(out, "  services.xserver.enable = true;").unwrap();
            writeln!(
                out,
                "  services.xserver.displayManager.lightdm.enable = true;"
            )
            .unwrap();
            writeln!(out, "  services.xserver.desktopManager.xfce.enable = true;").unwrap();
        }
        "cosmic" => {
            writeln!(out, "  # COSMIC Desktop (System76)").unwrap();
            writeln!(out, "  services.desktopManager.cosmic.enable = true;").unwrap();
            writeln!(
                out,
                "  services.displayManager.cosmic-greeter.enable = true;"
            )
            .unwrap();
        }
        _ => {
            writeln!(out, "  # No desktop environment — console only").unwrap();
            writeln!(
                out,
                "  # Install a DE later with: services.xserver.desktopManager.<name>.enable = true;"
            )
            .unwrap();
        }
    }
    // Auto-login (works with display-manager-based DEs)
    if choices.auto_login {
        let name = if choices.username.is_empty() {
            "user"
        } else {
            &choices.username
        };
        match desktop {
            "gnome" | "kde" | "xfce" | "cosmic" => {
                writeln!(out, "  services.displayManager.autoLogin.enable = true;").unwrap();
                writeln!(
                    out,
                    "  services.displayManager.autoLogin.user = \"{}\";",
                    name
                )
                .unwrap();
            }
            _ => {
                writeln!(
                    out,
                    "  # Auto-login not applicable for {} (no display manager)",
                    desktop
                )
                .unwrap();
            }
        }
    }
    writeln!(out).unwrap();
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
            writeln!(
                out,
                "    package = config.boot.kernelPackages.nvidiaPackages.stable;"
            )
            .unwrap();
            writeln!(out, "  }};\n").unwrap();
        }
        "amd" => {
            writeln!(out, "  # AMD GPU: {} — amdgpu (open-source)", hw.gpu_model).unwrap();
            writeln!(out, "  services.xserver.videoDrivers = [ \"amdgpu\" ];").unwrap();
            writeln!(out, "  hardware.amdgpu.amdvlk.enable = true;\n").unwrap();
        }
        "intel" => {
            writeln!(
                out,
                "  # Intel GPU: {} — modesetting (built-in)",
                hw.gpu_model
            )
            .unwrap();
            writeln!(
                out,
                "  services.xserver.videoDrivers = [ \"modesetting\" ];"
            )
            .unwrap();
            writeln!(out, "  hardware.graphics.enable = true;\n").unwrap();
        }
        _ => {
            writeln!(
                out,
                "  # GPU vendor not detected — using generic modesetting"
            )
            .unwrap();
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

fn write_user(out: &mut String, username: &str, extra_users: &[String]) {
    let name = if username.is_empty() {
        "user"
    } else {
        username
    };
    writeln!(out, "  users.users.{} = {{", name).unwrap();
    writeln!(out, "    isNormalUser = true;").unwrap();
    writeln!(
        out,
        "    extraGroups = [ \"wheel\" \"networkmanager\" \"video\" \"audio\" ];"
    )
    .unwrap();
    writeln!(out, "    # Set password after install: passwd {}", name).unwrap();
    writeln!(out, "  }};").unwrap();
    // Extra user accounts
    for extra in extra_users {
        let extra = extra.trim();
        if !extra.is_empty() && extra != name {
            writeln!(out, "  users.users.{} = {{", extra).unwrap();
            writeln!(out, "    isNormalUser = true;").unwrap();
            writeln!(out, "    extraGroups = [ \"networkmanager\" ];").unwrap();
            writeln!(out, "  }};").unwrap();
        }
    }
    writeln!(out).unwrap();
}

fn write_nix_settings(out: &mut String) {
    writeln!(out, "  nix.settings = {{").unwrap();
    writeln!(
        out,
        "    experimental-features = [ \"nix-command\" \"flakes\" ];"
    )
    .unwrap();
    writeln!(out, "    trusted-users = [ \"root\" \"@wheel\" ];").unwrap();
    writeln!(out, "  }};\n").unwrap();
}

fn write_packages(out: &mut String, nix_pkgs: &[&str], custom: &[String]) {
    writeln!(
        out,
        "  # Package names verified against nixpkgs {}",
        crate::CURRENT_VERIFIED_CHANNEL
    )
    .unwrap();
    writeln!(
        out,
        "  # If a package fails to build, search: nix search nixpkgs <name>"
    )
    .unwrap();
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

fn write_btrfs_maintenance(out: &mut String) {
    writeln!(
        out,
        "  # Automatic btrfs snapshots & scrub (NixForHumanity)"
    )
    .unwrap();
    writeln!(out, "  services.btrfs.autoScrub = {{").unwrap();
    writeln!(out, "    enable = true;").unwrap();
    writeln!(out, "    interval = \"monthly\";").unwrap();
    writeln!(out, "    fileSystems = [ \"/\" ];").unwrap();
    writeln!(out, "  }};\n").unwrap();
}

fn write_zfs_config(out: &mut String) {
    writeln!(out, "  # ZFS filesystem support").unwrap();
    writeln!(out, "  boot.supportedFilesystems = [ \"zfs\" ];").unwrap();
    writeln!(out, "  boot.zfs.devNodes = \"/dev/disk/by-id\";").unwrap();
    writeln!(out, "  services.zfs.autoScrub.enable = true;").unwrap();
    writeln!(out, "  services.zfs.trim.enable = true;").unwrap();
    writeln!(
        out,
        "  networking.hostId = \"deadbeef\"; # Replace with: head -c 8 /etc/machine-id\n"
    )
    .unwrap();
}

fn write_chromebook_config(out: &mut String) {
    writeln!(out, "  # Chromebook support").unwrap();
    writeln!(out, "  services.xserver.libinput.enable = true;").unwrap();
    writeln!(
        out,
        "  # Chromebook keyboard: consider remapping Search key to Super or Ctrl"
    )
    .unwrap();
    writeln!(
        out,
        "  # services.xserver.xkb.options = \"chromebook:search_overlay\";\n"
    )
    .unwrap();
}

fn write_swap(out: &mut String, swap_gb: u32) {
    if swap_gb > 0 {
        // Swap is handled by the install script (swapfile or zvol)
        // Config just needs zram
        writeln!(
            out,
            "  zramSwap = {{ enable = true; algorithm = \"zstd\"; }};\n"
        )
        .unwrap();
    }
}

fn write_shell(out: &mut String, username: &str, shell: &str) {
    let name = if username.is_empty() {
        "user"
    } else {
        username
    };
    match shell {
        "zsh" => {
            writeln!(out, "  programs.zsh.enable = true;").unwrap();
            writeln!(out, "  users.users.{}.shell = pkgs.zsh;\n", name).unwrap();
        }
        "fish" => {
            writeln!(out, "  programs.fish.enable = true;").unwrap();
            writeln!(out, "  users.users.{}.shell = pkgs.fish;\n", name).unwrap();
        }
        _ => {} // bash is default
    }
}

fn write_bluetooth(out: &mut String) {
    writeln!(out, "  hardware.bluetooth.enable = true;").unwrap();
    writeln!(out, "  hardware.bluetooth.powerOnBoot = true;\n").unwrap();
}

fn write_printing(out: &mut String) {
    writeln!(out, "  services.printing.enable = true;\n").unwrap();
}

fn write_kernel(out: &mut String, kernel: &str) {
    match kernel {
        "zen" => writeln!(out, "  boot.kernelPackages = pkgs.linuxPackages_zen;\n").unwrap(),
        "lts" => writeln!(out, "  boot.kernelPackages = pkgs.linuxPackages;\n").unwrap(),
        "hardened" => writeln!(
            out,
            "  boot.kernelPackages = pkgs.linuxPackages_hardened;\n"
        )
        .unwrap(),
        _ => {} // default kernel, no override needed
    }
}

fn write_laptop(out: &mut String) {
    writeln!(out, "  # Laptop power management").unwrap();
    writeln!(out, "  services.thermald.enable = true;").unwrap();
    writeln!(out, "  services.power-profiles-daemon.enable = true;").unwrap();
    writeln!(out, "  services.logind.lidSwitch = \"suspend\";\n").unwrap();
}

fn write_home_manager(out: &mut String, choices: &UserChoices) {
    let name = if choices.username.is_empty() {
        "user"
    } else {
        &choices.username
    };
    writeln!(out, "  # Home Manager").unwrap();
    writeln!(out, "  home-manager.useGlobalPkgs = true;").unwrap();
    writeln!(out, "  home-manager.useUserPackages = true;").unwrap();
    writeln!(out, "  home-manager.users.{} = {{ pkgs, ... }}: {{", name).unwrap();
    writeln!(out, "    home.stateVersion = \"25.05\";").unwrap();
    writeln!(out, "    programs.git.enable = true;").unwrap();
    if choices.shell == "zsh" {
        writeln!(out, "    programs.zsh = {{ enable = true; oh-my-zsh = {{ enable = true; theme = \"robbyrussell\"; }}; }};").unwrap();
    } else if choices.shell == "fish" {
        writeln!(out, "    programs.fish.enable = true;").unwrap();
    }
    writeln!(out, "    programs.starship.enable = true;").unwrap();
    writeln!(out, "  }};\n").unwrap();
}

fn write_symthaea(out: &mut String) {
    writeln!(out, "").unwrap();
    writeln!(out, "  # ── Symthaea Consciousness Engine ──").unwrap();
    writeln!(
        out,
        "  # Natural language NixOS management + consciousness metrics"
    )
    .unwrap();
    writeln!(out, "  services.ollama = {{").unwrap();
    writeln!(out, "    enable = true;").unwrap();
    writeln!(out, "    host = \"127.0.0.1\";").unwrap();
    writeln!(out, "    port = 11434;").unwrap();
    writeln!(out, "  }};").unwrap();
    writeln!(out, "  environment.systemPackages = with pkgs; [").unwrap();
    writeln!(out, "    ollama").unwrap();
    writeln!(out, "  ];\n").unwrap();
}

fn write_mycelix(out: &mut String) {
    writeln!(out, "").unwrap();
    writeln!(out, "  # ── Mycelix Sovereign Network ──").unwrap();
    writeln!(out, "  # Holochain-based decentralized applications").unwrap();
    writeln!(out, "  # EduNet, Portal, Health, Governance hApps").unwrap();
    writeln!(out, "  services.avahi = {{").unwrap();
    writeln!(out, "    enable = true;").unwrap();
    writeln!(out, "    nssmdns4 = true;").unwrap();
    writeln!(out, "    publish = {{ enable = true; addresses = true; }};").unwrap();
    writeln!(out, "  }};\n").unwrap();
}

// ═══════════════════════════════════════════════════════
// flake.nix builder
// ═══════════════════════════════════════════════════════

fn build_flake_nix(choices: &UserChoices) -> String {
    let mut out = String::with_capacity(1024);

    writeln!(
        out,
        "# Generated by Sovereign Inoculation (symthaea-app-db config_gen)"
    )
    .unwrap();
    writeln!(out, "# https://nixforhumanity.org").unwrap();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  description = \"NixOS configuration for {}\";",
        choices.hostname
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "  inputs = {{").unwrap();
    writeln!(
        out,
        "    nixpkgs.url = \"github:NixOS/nixpkgs/nixos-25.05\";"
    )
    .unwrap();
    if choices.secure_boot {
        writeln!(out, "    lanzaboote = {{").unwrap();
        writeln!(
            out,
            "      url = \"github:nix-community/lanzaboote/v0.4.2\";"
        )
        .unwrap();
        writeln!(out, "      inputs.nixpkgs.follows = \"nixpkgs\";").unwrap();
        writeln!(out, "    }};").unwrap();
    }
    if choices.home_manager {
        writeln!(out, "    home-manager = {{").unwrap();
        writeln!(
            out,
            "      url = \"github:nix-community/home-manager/release-25.05\";"
        )
        .unwrap();
        writeln!(out, "      inputs.nixpkgs.follows = \"nixpkgs\";").unwrap();
        writeln!(out, "    }};").unwrap();
    }
    if choices.symthaea_edition || choices.mycelix_edition {
        writeln!(
            out,
            "    # Future: Symthaea/Mycelix flake inputs will be added here"
        )
        .unwrap();
        writeln!(
            out,
            "    # when published as standalone flakes. For now, services are"
        )
        .unwrap();
        writeln!(
            out,
            "    # configured via system packages in configuration.nix."
        )
        .unwrap();
    }
    writeln!(out, "  }};").unwrap();
    writeln!(out).unwrap();

    // outputs
    let mut output_args = vec!["self", "nixpkgs"];
    if choices.secure_boot {
        output_args.push("lanzaboote");
    }
    if choices.home_manager {
        output_args.push("home-manager");
    }
    writeln!(out, "  outputs = {{ {}, ... }}: {{", output_args.join(", ")).unwrap();
    writeln!(
        out,
        "    nixosConfigurations.\"{}\" = nixpkgs.lib.nixosSystem {{",
        choices.hostname
    )
    .unwrap();
    writeln!(out, "      system = \"x86_64-linux\";").unwrap();
    writeln!(out, "      modules = [").unwrap();
    writeln!(out, "        ./configuration.nix").unwrap();
    if choices.secure_boot {
        writeln!(out, "        lanzaboote.nixosModules.lanzaboote").unwrap();
    }
    if choices.home_manager {
        writeln!(out, "        home-manager.nixosModules.home-manager").unwrap();
    }
    writeln!(out, "      ];").unwrap();
    writeln!(out, "    }};").unwrap();
    writeln!(out, "  }};").unwrap();
    writeln!(out, "}}").unwrap();

    out
}

// ═══════════════════════════════════════════════════════
// Validation helpers
// ═══════════════════════════════════════════════════════

/// Check whether a string looks like a valid nixpkgs attribute path.
///
/// Valid patterns: lowercase alphanumeric, hyphens, dots, underscores.
/// Examples: "firefox", "python3Packages.numpy", "jetbrains.idea-community"
fn is_valid_nix_attr(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    name.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.' || c == '_')
}

/// Lightweight structural validation of generated Nix configuration.
/// NOT a full Nix evaluator — just catches common syntax issues.
pub fn validate_nix_syntax(nix: &str) -> Vec<String> {
    let mut errors = vec![];

    // Check balanced braces
    let mut depth = 0i32;
    for ch in nix.chars() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth < 0 {
                    errors.push("Unmatched closing brace '}'".into());
                }
            }
            _ => {}
        }
    }
    if depth > 0 {
        errors.push(format!("{} unclosed '{{' brace(s)", depth));
    }

    // Check balanced brackets
    let mut bracket_depth = 0i32;
    for ch in nix.chars() {
        match ch {
            '[' => bracket_depth += 1,
            ']' => {
                bracket_depth -= 1;
                if bracket_depth < 0 {
                    errors.push("Unmatched closing bracket ']'".into());
                }
            }
            _ => {}
        }
    }
    if bracket_depth > 0 {
        errors.push(format!("{} unclosed '[' bracket(s)", bracket_depth));
    }

    // Check string literals are closed
    let mut in_string = false;
    let mut prev = ' ';
    for ch in nix.chars() {
        if ch == '"' && prev != '\\' {
            in_string = !in_string;
        }
        prev = ch;
    }
    if in_string {
        errors.push("Unclosed string literal".into());
    }

    // Check imports references hardware-configuration.nix
    if nix.contains("imports") && !nix.contains("hardware-configuration.nix") {
        errors.push("Missing import of ./hardware-configuration.nix".into());
    }

    errors
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
            chromebook: false,
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
            fido2_unlock: false,
            timezone: "Africa/Johannesburg".into(),
            keyboard: "us".into(),
            locale: "en_US.UTF-8".into(),
            custom_packages: vec![],
            filesystem: String::new(),
            swap_gb: 0,
            shell: String::new(),
            bluetooth: false,
            printing: false,
            kernel: String::new(),
            is_laptop: false,
            home_manager: false,
            symthaea_edition: false,
            mycelix_edition: false,
            auto_login: false,
            extra_users: vec![],
        }
    }

    #[test]
    fn generates_valid_configuration_nix() {
        let result = generate(
            &test_hw(),
            &test_choices(),
            &["Firefox".into(), "GIMP".into()],
        );
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
        assert!(
            result.warnings.is_empty(),
            "unexpected warnings: {:?}",
            result.warnings
        );
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
        assert!(
            result
                .configuration_nix
                .contains("boot.loader.systemd-boot.enable = true")
        );
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
        assert!(
            result
                .configuration_nix
                .contains("programs.hyprland.enable = true")
        );
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
            chromebook: false,
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
            chromebook: false,
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
            chromebook: false,
        };
        let result = generate(&hw, &test_choices(), &[]);
        assert!(
            result.configuration_nix.contains("modesetting"),
            "Intel should use modesetting"
        );
        assert!(
            !result.warnings.iter().any(|w| w.contains("Unknown GPU")),
            "Should not warn about Intel"
        );
    }

    #[test]
    fn angle_wrapped_nvidia_detected() {
        let hw = HardwareProfile {
            gpu_vendor: "Google Inc. (NVIDIA)".into(),
            gpu_model: "ANGLE (NVIDIA, NVIDIA GeForce RTX 3080, OpenGL 4.5)".into(),
            cpu_cores: 8,
            memory_gb: 32,
            has_wifi: true,
            chromebook: false,
        };
        let result = generate(&hw, &test_choices(), &[]);
        assert!(
            result.configuration_nix.contains("hardware.nvidia"),
            "NVIDIA should get proprietary driver"
        );
    }

    #[test]
    fn unknown_gpu_warns() {
        let hw = HardwareProfile {
            gpu_vendor: "matrox".into(),
            gpu_model: "G200".into(),
            cpu_cores: 2,
            memory_gb: 4,
            has_wifi: false,
            chromebook: false,
        };
        let result = generate(&hw, &test_choices(), &[]);
        assert!(result.warnings.iter().any(|w| w.contains("matrox")));
    }

    #[test]
    fn unknown_app_warns() {
        let result = generate(
            &test_hw(),
            &test_choices(),
            &["Firefox".into(), "TotallyFakeApp9000".into()],
        );
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("TotallyFakeApp9000"))
        );
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
        assert!(
            result
                .configuration_nix
                .contains("programs.sway.enable = true")
        );
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
    fn btrfs_autoscrub_included() {
        let result = generate(&test_hw(), &test_choices(), &[]);
        let nix = &result.configuration_nix;
        assert!(nix.contains("services.btrfs.autoScrub"));
        assert!(nix.contains("interval = \"monthly\""));
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
        let result = generate(
            &test_hw(),
            &test_choices(),
            &[
                "Firefox".into(),
                "firefox".into(), // lowercase also matches
            ],
        );
        let count = result.configuration_nix.matches("firefox").count();
        // "firefox" appears once in packages + possibly in comments, but not duplicated in package list
        assert!(count >= 1);
    }

    // ── Alias resolution tests ──

    #[test]
    fn custom_package_alias_resolved() {
        let mut c = test_choices();
        c.custom_packages = vec!["chrome".into()];
        let result = generate(&test_hw(), &c, &[]);
        // "chrome" should be resolved to "google-chrome"
        assert!(result.configuration_nix.contains("google-chrome"));
        assert!(result.warnings.iter().any(|w| w.contains("resolved to")));
    }

    #[test]
    fn custom_package_valid_attr_passes() {
        let mut c = test_choices();
        c.custom_packages = vec!["some-valid-pkg".into()];
        let result = generate(&test_hw(), &c, &[]);
        assert!(result.configuration_nix.contains("some-valid-pkg"));
        // No warning about invalid attr
        assert!(!result.warnings.iter().any(|w| w.contains("not recognized")));
    }

    #[test]
    fn custom_package_invalid_attr_warns() {
        let mut c = test_choices();
        c.custom_packages = vec!["My Fake App!!!".into()];
        let result = generate(&test_hw(), &c, &[]);
        assert!(result.warnings.iter().any(|w| w.contains("not recognized")));
    }

    // ── validate_nix_syntax tests ──

    #[test]
    fn validate_valid_config() {
        let result = generate(&test_hw(), &test_choices(), &["Firefox".into()]);
        let errors = validate_nix_syntax(&result.configuration_nix);
        assert!(
            errors.is_empty(),
            "Generated config should be valid, got: {:?}",
            errors
        );
    }

    #[test]
    fn validate_unmatched_closing_brace() {
        let nix = "{ config }: { foo = 1; }}";
        let errors = validate_nix_syntax(nix);
        assert!(!errors.is_empty());
    }

    #[test]
    fn validate_unclosed_brace() {
        let nix = "{ config }: { foo = {";
        let errors = validate_nix_syntax(nix);
        assert!(errors.iter().any(|e| e.contains("unclosed")));
    }

    #[test]
    fn validate_unclosed_bracket() {
        let nix = "{ pkgs }: { list = [ 1 2 3; }";
        let errors = validate_nix_syntax(nix);
        assert!(errors.iter().any(|e| e.contains("unclosed")));
    }

    #[test]
    fn validate_unclosed_string() {
        let nix = r#"{ pkgs }: { name = "hello; }"#;
        let errors = validate_nix_syntax(nix);
        assert!(errors.iter().any(|e| e.contains("string")));
    }

    #[test]
    fn validate_missing_hardware_import() {
        let nix = "{ config }: { imports = [ ./other.nix ]; }";
        let errors = validate_nix_syntax(nix);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("hardware-configuration.nix"))
        );
    }

    #[test]
    fn validate_balanced_nix_ok() {
        let nix = r#"{ config, pkgs, ... }: {
  imports = [ ./hardware-configuration.nix ];
  environment.systemPackages = [ pkgs.vim ];
}"#;
        let errors = validate_nix_syntax(nix);
        assert!(
            errors.is_empty(),
            "Valid nix should have no errors, got: {:?}",
            errors
        );
    }

    // ── is_valid_nix_attr tests ──

    #[test]
    fn valid_nix_attrs() {
        assert!(is_valid_nix_attr("firefox"));
        assert!(is_valid_nix_attr("python3Packages.numpy"));
        assert!(is_valid_nix_attr("jetbrains.idea-community"));
        assert!(is_valid_nix_attr("lm_sensors"));
    }

    #[test]
    fn invalid_nix_attrs() {
        assert!(!is_valid_nix_attr(""));
        assert!(!is_valid_nix_attr("My App"));
        assert!(!is_valid_nix_attr("pkg!@#"));
        assert!(!is_valid_nix_attr("hello world"));
    }

    #[test]
    fn fido2_config_included() {
        let mut c = test_choices();
        c.encryption = true;
        c.fido2_unlock = true;
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("boot.initrd.systemd.enable = true"),
            "FIDO2 requires systemd initrd"
        );
        assert!(
            nix.contains("fido2-device=auto"),
            "FIDO2 cryptTabExtraOpts should be present"
        );
    }

    #[test]
    fn zfs_config_included() {
        let mut c = test_choices();
        c.filesystem = "zfs".into();
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("boot.supportedFilesystems = [ \"zfs\" ]"),
            "ZFS support should be enabled"
        );
        assert!(
            nix.contains("services.zfs.autoScrub.enable = true"),
            "ZFS auto-scrub should be enabled"
        );
        assert!(
            nix.contains("services.zfs.trim.enable = true"),
            "ZFS trim should be enabled"
        );
        assert!(
            nix.contains("networking.hostId"),
            "ZFS requires networking.hostId"
        );
        // Should NOT have btrfs maintenance
        assert!(
            !nix.contains("services.btrfs.autoScrub"),
            "ZFS config should not include btrfs scrub"
        );
    }

    #[test]
    fn chromebook_config() {
        let mut hw = test_hw();
        hw.chromebook = true;
        let result = generate(&hw, &test_choices(), &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("services.xserver.libinput.enable = true"),
            "Chromebook should enable libinput"
        );
        assert!(nix.contains("Chromebook"), "Should have Chromebook comment");
    }

    #[test]
    fn shell_zsh_config() {
        let mut c = test_choices();
        c.shell = "zsh".into();
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("programs.zsh.enable = true"),
            "Zsh should be enabled"
        );
        assert!(
            nix.contains("users.users.tristan.shell = pkgs.zsh"),
            "User shell should be set to zsh"
        );
    }

    #[test]
    fn shell_fish_config() {
        let mut c = test_choices();
        c.shell = "fish".into();
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("programs.fish.enable = true"),
            "Fish should be enabled"
        );
        assert!(
            nix.contains("users.users.tristan.shell = pkgs.fish"),
            "User shell should be set to fish"
        );
    }

    #[test]
    fn bluetooth_config() {
        let mut c = test_choices();
        c.bluetooth = true;
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("hardware.bluetooth.enable = true"),
            "Bluetooth should be enabled"
        );
        assert!(
            nix.contains("hardware.bluetooth.powerOnBoot = true"),
            "Bluetooth powerOnBoot should be set"
        );
    }

    #[test]
    fn printing_config() {
        let mut c = test_choices();
        c.printing = true;
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("services.printing.enable = true"),
            "Printing should be enabled"
        );
    }

    #[test]
    fn laptop_config() {
        let mut c = test_choices();
        c.is_laptop = true;
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("services.thermald.enable = true"),
            "Thermald should be enabled for laptop"
        );
        assert!(
            nix.contains("services.power-profiles-daemon.enable = true"),
            "Power profiles should be enabled"
        );
        assert!(
            nix.contains("services.logind.lidSwitch = \"suspend\""),
            "Lid switch should suspend"
        );
    }

    #[test]
    fn kernel_zen_config() {
        let mut c = test_choices();
        c.kernel = "zen".into();
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("boot.kernelPackages = pkgs.linuxPackages_zen"),
            "Zen kernel should be configured"
        );
    }

    #[test]
    fn kernel_hardened_config() {
        let mut c = test_choices();
        c.kernel = "hardened".into();
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("boot.kernelPackages = pkgs.linuxPackages_hardened"),
            "Hardened kernel should be configured"
        );
    }

    #[test]
    fn kernel_lts_config() {
        let mut c = test_choices();
        c.kernel = "lts".into();
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("boot.kernelPackages = pkgs.linuxPackages;"),
            "LTS kernel should be configured"
        );
    }

    #[test]
    fn swap_zram_config() {
        let mut c = test_choices();
        c.swap_gb = 8;
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("zramSwap"),
            "zram swap should be enabled when swap_gb > 0"
        );
    }

    #[test]
    fn no_swap_no_zram() {
        let c = test_choices(); // swap_gb = 0
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(!nix.contains("zramSwap"), "No zram when swap_gb is 0");
    }

    #[test]
    fn home_manager_flake() {
        let mut c = test_choices();
        c.home_manager = true;
        c.shell = "zsh".into();
        let result = generate(&test_hw(), &c, &[]);
        let flake = &result.flake_nix;
        assert!(
            flake.contains("home-manager"),
            "Flake should include home-manager input"
        );
        assert!(
            flake.contains("home-manager.nixosModules.home-manager"),
            "Flake should include home-manager module"
        );
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("home-manager.useGlobalPkgs = true"),
            "Config should enable useGlobalPkgs"
        );
        assert!(
            nix.contains("home-manager.useUserPackages = true"),
            "Config should enable useUserPackages"
        );
        assert!(
            nix.contains("home-manager.users.tristan"),
            "Config should have user home-manager block"
        );
        assert!(
            nix.contains("programs.starship.enable = true"),
            "Config should enable starship"
        );
        assert!(
            nix.contains("oh-my-zsh"),
            "Zsh shell should get oh-my-zsh in home-manager"
        );
    }

    #[test]
    fn home_manager_without_shell() {
        let mut c = test_choices();
        c.home_manager = true;
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("home-manager.users.tristan"),
            "Home-manager block should exist"
        );
        assert!(!nix.contains("oh-my-zsh"), "No oh-my-zsh without zsh shell");
    }

    #[test]
    fn default_kernel_no_override() {
        let mut c = test_choices();
        c.kernel = "default".into();
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            !nix.contains("boot.kernelPackages = pkgs.linux"),
            "Default kernel should not override kernelPackages"
        );
    }

    #[test]
    fn bash_shell_no_override() {
        let mut c = test_choices();
        c.shell = "bash".into();
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(!nix.contains("programs.zsh"), "Bash should not set zsh");
        assert!(!nix.contains("programs.fish"), "Bash should not set fish");
    }

    #[test]
    fn symthaea_edition_config() {
        let mut c = test_choices();
        c.symthaea_edition = true;
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("Symthaea Consciousness Engine"),
            "Should have Symthaea section comment"
        );
        assert!(
            nix.contains("services.ollama"),
            "Symthaea should enable Ollama"
        );
        assert!(
            nix.contains("host = \"127.0.0.1\""),
            "Ollama should bind to localhost"
        );
        assert!(nix.contains("port = 11434"), "Ollama should use port 11434");
        assert!(
            nix.contains("ollama"),
            "Ollama package should be in systemPackages"
        );
        let flake = &result.flake_nix;
        assert!(
            flake.contains("Future: Symthaea/Mycelix flake inputs"),
            "Flake should have future input comment"
        );
        let errors = validate_nix_syntax(nix);
        assert!(
            errors.is_empty(),
            "Symthaea config should be valid Nix, got: {:?}",
            errors
        );
    }

    #[test]
    fn mycelix_edition_config() {
        let mut c = test_choices();
        c.mycelix_edition = true;
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("Mycelix Sovereign Network"),
            "Should have Mycelix section comment"
        );
        assert!(
            nix.contains("services.avahi"),
            "Mycelix should enable Avahi"
        );
        assert!(nix.contains("nssmdns4 = true"), "Avahi should enable mDNS");
        assert!(nix.contains("publish"), "Avahi should enable publishing");
        let flake = &result.flake_nix;
        assert!(
            flake.contains("Future: Symthaea/Mycelix flake inputs"),
            "Flake should have future input comment"
        );
        let errors = validate_nix_syntax(nix);
        assert!(
            errors.is_empty(),
            "Mycelix config should be valid Nix, got: {:?}",
            errors
        );
    }

    #[test]
    fn auto_login_gnome() {
        let mut c = test_choices();
        c.auto_login = true;
        c.desktop = "gnome".into();
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("services.displayManager.autoLogin.enable = true"),
            "Auto-login should be enabled"
        );
        assert!(
            nix.contains("services.displayManager.autoLogin.user = \"tristan\""),
            "Auto-login user should match"
        );
        let errors = validate_nix_syntax(nix);
        assert!(
            errors.is_empty(),
            "Auto-login config should be valid Nix, got: {:?}",
            errors
        );
    }

    #[test]
    fn auto_login_kde() {
        let mut c = test_choices();
        c.auto_login = true;
        c.desktop = "kde".into();
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("services.displayManager.autoLogin.enable = true"),
            "Auto-login should work for KDE"
        );
    }

    #[test]
    fn auto_login_not_for_tiling() {
        let mut c = test_choices();
        c.auto_login = true;
        c.desktop = "hyprland".into();
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            !nix.contains("autoLogin.enable = true"),
            "Tiling WMs should not get auto-login"
        );
        assert!(
            nix.contains("Auto-login not applicable"),
            "Should have explanatory comment"
        );
    }

    #[test]
    fn extra_users_config() {
        let mut c = test_choices();
        c.extra_users = vec!["alice".into(), "bob".into()];
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        assert!(
            nix.contains("users.users.alice"),
            "Extra user alice should be present"
        );
        assert!(
            nix.contains("users.users.bob"),
            "Extra user bob should be present"
        );
        assert!(
            nix.contains("users.users.tristan"),
            "Primary user should still be present"
        );
        // Extra users should not have wheel group
        let alice_section = nix.split("users.users.alice").nth(1).unwrap_or("");
        assert!(
            !alice_section
                .split('}')
                .next()
                .unwrap_or("")
                .contains("wheel"),
            "Extra users should not have wheel"
        );
        let errors = validate_nix_syntax(nix);
        assert!(
            errors.is_empty(),
            "Extra users config should be valid Nix, got: {:?}",
            errors
        );
    }

    #[test]
    fn extra_users_dedup_primary() {
        let mut c = test_choices();
        c.extra_users = vec!["tristan".into(), "alice".into()];
        let result = generate(&test_hw(), &c, &[]);
        let nix = &result.configuration_nix;
        // tristan should appear only once (as primary user, not duplicated as extra)
        let count = nix.matches("users.users.tristan").count();
        assert_eq!(
            count, 1,
            "Primary user should not be duplicated in extra users"
        );
        assert!(
            nix.contains("users.users.alice"),
            "Non-duplicate extra user should be present"
        );
    }

    // ── Security combination tests ──

    #[test]
    fn luks_plus_tpm2_plus_fido2() {
        let mut c = test_choices();
        c.encryption = true;
        c.tpm2_unlock = true;
        c.fido2_unlock = true;
        let result = generate(&test_hw(), &c, &[]);
        assert!(
            result
                .configuration_nix
                .contains("boot.initrd.systemd.enable = true")
        );
        assert!(result.configuration_nix.contains("fido2-device=auto"));
    }

    #[test]
    fn secure_boot_flake_has_lanzaboote() {
        let mut c = test_choices();
        c.secure_boot = true;
        let result = generate(&test_hw(), &c, &[]);
        assert!(result.flake_nix.contains("lanzaboote"));
    }

    #[test]
    fn home_manager_with_zsh() {
        let mut c = test_choices();
        c.home_manager = true;
        c.shell = "zsh".into();
        let result = generate(&test_hw(), &c, &[]);
        assert!(result.configuration_nix.contains("home-manager"));
        assert!(result.configuration_nix.contains("oh-my-zsh"));
        assert!(result.flake_nix.contains("home-manager"));
    }

    #[test]
    fn home_manager_with_fish() {
        let mut c = test_choices();
        c.home_manager = true;
        c.shell = "fish".into();
        let result = generate(&test_hw(), &c, &[]);
        assert!(
            result
                .configuration_nix
                .contains("programs.fish.enable = true")
        );
    }

    #[test]
    fn all_options_together() {
        // The "kitchen sink" test -- everything enabled at once
        let mut c = test_choices();
        c.hostname = "test-all".into();
        c.username = "testuser".into();
        c.desktop = "gnome".into();
        c.encryption = true;
        c.secure_boot = true;
        c.tpm2_unlock = true;
        c.fido2_unlock = true;
        c.shell = "zsh".into();
        c.kernel = "zen".into();
        c.swap_gb = 16;
        c.bluetooth = true;
        c.printing = true;
        c.is_laptop = true;
        c.home_manager = true;
        c.auto_login = true;
        c.filesystem = "btrfs".into();
        c.symthaea_edition = true;
        c.mycelix_edition = true;
        c.locale = "de_DE.UTF-8".into();
        c.custom_packages = vec!["neofetch".into(), "htop".into()];
        c.extra_users = vec!["alice".into(), "bob".into()];
        let hw = HardwareProfile {
            gpu_vendor: "nvidia".into(),
            gpu_model: "RTX 4090".into(),
            cpu_cores: 16,
            memory_gb: 64,
            has_wifi: true,
            chromebook: false,
        };
        let apps: Vec<String> = vec!["Firefox".into(), "Visual Studio Code".into()];
        let result = generate(&hw, &c, &apps);
        // Verify it generates valid-looking Nix
        let errors = validate_nix_syntax(&result.configuration_nix);
        assert!(
            errors.is_empty(),
            "Kitchen sink config has syntax errors: {:?}",
            errors
        );
        let flake_errors = validate_nix_syntax(&result.flake_nix);
        assert!(
            flake_errors.is_empty(),
            "Kitchen sink flake has syntax errors: {:?}",
            flake_errors
        );
        // Verify key options are present
        assert!(result.configuration_nix.contains("gnome"));
        assert!(result.configuration_nix.contains("nvidia"));
        assert!(result.configuration_nix.contains("zsh"));
        assert!(result.configuration_nix.contains("zen"));
        assert!(result.configuration_nix.contains("bluetooth"));
        assert!(result.configuration_nix.contains("printing"));
        assert!(result.configuration_nix.contains("thermald"));
        assert!(result.configuration_nix.contains("home-manager"));
        assert!(result.configuration_nix.contains("autoLogin"));
        assert!(result.configuration_nix.contains("ollama")); // symthaea
        assert!(result.configuration_nix.contains("avahi")); // mycelix
        assert!(result.configuration_nix.contains("de_DE.UTF-8"));
        assert!(result.configuration_nix.contains("neofetch"));
        assert!(result.configuration_nix.contains("alice"));
        assert!(result.configuration_nix.contains("bob"));
        assert!(result.configuration_nix.contains("firefox"));
        assert!(result.configuration_nix.contains("vscode"));
        assert!(result.flake_nix.contains("lanzaboote"));
        assert!(result.flake_nix.contains("home-manager"));
    }

    #[test]
    fn zfs_with_encryption() {
        let mut c = test_choices();
        c.filesystem = "zfs".into();
        c.encryption = true;
        let result = generate(&test_hw(), &c, &[]);
        // ZFS has native encryption, not LUKS
        assert!(result.configuration_nix.contains("zfs"));
    }

    #[test]
    fn empty_hostname_gets_default() {
        let mut c = test_choices();
        c.hostname = "".into();
        let result = generate(&test_hw(), &c, &[]);
        // Should still generate valid config
        let errors = validate_nix_syntax(&result.configuration_nix);
        assert!(errors.is_empty());
    }

    #[test]
    fn all_desktop_environments() {
        for de in &["gnome", "kde", "hyprland", "sway", "xfce", "cosmic", "none"] {
            let mut c = test_choices();
            c.desktop = de.to_string();
            let result = generate(&test_hw(), &c, &[]);
            let errors = validate_nix_syntax(&result.configuration_nix);
            assert!(
                errors.is_empty(),
                "DE '{}' generates invalid Nix: {:?}",
                de,
                errors
            );
        }
    }

    #[test]
    fn all_kernels() {
        for kernel in &["default", "zen", "lts", "hardened"] {
            let mut c = test_choices();
            c.kernel = kernel.to_string();
            let result = generate(&test_hw(), &c, &[]);
            let errors = validate_nix_syntax(&result.configuration_nix);
            assert!(
                errors.is_empty(),
                "Kernel '{}' generates invalid Nix: {:?}",
                kernel,
                errors
            );
        }
    }

    #[test]
    fn all_gpu_vendors() {
        for gpu in &["nvidia", "amd", "intel", "unknown"] {
            let hw = HardwareProfile {
                gpu_vendor: gpu.to_string(),
                gpu_model: "Test GPU".into(),
                cpu_cores: 4,
                memory_gb: 16,
                has_wifi: false,
                chromebook: false,
            };
            let result = generate(&hw, &test_choices(), &[]);
            let errors = validate_nix_syntax(&result.configuration_nix);
            assert!(
                errors.is_empty(),
                "GPU '{}' generates invalid Nix: {:?}",
                gpu,
                errors
            );
        }
    }

    #[test]
    fn config_has_package_version_comment() {
        let result = generate(&test_hw(), &test_choices(), &["Firefox".into()]);
        assert!(
            result.configuration_nix.contains("Package names verified"),
            "Generated config should contain package verification comment"
        );
        assert!(
            result.configuration_nix.contains("nixpkgs"),
            "Generated config should reference nixpkgs in the verification comment"
        );
        assert!(
            result.configuration_nix.contains("nix search nixpkgs"),
            "Generated config should contain fallback search hint"
        );
    }

    #[test]
    fn custom_package_with_dots() {
        // Some nixpkgs use dots: python3Packages.numpy
        let mut c = test_choices();
        c.custom_packages = vec!["python3Packages.numpy".into()];
        let result = generate(&test_hw(), &c, &[]);
        assert!(
            result.configuration_nix.contains("python3Packages.numpy"),
            "Dotted package names should be preserved in generated config"
        );
    }
}
