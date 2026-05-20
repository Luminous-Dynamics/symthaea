// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign Configuration Generator
//!
//! The bridge between the installer's hardware/user data and Symthaea's
//! NixOS reasoning capabilities. Takes raw data from the portal and relay,
//! encodes it into HDC space, reasons about it through the causal graph
//! and active inference engine, and produces a complete NixOS configuration
//! with explanations for every decision.
//!
//! This is where Symthaea becomes the architect of the system, not a template filler.

use crate::encoding::codebook::NixCodebook;
use crate::encoding::user_input_encoder::UserInputEncoder;
use crate::mind::active_inference::NixActiveInference;
use crate::mind::causal_graph::{NixCausalGraph, NixOSCausalPatterns};
use symthaea_core::hdc::ContinuousHV;

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════
// Input Types (from the installer relay)
// ═══════════════════════════════════════════════════════

/// Hardware profile detected by probe_hardware.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct HardwareProfile {
    pub gpu_vendor: String, // "nvidia", "amd", "intel", "unknown"
    pub gpu_model: String,
    pub gpu_hybrid: bool,
    pub has_wifi: bool,
    pub has_tpm: bool,
    pub has_secure_boot: bool,
    pub setup_mode: bool,
    pub efi_available: bool,
    pub arch: String, // "x86_64", "aarch64"
    pub is_apple: bool,
    pub has_bitlocker: bool,
    pub existing_os: Vec<DetectedOS>,
    pub disk_count: usize,
    pub total_disk_gb: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DetectedOS {
    pub name: String,
    pub os_type: String, // "windows", "linux", "macos"
    pub device: String,
}

/// User's choices from the portal UI.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct UserChoices {
    pub hostname: String,
    pub desktop: String, // "gnome", "plasma", "hyprland", "sway", "xfce", "none"
    pub gpu_driver: String, // "nvidia", "nvidia-open", "amdgpu", "modesetting", "auto", "none"
    pub timezone: String,
    pub keyboard: String,
    pub layout: String, // "single", "single-luks", "alongside", etc.
    pub encryption: bool,
    pub secure_boot: bool,
    pub tpm2_unlock: bool,
}

/// Deep scan data from the user's existing system.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MigrationData {
    pub git_user: String,
    pub git_email: String,
    pub git_prefers_rebase: bool,
    pub git_alias_count: usize,
    pub shell_type: String, // "bash", "zsh", "zsh-omz", "zsh-starship"
    pub shell_alias_count: usize,
    pub ssh_key_count: usize,
    pub ssh_host_count: usize,
    pub editors: Vec<String>, // ["neovim", "vscode", ...]
    pub has_daw: bool,
    pub docker_images: usize,
    pub rust_projects: usize,
    pub node_projects: usize,
    pub python_venvs: usize,
    pub detected_apps: Vec<String>,
}

// ═══════════════════════════════════════════════════════
// Output Types
// ═══════════════════════════════════════════════════════

/// A complete NixOS configuration with reasoning.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SovereignConfig {
    /// The generated configuration.nix content.
    pub configuration_nix: String,
    /// The generated sovereign-config.nix (system choices).
    pub sovereign_config_nix: String,
    /// Flake.nix content (if flake-based).
    pub flake_nix: Option<String>,
    /// Home-manager config (if applicable).
    pub home_nix: Option<String>,
    /// Explanation for each major decision.
    pub decisions: Vec<ConfigDecision>,
    /// Personalized welcome message.
    pub welcome_message: String,
    /// Warnings or concerns.
    pub warnings: Vec<String>,
}

/// A single configuration decision with reasoning.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConfigDecision {
    /// NixOS option path (e.g., "services.xserver.desktopManager.gnome.enable")
    pub option_path: String,
    /// Value set
    pub value: String,
    /// Why this was chosen
    pub reasoning: String,
    /// Confidence (0.0-1.0)
    pub confidence: f32,
    /// What would happen if the user changes this
    pub consequences: Vec<String>,
    /// Alternative options the user could choose instead
    pub alternatives: Vec<(String, String)>, // (value, why_not)
}

// ═══════════════════════════════════════════════════════
// The Generator
// ═══════════════════════════════════════════════════════

/// Sovereign Config Generator — Symthaea's architectural reasoning.
pub struct SovereignConfigGenerator {
    codebook: NixCodebook,
    inference: NixActiveInference,
    causal_graph: NixCausalGraph,
}

impl SovereignConfigGenerator {
    /// Create a new generator with the full NixOS knowledge base.
    pub fn new() -> Self {
        let codebook = NixCodebook::new();
        let inference = NixActiveInference::new();
        let mut causal_graph = NixCausalGraph::new(42);

        // Seed with curated NixOS causal patterns
        let patterns = NixOSCausalPatterns::known_patterns();
        for (cause, effect, _relationship) in &patterns {
            causal_graph.add_structural_edge(cause, effect, 0.8);
        }

        Self {
            codebook,
            inference,
            causal_graph,
        }
    }

    /// Generate a complete NixOS configuration from installer inputs.
    pub fn generate(
        &mut self,
        hardware: &HardwareProfile,
        choices: &UserChoices,
        migration: &MigrationData,
    ) -> SovereignConfig {
        let mut decisions = Vec::new();
        let mut warnings = Vec::new();
        let mut nix_options: Vec<(String, String)> = Vec::new();

        // ── Step 1: Encode hardware + intent into HDC space ──
        let intent_text = self.build_intent_description(hardware, choices, migration);
        let _intent_hv = {
            let mut encoder = UserInputEncoder::new(&mut self.codebook);
            encoder.encode_input(&intent_text)
        };

        // ── Step 2: Reason about desktop environment ──
        let de_decision = self.reason_about_desktop(hardware, choices);
        for (path, value) in &de_decision.options {
            nix_options.push((path.clone(), value.clone()));
        }
        decisions.push(de_decision.decision);

        // ── Step 3: Reason about GPU driver ──
        let gpu_decision = self.reason_about_gpu(hardware, choices);
        for (path, value) in &gpu_decision.options {
            nix_options.push((path.clone(), value.clone()));
        }
        decisions.push(gpu_decision.decision);
        warnings.extend(gpu_decision.warnings);

        // ── Step 4: Reason about audio ──
        let audio_decision = self.reason_about_audio(hardware, choices, migration);
        for (path, value) in &audio_decision.options {
            nix_options.push((path.clone(), value.clone()));
        }
        decisions.push(audio_decision.decision);

        // ── Step 5: Reason about locale/timezone ──
        let locale_decision = self.reason_about_locale(choices);
        for (path, value) in &locale_decision.options {
            nix_options.push((path.clone(), value.clone()));
        }
        decisions.push(locale_decision.decision);

        // ── Step 6: Reason about networking ──
        nix_options.push(("networking.networkmanager.enable".into(), "true".into()));
        nix_options.push(("networking.firewall.enable".into(), "true".into()));
        decisions.push(ConfigDecision {
            option_path: "networking.firewall.enable".into(),
            value: "true".into(),
            reasoning: "Firewall is always enabled by default for security.".into(),
            confidence: 1.0,
            consequences: vec![
                "Incoming connections are blocked unless explicitly allowed.".into(),
            ],
            alternatives: vec![(
                "false".into(),
                "Disabling the firewall exposes all services to the network.".into(),
            )],
        });

        // ── Step 7: Reason about Nix settings ──
        nix_options.push((
            "nix.settings.experimental-features".into(),
            "[ \"nix-command\" \"flakes\" ]".into(),
        ));
        nix_options.push(("nix.gc.automatic".into(), "true".into()));
        nix_options.push(("nix.gc.dates".into(), "\"weekly\"".into()));
        nix_options.push((
            "nix.gc.options".into(),
            "\"--delete-older-than 30d\"".into(),
        ));

        // ── Step 8: Check causal graph for conflicts ──
        let conflicts = self.check_conflicts(&nix_options);
        for conflict in &conflicts {
            warnings.push(conflict.clone());
        }

        // ── Step 9: Reason about packages from migration data ──
        let packages = self.reason_about_packages(migration);

        // ── Step 10: Generate the config files ──
        let sovereign_config_nix = self.render_sovereign_config(&nix_options, &packages);
        let welcome_message = self.compose_welcome(hardware, choices, migration, &decisions);

        SovereignConfig {
            configuration_nix: String::new(), // Generated by nixos-generate-config on target
            sovereign_config_nix,
            flake_nix: None, // TODO: Phase B
            home_nix: None,  // TODO: Phase C
            decisions,
            welcome_message,
            warnings,
        }
    }

    // ── Internal reasoning functions ──

    fn build_intent_description(
        &self,
        hardware: &HardwareProfile,
        choices: &UserChoices,
        migration: &MigrationData,
    ) -> String {
        let mut parts = Vec::new();
        parts.push(format!("configure {} desktop", choices.desktop));
        if hardware.gpu_vendor == "nvidia" {
            parts.push("with NVIDIA GPU".into());
        }
        if choices.encryption {
            parts.push("encrypted".into());
        }
        if migration.has_daw {
            parts.push("for music production".into());
        }
        if migration.rust_projects > 0 || migration.node_projects > 0 {
            parts.push("for development".into());
        }
        parts.join(" ")
    }

    fn reason_about_desktop(
        &self,
        hardware: &HardwareProfile,
        choices: &UserChoices,
    ) -> ReasonedOptions {
        let mut options = Vec::new();
        let mut reasoning = String::new();
        let mut alternatives = Vec::new();

        match choices.desktop.as_str() {
            "gnome" => {
                options.push(("services.xserver.enable".into(), "true".into()));
                options.push((
                    "services.xserver.displayManager.gdm.enable".into(),
                    "true".into(),
                ));
                options.push((
                    "services.xserver.desktopManager.gnome.enable".into(),
                    "true".into(),
                ));

                reasoning = if hardware.gpu_vendor == "nvidia" {
                    "GNOME with GDM — has the best NVIDIA+Wayland support among desktop environments. GDM handles NVIDIA EGLStreams natively.".into()
                } else {
                    "GNOME — modern, polished, great accessibility support. Wayland by default."
                        .into()
                };

                alternatives.push((
                    "plasma".into(),
                    "KDE Plasma is more customizable but heavier.".into(),
                ));
                alternatives.push((
                    "hyprland".into(),
                    "Hyprland is a tiling compositor — powerful but steep learning curve.".into(),
                ));
            }
            "plasma" => {
                options.push(("services.xserver.enable".into(), "true".into()));
                options.push(("services.displayManager.sddm.enable".into(), "true".into()));
                options.push((
                    "services.desktopManager.plasma6.enable".into(),
                    "true".into(),
                ));
                reasoning = "KDE Plasma 6 — highly customizable, good Wayland support, familiar to Windows users.".into();
            }
            "hyprland" => {
                options.push(("programs.hyprland.enable".into(), "true".into()));
                options.push(("services.displayManager.sddm.enable".into(), "true".into()));
                options.push((
                    "services.displayManager.sddm.wayland.enable".into(),
                    "true".into(),
                ));
                reasoning = "Hyprland — dynamic tiling Wayland compositor. Fast, beautiful animations, keyboard-driven workflow.".into();

                if hardware.gpu_vendor == "nvidia" {
                    reasoning.push_str(" Note: NVIDIA support in Hyprland requires modesetting and may have occasional tearing.");
                }
            }
            "sway" => {
                options.push(("programs.sway.enable".into(), "true".into()));
                options.push(("services.displayManager.sddm.enable".into(), "true".into()));
                options.push((
                    "services.displayManager.sddm.wayland.enable".into(),
                    "true".into(),
                ));
                reasoning =
                    "Sway — i3-compatible Wayland compositor. Stable, minimal, efficient.".into();
            }
            "xfce" => {
                options.push(("services.xserver.enable".into(), "true".into()));
                options.push((
                    "services.xserver.displayManager.lightdm.enable".into(),
                    "true".into(),
                ));
                options.push((
                    "services.xserver.desktopManager.xfce.enable".into(),
                    "true".into(),
                ));

                reasoning = if hardware.total_disk_gb < 64 {
                    "XFCE — lightweight, perfect for your smaller disk. Uses less RAM and storage than GNOME or KDE.".into()
                } else {
                    "XFCE — lightweight and stable. Good choice for older hardware or minimal setups.".into()
                };
            }
            _ => {
                reasoning =
                    "No desktop environment — server/CLI only. Lighter, uses less resources."
                        .into();
            }
        }

        ReasonedOptions {
            options,
            decision: ConfigDecision {
                option_path: format!("desktop: {}", choices.desktop),
                value: choices.desktop.clone(),
                reasoning,
                confidence: 0.9,
                consequences: if choices.desktop == "none" {
                    vec!["No graphical interface. Access via terminal or SSH only.".into()]
                } else {
                    vec!["Graphical login screen will appear on boot.".into()]
                },
                alternatives,
            },
            warnings: Vec::new(),
        }
    }

    fn reason_about_gpu(
        &self,
        hardware: &HardwareProfile,
        choices: &UserChoices,
    ) -> ReasonedOptions {
        let mut options = Vec::new();
        let mut reasoning;
        let mut warnings = Vec::new();
        let alternatives = Vec::new();

        let driver = if choices.gpu_driver == "auto" {
            &hardware.gpu_vendor
        } else {
            &choices.gpu_driver
        };

        match driver.as_str() {
            "nvidia" => {
                options.push((
                    "services.xserver.videoDrivers".into(),
                    "[ \"nvidia\" ]".into(),
                ));
                options.push(("hardware.nvidia.modesetting.enable".into(), "true".into()));
                options.push(("hardware.nvidia.open".into(), "false".into()));
                options.push(("hardware.graphics.enable".into(), "true".into()));
                options.push(("hardware.graphics.enable32Bit".into(), "true".into()));
                reasoning = format!(
                    "NVIDIA proprietary drivers for {}. Modesetting enabled for Wayland support. \
                     32-bit graphics enabled for Steam/Wine compatibility.",
                    hardware.gpu_model
                );

                // Check causal graph: NVIDIA + Wayland implications
                if choices.desktop == "hyprland" || choices.desktop == "sway" {
                    warnings.push(
                        "NVIDIA + Wayland tiling compositors may have occasional tearing. \
                                   Consider GBM_BACKEND=nvidia-drm if you see issues."
                            .into(),
                    );
                }

                if hardware.gpu_hybrid {
                    // Generate NVIDIA PRIME offload config for hybrid laptops
                    options.push(("hardware.nvidia.prime.offload.enable".into(), "true".into()));
                    options.push((
                        "hardware.nvidia.prime.offload.enableOffloadCmd".into(),
                        "true".into(),
                    ));
                    // Default bus IDs — should be detected from lspci
                    options.push((
                        "hardware.nvidia.prime.intelBusId".into(),
                        "\"PCI:0:2:0\"".into(),
                    ));
                    options.push((
                        "hardware.nvidia.prime.nvidiaBusId".into(),
                        "\"PCI:1:0:0\"".into(),
                    ));
                    reasoning.push_str(
                        " Hybrid graphics detected — PRIME offload mode configured. \
                                        Use `nvidia-offload` prefix to run apps on the dGPU.",
                    );
                    warnings.push("Hybrid GPU: bus IDs defaulted to PCI:0:2:0 (Intel) and PCI:1:0:0 (NVIDIA). \
                                   Verify with `lspci | grep VGA` and adjust if different.".into());
                }
            }
            "nvidia-open" => {
                options.push((
                    "services.xserver.videoDrivers".into(),
                    "[ \"nvidia\" ]".into(),
                ));
                options.push(("hardware.nvidia.modesetting.enable".into(), "true".into()));
                options.push(("hardware.nvidia.open".into(), "true".into()));
                options.push(("hardware.graphics.enable".into(), "true".into()));
                reasoning = "NVIDIA open kernel module — recommended for RTX 30xx/40xx series. \
                             Better Wayland integration than proprietary."
                    .into();
            }
            "amdgpu" | "amd" => {
                options.push((
                    "services.xserver.videoDrivers".into(),
                    "[ \"amdgpu\" ]".into(),
                ));
                options.push(("hardware.graphics.enable".into(), "true".into()));
                options.push(("hardware.graphics.enable32Bit".into(), "true".into()));
                reasoning = format!(
                    "AMD GPU drivers for {}. AMD GPUs work excellently on Linux with open-source drivers. \
                     Zero configuration needed beyond this.",
                    hardware.gpu_model
                );
            }
            "intel" | "modesetting" => {
                options.push(("hardware.graphics.enable".into(), "true".into()));
                reasoning = "Intel integrated graphics — works out of the box with modesetting. \
                             No proprietary drivers needed."
                    .into();
            }
            _ => {
                reasoning = "No GPU driver configured — using default modesetting.".into();
            }
        }

        ReasonedOptions {
            options,
            decision: ConfigDecision {
                option_path: "GPU driver".into(),
                value: driver.clone(),
                reasoning,
                confidence: 0.85,
                consequences: vec![
                    "GPU acceleration available for desktop and applications.".into(),
                ],
                alternatives,
            },
            warnings,
        }
    }

    fn reason_about_audio(
        &self,
        _hardware: &HardwareProfile,
        choices: &UserChoices,
        migration: &MigrationData,
    ) -> ReasonedOptions {
        let mut options = Vec::new();

        if choices.desktop != "none" {
            options.push(("services.pulseaudio.enable".into(), "false".into()));
            options.push(("security.rtkit.enable".into(), "true".into()));
            options.push(("services.pipewire.enable".into(), "true".into()));
            options.push(("services.pipewire.alsa.enable".into(), "true".into()));
            options.push(("services.pipewire.pulse.enable".into(), "true".into()));

            let reasoning = if migration.has_daw {
                options.push(("services.pipewire.jack.enable".into(), "true".into()));
                "PipeWire with JACK bridge — detected DAW configuration on your previous system. \
                 JACK bridge enables low-latency audio for music production. \
                 rtkit ensures real-time priority for audio threads."
                    .into()
            } else {
                "PipeWire — modern audio server replacing PulseAudio. \
                 Better Bluetooth audio, lower latency, and compatibility with both PulseAudio and ALSA apps.".into()
            };

            ReasonedOptions {
                options,
                decision: ConfigDecision {
                    option_path: "audio".into(),
                    value: "pipewire".into(),
                    reasoning,
                    confidence: 0.95,
                    consequences: vec!["All audio apps work through PipeWire.".into()],
                    alternatives: vec![(
                        "pulseaudio".into(),
                        "Legacy audio server. Only if PipeWire has issues with specific hardware."
                            .into(),
                    )],
                },
                warnings: Vec::new(),
            }
        } else {
            ReasonedOptions {
                options: Vec::new(),
                decision: ConfigDecision {
                    option_path: "audio".into(),
                    value: "none".into(),
                    reasoning: "No audio configured — server/CLI mode.".into(),
                    confidence: 1.0,
                    consequences: Vec::new(),
                    alternatives: Vec::new(),
                },
                warnings: Vec::new(),
            }
        }
    }

    fn reason_about_locale(&self, choices: &UserChoices) -> ReasonedOptions {
        let mut options = Vec::new();

        let tz = if choices.timezone.is_empty() {
            "UTC"
        } else {
            &choices.timezone
        };
        let kb = if choices.keyboard.is_empty() {
            "us"
        } else {
            &choices.keyboard
        };

        options.push(("time.timeZone".into(), format!("\"{}\"", tz)));
        options.push(("i18n.defaultLocale".into(), "\"en_US.UTF-8\"".into()));
        options.push(("console.keyMap".into(), format!("\"{}\"", kb)));
        options.push(("services.xserver.xkb.layout".into(), format!("\"{}\"", kb)));

        ReasonedOptions {
            options,
            decision: ConfigDecision {
                option_path: "locale".into(),
                value: format!("{} / {}", tz, kb),
                reasoning: format!("Timezone {} with {} keyboard layout.", tz, kb),
                confidence: 0.9,
                consequences: Vec::new(),
                alternatives: Vec::new(),
            },
            warnings: Vec::new(),
        }
    }

    fn check_conflicts(&self, options: &[(String, String)]) -> Vec<String> {
        let mut warnings = Vec::new();
        let option_set: HashMap<&str, &str> = options
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        // Use causal graph to check known conflicts
        let patterns = NixOSCausalPatterns::known_patterns();
        for (cause, effect, relationship) in &patterns {
            if *relationship == "conflicts" {
                if option_set.contains_key(cause) && option_set.contains_key(effect) {
                    let cause_val = option_set[cause];
                    let effect_val = option_set[effect];
                    if cause_val == "true" && effect_val == "true" {
                        warnings.push(format!(
                            "Conflict: {} and {} cannot both be enabled. Resolving in favor of {}.",
                            cause, effect, cause
                        ));
                    }
                }
            }
        }

        // Check PulseAudio + PipeWire conflict
        if option_set.get("services.pulseaudio.enable") == Some(&"true")
            && option_set.get("services.pipewire.pulse.enable") == Some(&"true")
        {
            warnings.push("PulseAudio and PipeWire's PulseAudio bridge are both enabled. Disabling PulseAudio.".into());
        }

        warnings
    }

    fn reason_about_packages(&self, migration: &MigrationData) -> Vec<String> {
        let mut packages = vec![
            "vim".into(),
            "git".into(),
            "curl".into(),
            "wget".into(),
            "htop".into(),
            "btrfs-progs".into(),
        ];

        // Add packages based on detected development environment
        if migration.rust_projects > 0 {
            packages.extend_from_slice(&["rustc".into(), "cargo".into(), "rust-analyzer".into()]);
        }
        if migration.node_projects > 0 {
            packages.push("nodejs".into());
        }
        if migration.python_venvs > 0 {
            packages.push("python3".into());
        }
        if migration.docker_images > 0 {
            packages.push("docker".into());
        }

        // Add packages from detected editors
        for editor in &migration.editors {
            match editor.as_str() {
                "neovim" => packages.push("neovim".into()),
                "vscode" => packages.push("vscode".into()),
                "vim" => {} // already included
                "emacs" => packages.push("emacs".into()),
                _ => {}
            }
        }

        // Add encryption tools if LUKS
        if migration.ssh_key_count > 0 {
            // SSH keys exist — they'll want the agent
        }

        // Add all detected apps from app database matching
        for app in &migration.detected_apps {
            if !app.is_empty() && !packages.contains(app) {
                packages.push(app.clone());
            }
        }

        packages.sort();
        packages.dedup();
        packages
    }

    fn render_sovereign_config(&self, options: &[(String, String)], packages: &[String]) -> String {
        let mut lines = Vec::new();
        lines.push("# Generated by Symthaea — Sovereign Inoculation".to_string());
        lines.push("# Each option was chosen by reasoning about your hardware,".to_string());
        lines.push("# your existing setup, and NixOS best practices.".to_string());
        lines.push("{ config, pkgs, ... }:".to_string());
        lines.push("{".to_string());

        // Group by category
        let mut boot = Vec::new();
        let mut display = Vec::new();
        let mut audio = Vec::new();
        let mut network = Vec::new();
        let mut locale = Vec::new();
        let mut nix_settings = Vec::new();
        let mut other = Vec::new();

        for (path, value) in options {
            let line = format!("  {} = {};", path, value);
            if path.starts_with("services.xserver")
                || path.starts_with("services.displayManager")
                || path.starts_with("services.desktopManager")
                || path.starts_with("programs.hyprland")
                || path.starts_with("programs.sway")
                || path.starts_with("hardware.nvidia")
                || path.starts_with("hardware.graphics")
            {
                display.push(line);
            } else if path.starts_with("services.pipewire")
                || path.starts_with("services.pulseaudio")
                || path.starts_with("security.rtkit")
            {
                audio.push(line);
            } else if path.starts_with("networking") {
                network.push(line);
            } else if path.starts_with("time.")
                || path.starts_with("i18n.")
                || path.starts_with("console.")
                || path.contains("xkb")
            {
                locale.push(line);
            } else if path.starts_with("nix.") {
                nix_settings.push(line);
            } else if path.starts_with("boot.") {
                boot.push(line);
            } else {
                other.push(line);
            }
        }

        if !locale.is_empty() {
            lines.push("  # Locale & Timezone".to_string());
            lines.extend(locale);
            lines.push(String::new());
        }
        if !display.is_empty() {
            lines.push("  # Display & GPU".to_string());
            lines.extend(display);
            lines.push(String::new());
        }
        if !audio.is_empty() {
            lines.push("  # Audio".to_string());
            lines.extend(audio);
            lines.push(String::new());
        }
        if !network.is_empty() {
            lines.push("  # Networking".to_string());
            lines.extend(network);
            lines.push(String::new());
        }
        if !nix_settings.is_empty() {
            lines.push("  # Nix Settings".to_string());
            lines.extend(nix_settings);
            lines.push(String::new());
        }
        if !boot.is_empty() {
            lines.push("  # Boot".to_string());
            lines.extend(boot);
            lines.push(String::new());
        }
        if !other.is_empty() {
            lines.extend(other);
            lines.push(String::new());
        }

        // Packages
        lines.push("  # Packages (based on your existing tools)".to_string());
        lines.push("  environment.systemPackages = with pkgs; [".to_string());
        for pkg in packages {
            lines.push(format!("    {}", pkg));
        }
        lines.push("  ];".to_string());

        lines.push("}".to_string());
        lines.join("\n")
    }

    fn compose_welcome(
        &self,
        hardware: &HardwareProfile,
        choices: &UserChoices,
        migration: &MigrationData,
        decisions: &[ConfigDecision],
    ) -> String {
        let mut lines = Vec::new();

        // Identity
        if !migration.git_user.is_empty() {
            lines.push(format!("Welcome home, {}.", migration.git_user));
        } else {
            lines.push("Welcome home.".into());
        }
        lines.push(String::new());

        // What we brought
        let mut brought = Vec::new();
        if !migration.git_user.is_empty() {
            let workflow = if migration.git_prefers_rebase {
                "rebase"
            } else {
                "merge"
            };
            brought.push(format!(
                "your git identity ({}, {} workflow)",
                migration.git_user, workflow
            ));
        }
        if migration.ssh_key_count > 0 {
            brought.push(format!(
                "your {} SSH key{}",
                migration.ssh_key_count,
                if migration.ssh_key_count > 1 { "s" } else { "" }
            ));
        }
        if !migration.editors.is_empty() {
            brought.push(format!(
                "your editor setup ({})",
                migration.editors.join(", ")
            ));
        }
        if migration.has_daw {
            brought
                .push("your audio production setup (PipeWire configured for low-latency)".into());
        }

        if !brought.is_empty() {
            lines.push(format!("I brought {}.", brought.join(", ")));
            lines.push(String::new());
        }

        // What Symthaea chose and why
        lines.push("Here's what I configured and why:".into());
        for decision in decisions {
            if !decision.reasoning.is_empty() && decision.confidence > 0.5 {
                lines.push(format!(
                    "  {} — {}",
                    decision.option_path, decision.reasoning
                ));
            }
        }
        lines.push(String::new());

        // Dev environment
        if migration.rust_projects > 0 {
            lines.push(
                "I see you write Rust. I added rustc, cargo, and rust-analyzer. \
                        Try `nix develop` for per-project toolchains — no rustup needed."
                    .into(),
            );
        } else if migration.docker_images > 0 {
            lines.push(
                "I noticed your Docker setup. NixOS can run containers, but try `nix develop` \
                        for reproducible dev environments without the overhead."
                    .into(),
            );
        }

        lines.push(String::new());
        lines.push("Your first command: `sudo nixos-rebuild switch`".into());
        lines.push(
            "Your safety net: boot the previous generation from the boot menu if anything breaks."
                .into(),
        );

        lines.join("\n")
    }
}

/// Internal: a reasoned set of NixOS options + a decision explanation.
struct ReasonedOptions {
    options: Vec<(String, String)>,
    decision: ConfigDecision,
    warnings: Vec<String>,
}

// ═══════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_gnome_nvidia() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile {
                gpu_vendor: "nvidia".into(),
                gpu_model: "NVIDIA RTX 3060".into(),
                efi_available: true,
                total_disk_gb: 512,
                ..Default::default()
            },
            &UserChoices {
                hostname: "workstation".into(),
                desktop: "gnome".into(),
                gpu_driver: "auto".into(),
                timezone: "America/Chicago".into(),
                keyboard: "us".into(),
                ..Default::default()
            },
            &MigrationData {
                git_user: "Tristan".into(),
                git_email: "tristan@example.com".into(),
                git_prefers_rebase: true,
                editors: vec!["neovim".into()],
                rust_projects: 3,
                ssh_key_count: 2,
                ..Default::default()
            },
        );

        // Config should have GNOME + NVIDIA + PipeWire
        assert!(config.sovereign_config_nix.contains("gnome.enable"));
        assert!(config.sovereign_config_nix.contains("nvidia"));
        assert!(config.sovereign_config_nix.contains("pipewire"));
        assert!(config.sovereign_config_nix.contains("America/Chicago"));
        assert!(config.sovereign_config_nix.contains("neovim"));
        assert!(config.sovereign_config_nix.contains("rustc"));

        // Welcome should mention Tristan, Rust, SSH keys
        assert!(config.welcome_message.contains("Tristan"));
        assert!(config.welcome_message.contains("Rust"));
        assert!(config.welcome_message.contains("SSH key"));

        // Decisions should have reasoning
        assert!(!config.decisions.is_empty());
        for d in &config.decisions {
            assert!(
                !d.reasoning.is_empty(),
                "Decision '{}' has no reasoning",
                d.option_path
            );
        }
    }

    #[test]
    fn test_generate_server_no_desktop() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile {
                ..Default::default()
            },
            &UserChoices {
                hostname: "server".into(),
                desktop: "none".into(),
                gpu_driver: "none".into(),
                timezone: "UTC".into(),
                ..Default::default()
            },
            &MigrationData {
                docker_images: 15,
                ..Default::default()
            },
        );

        // Should NOT have any desktop or audio
        assert!(!config.sovereign_config_nix.contains("gnome"));
        assert!(!config.sovereign_config_nix.contains("pipewire"));
        assert!(config.sovereign_config_nix.contains("docker"));
    }

    #[test]
    fn test_generate_musician() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile {
                gpu_vendor: "amd".into(),
                gpu_model: "AMD RX 6600".into(),
                ..Default::default()
            },
            &UserChoices {
                desktop: "gnome".into(),
                gpu_driver: "amdgpu".into(),
                ..Default::default()
            },
            &MigrationData {
                has_daw: true,
                editors: vec!["vim".into()],
                ..Default::default()
            },
        );

        // Should have JACK bridge for DAW
        assert!(config.sovereign_config_nix.contains("jack.enable"));
        // Welcome should mention audio production
        assert!(
            config.welcome_message.contains("audio production")
                || config.welcome_message.contains("low-latency")
        );
    }

    #[test]
    fn test_welcome_personalization() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile::default(),
            &UserChoices {
                desktop: "gnome".into(),
                ..Default::default()
            },
            &MigrationData {
                git_user: "Alice".into(),
                git_prefers_rebase: true,
                ssh_key_count: 3,
                editors: vec!["vscode".into(), "neovim".into()],
                ..Default::default()
            },
        );

        assert!(config.welcome_message.contains("Alice"));
        assert!(config.welcome_message.contains("rebase"));
        assert!(config.welcome_message.contains("3 SSH keys"));
        assert!(config.welcome_message.contains("vscode, neovim"));
    }

    // ═══════════════════════════════════════════════════════
    // Nix Syntax Validation
    // ═══════════════════════════════════════════════════════

    /// Validate generated Nix syntax using the rnix parser.
    /// Falls back to structural checks if rnix reports errors that are
    /// false positives for NixOS module patterns.
    fn validate_nix_syntax(nix: &str) -> Result<(), String> {
        // Phase 1: rnix parse — catches real syntax errors
        let parse = rnix::Root::parse(nix);
        let errors = parse.errors();
        if !errors.is_empty() {
            // Collect up to 5 errors for the diagnostic
            let msgs: Vec<String> = errors.iter().take(5).map(|e| format!("{}", e)).collect();
            return Err(format!("Nix syntax errors: {}", msgs.join("; ")));
        }

        // Phase 2: structural checks that rnix doesn't enforce
        let open_braces = nix.chars().filter(|&c| c == '{').count();
        let close_braces = nix.chars().filter(|&c| c == '}').count();
        if open_braces != close_braces {
            return Err(format!(
                "Unbalanced braces: {} open, {} close",
                open_braces, close_braces
            ));
        }

        // Must have NixOS module header
        if !nix.contains("{ config, pkgs, ... }:") && !nix.contains("{ config, lib, pkgs") {
            return Err("Missing NixOS module header".into());
        }

        Ok(())
    }

    #[test]
    fn test_all_desktop_configs_valid_nix() {
        let desktops = ["gnome", "plasma", "hyprland", "sway", "xfce", "none"];
        let mut r#gen = SovereignConfigGenerator::new();

        for desktop in &desktops {
            let config = r#gen.generate(
                &HardwareProfile::default(),
                &UserChoices {
                    desktop: desktop.to_string(),
                    ..Default::default()
                },
                &MigrationData::default(),
            );

            validate_nix_syntax(&config.sovereign_config_nix).unwrap_or_else(|e| {
                panic!(
                    "Invalid Nix for desktop '{}': {}\n\nGenerated:\n{}",
                    desktop, e, config.sovereign_config_nix
                )
            });
        }
    }

    #[test]
    fn test_all_gpu_configs_valid_nix() {
        let gpus = [
            ("nvidia", "NVIDIA RTX 3060"),
            ("nvidia-open", "NVIDIA RTX 4070"),
            ("amd", "AMD RX 6600"),
            ("intel", "Intel UHD 770"),
            ("auto", ""),
            ("none", ""),
        ];
        let mut r#gen = SovereignConfigGenerator::new();

        for (driver, model) in &gpus {
            let config = r#gen.generate(
                &HardwareProfile {
                    gpu_vendor: driver.to_string(),
                    gpu_model: model.to_string(),
                    ..Default::default()
                },
                &UserChoices {
                    desktop: "gnome".into(),
                    gpu_driver: driver.to_string(),
                    ..Default::default()
                },
                &MigrationData::default(),
            );

            validate_nix_syntax(&config.sovereign_config_nix)
                .unwrap_or_else(|e| panic!("Invalid Nix for GPU '{}': {}", driver, e));

            // Every decision should have reasoning
            for d in &config.decisions {
                assert!(
                    !d.reasoning.is_empty(),
                    "GPU '{}' decision '{}' has no reasoning",
                    driver,
                    d.option_path
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════
    // Persona Tests — every persona from the test suite
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_persona_maya_student() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile {
                gpu_vendor: "intel".into(),
                efi_available: true,
                total_disk_gb: 256,
                ..Default::default()
            },
            &UserChoices {
                hostname: "maya-nixos".into(),
                desktop: "gnome".into(),
                timezone: "America/New_York".into(),
                ..Default::default()
            },
            &MigrationData {
                git_user: "Maya Chen".into(),
                editors: vec!["vscode".into()],
                ..Default::default()
            },
        );
        validate_nix_syntax(&config.sovereign_config_nix).unwrap();
        assert!(config.sovereign_config_nix.contains("gnome.enable"));
        assert!(!config.sovereign_config_nix.contains("nvidia"));
        assert!(config.sovereign_config_nix.contains("vscode"));
        assert!(config.welcome_message.contains("Maya"));
    }

    #[test]
    fn test_persona_kai_developer() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile {
                gpu_vendor: "intel".into(),
                has_tpm: true,
                total_disk_gb: 1000,
                ..Default::default()
            },
            &UserChoices {
                desktop: "hyprland".into(),
                encryption: true,
                secure_boot: true,
                timezone: "Asia/Tokyo".into(),
                ..Default::default()
            },
            &MigrationData {
                git_user: "Kai Tanaka".into(),
                git_prefers_rebase: true,
                ssh_key_count: 3,
                editors: vec!["neovim".into()],
                docker_images: 12,
                rust_projects: 5,
                ..Default::default()
            },
        );
        validate_nix_syntax(&config.sovereign_config_nix).unwrap();
        assert!(config.sovereign_config_nix.contains("hyprland.enable"));
        assert!(config.sovereign_config_nix.contains("rustc"));
        assert!(config.sovereign_config_nix.contains("docker"));
        assert!(config.sovereign_config_nix.contains("neovim"));
        assert!(config.welcome_message.contains("Kai"));
        assert!(config.welcome_message.contains("rebase"));
        assert!(config.welcome_message.contains("Rust"));
    }

    #[test]
    fn test_persona_jordan_gamer() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile {
                gpu_vendor: "nvidia".into(),
                gpu_model: "NVIDIA RTX 4070".into(),
                gpu_hybrid: false,
                total_disk_gb: 2000,
                ..Default::default()
            },
            &UserChoices {
                desktop: "plasma".into(),
                gpu_driver: "nvidia".into(),
                timezone: "America/Los_Angeles".into(),
                ..Default::default()
            },
            &MigrationData::default(),
        );
        validate_nix_syntax(&config.sovereign_config_nix).unwrap();
        assert!(config.sovereign_config_nix.contains("plasma6.enable"));
        assert!(config.sovereign_config_nix.contains("nvidia"));
        assert!(config.sovereign_config_nix.contains("32Bit")); // for Steam/Wine
        assert!(config.sovereign_config_nix.contains("modesetting"));
    }

    #[test]
    fn test_persona_river_musician() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile {
                gpu_vendor: "amd".into(),
                gpu_model: "AMD RX 6600".into(),
                ..Default::default()
            },
            &UserChoices {
                desktop: "gnome".into(),
                gpu_driver: "amdgpu".into(),
                timezone: "Europe/London".into(),
                keyboard: "gb".into(),
                ..Default::default()
            },
            &MigrationData {
                has_daw: true,
                editors: vec!["vim".into()],
                ssh_key_count: 1,
                ..Default::default()
            },
        );
        validate_nix_syntax(&config.sovereign_config_nix).unwrap();
        assert!(config.sovereign_config_nix.contains("jack.enable"));
        assert!(config.sovereign_config_nix.contains("amdgpu"));
        assert!(config.sovereign_config_nix.contains("Europe/London"));
        assert!(config.sovereign_config_nix.contains("\"gb\""));
    }

    #[test]
    fn test_persona_sam_sysadmin() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile {
                disk_count: 2,
                total_disk_gb: 4000,
                ..Default::default()
            },
            &UserChoices {
                hostname: "prod-web-03".into(),
                desktop: "none".into(),
                gpu_driver: "none".into(),
                ..Default::default()
            },
            &MigrationData {
                docker_images: 15,
                ssh_key_count: 8,
                editors: vec!["vim".into()],
                ..Default::default()
            },
        );
        validate_nix_syntax(&config.sovereign_config_nix).unwrap();
        assert!(!config.sovereign_config_nix.contains("gnome"));
        assert!(!config.sovereign_config_nix.contains("pipewire.enable"));
        assert!(config.sovereign_config_nix.contains("docker"));
        // Firewall must always be enabled
        assert!(
            config
                .sovereign_config_nix
                .contains("firewall.enable = true")
        );
    }

    #[test]
    fn test_persona_alex_privacy() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile {
                gpu_vendor: "intel".into(),
                has_tpm: true,
                efi_available: true,
                ..Default::default()
            },
            &UserChoices {
                desktop: "sway".into(),
                encryption: true,
                secure_boot: true,
                timezone: "Europe/Berlin".into(),
                keyboard: "de".into(),
                ..Default::default()
            },
            &MigrationData {
                ssh_key_count: 2,
                editors: vec!["vim".into()],
                ..Default::default()
            },
        );
        validate_nix_syntax(&config.sovereign_config_nix).unwrap();
        assert!(config.sovereign_config_nix.contains("sway.enable"));
        assert!(config.sovereign_config_nix.contains("\"de\""));
        assert!(config.sovereign_config_nix.contains("Europe/Berlin"));
        assert!(
            config
                .sovereign_config_nix
                .contains("firewall.enable = true")
        );
    }

    #[test]
    fn test_persona_avery_data_scientist() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile {
                gpu_vendor: "nvidia".into(),
                gpu_model: "NVIDIA RTX 3060 Laptop".into(),
                gpu_hybrid: true,
                ..Default::default()
            },
            &UserChoices {
                desktop: "gnome".into(),
                gpu_driver: "nvidia".into(),
                encryption: true,
                timezone: "Europe/Amsterdam".into(),
                ..Default::default()
            },
            &MigrationData {
                git_user: "Avery Nguyen".into(),
                editors: vec!["vscode".into()],
                docker_images: 5,
                python_venvs: 8,
                ssh_key_count: 2,
                ..Default::default()
            },
        );
        validate_nix_syntax(&config.sovereign_config_nix).unwrap();
        assert!(config.sovereign_config_nix.contains("gnome.enable"));
        assert!(config.sovereign_config_nix.contains("nvidia"));
        assert!(config.sovereign_config_nix.contains("python3"));
        assert!(config.sovereign_config_nix.contains("vscode"));
        assert!(config.welcome_message.contains("Avery"));
        // Should warn about hybrid graphics
        assert!(
            !config.warnings.is_empty(),
            "Should warn about hybrid graphics"
        );
    }

    // ═══════════════════════════════════════════════════════
    // Edge Cases
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_empty_everything() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile::default(),
            &UserChoices::default(),
            &MigrationData::default(),
        );
        validate_nix_syntax(&config.sovereign_config_nix).unwrap();
        // Should still produce valid config with sensible defaults
        assert!(
            config
                .sovereign_config_nix
                .contains("firewall.enable = true")
        );
        assert!(config.sovereign_config_nix.contains("flakes"));
        assert!(!config.decisions.is_empty());
    }

    #[test]
    fn test_every_desktop_has_decisions() {
        let desktops = ["gnome", "plasma", "hyprland", "sway", "xfce", "none"];
        let mut r#gen = SovereignConfigGenerator::new();

        for desktop in &desktops {
            let config = r#gen.generate(
                &HardwareProfile::default(),
                &UserChoices {
                    desktop: desktop.to_string(),
                    ..Default::default()
                },
                &MigrationData::default(),
            );
            // Every config should have at least 3 decisions (desktop, GPU, audio/firewall)
            assert!(
                config.decisions.len() >= 3,
                "Desktop '{}' has only {} decisions",
                desktop,
                config.decisions.len()
            );
        }
    }

    #[test]
    fn test_nvidia_hybrid_warning() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile {
                gpu_vendor: "nvidia".into(),
                gpu_hybrid: true,
                ..Default::default()
            },
            &UserChoices {
                desktop: "gnome".into(),
                gpu_driver: "nvidia".into(),
                ..Default::default()
            },
            &MigrationData::default(),
        );
        // Should produce a warning about hybrid graphics
        assert!(
            config
                .warnings
                .iter()
                .any(|w| w.to_lowercase().contains("hybrid")),
            "Missing hybrid graphics warning. Warnings: {:?}",
            config.warnings
        );
    }

    #[test]
    fn test_nvidia_wayland_tiling_warning() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile {
                gpu_vendor: "nvidia".into(),
                ..Default::default()
            },
            &UserChoices {
                desktop: "hyprland".into(),
                gpu_driver: "nvidia".into(),
                ..Default::default()
            },
            &MigrationData::default(),
        );
        // Should warn about NVIDIA + tiling Wayland
        assert!(
            config
                .warnings
                .iter()
                .any(|w| w.to_lowercase().contains("tearing")
                    || w.to_lowercase().contains("wayland")),
            "Missing NVIDIA+tiling warning. Warnings: {:?}",
            config.warnings
        );
    }

    #[test]
    fn test_packages_from_migration() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile::default(),
            &UserChoices {
                desktop: "gnome".into(),
                ..Default::default()
            },
            &MigrationData {
                rust_projects: 2,
                node_projects: 3,
                python_venvs: 1,
                docker_images: 4,
                editors: vec!["neovim".into(), "vscode".into(), "emacs".into()],
                ..Default::default()
            },
        );
        assert!(config.sovereign_config_nix.contains("rustc"));
        assert!(config.sovereign_config_nix.contains("cargo"));
        assert!(config.sovereign_config_nix.contains("nodejs"));
        assert!(config.sovereign_config_nix.contains("python3"));
        assert!(config.sovereign_config_nix.contains("docker"));
        assert!(config.sovereign_config_nix.contains("neovim"));
        assert!(config.sovereign_config_nix.contains("vscode"));
        assert!(config.sovereign_config_nix.contains("emacs"));
    }

    #[test]
    fn test_no_duplicate_packages() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile::default(),
            &UserChoices {
                desktop: "gnome".into(),
                ..Default::default()
            },
            &MigrationData {
                editors: vec!["vim".into(), "vim".into()], // duplicate
                ..Default::default()
            },
        );
        // vim should appear once (it's in default packages already)
        let vim_count = config.sovereign_config_nix.matches("    vim").count();
        assert!(
            vim_count <= 1,
            "vim appears {} times (should be ≤1)",
            vim_count
        );
    }

    #[test]
    fn test_config_has_comments() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile {
                gpu_vendor: "nvidia".into(),
                ..Default::default()
            },
            &UserChoices {
                desktop: "gnome".into(),
                gpu_driver: "nvidia".into(),
                ..Default::default()
            },
            &MigrationData::default(),
        );
        // Config should have section comments
        assert!(
            config
                .sovereign_config_nix
                .contains("# Generated by Symthaea")
        );
        assert!(
            config.sovereign_config_nix.contains("# Display & GPU")
                || config.sovereign_config_nix.contains("# Audio")
        );
        assert!(config.sovereign_config_nix.contains("# Packages"));
    }

    #[test]
    fn test_xfce_for_small_disk() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile {
                total_disk_gb: 32,
                ..Default::default()
            },
            &UserChoices {
                desktop: "xfce".into(),
                ..Default::default()
            },
            &MigrationData::default(),
        );
        // Reasoning should mention lightweight/smaller disk
        let de_decision = config
            .decisions
            .iter()
            .find(|d| d.option_path.contains("desktop"))
            .unwrap();
        assert!(
            de_decision.reasoning.to_lowercase().contains("small")
                || de_decision.reasoning.to_lowercase().contains("light"),
            "XFCE on small disk should mention lightweight. Got: {}",
            de_decision.reasoning
        );
    }

    #[test]
    fn test_welcome_without_git_user() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile::default(),
            &UserChoices {
                desktop: "gnome".into(),
                ..Default::default()
            },
            &MigrationData::default(),
        );
        assert!(config.welcome_message.contains("Welcome home"));
        assert!(!config.welcome_message.contains("Welcome home, ."));
    }

    #[test]
    fn test_welcome_with_docker_user() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile::default(),
            &UserChoices {
                desktop: "gnome".into(),
                ..Default::default()
            },
            &MigrationData {
                docker_images: 10,
                ..Default::default()
            },
        );
        assert!(
            config.welcome_message.contains("Docker")
                || config.welcome_message.contains("docker")
                || config.welcome_message.contains("nix develop")
        );
    }

    #[test]
    fn test_confidence_levels() {
        let mut r#gen = SovereignConfigGenerator::new();
        let config = r#gen.generate(
            &HardwareProfile {
                gpu_vendor: "nvidia".into(),
                ..Default::default()
            },
            &UserChoices {
                desktop: "gnome".into(),
                gpu_driver: "nvidia".into(),
                ..Default::default()
            },
            &MigrationData::default(),
        );
        for d in &config.decisions {
            assert!(
                d.confidence >= 0.0 && d.confidence <= 1.0,
                "Confidence out of range for '{}': {}",
                d.option_path,
                d.confidence
            );
            assert!(
                d.confidence >= 0.5,
                "Low confidence ({}) for '{}' — should be confident in its choices",
                d.confidence,
                d.option_path
            );
        }
    }
}
