// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Nix codegen — neurosymbolic NixOS Flake configuration generation.
//!
//! Mirrors the brain_codegen pipeline (HDC encoder → System 1 classify →
//! System 2 idiom assembly → self-repair) but emits Nix expressions
//! instead of Rust. Uses `nix-instantiate --parse` for fast verification.
//!
//! Idioms were extracted from the user's actual ~/etc/nixos/ flake-based
//! config to ensure templates reflect real-world patterns.

use crate::language::learned_idioms::LearnedIdiomCache;
use crate::language::nix_kg::{ConflictClaim, NixKg};
use crate::language::nixpkgs_index::NixpkgsIndex;
use std::process::Command;
use std::sync::OnceLock;

/// Process-level shared knowledge graph (loads once, reuses bundled defaults
/// merged with `~/.cache/symthaea/nix-kg.json` if present).
static SHARED_KG: OnceLock<NixKg> = OnceLock::new();

pub fn shared_nix_kg() -> &'static NixKg {
    SHARED_KG.get_or_init(NixKg::from_disk_or_default)
}

// ─── Nix Intent Categories ─────────────────────────────────────────────────

/// High-level NixOS configuration intent — the "class" for Nix codegen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NixIntent {
    /// Dev environment (mkShell with language toolchain).
    DevShell,
    /// System-level service (services.X.enable + config).
    Service,
    /// Hardware / kernel config (NVIDIA, kernel params, sysctl).
    Hardware,
    /// Desktop environment (KDE, Sway, Wayland).
    Desktop,
    /// User account / groups / permissions.
    User,
    /// Home Manager (programs.X for user-level config).
    HomeManager,
    /// Networking / firewall.
    Networking,
    /// Secrets management (sops-nix, agenix).
    Secrets,
    /// Flake template (full project / system scaffolding).
    FlakeTemplate,
    /// Generic / unknown.
    Generic,
}

impl NixIntent {
    pub const ALL: [Self; 10] = [
        Self::DevShell,
        Self::Service,
        Self::Hardware,
        Self::Desktop,
        Self::User,
        Self::HomeManager,
        Self::Networking,
        Self::Secrets,
        Self::FlakeTemplate,
        Self::Generic,
    ];
}

/// What kind of Nix file fragment we're generating.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NixTarget {
    /// Standalone shell.nix (uses `pkgs ? import <nixpkgs> {}`).
    ShellNix,
    /// NixOS module (curly-brace block with services/hardware/users).
    NixosModule,
    /// home-manager module (programs/xdg/home blocks).
    HomeManager,
    /// flake.nix outputs fragment.
    FlakeFragment,
}

// ─── Nix Channels (16D feature vector for Nix intent) ─────────────────────

/// Lightweight feature vector for Nix prompts. Smaller than the 32-channel
/// algorithm encoder because Nix domain is more bounded.
#[derive(Clone, Debug, Default)]
pub struct NixChannels {
    /// 10 intent class one-hot (matches NixIntent::ALL order).
    pub intent: [f32; 10],
    /// Which programming language is mentioned (for dev shells).
    /// 0=none, 1=rust, 2=python, 3=node, 4=go, 5=haskell, 6=other
    pub language: f32,
    /// How many distinct packages/services mentioned.
    pub item_count: f32,
    /// Wants extras (buildInputs, plugins, extensions).
    pub has_extras: f32,
    /// Mentions specific port/IP (networking config).
    pub has_network_spec: f32,
    /// Mentions GPU / hardware acceleration.
    pub has_hardware: f32,
    /// Mentions group/user/permission.
    pub has_permission: f32,
    /// Mentions Wayland or Sway specifically.
    pub has_wayland: f32,
}

/// Extract Nix-relevant features from a natural-language prompt.
pub fn build_nix_channels(prompt: &str) -> NixChannels {
    let lower = prompt.to_lowercase();
    let mut ch = NixChannels::default();

    // ── Intent classification ──
    let intent = classify_nix_intent(&lower);
    let idx = NixIntent::ALL
        .iter()
        .position(|i| *i == intent)
        .unwrap_or(9);
    ch.intent[idx] = 1.0;

    // ── Language detection ──
    ch.language = if lower.contains("rust") || lower.contains("cargo") {
        1.0
    } else if lower.contains("python") || lower.contains("jupyter") || lower.contains("pandas") {
        2.0
    } else if lower.contains("node") || lower.contains("npm") || lower.contains("javascript") {
        3.0
    } else if lower.contains(" go ") || lower.contains("golang") {
        4.0
    } else if lower.contains("haskell") || lower.contains("ghc") {
        5.0
    } else {
        0.0
    };

    // ── Counts and flags ──
    ch.item_count = count_items(&lower) as f32;
    ch.has_extras = if lower.contains("with ") || lower.contains("plus ") || lower.contains("and ")
    {
        1.0
    } else {
        0.0
    };
    ch.has_network_spec = if lower.contains("port ")
        || lower.contains("firewall")
        || lower.contains("listen")
        || lower.contains("address")
    {
        1.0
    } else {
        0.0
    };
    ch.has_hardware = if lower.contains("nvidia")
        || lower.contains("gpu")
        || lower.contains("amd")
        || lower.contains("intel")
        || lower.contains("kernel")
    {
        1.0
    } else {
        0.0
    };
    ch.has_permission = if lower.contains("group")
        || lower.contains("user")
        || lower.contains("sudo")
        || lower.contains("permission")
    {
        1.0
    } else {
        0.0
    };
    ch.has_wayland =
        if lower.contains("wayland") || lower.contains("sway") || lower.contains("hyprland") {
            1.0
        } else {
            0.0
        };

    ch
}

fn count_items(lower: &str) -> usize {
    // Estimate: count " and " + commas + 1
    let ands = lower.matches(" and ").count();
    let commas = lower.matches(',').count();
    (ands + commas + 1).min(10)
}

/// Strong-keyword intent classifier (System 1 keyword prior for Nix).
pub fn classify_nix_intent(lower: &str) -> NixIntent {
    // Dev shell — broaden to catch "configure a python environment" etc.
    let dev_kw = lower.contains("dev environment")
        || lower.contains("development environment")
        || lower.contains("dev shell")
        || lower.contains("shell.nix");
    let lang_with_env = (lower.contains("rust") && lower.contains("environment"))
        || (lower.contains("python") && lower.contains("environment"))
        || (lower.contains("node") && lower.contains("environment"))
        || (lower.contains("haskell") && lower.contains("environment"))
        || (lower.contains("data-science") || lower.contains("data science"));
    let setup_lang = lower.contains("set up")
        && (lower.contains("rust")
            || lower.contains("python")
            || lower.contains("nodejs")
            || lower.contains("node.js")
            || lower.contains("node development")
            || lower.contains("haskell"));
    // Secrets management — check before generic Service so "encrypt" etc. routes here
    if lower.contains("sops")
        || lower.contains("agenix")
        || lower.contains("secret")
        || lower.contains("encrypted")
        || lower.contains("encrypt password")
        || lower.contains("manage credentials")
        || lower.contains("age key")
    {
        return NixIntent::Secrets;
    }
    // Flake template — full project/system scaffolding
    if lower.contains("flake template")
        || lower.contains("flake.nix template")
        || lower.contains("project scaffold")
        || lower.contains("complete flake")
        || (lower.contains("full") && lower.contains("flake"))
        || (lower.contains("system flake") || lower.contains("nixosconfigurations"))
    {
        return NixIntent::FlakeTemplate;
    }

    if dev_kw || lang_with_env || setup_lang {
        NixIntent::DevShell
    } else if lower.contains("nvidia")
        || lower.contains("gpu")
        || lower.contains("kernel")
        || lower.contains("hardware")
        || lower.contains("driver")
    {
        NixIntent::Hardware
    } else if lower.contains("wayland")
        || lower.contains("sway")
        || lower.contains("hyprland")
        || lower.contains("kde")
        || lower.contains("gnome")
        || lower.contains("desktop")
        || lower.contains("font")
    {
        NixIntent::Desktop
    } else if lower.contains("port ") || lower.contains("firewall") {
        NixIntent::Networking
    } else if lower.contains("home-manager")
        || lower.contains("home manager")
        || lower.contains("dotfile")
        || lower.contains("user-level")
    {
        NixIntent::HomeManager
    } else if lower.contains("enable ") || shared_nix_kg().matches_service_keyword(lower) {
        // Service keywords (postgres, nginx, tailscale, prometheus, …) live
        // in the shared knowledge graph — extend via
        // ~/.cache/symthaea/nix-kg.json without recompiling.
        NixIntent::Service
    } else if lower.contains("user")
        || lower.contains("group")
        || lower.contains("permission")
        || lower.contains("sudo")
    {
        NixIntent::User
    } else {
        NixIntent::Generic
    }
}

// ─── Nix Idiom Library ─────────────────────────────────────────────────────

/// A reusable Nix idiom — intent matcher + expression template.
///
/// Templates use `{LANG}`, `{EXTRAS}`, etc. as substitution markers.
pub struct NixIdiom {
    pub name: &'static str,
    pub intent: NixIntent,
    pub target: NixTarget,
}

/// Match prompt to the best idiom and return generated Nix expression.
///
/// Returns None if no idiom matches; falls through to template assembly.
pub fn nix_idiom_body(prompt: &str) -> Option<String> {
    let lower = prompt.to_lowercase();

    // Fast path: prompts like "set time zone to Africa/Johannesburg"
    // map to `time.timeZone = "..."`. Detected here (before intent
    // classification) rather than adding a new NixIntent variant —
    // time-zone prompts are rare and the taxonomy is stable at 10
    // variants. Uses the ORIGINAL prompt casing so IANA zone names
    // emit verbatim, not lowercased.
    if let Some(body) = emit_time_zone(prompt, &lower) {
        return Some(body);
    }

    let intent = classify_nix_intent(&lower);

    match intent {
        NixIntent::DevShell => emit_dev_shell(&lower),
        NixIntent::Service => emit_service(&lower),
        NixIntent::Hardware => emit_hardware(&lower),
        NixIntent::Desktop => emit_desktop(&lower),
        NixIntent::User => emit_user_group(&lower),
        NixIntent::Networking => emit_networking(&lower),
        NixIntent::HomeManager => emit_home_manager(&lower),
        NixIntent::Secrets => emit_secrets(&lower),
        NixIntent::FlakeTemplate => emit_flake_template(&lower),
        NixIntent::Generic => None,
    }
}

/// Emit a `time.timeZone = "..."` config when the prompt is a time-zone
/// set request. Returns None if the prompt doesn't look time-zone-shaped.
/// Uses both the original-cased `prompt` (to recover IANA zone names
/// with their proper capitalization) and the pre-lowercased `lower`
/// (for matching).
fn emit_time_zone(prompt: &str, lower: &str) -> Option<String> {
    if !(lower.contains("time zone") || lower.contains("timezone")) {
        return None;
    }
    // Scan the ORIGINAL prompt for an IANA-zone token: looks like
    // `Africa/Johannesburg`, `America/New_York`, `Asia/Tokyo`. Must
    // contain `/` and be composed of ASCII alpha + underscores.
    let zone = prompt.split_whitespace().find_map(|w| {
        let cleaned = w.trim_end_matches(',').trim_end_matches('.');
        if cleaned.contains('/')
            && cleaned
                .chars()
                .all(|c| c.is_ascii_alphabetic() || c == '/' || c == '_')
        {
            Some(cleaned.to_string())
        } else {
            None
        }
    })?;
    Some(format!(
        r#"{{
  time.timeZone = "{zone}";
}}
"#
    ))
}

// ── Dev Shell Emitters ──

fn emit_dev_shell(lower: &str) -> Option<String> {
    // Rust dev shell — extracted from /home/tstoltz/nix-templates/rust-shell.nix
    if lower.contains("rust") {
        let mut tools: Vec<&str> = vec!["rustc", "cargo", "rustfmt", "clippy"];
        if lower.contains("rust-analyzer") || lower.contains("analyzer") {
            tools.push("rust-analyzer");
        }
        if lower.contains("mold") {
            tools.push("mold");
        }
        let has_sccache = lower.contains("sccache");
        if has_sccache {
            tools.push("sccache");
        }
        if !tools.contains(&"pkg-config") && lower.contains("openssl") {
            tools.extend_from_slice(&["pkg-config", "openssl", "openssl.dev"]);
        }
        let pkg_list = tools.join(" ");
        // sccache is useless as a plain package — it must be set as
        // RUSTC_WRAPPER for rustc to route through it. The scorer caught
        // a case where sccache was added to buildInputs but the env var
        // was missing, resulting in a silent-footgun shell.
        let rustc_wrapper_line = if has_sccache {
            "  RUSTC_WRAPPER = \"sccache\";\n"
        } else {
            ""
        };
        return Some(format!(
            r#"{{ pkgs ? import <nixpkgs> {{}} }}:
pkgs.mkShell {{
  name = "rust-dev-shell";
  buildInputs = with pkgs; [ {pkg_list} ];
{rustc_wrapper_line}  shellHook = ''
    export RUST_BACKTRACE=1
  '';
}}
"#
        ));
    }

    // Python dev shell — extracted from python-shell.nix
    if lower.contains("python") {
        let mut py_pkgs: Vec<&str> = vec!["pip", "setuptools", "wheel"];
        if lower.contains("jupyter") {
            py_pkgs.push("jupyter");
            py_pkgs.push("ipykernel");
        }
        if lower.contains("pandas")
            || lower.contains("data")
            || lower.contains("data-science")
            || lower.contains("data science")
        {
            py_pkgs.extend_from_slice(&["numpy", "pandas", "matplotlib"]);
        }
        if lower.contains("scikit") || lower.contains("ml") || lower.contains("machine") {
            py_pkgs.push("scikit-learn");
        }
        if lower.contains("test") {
            py_pkgs.extend_from_slice(&["pytest", "mypy"]);
        }
        let py_list = py_pkgs.join(" ");
        return Some(format!(
            r#"{{ pkgs ? import <nixpkgs> {{}} }}:
let
  pythonEnv = pkgs.python311.withPackages (ps: with ps; [ {py_list} ]);
in
pkgs.mkShell {{
  name = "python-dev-shell";
  buildInputs = with pkgs; [ pythonEnv poetry ];
  shellHook = ''
    export PYTHONDONTWRITEBYTECODE=1
  '';
}}
"#
        ));
    }

    // Node dev shell
    if lower.contains("node") || lower.contains("npm") {
        let mut tools: Vec<&str> = vec!["nodejs_20", "nodePackages.npm"];
        if lower.contains("typescript") {
            tools.push("nodePackages.typescript");
        }
        if lower.contains("webpack") {
            tools.push("nodePackages.webpack");
        }
        let pkg_list = tools.join(" ");
        return Some(format!(
            r#"{{ pkgs ? import <nixpkgs> {{}} }}:
pkgs.mkShell {{
  name = "nodejs-dev-shell";
  buildInputs = with pkgs; [ {pkg_list} ];
  shellHook = ''
    export NODE_ENV="development"
  '';
}}
"#
        ));
    }

    None
}

// ── Service Emitters ──

fn emit_service(lower: &str) -> Option<String> {
    // PostgreSQL — extracted from configuration.nix:454
    if lower.contains("postgres") || lower.contains("postgresql") {
        let extensions = if lower.contains("pgvector") {
            "    extraPlugins = with pkgs.postgresql_16.pkgs; [ pgvector ];\n"
        } else if lower.contains("postgis") {
            "    extraPlugins = with pkgs.postgresql_16.pkgs; [ postgis ];\n"
        } else {
            ""
        };
        return Some(format!(
            r#"{{ pkgs, ... }}: {{
  services.postgresql = {{
    enable = true;
    package = pkgs.postgresql_16;
    enableTCPIP = true;
    settings.listen_addresses = "127.0.0.1";
{extensions}    authentication = ''
      local all all trust
      host all all 127.0.0.1/32 trust
    '';
  }};
}}
"#
        ));
    }

    // Podman — check BEFORE docker since "podman as docker alternative"
    // contains both keywords and the user wants podman.
    if lower.contains("podman") {
        return Some(
            r#"{
  virtualisation.podman = {
    enable = true;
    dockerCompat = true;
    defaultNetwork.settings.dns_enabled = true;
  };
}
"#
            .to_string(),
        );
    }

    // Docker — combines virtualisation.docker + user group
    if lower.contains("docker") {
        let user_block =
            if lower.contains("my user") || lower.contains("add user") || lower.contains("group") {
                "  users.users.tstoltz.extraGroups = [ \"docker\" ];\n"
            } else {
                ""
            };
        return Some(format!(
            r#"{{ config, pkgs, ... }}: {{
  virtualisation.docker = {{
    enable = true;
    enableOnBoot = true;
  }};
{user_block}}}
"#
        ));
    }

    // Redis
    if lower.contains("redis") {
        return Some(
            r#"{
  services.redis.servers."".enable = true;
}
"#
            .to_string(),
        );
    }

    // Nginx
    if lower.contains("nginx") {
        return Some(
            r#"{
  services.nginx = {
    enable = true;
    recommendedGzipSettings = true;
    recommendedOptimisation = true;
    recommendedTlsSettings = true;
  };
  networking.firewall.allowedTCPPorts = [ 80 443 ];
}
"#
            .to_string(),
        );
    }

    // IPFS / Kubo
    if lower.contains("ipfs") || lower.contains("kubo") {
        return Some(
            r#"{
  services.kubo = {
    enable = true;
    settings = {
      Addresses = {
        API = "/ip4/127.0.0.1/tcp/5001";
        Gateway = "/ip4/0.0.0.0/tcp/8081";
        Swarm = [ "/ip4/0.0.0.0/tcp/4001" ];
      };
      Datastore.StorageMax = "50GB";
    };
  };
  networking.firewall.allowedTCPPorts = [ 8081 4001 ];
}
"#
            .to_string(),
        );
    }

    // OpenSSH
    if lower.contains("openssh") || lower.contains("ssh server") {
        return Some(
            r#"{
  services.openssh = {
    enable = true;
    settings = {
      PasswordAuthentication = false;
      PermitRootLogin = "no";
      KbdInteractiveAuthentication = false;
    };
  };
  networking.firewall.allowedTCPPorts = [ 22 ];
}
"#
            .to_string(),
        );
    }

    // Tailscale
    if lower.contains("tailscale") {
        return Some(
            r#"{
  services.tailscale = {
    enable = true;
    openFirewall = true;
  };
}
"#
            .to_string(),
        );
    }

    // Prometheus
    if lower.contains("prometheus") {
        return Some(
            r#"{
  services.prometheus = {
    enable = true;
    port = 9090;
    exporters.node = {
      enable = true;
      enabledCollectors = [ "systemd" ];
      port = 9100;
    };
  };
  networking.firewall.allowedTCPPorts = [ 9090 9100 ];
}
"#
            .to_string(),
        );
    }

    // Grafana
    if lower.contains("grafana") {
        return Some(
            r#"{
  services.grafana = {
    enable = true;
    settings.server = {
      http_addr = "127.0.0.1";
      http_port = 3000;
    };
  };
}
"#
            .to_string(),
        );
    }

    // Jellyfin / Plex media server
    if lower.contains("jellyfin") {
        return Some(
            r#"{
  services.jellyfin = {
    enable = true;
    openFirewall = true;
  };
}
"#
            .to_string(),
        );
    }
    if lower.contains("plex") {
        return Some(
            r#"{
  services.plex = {
    enable = true;
    openFirewall = true;
  };
}
"#
            .to_string(),
        );
    }

    // (podman moved up — checked BEFORE docker)

    // Libvirtd / QEMU virtual machines
    if lower.contains("libvirt") || lower.contains("virtual machine") {
        return Some(
            r#"{ pkgs, ... }: {
  virtualisation.libvirtd = {
    enable = true;
    qemu = {
      package = pkgs.qemu_kvm;
      runAsRoot = true;
      swtpm.enable = true;
      ovmf = {
        enable = true;
        packages = [ pkgs.OVMFFull.fd ];
      };
    };
  };
  programs.virt-manager.enable = true;
}
"#
            .to_string(),
        );
    }

    // CUPS printing
    if lower.contains("cups") || lower.contains("printing") {
        return Some(
            r#"{ pkgs, ... }: {
  services.printing = {
    enable = true;
    drivers = with pkgs; [ gutenprint hplip ];
  };
  services.avahi = {
    enable = true;
    nssmdns4 = true;
    openFirewall = true;
  };
}
"#
            .to_string(),
        );
    }

    // systemd-resolved (DNS)
    if lower.contains("systemd-resolved") || lower.contains("resolved") {
        return Some(
            r#"{
  services.resolved = {
    enable = true;
    dnssec = "true";
    fallbackDns = [ "1.1.1.1" "9.9.9.9" ];
  };
}
"#
            .to_string(),
        );
    }

    // Steam (gaming)
    if lower.contains("steam") {
        return Some(
            r#"{
  programs.steam = {
    enable = true;
    remotePlay.openFirewall = true;
    dedicatedServer.openFirewall = true;
    gamescopeSession.enable = true;
  };
  hardware.graphics.enable32Bit = true;
}
"#
            .to_string(),
        );
    }

    None
}

// ── Hardware Emitters ──

fn emit_hardware(lower: &str) -> Option<String> {
    if lower.contains("nvidia") {
        return Some(
            r#"{ pkgs, ... }: {
  hardware.nvidia = {
    modesetting.enable = true;
    powerManagement.enable = true;
    open = false;
    nvidiaSettings = true;
  };
  hardware.graphics = {
    enable = true;
    enable32Bit = true;
    extraPackages = with pkgs; [
      vulkan-loader
      vulkan-validation-layers
      nvidia-vaapi-driver
    ];
  };
  services.xserver.videoDrivers = [ "nvidia" ];
}
"#
            .to_string(),
        );
    }
    None
}

// ── Desktop Emitters ──

fn emit_desktop(lower: &str) -> Option<String> {
    // Hyprland — check first so "hyprland wayland" doesn't match Sway via "wayland"
    if lower.contains("hyprland") {
        let fonts = if lower.contains("font") {
            "  fonts.packages = with pkgs; [\n    nerd-fonts.fira-code\n    noto-fonts\n    noto-fonts-emoji\n  ];\n"
        } else {
            ""
        };
        return Some(format!(
            r#"{{ config, pkgs, ... }}: {{
  programs.hyprland = {{
    enable = true;
    xwayland.enable = true;
  }};
  xdg.portal = {{
    enable = true;
    extraPortals = with pkgs; [
      xdg-desktop-portal-hyprland
      xdg-desktop-portal-gtk
    ];
  }};
  environment.systemPackages = with pkgs; [
    waybar wofi mako kitty grim slurp wl-clipboard
  ];
{fonts}}}
"#
        ));
    }

    // Sway / Wayland (without Hyprland keyword)
    if lower.contains("sway") || lower.contains("wayland") {
        let fonts = if lower.contains("font") {
            r#"  fonts.packages = with pkgs; [
    noto-fonts
    noto-fonts-cjk-sans
    noto-fonts-emoji
    fira-code
    fira-code-symbols
    jetbrains-mono
  ];
"#
        } else {
            ""
        };
        return Some(format!(
            r#"{{ config, pkgs, ... }}: {{
  programs.sway = {{
    enable = true;
    wrapperFeatures.gtk = true;
  }};
  xdg.portal = {{
    enable = true;
    wlr.enable = true;
    extraPortals = [ pkgs.xdg-desktop-portal-gtk ];
  }};
{fonts}}}
"#
        ));
    }

    // KDE
    if lower.contains("kde") || lower.contains("plasma") {
        return Some(
            r#"{
  services.desktopManager.plasma6.enable = true;
  services.displayManager.sddm.enable = true;
  services.displayManager.sddm.wayland.enable = true;
  programs.kdeconnect.enable = true;
}
"#
            .to_string(),
        );
    }

    // Hyprland is now handled at the top of the function — keep a marker
    if false {
        return Some(String::new());
    }

    // GNOME desktop
    if lower.contains("gnome") {
        let extra = if lower.contains("extension") {
            "  environment.systemPackages = with pkgs; [\n    gnomeExtensions.appindicator\n    gnomeExtensions.dash-to-dock\n    gnome-tweaks\n  ];\n"
        } else {
            ""
        };
        return Some(format!(
            r#"{{ config, pkgs, ... }}: {{
  services.xserver.enable = true;
  services.xserver.displayManager.gdm.enable = true;
  services.xserver.desktopManager.gnome.enable = true;
  services.gnome.gnome-keyring.enable = true;
{extra}}}
"#
        ));
    }

    None
}

// ── User/Group Emitters ──

fn emit_user_group(lower: &str) -> Option<String> {
    let mut groups: Vec<&str> = vec!["wheel"];
    if lower.contains("docker") {
        groups.push("docker");
    }
    if lower.contains("audio") {
        groups.push("audio");
    }
    if lower.contains("video") {
        groups.push("video");
    }
    if lower.contains("network") {
        groups.push("networkmanager");
    }
    let group_list = groups
        .iter()
        .map(|g| format!("\"{g}\""))
        .collect::<Vec<_>>()
        .join(" ");

    Some(format!(
        r#"{{ pkgs, ... }}: {{
  users.users.tstoltz = {{
    isNormalUser = true;
    extraGroups = [ {group_list} ];
    shell = pkgs.bash;
  }};
}}
"#
    ))
}

// ── Networking Emitters ──

fn emit_networking(lower: &str) -> Option<String> {
    // Extract port numbers from prompt.
    let mut ports = Vec::new();
    for word in lower.split_whitespace() {
        if let Ok(p) = word.trim_end_matches(',').parse::<u16>() {
            if p >= 1 {
                ports.push(p);
            }
        }
    }
    if ports.is_empty() && (lower.contains("port 80") || lower.contains("http")) {
        ports.push(80);
    }
    if ports.is_empty() && lower.contains("https") {
        ports.push(443);
    }
    if ports.is_empty() {
        return None;
    }
    let port_list = ports
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    // Protocol detection: prompts mentioning "udp" or wireguard (which
    // is UDP-only) emit `allowedUDPPorts`; everything else defaults to
    // TCP. Structural scorer surfaced this defect on
    // "open udp port 51820 for wireguard" — the prior emitter always
    // emitted TCP regardless of the explicit UDP mention.
    let option_name =
        if lower.contains("udp") || lower.contains("wireguard") || lower.contains("quic") {
            "allowedUDPPorts"
        } else {
            "allowedTCPPorts"
        };
    Some(format!(
        r#"{{
  networking.firewall.{option_name} = [ {port_list} ];
}}
"#
    ))
}

// ── Home Manager Emitters ──

fn emit_home_manager(lower: &str) -> Option<String> {
    if lower.contains("git") {
        return Some(
            r#"{ pkgs, ... }: {
  programs.git = {
    enable = true;
    extraConfig = {
      init.defaultBranch = "main";
      pull.rebase = true;
    };
  };
}
"#
            .to_string(),
        );
    }
    None
}

// ── Secrets Emitters ──

fn emit_secrets(lower: &str) -> Option<String> {
    // sops-nix — declarative encrypted secrets backed by sops + age/PGP
    if lower.contains("sops") {
        return Some(
            r#"{ config, pkgs, ... }: {
  imports = [ <sops-nix/modules/sops> ];

  sops = {
    defaultSopsFile = ./secrets.yaml;
    age = {
      keyFile = "/var/lib/sops-nix/key.txt";
      generateKey = true;
    };
    secrets = {
      "example_secret" = {
        owner = config.users.users.tstoltz.name;
        mode = "0400";
      };
    };
  };
}
"#
            .to_string(),
        );
    }

    // agenix — age-encrypted secrets (simpler than sops-nix)
    if lower.contains("agenix") || lower.contains("age key") {
        return Some(
            r#"{ config, pkgs, ... }: {
  imports = [
    (builtins.fetchTarball
      "https://github.com/ryantm/agenix/archive/main.tar.gz"
      + "/modules/age.nix")
  ];

  age = {
    identityPaths = [ "/home/tstoltz/.ssh/id_ed25519" ];
    secrets = {
      example_secret = {
        file = ./secrets/example.age;
        owner = "tstoltz";
        mode = "0400";
      };
    };
  };
}
"#
            .to_string(),
        );
    }

    // Generic secret hint — give them sops-nix as the recommended default
    if lower.contains("secret") || lower.contains("encrypted") || lower.contains("credential") {
        return Some(
            r#"{ config, pkgs, ... }: {
  # Recommended: use sops-nix for declarative encrypted secrets.
  # See https://github.com/Mic92/sops-nix
  imports = [ <sops-nix/modules/sops> ];

  sops = {
    defaultSopsFile = ./secrets.yaml;
    age.keyFile = "/var/lib/sops-nix/key.txt";
    secrets.example = {
      owner = config.users.users.tstoltz.name;
      mode = "0400";
    };
  };
}
"#
            .to_string(),
        );
    }
    None
}

fn emit_flake_template(lower: &str) -> Option<String> {
    // Dev-project flake — multi-language devShells
    if lower.contains("dev") || lower.contains("project") || lower.contains("devshell") {
        let mut langs: Vec<&str> = Vec::new();
        if lower.contains("rust") {
            langs.push("rust");
        }
        if lower.contains("python") {
            langs.push("python");
        }
        if lower.contains("node") || lower.contains("typescript") {
            langs.push("nodejs");
        }
        if lower.contains("go ") || lower.contains("golang") {
            langs.push("go");
        }
        if langs.is_empty() {
            langs.push("rust");
        }

        // Build buildInputs list
        let mut inputs: Vec<&str> = Vec::new();
        for lang in &langs {
            match *lang {
                "rust" => inputs.extend_from_slice(&["rustc", "cargo", "rustfmt", "rust-analyzer"]),
                "python" => inputs.push("python311"),
                "nodejs" => inputs.extend_from_slice(&["nodejs_20", "nodePackages.npm"]),
                "go" => inputs.push("go"),
                _ => {}
            }
        }
        let inputs_str = inputs.join(" ");
        let lang_summary = langs.join(", ");

        return Some(format!(
            r#"{{
  description = "Dev project: {lang_summary}";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = {{ self, nixpkgs }}:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {{ inherit system; }};
    in {{
      devShells.${{system}}.default = pkgs.mkShell {{
        buildInputs = with pkgs; [ {inputs_str} ];
        shellHook = ''
          echo "Dev shell: {lang_summary}"
        '';
      }};
    }};
}}
"#
        ));
    }

    // Full NixOS system flake — combines configuration.nix + home-manager
    if lower.contains("system") || lower.contains("nixos") || lower.contains("complete") {
        return Some(
            r#"{
  description = "NixOS system flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    home-manager = {
      url = "github:nix-community/home-manager";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, home-manager, ... }@inputs:
    let
      system = "x86_64-linux";
    in {
      nixosConfigurations.default = nixpkgs.lib.nixosSystem {
        inherit system;
        specialArgs = { inherit inputs; };
        modules = [
          ./configuration.nix
          home-manager.nixosModules.home-manager
          {
            home-manager.useGlobalPkgs = true;
            home-manager.useUserPackages = true;
            home-manager.users.tstoltz = import ./home.nix;
          }
        ];
      };
    };
}
"#
            .to_string(),
        );
    }

    None
}

// ─── Verification ──────────────────────────────────────────────────────────

/// Outcome of attempting to verify a Nix expression.
#[derive(Debug)]
pub enum NixVerdict {
    /// `nix-instantiate --parse` succeeded — syntactically valid.
    ParseOk,
    /// Parse failed with a specific error.
    ParseError(String),
}

impl NixVerdict {
    pub fn is_ok(&self) -> bool {
        matches!(self, NixVerdict::ParseOk)
    }
    pub fn message(&self) -> String {
        match self {
            NixVerdict::ParseOk => String::new(),
            NixVerdict::ParseError(e) => e.clone(),
        }
    }
}

/// Wrap an expression with a `pkgs` binding so it can be evaluated in
/// isolation. Handles three cases:
/// 1. Already a function `{ pkgs, ... }: BODY` → applies `pkgs = import <nixpkgs> {}`
/// 2. Already a function `{ pkgs ? import <nixpkgs> {} }: BODY` → calls with `{}`
/// 3. Bare module body `{ ... }` → wraps as `(let pkgs = import <nixpkgs> {}; in BODY)`
fn wrap_for_eval(expr: &str) -> String {
    let trimmed = expr.trim();
    if trimmed.starts_with("{ pkgs ? import <nixpkgs>") || trimmed.starts_with("{ pkgs ? import") {
        // Already self-contained shell.nix style — call with no args
        return format!("({trimmed}) {{}}");
    }
    if trimmed.starts_with("{ pkgs, ...") || trimmed.starts_with("{ config, pkgs, ...") {
        // Module that takes pkgs — apply with synthetic pkgs
        return format!(
            "let pkgs = import <nixpkgs> {{ config.allowUnfree = true; }};\n    config = {{}};\nin ({trimmed}) {{ inherit pkgs config; }}"
        );
    }
    // Bare attrset — wrap so any pkgs references inside resolve
    format!("let pkgs = import <nixpkgs> {{ config.allowUnfree = true; }};\nin {trimmed}")
}

/// Verify a Nix expression by evaluating it with `nix-instantiate --eval`.
///
/// Catches strictly more errors than `try_nix_parse`:
/// - Undefined variables (typos in package names like `pkgs.firefoxx`)
/// - Type mismatches that get evaluated eagerly
/// - Some attribute path errors
///
/// Slower than parse (~50-200ms) but still much faster than full
/// `nixos-rebuild dry-run` (which can take 30s+).
pub fn try_nix_eval(expr: &str) -> NixVerdict {
    let wrapped = wrap_for_eval(expr);
    let tmp = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = tmp.join(format!(
        "symthaea_nix_codegen_eval_{}_{}.nix",
        std::process::id(),
        nonce
    ));
    if let Err(e) = std::fs::write(&path, &wrapped) {
        return NixVerdict::ParseError(format!("write: {e}"));
    }

    // --strict forces evaluation of attrset values (catches more lazy errors)
    let out = Command::new("nix-instantiate")
        .args([
            "--eval",
            "--strict",
            "--read-write-mode",
            path.to_str().unwrap_or(""),
        ])
        .output();

    let _ = std::fs::remove_file(&path);

    match out {
        Ok(o) if o.status.success() => NixVerdict::ParseOk,
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr).to_string();
            // Find the first error line
            let msg = stderr
                .lines()
                .find(|l| l.trim_start().starts_with("error:") || l.contains("error:"))
                .unwrap_or(stderr.lines().next().unwrap_or(""))
                .trim()
                .to_string();
            NixVerdict::ParseError(msg)
        }
        Err(e) => NixVerdict::ParseError(format!("nix-instantiate: {e}")),
    }
}

/// Tier 1.3 — full NixOS module evaluation.
///
/// Wraps the snippet as an `imports = [ ... ]` entry inside a minimal
/// configuration that's complete enough to evaluate (root filesystem,
/// bootloader, stateVersion stubs), then runs
/// `nix-instantiate '<nixpkgs/nixos>' -A system`. This is the strongest
/// pre-build verification we can perform without actually realizing
/// any derivations.
///
/// Catches:
/// - All parse + eval errors
/// - Option-name typos (`services.openssh.enabled` → suggests `enable`)
/// - Type errors on option values
/// - Cross-module assertion failures (most NixOS sanity checks)
///
/// Cost: ~5–30s on a warm nixpkgs eval cache, much longer cold. Should
/// only be invoked when caller opts in (`generate_nix_with_dry_build`)
/// AND the prompt is for a NixOS module (not a shell.nix or flake fragment).
///
/// Returns `Some(verdict)` if the verifier ran, `None` if the snippet is
/// not a valid module shape (e.g. shell.nix that takes `pkgs ? import …`).
pub fn try_nix_module_eval(snippet: &str) -> Option<NixVerdict> {
    if !looks_like_module(snippet) {
        return None;
    }

    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    // Per-call nonce so concurrent calls (test threads, services) don't
    // race on the same tempfile names.
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let snippet_path = tmp.join(format!("symthaea_nix_module_snippet_{pid}_{nonce}.nix"));
    let wrapper_path = tmp.join(format!("symthaea_nix_module_wrap_{pid}_{nonce}.nix"));

    if let Err(e) = std::fs::write(&snippet_path, snippet) {
        return Some(NixVerdict::ParseError(format!("write snippet: {e}")));
    }

    // Minimal NixOS config that imports the snippet. The fileSystems +
    // bootloader + stateVersion stubs are required for a NixOS eval to
    // succeed; otherwise assertions fail before the snippet is checked.
    let snippet_path_str = snippet_path.to_string_lossy();
    let wrapper = format!(
        r#"{{ pkgs, ... }}: {{
  imports = [ {snippet_path_str} ];
  fileSystems."/" = {{ device = "/dev/null"; fsType = "tmpfs"; }};
  boot.loader.grub.device = "nodev";
  system.stateVersion = "24.05";
}}
"#
    );
    if let Err(e) = std::fs::write(&wrapper_path, wrapper) {
        let _ = std::fs::remove_file(&snippet_path);
        return Some(NixVerdict::ParseError(format!("write wrapper: {e}")));
    }

    let out = Command::new("nix-instantiate")
        .args([
            "<nixpkgs/nixos>",
            "-A",
            "system",
            "--arg",
            "configuration",
            wrapper_path.to_str().unwrap_or(""),
        ])
        .output();

    let _ = std::fs::remove_file(&snippet_path);
    let _ = std::fs::remove_file(&wrapper_path);

    let verdict = match out {
        Ok(o) if o.status.success() => NixVerdict::ParseOk,
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr).to_string();
            // The most informative line for module-eval errors is usually the
            // last `error: ...` line (which has the option-name suggestion).
            let msg = stderr
                .lines()
                .rev()
                .find(|l| l.trim_start().starts_with("error:"))
                .or_else(|| stderr.lines().find(|l| l.contains("error")))
                .unwrap_or(stderr.lines().next().unwrap_or(""))
                .trim()
                .to_string();
            NixVerdict::ParseError(msg)
        }
        Err(e) => NixVerdict::ParseError(format!("nix-instantiate: {e}")),
    };
    Some(verdict)
}

/// Heuristic: does this snippet look like a NixOS module (vs. a shell.nix
/// or flake.nix fragment)?
///
/// Module shapes we accept:
/// - `{ ... }: { services.X = …; }`
/// - `{ pkgs, ... }: { ... }`
/// - bare attrset `{ services.X = …; }`
///
/// We reject:
/// - `{ pkgs ? import <nixpkgs> {} }: pkgs.mkShell { … }` (shell.nix)
/// - `{ description = …; outputs = …; }` (flake)
fn looks_like_module(snippet: &str) -> bool {
    let trimmed = snippet.trim_start();
    let lower = snippet.to_lowercase();
    if lower.contains("mkshell") || lower.contains("mkderivation") {
        return false;
    }
    if lower.contains("description") && lower.contains("outputs") && lower.contains("inputs") {
        return false;
    }
    // Either bare attrset or function-of-attrset
    trimmed.starts_with('{') || trimmed.starts_with('(')
}

/// Verify a Nix expression by running `nix-instantiate --parse`.
///
/// This is the fastest verification (millisecond) — checks syntax only,
/// not type-correctness or semantic validity. Sufficient for first-cut
/// repair signal.
pub fn try_nix_parse(expr: &str) -> NixVerdict {
    let tmp = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = tmp.join(format!(
        "symthaea_nix_codegen_check_{}_{}.nix",
        std::process::id(),
        nonce
    ));
    if let Err(e) = std::fs::write(&path, expr) {
        return NixVerdict::ParseError(format!("write: {e}"));
    }

    let out = Command::new("nix-instantiate")
        .args(["--parse", path.to_str().unwrap_or("")])
        .output();

    let _ = std::fs::remove_file(&path);

    match out {
        Ok(o) if o.status.success() => NixVerdict::ParseOk,
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr).to_string();
            // First non-empty line tends to be the most useful error
            let msg = stderr
                .lines()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("")
                .to_string();
            NixVerdict::ParseError(msg)
        }
        Err(e) => NixVerdict::ParseError(format!("nix-instantiate: {e}")),
    }
}

// ─── Generation Pipeline ───────────────────────────────────────────────────

/// Result of attempting to generate Nix from a prompt.
#[derive(Debug)]
pub struct NixGenResult {
    pub prompt: String,
    pub intent: NixIntent,
    pub code: String,
    pub iterations: usize,
    pub parses: bool,
    pub last_error: Option<String>,
    pub source: NixGenSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NixGenSource {
    /// Idiom library matched + emitted directly.
    Idiom,
    /// No idiom match — used a fallback skeleton.
    Skeleton,
    /// No idiom + no cache hit — drafted from `search.nixos.org` option
    /// hits as commented suggestions. Caller MUST review (not verified).
    RagDraft,
    /// All paths failed.
    Empty,
}

/// Generate a Nix expression for the given prompt with verification.
///
/// Pipeline: classify intent → match idiom → verify with nix-instantiate.
/// Falls through to a skeleton if no idiom matches.
pub fn generate_nix(prompt: &str) -> NixGenResult {
    let channels = build_nix_channels(prompt);
    let intent = classify_nix_intent(&prompt.to_lowercase());

    // Try idiom first
    if let Some(code) = nix_idiom_body(prompt) {
        let verdict = try_nix_parse(&code);
        let parses = verdict.is_ok();
        let last_error = if parses {
            None
        } else {
            Some(verdict.message())
        };
        return NixGenResult {
            prompt: prompt.to_string(),
            intent,
            code,
            iterations: 1,
            parses,
            last_error,
            source: NixGenSource::Idiom,
        };
    }

    // Fallback: minimal skeleton based on intent
    let skeleton = match intent {
        NixIntent::DevShell => {
            "{ pkgs ? import <nixpkgs> {} }:\npkgs.mkShell {\n  buildInputs = [ ];\n}\n"
        }
        NixIntent::Service => "{\n  # service config\n}\n",
        NixIntent::Hardware => "{\n  # hardware config\n}\n",
        NixIntent::Desktop => "{\n  # desktop config\n}\n",
        NixIntent::User => "{\n  users.users.tstoltz = {\n    isNormalUser = true;\n  };\n}\n",
        NixIntent::Networking => "{\n  networking = { };\n}\n",
        NixIntent::HomeManager => "{ pkgs, ... }: {\n  home.packages = [ ];\n}\n",
        NixIntent::Secrets => "{\n  # secrets config — see sops-nix or agenix\n}\n",
        NixIntent::FlakeTemplate => "{\n  description = \"flake\";\n  inputs.nixpkgs.url = \"github:NixOS/nixpkgs/nixos-unstable\";\n  outputs = { self, nixpkgs }: { };\n}\n",
        NixIntent::Generic => "{ }\n",
    };
    let verdict = try_nix_parse(skeleton);
    let parses = verdict.is_ok();
    let last_error = if parses {
        None
    } else {
        Some(verdict.message())
    };

    let _ = channels;

    NixGenResult {
        prompt: prompt.to_string(),
        intent,
        code: skeleton.to_string(),
        iterations: 1,
        parses,
        last_error,
        source: NixGenSource::Skeleton,
    }
}

/// Classify what kind of Nix error this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NixErrorKind {
    /// Syntactic error (typo, unbalanced braces, missing `;`).
    Syntax,
    /// Reference to a name that doesn't exist (`pkgs.firefoxx`).
    UndefinedName,
    /// Type mismatch (string where bool expected, etc.).
    TypeMismatch,
    /// Attribute path missing.
    MissingAttribute,
    /// Unrecognized/other.
    Other,
}

/// Classify the kind of error from a nix-instantiate stderr message.
pub fn classify_nix_error(msg: &str) -> NixErrorKind {
    let lower = msg.to_lowercase();
    if lower.contains("undefined variable")
        || lower.contains("attribute missing")
        || lower.contains("not found in attribute set")
    {
        NixErrorKind::UndefinedName
    } else if lower.contains("expected") && lower.contains("got") {
        NixErrorKind::TypeMismatch
    } else if lower.contains("attribute") && lower.contains("missing") {
        NixErrorKind::MissingAttribute
    } else if lower.contains("syntax error")
        || lower.contains("unexpected")
        || lower.contains("unbalanced")
    {
        NixErrorKind::Syntax
    } else {
        NixErrorKind::Other
    }
}

/// Enrich Nix channels based on an error message.
///
/// The channel adjustments push subsequent idiom selection toward
/// alternatives that are more likely to satisfy the failure mode:
/// - Undefined name → was probably a stale package name; broaden via
///   item_count (try simpler set of packages)
/// - Type mismatch → boost has_extras (we may have used the wrong shape)
pub fn enrich_nix_channels_from_error(channels: &mut NixChannels, error: &str) {
    let kind = classify_nix_error(error);
    match kind {
        NixErrorKind::UndefinedName => {
            // Likely a missing/typo package — drop one item to try simpler set
            channels.item_count = (channels.item_count - 1.0).max(1.0);
        }
        NixErrorKind::TypeMismatch => {
            channels.has_extras = 0.0;
        }
        NixErrorKind::MissingAttribute => {
            channels.has_extras = (channels.has_extras + 1.0).min(3.0);
        }
        NixErrorKind::Syntax => {
            // Syntax errors are usually template bugs we can't repair via channels
        }
        NixErrorKind::Other => {}
    }
}

// ─── Causal Validation (conflict detection) ────────────────────────────────

/// A known conflict between two Nix options. If both are enabled the system
/// won't boot or will misbehave.
#[derive(Clone, Debug)]
pub struct CausalConflict {
    pub a: &'static str,
    pub b: &'static str,
    pub reason: &'static str,
}

/// Conflict patterns sourced from the shared knowledge graph.
/// The `&'static str` form is preserved so existing callers don't break.
/// Strings are leaked exactly once via `OnceLock` — KG entries don't change
/// at runtime so this is bounded.
pub fn known_conflicts() -> Vec<CausalConflict> {
    static CACHED: OnceLock<Vec<CausalConflict>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            shared_nix_kg()
                .conflicts
                .iter()
                .map(|c: &ConflictClaim| CausalConflict {
                    a: Box::leak(c.a.clone().into_boxed_str()),
                    b: Box::leak(c.b.clone().into_boxed_str()),
                    reason: Box::leak(c.reason.clone().into_boxed_str()),
                })
                .collect()
        })
        .clone()
}

/// Scan a generated Nix expression for known conflicts.
/// Returns a list of (conflict, true_if_both_present) — a warning record.
pub fn validate_conflicts(code: &str) -> Vec<CausalConflict> {
    known_conflicts()
        .into_iter()
        .filter(|c| {
            // Normalize whitespace for substring matching
            let norm = code.split_whitespace().collect::<Vec<_>>().join(" ");
            let a_norm = c.a.split_whitespace().collect::<Vec<_>>().join(" ");
            let b_norm = c.b.split_whitespace().collect::<Vec<_>>().join(" ");
            norm.contains(&a_norm) && norm.contains(&b_norm)
        })
        .collect()
}

/// Generate Nix with a self-repair loop that uses semantic verification.
///
/// Strategy:
/// 1. Generate via idiom (parse-verify)
/// 2. If parses, run nix-eval verification (catches typos, undef names)
/// 3. If eval fails, classify error → enrich channels → retry up to N times
///
/// The repair distinguishes between syntactic and semantic errors so the
/// adjustment is targeted (e.g. drop an extra package vs change template).
pub fn generate_nix_with_repair(prompt: &str, max_iterations: usize) -> NixGenResult {
    let mut channels = build_nix_channels(prompt);
    let intent = classify_nix_intent(&prompt.to_lowercase());
    let mut error_history: Vec<String> = Vec::new();
    let mut current_code = String::new();
    let mut last_source = NixGenSource::Empty;

    for iteration in 0..max_iterations {
        // Attempt: idiom path (channels may have been enriched between iterations)
        let attempt = nix_idiom_body(prompt);
        let (code, source) = match attempt {
            Some(c) => (c, NixGenSource::Idiom),
            None => {
                let skeleton = match intent {
                    NixIntent::DevShell => {
                        "{ pkgs ? import <nixpkgs> {} }:\npkgs.mkShell {\n  buildInputs = [ ];\n}\n"
                    }
                    _ => "{ }\n",
                };
                (skeleton.to_string(), NixGenSource::Skeleton)
            }
        };
        current_code = code.clone();
        last_source = source;

        // Step 1: parse check
        let parse_verdict = try_nix_parse(&code);
        if !parse_verdict.is_ok() {
            let msg = parse_verdict.message();
            error_history.push(format!("parse: {msg}"));
            enrich_nix_channels_from_error(&mut channels, &msg);
            continue;
        }

        // Step 2: eval check (catches typos, undefined names)
        let eval_verdict = try_nix_eval(&code);
        if eval_verdict.is_ok() {
            return NixGenResult {
                prompt: prompt.to_string(),
                intent,
                code: current_code,
                iterations: iteration + 1,
                parses: true,
                last_error: None,
                source: last_source,
            };
        }
        let msg = eval_verdict.message();
        error_history.push(format!("eval: {msg}"));
        enrich_nix_channels_from_error(&mut channels, &msg);
    }

    // Even if eval failed, parse succeeded somewhere in the loop —
    // return the most recent attempt with the last error.
    let parses_at_minimum = try_nix_parse(&current_code).is_ok();
    NixGenResult {
        prompt: prompt.to_string(),
        intent,
        code: current_code,
        iterations: error_history.len(),
        parses: parses_at_minimum,
        last_error: error_history.last().cloned(),
        source: last_source,
    }
}

// ─── Tier 1.1: option-index verification ──────────────────────────────────

/// One unrecognized NixOS option path found in generated code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownOption {
    /// Dotted path as seen in the source (e.g. `services.nginxx.enable`).
    pub path: String,
    /// Suggested nearest-known prefix, if we can find one.
    pub nearest_known_prefix: Option<String>,
}

/// Top-level NixOS option roots — sourced from `shared_nix_kg().option_roots`.
/// Replaces the previous `KNOWN_OPTION_ROOTS` constant array; see
/// `language::nix_kg` for the canonical list and how user overrides extend it.

/// Extract candidate NixOS option paths from a Nix source fragment.
///
/// Heuristics: matches the longest dotted identifier whose root is in
/// `KNOWN_OPTION_ROOTS` and whose tail looks like an option path (no `${`
/// interpolation, no leading number). Returns paths like `services.nginx.enable`.
pub fn extract_option_paths(code: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let bytes = code.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        // Skip strings (rough — only handles "...")
        if bytes[i] == b'"' {
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
            i += 1;
            continue;
        }
        // Skip line comments
        if i + 1 < bytes.len() && &bytes[i..i + 1] == b"#" {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Identifier start?
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric()
                    || bytes[i] == b'_'
                    || bytes[i] == b'-'
                    || bytes[i] == b'.')
            {
                i += 1;
            }
            let span = &code[start..i];
            // Must contain at least one dot and start with a known root.
            if span.contains('.')
                && !span.contains("..")
                && !span.ends_with('.')
                && !span.starts_with('.')
            {
                let root = span.split('.').next().unwrap_or("");
                if shared_nix_kg().is_known_option_root(root) {
                    // Skip if it looks like a function call (followed by `(`)
                    // or attr selection from pkgs (`pkgs.xxx`).
                    out.push(span.to_string());
                }
            }
            continue;
        }
        i += 1;
    }
    // De-dup while preserving order.
    let mut seen: Vec<String> = Vec::new();
    for p in out {
        if !seen.contains(&p) {
            seen.push(p);
        }
    }
    seen
}

/// Process-level shared index handle (lazy-init).
static SHARED_INDEX: OnceLock<NixpkgsIndex> = OnceLock::new();

/// Get the shared, process-wide nixpkgs index (cache at default path).
pub fn shared_nixpkgs_index() -> &'static NixpkgsIndex {
    SHARED_INDEX.get_or_init(NixpkgsIndex::default_cache)
}

/// Validate option paths in `code` against the nixpkgs option index.
/// Returns the list of paths that did NOT resolve.
///
/// Trims trailing assignment fragments — only the head path is verified.
/// E.g. `services.nginx.enable = true;` -> we verify `services.nginx.enable`.
pub fn validate_options(code: &str, index: &NixpkgsIndex) -> Vec<UnknownOption> {
    let paths = extract_option_paths(code);
    let mut unknown: Vec<UnknownOption> = Vec::new();

    for p in paths {
        // Try the full path first; if missing, walk back to find longest known prefix.
        if index.option_exists(&p) {
            continue;
        }
        // Walk backwards by '.' to find the longest known prefix.
        let mut nearest: Option<String> = None;
        let mut parts: Vec<&str> = p.split('.').collect();
        while parts.len() > 1 {
            parts.pop();
            let candidate = parts.join(".");
            if index.option_exists(&candidate) {
                nearest = Some(candidate);
                break;
            }
        }
        unknown.push(UnknownOption {
            path: p,
            nearest_known_prefix: nearest,
        });
    }
    unknown
}

/// Result of running the generation pipeline with option-index verification.
#[derive(Debug)]
pub struct NixGenWithIndexResult {
    pub base: NixGenResult,
    /// Options that the index could not resolve (only meaningful when
    /// `base.parses == true` — otherwise the syntax error precedes any
    /// semantic check).
    pub unknown_options: Vec<UnknownOption>,
}

/// Like `generate_nix_with_repair`, but additionally validates emitted option
/// paths against the live nixpkgs option index. The result includes a list of
/// unresolved option paths so callers can decide whether to surface a warning
/// or trigger another repair pass.
///
/// Note: this does NOT yet feed `unknown_options` back into the channel
/// enrichment — it's an out-of-band signal for the caller. Tier 3
/// (self-improvement loop) will close that feedback loop.
pub fn generate_nix_with_index_verify(
    prompt: &str,
    max_iterations: usize,
) -> NixGenWithIndexResult {
    let base = generate_nix_with_repair(prompt, max_iterations);
    let unknown_options = if base.parses {
        validate_options(&base.code, shared_nixpkgs_index())
    } else {
        Vec::new()
    };
    NixGenWithIndexResult {
        base,
        unknown_options,
    }
}

/// Result of running the generation pipeline with the full verification
/// ladder (parse → eval → option-index → optional module-eval).
#[derive(Debug)]
pub struct NixGenWithDryBuildResult {
    pub base: NixGenResult,
    pub unknown_options: Vec<UnknownOption>,
    /// Module-eval verdict. `None` if the snippet was not module-shaped
    /// (e.g. a shell.nix), so the dry-build verifier was skipped.
    pub module_eval: Option<NixVerdict>,
}

impl NixGenWithDryBuildResult {
    /// True iff every available verification rung was clean.
    pub fn fully_verified(&self) -> bool {
        self.base.parses
            && self.unknown_options.is_empty()
            && match &self.module_eval {
                Some(v) => v.is_ok(),
                None => true, // not applicable for shell.nix etc.
            }
    }
}

/// Like `generate_nix_with_index_verify`, but adds the `try_nix_module_eval`
/// rung when the snippet is module-shaped. SLOW: 5-30s per call (full
/// `<nixpkgs/nixos>` evaluation), so callers should opt in explicitly.
pub fn generate_nix_with_dry_build(
    prompt: &str,
    max_iterations: usize,
) -> NixGenWithDryBuildResult {
    let mid = generate_nix_with_index_verify(prompt, max_iterations);
    let module_eval = if mid.base.parses {
        // P3: route through the process-wide content-hash cache.
        // First call per (snippet, nixpkgs_rev) pays ~5–30s; repeats
        // are effectively free. Transient errors (nix-instantiate
        // missing, IO failures) are deliberately NOT cached.
        crate::language::nix_eval_cache::cached_module_eval(&mid.base.code)
    } else {
        None
    };
    NixGenWithDryBuildResult {
        base: mid.base,
        unknown_options: mid.unknown_options,
        module_eval,
    }
}

// ─── Tier 3: self-improvement loop ────────────────────────────────────────

/// Process-level shared cache of verified idioms.
static SHARED_LEARNED: OnceLock<LearnedIdiomCache> = OnceLock::new();

/// Get the shared learned-idiom cache (default disk path).
pub fn shared_learned_cache() -> &'static LearnedIdiomCache {
    SHARED_LEARNED.get_or_init(LearnedIdiomCache::default_cache)
}

/// Whether a generation result was retrieved from the learned cache or
/// freshly generated (and possibly recorded back into the cache).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelfImproveSource {
    /// Found a matching learned idiom — used it directly.
    LearnedCache,
    /// No cache hit; regenerated via the standard pipeline. If `recorded` is
    /// true, the result was written back into the cache for next time.
    FreshlyGenerated { recorded: bool },
}

/// Self-improving wrapper: tries the learned-idiom cache first, falls
/// through to `generate_nix_with_index_verify`, and records every fully
/// verified output (parses + zero unknown options) back into the cache.
///
/// Subsequent calls with paraphrased prompts get the cached answer at
/// O(keyword-set-overlap) instead of redoing the whole pipeline.
pub fn generate_nix_with_self_improve(
    prompt: &str,
    max_iterations: usize,
) -> (NixGenWithIndexResult, SelfImproveSource) {
    generate_nix_with_cache(prompt, max_iterations, shared_learned_cache())
}

/// Lower-level self-improve entry point with an injectable cache. Used by
/// `generate_nix_with_self_improve` (with the process-wide singleton) and
/// by benchmarks that need a per-run cache to avoid polluting the user's
/// real `~/.cache/symthaea/learned-idioms.json`.
pub fn generate_nix_with_cache(
    prompt: &str,
    max_iterations: usize,
    cache: &LearnedIdiomCache,
) -> (NixGenWithIndexResult, SelfImproveSource) {
    // P2 cache integrity: classify intent BEFORE query so find_match
    // can enforce the intent gate. A prompt for Service intent must
    // never receive a DevShell cache hit.
    let intent = classify_nix_intent(&prompt.to_lowercase());

    if let Some(idiom) = cache.find_match(prompt, intent) {
        // P2 re-verification: option-index may have drifted since the
        // cached entry was recorded (nixpkgs rename, option removed).
        // Re-run verify_options on the cached code. If still clean,
        // serve the hit. If stale, invalidate and fall through to
        // fresh generation.
        let unknown = validate_options(&idiom.code, shared_nixpkgs_index());
        if unknown.is_empty() {
            let result = NixGenWithIndexResult {
                base: NixGenResult {
                    prompt: prompt.to_string(),
                    intent,
                    code: idiom.code.clone(),
                    iterations: 0,
                    parses: true,
                    last_error: None,
                    source: NixGenSource::Idiom,
                },
                unknown_options: Vec::new(),
            };
            return (result, SelfImproveSource::LearnedCache);
        }
        // Stale — evict this entry and fall through. Other entries
        // stay intact (invalidate is prompt+code-specific).
        cache.invalidate(&idiom.prompt, &idiom.code);
    }

    let result = generate_nix_with_index_verify(prompt, max_iterations);
    // Only cache idiom-sourced results. Skeleton ("no idea, here's `{ }`")
    // would Jaccard-match anything later and poison the cache.
    let recorded = if result.base.parses
        && result.unknown_options.is_empty()
        && result.base.source == NixGenSource::Idiom
    {
        cache.record(prompt, &result.base.code, result.base.intent);
        true
    } else {
        false
    };
    (result, SelfImproveSource::FreshlyGenerated { recorded })
}

// ─── Tier 3 (#42): RAG fallback — draft from option-index search ──────────

use crate::language::nixos_search::NixosSearch;

/// Process-level shared `NixosSearch` handle (lazy-init).
static SHARED_SEARCH: OnceLock<NixosSearch> = OnceLock::new();

/// Get the shared search.nixos.org handle (default disk cache path).
pub fn shared_nixos_search() -> &'static NixosSearch {
    SHARED_SEARCH.get_or_init(NixosSearch::default_cache)
}

/// Stop-words used during RAG keyword extraction. Same spirit as
/// `learned_idioms::STOP` but tuned for option search (we want service
/// names + verbs, not boilerplate).
const RAG_STOP: &[&str] = &[
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "for",
    "with",
    "of",
    "in",
    "on",
    "at",
    "set",
    "up",
    "configure",
    "enable",
    "make",
    "please",
    "i",
    "want",
    "need",
    "this",
    "that",
    "my",
    "me",
    "do",
    "can",
    "you",
    "should",
    "will",
    "would",
    "could",
    "is",
    "be",
    "as",
    "by",
];

/// Pull short keywords from the prompt suitable for `prefix:services.X`-style
/// option queries.
fn rag_keywords(prompt: &str) -> Vec<String> {
    let lower = prompt.to_lowercase();
    let mut out: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for tok in lower.split(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_') {
        let t = tok.trim_matches(|c: char| !c.is_ascii_alphanumeric());
        if t.len() < 3 {
            continue;
        }
        if RAG_STOP.contains(&t) {
            continue;
        }
        if seen.insert(t.to_string()) {
            out.push(t.to_string());
        }
    }
    out
}

/// Prefixes we'll try when querying the option index. Caller intent guides
/// which we hit first. Sourced from `shared_nix_kg().rag_prefixes_for(intent)`.
fn rag_prefixes_for_intent(intent: NixIntent) -> Vec<String> {
    shared_nix_kg().rag_prefixes_for(intent).to_vec()
}

/// Build a draft Nix snippet from RAG search hits. Marks every line as a
/// suggestion (commented) so the user knows it's not auto-verified.
fn render_rag_draft(prompt: &str, hits: &[crate::language::nixos_search::SearchHit]) -> String {
    let mut s = String::new();
    s.push_str("# Draft generated from search.nixos.org option hits.\n");
    s.push_str("# This is NOT verified — review each line before applying.\n");
    s.push_str(&format!("# Prompt: {}\n", prompt));
    s.push_str("# ────────────────────────────────────────────────────────\n");
    s.push_str("{ ... }: {\n");
    for h in hits {
        // Skip nested attribute-set parents — only suggest leaf options.
        if h.option_type.is_empty() || h.option_type.contains("submodule") {
            continue;
        }
        let default = match h.option_type.to_lowercase().as_str() {
            t if t.contains("boolean") => "true",
            t if t.contains("string") => "\"\"",
            t if t.contains("int") => "0",
            t if t.contains("list") => "[ ]",
            t if t.contains("attribute set") || t.contains("attrset") => "{ }",
            _ => "null",
        };
        s.push_str(&format!(
            "  # {} :: {}\n  # {} = {};\n",
            h.option_name,
            h.option_type.chars().take(60).collect::<String>(),
            h.option_name,
            default,
        ));
    }
    s.push_str("}\n");
    s
}

/// Try a RAG-style draft for prompts that the idiom library + cache both miss.
/// Returns `None` if no useful options were found (e.g. prompt is OOD or
/// the search backend is unreachable).
pub fn rag_draft_from_search(prompt: &str) -> Option<String> {
    let lower = prompt.to_lowercase();
    let intent = classify_nix_intent(&lower);

    // For Generic intent, fall back to the most-likely top-level prefixes
    // when the prompt looks like a configuration request. Catches the
    // "set up jellyfin" / "enable redis" cases the keyword classifier
    // routes to Generic because the service name isn't in its dictionary.
    let setup_like = lower.contains("set up")
        || lower.contains("setup")
        || lower.contains("install")
        || lower.contains("enable")
        || lower.contains("configure")
        || lower.contains("provision");
    let mut prefixes: Vec<String> = rag_prefixes_for_intent(intent);
    if prefixes.is_empty() && intent == NixIntent::Generic && setup_like {
        prefixes = vec![
            "services.".to_string(),
            "programs.".to_string(),
            "hardware.".to_string(),
            "networking.".to_string(),
        ];
    }
    if prefixes.is_empty() {
        return None;
    }
    let keywords = rag_keywords(prompt);
    if keywords.is_empty() {
        return None;
    }

    let search = shared_nixos_search();
    let mut hits: Vec<crate::language::nixos_search::SearchHit> = Vec::new();

    // Strategy: for each prefix × keyword, ask for top-5; merge + de-dup.
    for prefix in prefixes {
        for kw in &keywords {
            let probe = format!("{prefix}{kw}");
            for h in search.search_options_by_prefix(&probe, 5) {
                if !hits.iter().any(|x| x.option_name == h.option_name) {
                    hits.push(h);
                }
                if hits.len() >= 12 {
                    break;
                }
            }
            if hits.len() >= 12 {
                break;
            }
        }
        if hits.len() >= 12 {
            break;
        }
    }

    if hits.is_empty() {
        return None;
    }
    Some(render_rag_draft(prompt, &hits))
}

/// Self-improve + RAG-fallback wrapper.
///
/// Pipeline order:
/// 1. Cache hit → return immediately
/// 2. `generate_nix_with_index_verify` — idiom library path
/// 3. If unknown_options OR no idiom matched, try `rag_draft_from_search`
/// 4. On full success (parses + zero unknown options), record into cache
///
/// RAG drafts are NOT recorded into the cache because they are not
/// machine-verified — they're suggestions for the user to review.
pub fn generate_nix_with_rag(
    prompt: &str,
    max_iterations: usize,
) -> (NixGenWithIndexResult, SelfImproveSource) {
    // Step 1: cache. Intent-gated + re-verified on hit (P2 integrity).
    let intent = classify_nix_intent(&prompt.to_lowercase());
    let cache = shared_learned_cache();
    if let Some(idiom) = cache.find_match(prompt, intent) {
        let unknown = validate_options(&idiom.code, shared_nixpkgs_index());
        if unknown.is_empty() {
            let result = NixGenWithIndexResult {
                base: NixGenResult {
                    prompt: prompt.to_string(),
                    intent,
                    code: idiom.code.clone(),
                    iterations: 0,
                    parses: true,
                    last_error: None,
                    source: NixGenSource::Idiom,
                },
                unknown_options: Vec::new(),
            };
            return (result, SelfImproveSource::LearnedCache);
        }
        cache.invalidate(&idiom.prompt, &idiom.code);
    }

    // Step 2: standard pipeline.
    let mut result = generate_nix_with_index_verify(prompt, max_iterations);

    // Step 3: if the idiom path produced a Skeleton (no idiom matched),
    // try a RAG draft. We do NOT trigger RAG on unknown_options alone —
    // hand-written idioms often emit dynamic attribute paths
    // (`users.users.<name>`, `hardware.opengl.driSupport`) that the
    // option-index can't resolve, but the idiom output is still canonical.
    let weak = matches!(result.base.source, NixGenSource::Skeleton);
    if weak {
        if let Some(draft) = rag_draft_from_search(prompt) {
            // RAG draft replaces the weak result. We mark parses=true
            // because the rendered draft is syntactically valid Nix
            // (commented suggestions inside `{ ... }: { }`).
            let intent = classify_nix_intent(&prompt.to_lowercase());
            result = NixGenWithIndexResult {
                base: NixGenResult {
                    prompt: prompt.to_string(),
                    intent,
                    code: draft,
                    iterations: result.base.iterations + 1,
                    parses: true,
                    last_error: None,
                    source: NixGenSource::RagDraft,
                },
                unknown_options: Vec::new(),
            };
        }
    }

    // Step 4: record verified, non-RAG results.
    let recorded = if result.base.parses
        && result.unknown_options.is_empty()
        && result.base.source == NixGenSource::Idiom
    {
        shared_learned_cache().record(prompt, &result.base.code, result.base.intent);
        true
    } else {
        false
    };
    (result, SelfImproveSource::FreshlyGenerated { recorded })
}

/// Fast-mode variant — same pipeline as `generate_nix_with_rag` but skips
/// `try_nix_eval` (which re-imports nixpkgs each call ≈100s) and instead
/// relies on `try_nix_parse` (millisecond). Cold-prompt latency drops from
/// ~100s to ~10ms. Use this for interactive paths (HTTP API, REPL, IDE).
///
/// What it gives up: the deeper semantic checks `try_nix_eval` provides
/// (undefined names, attribute mismatches). Callers who need that should
/// use `generate_nix_with_rag` (slow) or `generate_nix_with_dry_build`
/// (slowest, full nixos eval) for batch verification.
///
/// What it keeps: idiom library, learned-idiom cache, RAG fallback,
/// nixpkgs option-index validation. Cache writes are gated on `parses ==
/// true` only (not eval) so the cache may include code that would have
/// failed `try_nix_eval`. In practice the option-index check catches
/// most semantic errors that eval would; the gap is narrow.
pub fn generate_nix_with_rag_fast(
    prompt: &str,
    max_iterations: usize,
) -> (NixGenWithIndexResult, SelfImproveSource) {
    generate_nix_with_rag_fast_using(prompt, max_iterations, shared_learned_cache())
}

/// Fast-mode RAG with an injectable cache (mirrors `generate_nix_with_cache`).
pub fn generate_nix_with_rag_fast_using(
    prompt: &str,
    max_iterations: usize,
    cache: &LearnedIdiomCache,
) -> (NixGenWithIndexResult, SelfImproveSource) {
    // Step 1: cache. Intent-gated + re-verified on hit (P2 integrity).
    let intent = classify_nix_intent(&prompt.to_lowercase());
    if let Some(idiom) = cache.find_match(prompt, intent) {
        let unknown = validate_options(&idiom.code, shared_nixpkgs_index());
        if unknown.is_empty() {
            let result = NixGenWithIndexResult {
                base: NixGenResult {
                    prompt: prompt.to_string(),
                    intent,
                    code: idiom.code.clone(),
                    iterations: 0,
                    parses: true,
                    last_error: None,
                    source: NixGenSource::Idiom,
                },
                unknown_options: Vec::new(),
            };
            return (result, SelfImproveSource::LearnedCache);
        }
        cache.invalidate(&idiom.prompt, &idiom.code);
    }

    // Step 2: idiom + parse-only verification (no eval).
    let mut iterations = 0;
    let mut base = generate_nix(prompt);
    iterations += base.iterations;

    // Step 3: option-index validation (cheap — pure HTTP/disk lookups).
    let unknown_options = if base.parses {
        validate_options(&base.code, shared_nixpkgs_index())
    } else {
        Vec::new()
    };

    // Step 4: RAG fallback if the idiom matcher punted to Skeleton.
    // (See `generate_nix_with_rag` step 3 for why we do NOT trigger on
    // unknown_options alone.)
    let weak = matches!(base.source, NixGenSource::Skeleton);
    let mut unknown_options = unknown_options;
    if weak {
        if let Some(draft) = rag_draft_from_search(prompt) {
            iterations += 1;
            base = NixGenResult {
                prompt: prompt.to_string(),
                intent: base.intent,
                code: draft,
                iterations,
                parses: true,
                last_error: None,
                source: NixGenSource::RagDraft,
            };
            unknown_options = Vec::new();
        }
    }

    let result = NixGenWithIndexResult {
        base,
        unknown_options,
    };

    // Step 5: record clean non-RAG results.
    let recorded = if result.base.parses
        && result.unknown_options.is_empty()
        && result.base.source == NixGenSource::Idiom
    {
        cache.record(prompt, &result.base.code, result.base.intent);
        true
    } else {
        false
    };
    (result, SelfImproveSource::FreshlyGenerated { recorded })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_intent_dev_shell() {
        assert_eq!(
            classify_nix_intent("set up a rust dev environment"),
            NixIntent::DevShell
        );
        assert_eq!(
            classify_nix_intent("python development environment with jupyter"),
            NixIntent::DevShell
        );
    }

    #[test]
    fn test_classify_intent_service() {
        assert_eq!(
            classify_nix_intent("set up postgresql with pgvector"),
            NixIntent::Service
        );
        assert_eq!(classify_nix_intent("enable docker"), NixIntent::Service);
    }

    #[test]
    fn test_classify_intent_desktop() {
        assert_eq!(
            classify_nix_intent("enable wayland with sway"),
            NixIntent::Desktop
        );
    }

    #[test]
    fn test_emit_rust_dev_shell() {
        let code = emit_dev_shell("set up rust dev environment with rust-analyzer and mold")
            .expect("rust idiom should match");
        assert!(code.contains("rustc"));
        assert!(code.contains("rust-analyzer"));
        assert!(code.contains("mold"));
        assert!(code.contains("mkShell"));
    }

    #[test]
    fn test_emit_postgres_with_pgvector() {
        let code = emit_service("set up postgresql with pgvector").expect("postgres idiom");
        assert!(code.contains("services.postgresql"));
        assert!(code.contains("pgvector"));
    }

    #[test]
    fn test_emit_docker_with_user_group() {
        let code = emit_service("enable docker and add my user to the docker group")
            .expect("docker idiom");
        assert!(code.contains("virtualisation.docker"));
        assert!(code.contains("docker"));
    }

    #[test]
    fn test_emit_sway_wayland() {
        let code = emit_desktop("enable wayland with sway and configure standard fonts")
            .expect("sway idiom");
        assert!(code.contains("programs.sway"));
        assert!(code.contains("xdg.portal"));
        assert!(code.contains("noto-fonts") || code.contains("fira-code"));
    }

    #[test]
    fn test_python_data_science() {
        let code =
            emit_dev_shell("configure a python data-science environment with jupyter and pandas")
                .expect("python idiom");
        assert!(code.contains("python311"));
        assert!(code.contains("jupyter"));
        assert!(code.contains("pandas"));
    }

    #[test]
    fn test_full_generate_nix_idiom_path() {
        let result = generate_nix("set up a rust dev environment with rust-analyzer");
        assert_eq!(result.intent, NixIntent::DevShell);
        assert!(matches!(result.source, NixGenSource::Idiom));
        assert!(result.code.contains("rustc"));
    }

    #[test]
    fn test_classify_nix_error_kinds() {
        assert_eq!(
            classify_nix_error("error: undefined variable 'firefoxx'"),
            NixErrorKind::UndefinedName
        );
        assert_eq!(
            classify_nix_error("syntax error, unexpected '}'"),
            NixErrorKind::Syntax
        );
        assert_eq!(
            classify_nix_error("expected a string but got an integer"),
            NixErrorKind::TypeMismatch
        );
    }

    #[test]
    fn test_eval_catches_typos_that_parse_misses() {
        // Nix 2.31's `--parse` resolves free variables (so a bare
        // `pkgs.firefoxxxx_undef` now fails parse too). To exercise the
        // parse-vs-eval gap, hide the bad attribute access behind a
        // function whose body is only invoked at eval time.
        let bad = "let f = x: x.firefoxxxx_undefined_attr; in f { }";
        let parse_ok = try_nix_parse(bad).is_ok();
        let eval_ok = try_nix_eval(bad).is_ok();
        assert!(parse_ok, "parse should accept syntactically valid input");
        assert!(!eval_ok, "eval should reject undefined attribute reference");
    }

    #[test]
    fn test_validate_conflicts_catches_dual_bootloader() {
        let code = r#"{
  boot.loader.grub.enable = true;
  boot.loader.systemd-boot.enable = true;
}"#;
        let conflicts = validate_conflicts(code);
        assert_eq!(
            conflicts.len(),
            1,
            "should detect grub vs systemd-boot conflict"
        );
        assert!(conflicts[0].reason.contains("boot loader"));
    }

    #[test]
    fn test_validate_conflicts_no_false_positive() {
        // Generated config from our pipeline should not flag conflicts
        let code = "{ services.postgresql.enable = true; }";
        assert!(
            validate_conflicts(code).is_empty(),
            "single-service config should have no conflicts"
        );
    }

    #[test]
    fn test_validate_conflicts_dual_dm() {
        let code = r#"{
  services.displayManager.gdm.enable = true;
  services.displayManager.sddm.enable = true;
}"#;
        let conflicts = validate_conflicts(code);
        assert_eq!(conflicts.len(), 1);
        assert!(conflicts[0].reason.contains("display manager"));
    }

    #[test]
    fn test_repair_returns_within_max_iterations() {
        let result = generate_nix_with_repair(
            "set up a rust dev environment with rust-analyzer and mold",
            3,
        );
        // Should succeed quickly (idiom matches first try)
        assert!(result.parses);
        assert!(result.iterations <= 3);
    }

    #[test]
    fn extract_option_paths_finds_known_roots() {
        let code = r#"{
  services.nginx.enable = true;
  services.postgresql.enable = true;
  hardware.opengl.enable = true;
  networking.firewall.allowedTCPPorts = [ 80 443 ];
  # comment with services.fake.path that should also be found by the scanner
  programs.git.enable = true;
}"#;
        let paths = extract_option_paths(code);
        // The scanner skips comments, so services.fake.path should NOT appear.
        assert!(paths.contains(&"services.nginx.enable".to_string()));
        assert!(paths.contains(&"services.postgresql.enable".to_string()));
        assert!(paths.contains(&"hardware.opengl.enable".to_string()));
        assert!(paths.contains(&"networking.firewall.allowedTCPPorts".to_string()));
        assert!(paths.contains(&"programs.git.enable".to_string()));
        assert!(!paths.contains(&"services.fake.path".to_string()));
    }

    #[test]
    fn extract_option_paths_ignores_pkgs_attrs() {
        // pkgs.* is NOT an option path — should be excluded.
        let code = r#"{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [ pkgs.rust-analyzer pkgs.mold ];
}"#;
        let paths = extract_option_paths(code);
        // pkgs is not in KNOWN_OPTION_ROOTS so none of those should appear.
        for p in &paths {
            assert!(
                !p.starts_with("pkgs."),
                "pkgs.* should be excluded, got {p}"
            );
        }
    }

    #[test]
    fn extract_option_paths_skips_strings() {
        // Strings should be skipped — `services.fake` inside a string is not
        // a real option reference.
        let code = r#"{
  services.openssh.enable = true;
  description = "see services.fake.path docs";
}"#;
        let paths = extract_option_paths(code);
        assert!(paths.contains(&"services.openssh.enable".to_string()));
        assert!(!paths.contains(&"services.fake.path".to_string()));
    }

    #[test]
    fn validate_options_flags_unknown() {
        // This test uses the live shared index — only meaningful if
        // nix-instantiate is available.
        if Command::new("nix-instantiate")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("[skip] nix-instantiate not available");
            return;
        }
        let code = r#"{
  services.nginx.enable = true;
  services.this_definitely_does_not_exist_xyz.enable = true;
}"#;
        let unknown = validate_options(code, shared_nixpkgs_index());
        let bad_path = "services.this_definitely_does_not_exist_xyz.enable";
        assert!(
            unknown.iter().any(|u| u.path == bad_path),
            "expected to flag {bad_path}, got {unknown:?}"
        );
        // services.nginx.enable is real — should NOT be flagged.
        assert!(
            !unknown.iter().any(|u| u.path == "services.nginx.enable"),
            "real path should not be flagged"
        );
    }

    #[test]
    fn generate_with_index_verify_clean_idiom() {
        if Command::new("nix-instantiate")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("[skip] nix-instantiate not available");
            return;
        }
        // The rust dev shell idiom doesn't reference any NixOS module options
        // (it's a standalone shell.nix), so there should be 0 unknown options.
        let result = generate_nix_with_index_verify(
            "set up a rust dev environment with rust-analyzer and mold",
            3,
        );
        assert!(result.base.parses);
        assert!(
            result.unknown_options.is_empty(),
            "shell.nix idiom should have no NixOS option references, got {:?}",
            result.unknown_options
        );
    }

    #[test]
    fn looks_like_module_classifies_correctly() {
        // Modules
        assert!(looks_like_module("{ services.openssh.enable = true; }"));
        assert!(looks_like_module(
            "{ pkgs, ... }: { services.nginx.enable = true; }"
        ));
        assert!(looks_like_module(
            "{ ... }: {\n  hardware.opengl.enable = true;\n}"
        ));

        // shell.nix — not a module
        assert!(!looks_like_module(
            "{ pkgs ? import <nixpkgs> {} }: pkgs.mkShell { buildInputs = [ ]; }"
        ));
        // flake.nix — not a module
        assert!(!looks_like_module(
            "{ description = \"x\"; inputs.nixpkgs.url = \"x\"; outputs = { ... }: { }; }"
        ));
    }

    #[test]
    fn try_nix_module_eval_skips_shell_nix() {
        let shell = "{ pkgs ? import <nixpkgs> {} }:\npkgs.mkShell { buildInputs = [ ]; }\n";
        // Returns None — caller knows to skip.
        assert!(try_nix_module_eval(shell).is_none());
    }

    #[test]
    #[ignore = "live: ~10-15s, slow even with warm cache"]
    fn try_nix_module_eval_live_clean_module() {
        if Command::new("nix-instantiate")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("[skip] nix-instantiate not available");
            return;
        }
        let module = "{ ... }: { services.openssh.enable = true; }";
        let verdict = try_nix_module_eval(module).expect("module should be evaluated");
        assert!(verdict.is_ok(), "clean module failed: {:?}", verdict);
    }

    #[test]
    #[ignore = "live: ~10-15s, slow even with warm cache"]
    fn try_nix_module_eval_live_catches_typo() {
        if Command::new("nix-instantiate")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("[skip] nix-instantiate not available");
            return;
        }
        // 'enabled' is a typo for 'enable' — module-eval should reject and
        // the error message should hint at the correct name.
        let bad = "{ ... }: { services.openssh.enabled = true; }";
        let verdict = try_nix_module_eval(bad).expect("module should be evaluated");
        assert!(!verdict.is_ok(), "typo'd module should fail eval");
        let msg = verdict.message().to_lowercase();
        assert!(
            msg.contains("enable") || msg.contains("does not exist"),
            "expected 'did you mean enable' hint, got: {}",
            verdict.message()
        );
    }

    #[test]
    fn self_improve_smoke() {
        // We can't easily inject a per-test cache (the function uses the
        // process-wide SHARED_LEARNED). What we CAN verify here is that
        // the function signature works end-to-end on a known-good prompt
        // and returns a SelfImproveSource of either variant.
        let (result, src) = generate_nix_with_self_improve(
            "set up a rust dev environment with rust-analyzer and mold",
            3,
        );
        assert!(result.base.parses, "rust dev shell should parse");
        // Either we hit the cache (a previous test left it warm) or we
        // generated fresh; both are valid outcomes.
        match src {
            SelfImproveSource::LearnedCache => {}
            SelfImproveSource::FreshlyGenerated { recorded: _ } => {}
        }
    }

    #[test]
    fn rag_keywords_filters_stop_and_short() {
        let kw = rag_keywords("Please configure the redis server with persistence");
        assert!(kw.contains(&"redis".to_string()));
        assert!(kw.contains(&"server".to_string()));
        assert!(kw.contains(&"persistence".to_string()));
        assert!(!kw.contains(&"please".to_string()));
        assert!(!kw.contains(&"configure".to_string()));
        assert!(!kw.contains(&"the".to_string()));
    }

    #[test]
    fn rag_prefixes_for_intent_matches_class() {
        assert!(rag_prefixes_for_intent(NixIntent::Service)
            .iter()
            .any(|p| *p == "services."));
        assert!(rag_prefixes_for_intent(NixIntent::Hardware)
            .iter()
            .any(|p| *p == "hardware."));
        assert!(rag_prefixes_for_intent(NixIntent::DevShell).is_empty());
    }

    #[test]
    fn render_rag_draft_includes_options_as_comments() {
        use crate::language::nixos_search::SearchHit;
        let hits = vec![
            SearchHit {
                option_name: "services.redis.servers".to_string(),
                option_type: "attribute set of submodules".to_string(),
                option_description: "Redis instances.".to_string(),
            },
            SearchHit {
                option_name: "services.redis.vmOverCommit".to_string(),
                option_type: "boolean".to_string(),
                option_description: "Enable vm overcommit.".to_string(),
            },
        ];
        let draft = render_rag_draft("enable redis", &hits);
        // Submodules are skipped (caller would have to write nested config).
        assert!(!draft.contains("services.redis.servers ="));
        // Leaf boolean is suggested with a `true` default.
        assert!(draft.contains("# services.redis.vmOverCommit = true;"));
        // Header callout is present.
        assert!(draft.contains("NOT verified"));
        // Output is module-shaped so it parses.
        assert!(try_nix_parse(&draft).is_ok(), "RAG draft must parse");
    }

    #[test]
    fn rag_draft_returns_none_for_devshell() {
        // DevShell intent has no RAG prefixes — we don't fall back to
        // search for shell.nix prompts.
        assert!(rag_draft_from_search("set up a rust dev shell").is_none());
    }

    #[test]
    fn generate_with_rag_uses_idiom_for_known_prompt() {
        // For a prompt the idiom library knows, the RAG path should NOT
        // fire — we should get NixGenSource::Idiom (or LearnedCache hit).
        let (result, _src) =
            generate_nix_with_rag("set up a rust dev environment with rust-analyzer", 3);
        assert!(result.base.parses);
        assert_ne!(
            result.base.source,
            NixGenSource::RagDraft,
            "known prompts should not invoke RAG"
        );
    }

    #[test]
    fn generate_with_rag_falls_back_for_unknown_service() {
        // No idiom for jellyfin yet — RAG should fire if search.nixos.org
        // is reachable, otherwise we get a Skeleton.
        if Command::new("curl").arg("--version").output().is_err() {
            eprintln!("[skip] curl not available");
            return;
        }
        let (result, _src) = generate_nix_with_rag("set up jellyfin media server", 3);
        // Must parse regardless of which path won.
        assert!(result.base.parses);
        // If we got RAG, the draft should mention jellyfin somewhere.
        if result.base.source == NixGenSource::RagDraft {
            assert!(
                result.base.code.to_lowercase().contains("jellyfin"),
                "RAG draft for 'jellyfin' should include the keyword: {}",
                result.base.code
            );
        }
    }

    #[test]
    fn rag_fast_cold_latency_is_sub_second() {
        // The headline guarantee: cold prompts return in <1s with the fast
        // pipeline. Real interactive use needs this — full eval mode takes
        // ~100s per cold prompt because of nixpkgs re-import.
        use std::path::PathBuf;
        use std::time::Instant;
        let cache_path: PathBuf = std::env::temp_dir().join(format!(
            "symthaea_rag_fast_latency_{}_{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::remove_file(&cache_path);
        let cache = LearnedIdiomCache::with_cache_path(cache_path.clone());

        // Pick a prompt the idiom library knows so we don't depend on
        // search.nixos.org being reachable.
        let t0 = Instant::now();
        let (result, _src) =
            generate_nix_with_rag_fast_using("set up postgresql with pgvector", 3, &cache);
        let elapsed = t0.elapsed();

        assert!(
            result.base.parses,
            "fast-mode RAG should produce parsing code"
        );
        assert!(
            elapsed.as_secs() < 1,
            "fast-mode cold latency must be <1s, got {:?}",
            elapsed
        );
        let _ = std::fs::remove_file(&cache_path);
    }
}
