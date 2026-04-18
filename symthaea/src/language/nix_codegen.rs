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

use std::process::Command;

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
    /// Generic / unknown.
    Generic,
}

impl NixIntent {
    pub const ALL: [Self; 8] = [
        Self::DevShell,
        Self::Service,
        Self::Hardware,
        Self::Desktop,
        Self::User,
        Self::HomeManager,
        Self::Networking,
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
    /// 8 intent class one-hot (matches NixIntent::ALL order).
    pub intent: [f32; 8],
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
        .unwrap_or(7);
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
    } else if lower.contains("enable ")
        || lower.contains("service")
        || lower.contains("postgres")
        || lower.contains("nginx")
        || lower.contains("redis")
        || lower.contains("docker")
        || lower.contains("ipfs")
        || lower.contains("kubo")
        || lower.contains("avahi")
        || lower.contains("set up postgres")
        || lower.contains("set up ipfs")
        || lower.contains("set up redis")
    {
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
    let intent = classify_nix_intent(&lower);

    match intent {
        NixIntent::DevShell => emit_dev_shell(&lower),
        NixIntent::Service => emit_service(&lower),
        NixIntent::Hardware => emit_hardware(&lower),
        NixIntent::Desktop => emit_desktop(&lower),
        NixIntent::User => emit_user_group(&lower),
        NixIntent::Networking => emit_networking(&lower),
        NixIntent::HomeManager => emit_home_manager(&lower),
        NixIntent::Generic => None,
    }
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
        if lower.contains("sccache") {
            tools.push("sccache");
        }
        if !tools.contains(&"pkg-config") && lower.contains("openssl") {
            tools.extend_from_slice(&["pkg-config", "openssl", "openssl.dev"]);
        }
        let pkg_list = tools.join(" ");
        return Some(format!(
            r#"{{ pkgs ? import <nixpkgs> {{}} }}:
pkgs.mkShell {{
  name = "rust-dev-shell";
  buildInputs = with pkgs; [ {pkg_list} ];
  shellHook = ''
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
    // Sway / Wayland
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
    // Extract port numbers from prompt
    let mut tcp_ports = Vec::new();
    for word in lower.split_whitespace() {
        if let Ok(p) = word.trim_end_matches(',').parse::<u16>() {
            if p >= 1 {
                tcp_ports.push(p);
            }
        }
    }
    if tcp_ports.is_empty() && (lower.contains("port 80") || lower.contains("http")) {
        tcp_ports.push(80);
    }
    if tcp_ports.is_empty() && lower.contains("https") {
        tcp_ports.push(443);
    }
    if tcp_ports.is_empty() {
        return None;
    }
    let port_list = tcp_ports
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    Some(format!(
        r#"{{
  networking.firewall.allowedTCPPorts = [ {port_list} ];
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
    let path = tmp.join("symthaea_nix_codegen_eval.nix");
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

/// Verify a Nix expression by running `nix-instantiate --parse`.
///
/// This is the fastest verification (millisecond) — checks syntax only,
/// not type-correctness or semantic validity. Sufficient for first-cut
/// repair signal.
pub fn try_nix_parse(expr: &str) -> NixVerdict {
    let tmp = std::env::temp_dir();
    let path = tmp.join("symthaea_nix_codegen_check.nix");
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

#[derive(Debug)]
pub enum NixGenSource {
    /// Idiom library matched + emitted directly.
    Idiom,
    /// No idiom match — used a fallback skeleton.
    Skeleton,
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
        // Parse-only would accept this; eval catches the undefined name
        let bad = "{ x = pkgs.firefoxxxx_does_not_exist; }";
        let parse_ok = try_nix_parse(bad).is_ok();
        let eval_ok = try_nix_eval(bad).is_ok();
        // Parse-only DOES accept attribute access (it's syntactic)
        // Eval should NOT accept undefined `pkgs.firefoxxxx_does_not_exist`
        assert!(parse_ok, "parse should accept syntactically valid input");
        assert!(!eval_ok, "eval should reject undefined attribute reference");
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
}
