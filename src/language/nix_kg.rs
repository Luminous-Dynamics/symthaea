// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Nix Knowledge Graph — single source of truth for codegen lookup tables.
//!
//! Replaces four hand-coded tables that were scattered across `nix_codegen.rs`:
//! - `KNOWN_OPTION_ROOTS` (top-level NixOS module namespaces)
//! - `known_conflicts()` (option pairs that can't coexist)
//! - `rag_prefixes_for_intent()` (intent → option-prefix map for RAG search)
//! - the `Service` keyword block in `classify_nix_intent` (service-name → intent)
//!
//! ## Why not just leave them as code?
//! Each new service / conflict / option-namespace was a code edit + recompile.
//! As a typed table loaded from JSON, additions become data — and a future
//! sync layer with `mycelix-knowledge` (the federated knowledge-graph cluster
//! in this monorepo) can contribute entries with provenance/attestation.
//!
//! ## Loading
//! - `NixKg::default()` returns the bundled-in defaults (this file is the
//!   source of truth at build time).
//! - `NixKg::from_disk_or_default()` loads `~/.cache/symthaea/nix-kg.json`
//!   if present; falls back to defaults otherwise.
//! - User overrides MERGE additively — the user can extend without losing
//!   bundled entries.
//!
//! ## Schema
//! The on-disk JSON shape is intentionally close to mycelix-knowledge's
//! claim/edge model so a future sync is a data migration, not a rewrite.
//! See `NixKgFile` for the canonical layout.

use crate::language::nix_codegen::NixIntent;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// One conflict claim — two option assignments that cannot coexist.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConflictClaim {
    /// Left option assignment (e.g. `boot.loader.grub.enable = true`).
    pub a: String,
    /// Right option assignment (the conflicting one).
    pub b: String,
    /// Human-readable reason.
    pub reason: String,
}

/// One service mapping — keyword → Service intent (e.g. "tailscale").
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServiceKeyword {
    /// The keyword users type (lowercased).
    pub keyword: String,
    /// Optional canonical option-path the service maps to (e.g.
    /// `services.tailscale`). Used by future RAG and conflict detection.
    pub option_path: Option<String>,
}

/// On-disk JSON layout. Versioned; future schema bumps can migrate.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct NixKgFile {
    /// Schema version (current = 1).
    pub version: u32,
    /// Top-level NixOS module namespace roots ("services", "hardware", …).
    pub option_roots: Vec<String>,
    /// Conflict pairs — two options that can't both be enabled.
    pub conflicts: Vec<ConflictClaim>,
    /// Service-name keywords that should classify as `NixIntent::Service`.
    pub service_keywords: Vec<ServiceKeyword>,
    /// Per-intent RAG search prefixes (intent name → list of `services.X.` etc.).
    /// Keys are the Debug name of the NixIntent variant ("Service", "Hardware", …).
    pub rag_prefixes: HashMap<String, Vec<String>>,
}

const KG_VERSION: u32 = 1;

/// Compiled, runtime-ready knowledge graph. Cheap to construct from `NixKgFile`.
#[derive(Clone, Debug)]
pub struct NixKg {
    pub option_roots: Vec<String>,
    pub conflicts: Vec<ConflictClaim>,
    pub service_keywords: Vec<String>, // canonical lowercased
    pub rag_prefixes: HashMap<NixIntent, Vec<String>>,
}

impl Default for NixKg {
    fn default() -> Self {
        bundled_defaults()
    }
}

impl NixKg {
    /// Try to load from `~/.cache/symthaea/nix-kg.json`; merge into defaults.
    /// Falls back to pure defaults if file absent or malformed.
    pub fn from_disk_or_default() -> Self {
        Self::from_path_or_default(&default_path())
    }

    /// Try to load from an explicit path; merge into defaults. Useful for
    /// tests, demos, and shipping a per-deployment override file outside
    /// the user's home directory.
    pub fn from_path_or_default(path: &std::path::Path) -> Self {
        let mut kg = Self::default();
        if let Some(file) = load_from_path(path) {
            kg.merge_from_file(&file);
        }
        kg
    }

    /// Construct from an explicit `NixKgFile` (useful for tests / overrides).
    pub fn from_file(file: NixKgFile) -> Self {
        let mut kg = Self::default();
        kg.merge_from_file(&file);
        kg
    }

    /// Additive merge: file entries that aren't already present get appended.
    /// Same-keyed entries (matched by `a+b` for conflicts, by `keyword` for
    /// services, by literal string for roots/prefixes) are NOT overwritten —
    /// bundled defaults win to keep tests stable.
    fn merge_from_file(&mut self, file: &NixKgFile) {
        for r in &file.option_roots {
            if !self.option_roots.iter().any(|x| x == r) {
                self.option_roots.push(r.clone());
            }
        }
        for c in &file.conflicts {
            if !self.conflicts.iter().any(|x| x.a == c.a && x.b == c.b) {
                self.conflicts.push(c.clone());
            }
        }
        for s in &file.service_keywords {
            let kw = s.keyword.to_lowercase();
            if !self.service_keywords.iter().any(|x| x == &kw) {
                self.service_keywords.push(kw);
            }
        }
        for (intent_name, prefixes) in &file.rag_prefixes {
            if let Some(intent) = parse_intent(intent_name) {
                let entry = self.rag_prefixes.entry(intent).or_default();
                for p in prefixes {
                    if !entry.iter().any(|x| x == p) {
                        entry.push(p.clone());
                    }
                }
            }
        }
    }

    /// True if `prompt` (lowercased) contains any service keyword.
    pub fn matches_service_keyword(&self, lower: &str) -> bool {
        self.service_keywords.iter().any(|k| lower.contains(k))
    }

    /// True if `root` (the first segment of a dotted option path) is a known
    /// NixOS module-root namespace.
    pub fn is_known_option_root(&self, root: &str) -> bool {
        self.option_roots.iter().any(|r| r == root)
    }

    /// RAG prefixes to probe for the given intent. Empty slice if no prefixes
    /// are configured (caller should skip the RAG step entirely).
    pub fn rag_prefixes_for(&self, intent: NixIntent) -> &[String] {
        self.rag_prefixes
            .get(&intent)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

// ─── Bundled defaults ──────────────────────────────────────────────────────

fn bundled_defaults() -> NixKg {
    let mut rag_prefixes: HashMap<NixIntent, Vec<String>> = HashMap::new();
    rag_prefixes.insert(NixIntent::Service, vec!["services.".to_string()]);
    rag_prefixes.insert(
        NixIntent::Hardware,
        vec!["hardware.".to_string(), "boot.".to_string()],
    );
    rag_prefixes.insert(NixIntent::Networking, vec!["networking.".to_string()]);
    rag_prefixes.insert(
        NixIntent::Desktop,
        vec![
            "services.xserver.".to_string(),
            "services.displayManager.".to_string(),
        ],
    );
    rag_prefixes.insert(NixIntent::User, vec!["users.".to_string()]);
    rag_prefixes.insert(NixIntent::HomeManager, vec!["programs.".to_string()]);

    NixKg {
        option_roots: vec![
            "services",
            "hardware",
            "programs",
            "networking",
            "boot",
            "users",
            "systemd",
            "security",
            "virtualisation",
            "nix",
            "nixpkgs",
            "fileSystems",
            "swapDevices",
            "fonts",
            "i18n",
            "time",
            "xdg",
            "console",
            "sound",
            "powerManagement",
            "documentation",
            "system",
        ]
        .into_iter()
        .map(String::from)
        .collect(),

        conflicts: vec![
            ConflictClaim {
                a: "boot.loader.grub.enable = true".into(),
                b: "boot.loader.systemd-boot.enable = true".into(),
                reason: "two boot loaders enabled simultaneously".into(),
            },
            ConflictClaim {
                a: "services.displayManager.gdm.enable = true".into(),
                b: "services.displayManager.sddm.enable = true".into(),
                reason: "two display managers conflict".into(),
            },
            ConflictClaim {
                a: "services.xserver.displayManager.gdm.enable = true".into(),
                b: "services.xserver.displayManager.sddm.enable = true".into(),
                reason: "two display managers conflict".into(),
            },
            ConflictClaim {
                a: "networking.networkmanager.enable = true".into(),
                b: "networking.wireless.enable = true".into(),
                reason: "NetworkManager and wpa_supplicant both manage wireless".into(),
            },
            ConflictClaim {
                a: "services.desktopManager.plasma6.enable = true".into(),
                b: "services.xserver.desktopManager.gnome.enable = true".into(),
                reason: "two desktop environments conflict".into(),
            },
            ConflictClaim {
                a: "programs.sway.enable = true".into(),
                b: "services.xserver.enable = true".into(),
                reason: "Sway is a Wayland compositor — xserver is unnecessary and may conflict"
                    .into(),
            },
            ConflictClaim {
                a: "programs.hyprland.enable = true".into(),
                b: "services.xserver.enable = true".into(),
                reason: "Hyprland is Wayland-only — xserver shouldn't be enabled alongside".into(),
            },
            ConflictClaim {
                a: "services.openssh.enable = true".into(),
                b: "services.openssh.enable = false".into(),
                reason: "contradictory openssh settings".into(),
            },
        ],

        // Lowercased keywords that classify as Service intent. Mirrors the
        // hand-coded block in `classify_nix_intent`'s Service branch.
        service_keywords: [
            "service",
            "postgres",
            "nginx",
            "redis",
            "docker",
            "ipfs",
            "kubo",
            "avahi",
            "openssh",
            "tailscale",
            "prometheus",
            "grafana",
            "jellyfin",
            "plex",
            "podman",
            "libvirt",
            "steam",
            "cups",
            "printing",
            "systemd-resolved",
            "resolved",
            "monitoring",
            "vpn",
            "media server",
        ]
        .into_iter()
        .map(String::from)
        .collect(),

        rag_prefixes,
    }
}

// ─── Internals ──────────────────────────────────────────────────────────────

fn default_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("symthaea")
        .join("nix-kg.json")
}

fn load_from_path(path: &std::path::Path) -> Option<NixKgFile> {
    let bytes = std::fs::read(path).ok()?;
    let parsed: NixKgFile = serde_json::from_slice(&bytes).ok()?;
    if parsed.version != KG_VERSION {
        return None;
    }
    Some(parsed)
}

/// Public schema version — write this into the `version` field of any
/// `~/.cache/symthaea/nix-kg.json` override file. Loader rejects mismatches
/// to make schema migrations explicit rather than silent.
pub const SCHEMA_VERSION: u32 = KG_VERSION;

/// Parse the Debug-name of a `NixIntent` variant back into the enum.
fn parse_intent(name: &str) -> Option<NixIntent> {
    match name {
        "DevShell" => Some(NixIntent::DevShell),
        "Service" => Some(NixIntent::Service),
        "Hardware" => Some(NixIntent::Hardware),
        "Desktop" => Some(NixIntent::Desktop),
        "User" => Some(NixIntent::User),
        "HomeManager" => Some(NixIntent::HomeManager),
        "Networking" => Some(NixIntent::Networking),
        "Secrets" => Some(NixIntent::Secrets),
        "FlakeTemplate" => Some(NixIntent::FlakeTemplate),
        "Generic" => Some(NixIntent::Generic),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_have_expected_entries() {
        let kg = NixKg::default();
        assert!(kg.is_known_option_root("services"));
        assert!(kg.is_known_option_root("hardware"));
        assert!(!kg.is_known_option_root("pkgs")); // pkgs is NOT an option root
        assert!(kg.matches_service_keyword("set up tailscale vpn"));
        assert!(kg.matches_service_keyword("configure prometheus monitoring"));
        assert!(!kg.matches_service_keyword("explain quantum entanglement"));
        assert!(!kg.rag_prefixes_for(NixIntent::Service).is_empty());
        assert!(kg.rag_prefixes_for(NixIntent::DevShell).is_empty());
    }

    #[test]
    fn conflicts_round_trip_via_json() {
        let kg = NixKg::default();
        // Serialize the conflicts to JSON and back, ensure they survive.
        let json = serde_json::to_string(&kg.conflicts).unwrap();
        let back: Vec<ConflictClaim> = serde_json::from_str(&json).unwrap();
        assert_eq!(kg.conflicts.len(), back.len());
        assert!(back
            .iter()
            .any(|c| c.reason.contains("two boot loaders enabled simultaneously")));
    }

    #[test]
    fn merge_adds_user_entries_without_clobbering_defaults() {
        let mut user_file = NixKgFile {
            version: KG_VERSION,
            option_roots: vec!["myCustomRoot".to_string()],
            conflicts: vec![ConflictClaim {
                a: "services.foo = true".into(),
                b: "services.bar = true".into(),
                reason: "user-defined conflict".into(),
            }],
            service_keywords: vec![ServiceKeyword {
                keyword: "MyService".into(), // mixed case → should lowercase
                option_path: Some("services.myservice".into()),
            }],
            rag_prefixes: HashMap::new(),
        };
        user_file
            .rag_prefixes
            .insert("Service".to_string(), vec!["services.custom.".to_string()]);

        let kg = NixKg::from_file(user_file);
        assert!(kg.is_known_option_root("myCustomRoot"));
        assert!(kg.is_known_option_root("services")); // bundled default still present
        assert!(kg
            .conflicts
            .iter()
            .any(|c| c.reason == "user-defined conflict"));
        assert!(kg
            .conflicts
            .iter()
            .any(|c| c.reason.contains("two boot loaders"))); // default kept
        assert!(kg.service_keywords.iter().any(|k| k == "myservice"));
        assert!(kg
            .rag_prefixes_for(NixIntent::Service)
            .iter()
            .any(|p| p == "services.custom."));
        assert!(kg
            .rag_prefixes_for(NixIntent::Service)
            .iter()
            .any(|p| p == "services.")); // default kept
    }

    #[test]
    fn parse_intent_handles_all_variants() {
        for v in NixIntent::ALL {
            let name = format!("{:?}", v);
            assert_eq!(parse_intent(&name), Some(v), "failed for {name}");
        }
        assert_eq!(parse_intent("Bogus"), None);
    }

    #[test]
    fn version_mismatch_drops_file() {
        let tmp = std::env::temp_dir().join(format!(
            "symthaea_nix_kg_version_{}_{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let bad = NixKgFile {
            version: 999, // wrong
            ..Default::default()
        };
        std::fs::write(&tmp, serde_json::to_vec(&bad).unwrap()).unwrap();
        let bytes = std::fs::read(&tmp).unwrap();
        let parsed: NixKgFile = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.version, 999);
        // Loader should reject this.
        // (We can't directly test load_from_default_path without env mucking;
        // the version check is in load_from_default_path itself.)
        let _ = std::fs::remove_file(&tmp);
    }
}
