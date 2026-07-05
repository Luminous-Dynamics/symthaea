// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Schema representation for NixOS options.
//!
//! Parses the JSON output of `nixos-option --json` into structured schemas
//! that can be used for causal graph bootstrapping and HDC encoding.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents the full schema of a NixOS option including type, default, and description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NixOptionSchema {
    /// Full dotted path (e.g., "services.nginx.enable")
    pub name: String,
    /// NixOS option type (e.g., "bool", "string", "listOf package")
    pub option_type: String,
    /// Default value as string representation
    pub default: Option<String>,
    /// Human-readable description
    pub description: String,
    /// Packages related to this option
    pub related_packages: Vec<String>,
    /// Module declaration file paths
    pub declarations: Vec<String>,
}

/// Raw JSON schema entry from `nixos-option --json`
#[derive(Debug, Deserialize)]
struct RawOptionEntry {
    #[serde(rename = "type")]
    option_type: Option<String>,
    default: Option<serde_json::Value>,
    description: Option<String>,
    declarations: Option<Vec<String>>,
    #[serde(rename = "readOnly")]
    #[allow(dead_code)]
    read_only: Option<bool>,
}

impl NixOptionSchema {
    /// Get the dot-separated path components.
    pub fn path_components(&self) -> Vec<&str> {
        self.name.split('.').collect()
    }

    /// Get the top-level module prefix (e.g., "services" from "services.nginx.enable").
    pub fn module_prefix(&self) -> &str {
        self.name.split('.').next().unwrap_or(&self.name)
    }

    /// Whether this is an `.enable` toggle option.
    pub fn is_enable_option(&self) -> bool {
        self.name.ends_with(".enable") && self.option_type == "bool"
    }

    /// Whether this is a package list option.
    pub fn is_package_list(&self) -> bool {
        self.option_type.contains("listOf")
            && (self.option_type.contains("package") || self.name.contains("Packages"))
    }

    /// Extract the service name if this is a service option.
    pub fn service_name(&self) -> Option<&str> {
        let parts = self.path_components();
        if parts.len() >= 2 && parts[0] == "services" {
            Some(parts[1])
        } else {
            None
        }
    }
}

/// Parse the JSON output of `nixos-option --json` into structured schemas.
pub fn parse_nixos_options(json: &str) -> Result<Vec<NixOptionSchema>, String> {
    let raw: HashMap<String, RawOptionEntry> =
        serde_json::from_str(json).map_err(|e| format!("Failed to parse options JSON: {e}"))?;

    let mut schemas = Vec::with_capacity(raw.len());

    for (name, entry) in raw {
        let option_type = entry.option_type.unwrap_or_else(|| "unknown".into());
        let description = entry.description.unwrap_or_default();
        let default = entry.default.map(|v| match v {
            serde_json::Value::String(s) => s,
            other => other.to_string(),
        });

        // Extract related packages from description heuristics
        let related_packages = extract_related_packages(&name, &description);

        schemas.push(NixOptionSchema {
            name,
            option_type,
            default,
            description,
            related_packages,
            declarations: entry.declarations.unwrap_or_default(),
        });
    }

    schemas.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(schemas)
}

/// Load NixOS options by executing `nixos-option --json`.
///
/// Returns an error if the command is not available (non-NixOS systems).
pub fn load_nixos_options() -> Result<Vec<NixOptionSchema>, String> {
    let output = std::process::Command::new("nixos-option")
        .arg("--json")
        .output()
        .map_err(|e| format!("Failed to execute nixos-option: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "nixos-option failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let json = String::from_utf8_lossy(&output.stdout);
    parse_nixos_options(&json)
}

/// Load bundled NixOS option schemas compiled into the binary.
///
/// This provides a fallback for non-NixOS systems and CI environments where
/// `nixos-option` is not available. The bundled set covers ~180 common options
/// across boot, hardware, services, networking, security, programs, and virtualisation.
pub fn load_bundled_options() -> Result<Vec<NixOptionSchema>, String> {
    let json = include_str!("bundled_options.json");
    parse_nixos_options(json)
}

/// Load NixOS options with automatic fallback.
///
/// 1. If on NixOS, tries `nixos-option --json` first
/// 2. Falls back to bundled options if the command fails or isn't available
///
/// This ensures the system always has a usable option set.
pub fn load_options_with_fallback() -> Vec<NixOptionSchema> {
    match load_nixos_options() {
        Ok(schemas) if !schemas.is_empty() => {
            tracing::info!(count = schemas.len(), "Loaded live NixOS options");
            schemas
        }
        Ok(_) | Err(_) => {
            tracing::info!("Using bundled NixOS option schemas (nixos-option unavailable)");
            load_bundled_options().unwrap_or_default()
        }
    }
}

/// Auto-refreshing option schema cache.
///
/// Caches loaded schemas and refreshes them periodically on NixOS systems.
pub struct OptionSchemaCache {
    schemas: Vec<NixOptionSchema>,
    last_refresh: std::time::Instant,
    refresh_interval: std::time::Duration,
}

impl OptionSchemaCache {
    /// Create a new cache, loading schemas immediately.
    pub fn new() -> Self {
        Self {
            schemas: load_options_with_fallback(),
            last_refresh: std::time::Instant::now(),
            refresh_interval: std::time::Duration::from_secs(3600), // 1 hour default
        }
    }

    /// Set the refresh interval.
    pub fn with_refresh_interval(mut self, interval: std::time::Duration) -> Self {
        self.refresh_interval = interval;
        self
    }

    /// Get the cached schemas, refreshing if stale.
    pub fn schemas(&mut self) -> &[NixOptionSchema] {
        if self.last_refresh.elapsed() >= self.refresh_interval {
            self.refresh();
        }
        &self.schemas
    }

    /// Force a refresh from the live system (with bundled fallback).
    pub fn refresh(&mut self) {
        let fresh = load_options_with_fallback();
        if !fresh.is_empty() {
            self.schemas = fresh;
            self.last_refresh = std::time::Instant::now();
        }
    }

    /// Number of cached schemas.
    pub fn len(&self) -> usize {
        self.schemas.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.schemas.is_empty()
    }
}

impl Default for OptionSchemaCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Heuristic extraction of related packages from option name and description.
fn extract_related_packages(name: &str, description: &str) -> Vec<String> {
    let mut packages = Vec::new();

    // Service options often relate to a package of the same name
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() >= 2 && parts[0] == "services" {
        packages.push(parts[1].to_string());
    }
    if parts.len() >= 2 && parts[0] == "programs" {
        packages.push(parts[1].to_string());
    }

    // Look for `pkgs.X` references in description
    for word in description.split_whitespace() {
        let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '-');
        if let Some(pkg) = trimmed.strip_prefix("pkgs.")
            && !pkg.is_empty()
            && !packages.contains(&pkg.to_string())
        {
            packages.push(pkg.to_string());
        }
    }

    packages
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_json() -> &'static str {
        r#"{
            "services.nginx.enable": {
                "type": "bool",
                "default": false,
                "description": "Whether to enable the nginx HTTP server.",
                "declarations": ["nixos/modules/services/web-servers/nginx/default.nix"],
                "readOnly": false
            },
            "services.nginx.package": {
                "type": "package",
                "default": "pkgs.nginx",
                "description": "The nginx package to use. Defaults to pkgs.nginx.",
                "declarations": ["nixos/modules/services/web-servers/nginx/default.nix"],
                "readOnly": false
            },
            "environment.systemPackages": {
                "type": "listOf package",
                "default": [],
                "description": "The set of packages that appear in /run/current-system/sw.",
                "declarations": ["nixos/modules/config/system-path.nix"],
                "readOnly": false
            }
        }"#
    }

    #[test]
    fn test_parse_nixos_options() {
        let schemas = parse_nixos_options(sample_json()).unwrap();
        assert_eq!(schemas.len(), 3);

        let nginx_enable = schemas
            .iter()
            .find(|s| s.name == "services.nginx.enable")
            .unwrap();
        assert!(nginx_enable.is_enable_option());
        assert_eq!(nginx_enable.module_prefix(), "services");
        assert_eq!(nginx_enable.service_name(), Some("nginx"));
        assert_eq!(
            nginx_enable.path_components(),
            vec!["services", "nginx", "enable"]
        );
    }

    #[test]
    fn test_package_list_detection() {
        let schemas = parse_nixos_options(sample_json()).unwrap();
        let sys_pkgs = schemas
            .iter()
            .find(|s| s.name == "environment.systemPackages")
            .unwrap();
        assert!(sys_pkgs.is_package_list());
        assert!(!sys_pkgs.is_enable_option());
    }

    #[test]
    fn test_load_bundled_options() {
        let schemas = load_bundled_options().unwrap();
        // Should have a substantial number of bundled options
        assert!(
            schemas.len() >= 100,
            "expected >= 100 bundled options, got {}",
            schemas.len()
        );
        // Should include common options
        assert!(schemas.iter().any(|s| s.name == "services.nginx.enable"));
        assert!(schemas.iter().any(|s| s.name == "boot.loader.grub.enable"));
        assert!(
            schemas
                .iter()
                .any(|s| s.name == "networking.firewall.enable")
        );
    }

    #[test]
    fn test_related_packages_extraction() {
        let schemas = parse_nixos_options(sample_json()).unwrap();
        let nginx_pkg = schemas
            .iter()
            .find(|s| s.name == "services.nginx.package")
            .unwrap();
        assert!(nginx_pkg.related_packages.contains(&"nginx".to_string()));
    }

    #[test]
    fn test_parse_invalid_json() {
        let result = parse_nixos_options("not json at all");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to parse"));
    }

    #[test]
    fn test_parse_empty_object() {
        let schemas = parse_nixos_options("{}").unwrap();
        assert!(schemas.is_empty());
    }

    #[test]
    fn test_missing_type_defaults_unknown() {
        let json = r#"{ "my.option": { "description": "no type field" } }"#;
        let schemas = parse_nixos_options(json).unwrap();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].option_type, "unknown");
    }

    #[test]
    fn test_default_value_variants() {
        let json = r#"{
            "a": { "type": "bool", "default": true, "description": "" },
            "b": { "type": "string", "default": "hello", "description": "" },
            "c": { "type": "int", "default": 42, "description": "" },
            "d": { "type": "null", "description": "" }
        }"#;
        let schemas = parse_nixos_options(json).unwrap();
        let a = schemas.iter().find(|s| s.name == "a").unwrap();
        assert_eq!(a.default.as_deref(), Some("true"));
        let b = schemas.iter().find(|s| s.name == "b").unwrap();
        assert_eq!(b.default.as_deref(), Some("hello"));
        let c = schemas.iter().find(|s| s.name == "c").unwrap();
        assert_eq!(c.default.as_deref(), Some("42"));
        let d = schemas.iter().find(|s| s.name == "d").unwrap();
        assert!(d.default.is_none());
    }

    #[test]
    fn test_schema_accessors() {
        let schema = NixOptionSchema {
            name: "services.postgresql.enable".to_string(),
            option_type: "bool".to_string(),
            default: None,
            description: "Enable PostgreSQL".to_string(),
            related_packages: vec![],
            declarations: vec![],
        };
        assert!(schema.is_enable_option());
        assert_eq!(schema.module_prefix(), "services");
        assert_eq!(schema.service_name(), Some("postgresql"));
        assert_eq!(
            schema.path_components(),
            vec!["services", "postgresql", "enable"]
        );
    }

    #[test]
    fn test_non_service_has_no_service_name() {
        let schema = NixOptionSchema {
            name: "boot.loader.grub.enable".to_string(),
            option_type: "bool".to_string(),
            default: None,
            description: "".to_string(),
            related_packages: vec![],
            declarations: vec![],
        };
        assert!(schema.is_enable_option());
        assert_eq!(schema.service_name(), None);
    }

    #[test]
    fn test_is_package_list_variants() {
        let pkg_list = NixOptionSchema {
            name: "environment.systemPackages".to_string(),
            option_type: "listOf package".to_string(),
            default: None,
            description: "".to_string(),
            related_packages: vec![],
            declarations: vec![],
        };
        assert!(pkg_list.is_package_list());

        let non_list = NixOptionSchema {
            name: "services.nginx.enable".to_string(),
            option_type: "bool".to_string(),
            default: None,
            description: "".to_string(),
            related_packages: vec![],
            declarations: vec![],
        };
        assert!(!non_list.is_package_list());
    }

    #[test]
    fn test_programs_related_packages() {
        let json = r#"{
            "programs.firefox.enable": {
                "type": "bool",
                "default": false,
                "description": "Enable Firefox via pkgs.firefox"
            }
        }"#;
        let schemas = parse_nixos_options(json).unwrap();
        // "programs.firefox" → "firefox" from prefix extraction
        assert!(schemas[0].related_packages.contains(&"firefox".to_string()));
        // pkgs.firefox from description also extracted (deduped)
        assert_eq!(
            schemas[0]
                .related_packages
                .iter()
                .filter(|p| *p == "firefox")
                .count(),
            1,
            "Should deduplicate"
        );
    }

    #[test]
    fn test_cache_len_and_is_empty() {
        let cache = OptionSchemaCache::new();
        // Bundled options should be loaded
        assert!(!cache.is_empty());
        assert!(cache.len() >= 100);
    }

    #[test]
    fn test_cache_with_refresh_interval() {
        let cache =
            OptionSchemaCache::new().with_refresh_interval(std::time::Duration::from_secs(60));
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_cache_schemas_returns_slice() {
        let mut cache = OptionSchemaCache::new();
        let schemas = cache.schemas();
        assert!(!schemas.is_empty());
        // All schemas should have non-empty names
        for s in schemas {
            assert!(!s.name.is_empty());
        }
    }

    #[test]
    fn test_sorted_output() {
        let schemas = parse_nixos_options(sample_json()).unwrap();
        for w in schemas.windows(2) {
            assert!(w[0].name <= w[1].name, "Results should be sorted by name");
        }
    }
}
