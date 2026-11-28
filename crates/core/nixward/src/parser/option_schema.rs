// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Schema representation for NixOS options.
//!
//! Parses the JSON output of `nixos-option --json` into structured schemas
//! that can be used for causal graph bootstrapping and HDC encoding.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Structured NixOS option metadata.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NixOptionSchema {
    /// Full dotted path (e.g., "services.nginx.enable").
    pub name: String,
    /// NixOS option type (e.g., "bool", "string", "listOf package").
    pub option_type: String,
    /// Human-readable display form of the default, retained for existing callers.
    pub default: Option<String>,
    /// Original default JSON value, preserving literal-expression metadata.
    pub default_value: Option<serde_json::Value>,
    /// Original example JSON value, preserving literal-expression metadata.
    pub example: Option<serde_json::Value>,
    /// Whether the option is read-only. None means the source did not say.
    pub read_only: Option<bool>,
    /// Structured option location components from the evaluated option docs.
    pub location: Option<Vec<String>>,
    /// Human-readable description.
    pub description: String,
    /// Packages related to this option, extracted heuristically for retrieval.
    pub related_packages: Vec<String>,
    /// Original relatedPackages documentation field, when supplied.
    pub related_packages_markdown: Option<String>,
    /// Module declaration file paths.
    pub declarations: Vec<String>,
}

/// Raw JSON option entry from nixosOptionsDoc/options JSON.
#[derive(Debug, Deserialize)]
struct RawOptionEntry {
    #[serde(rename = "type")]
    option_type: Option<String>,
    default: Option<serde_json::Value>,
    example: Option<serde_json::Value>,
    description: Option<String>,
    declarations: Option<Vec<String>>,
    #[serde(rename = "loc")]
    location: Option<Vec<String>>,
    #[serde(rename = "relatedPackages")]
    related_packages_markdown: Option<String>,
    #[serde(rename = "readOnly")]
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
        let default_value = entry.default;
        let default = default_value.as_ref().map(|value| match value {
            serde_json::Value::String(text) => text.clone(),
            other => other.to_string(),
        });

        // Keep structured source metadata; heuristics remain retrieval hints only.
        let related_packages = extract_related_packages(&name, &description);

        schemas.push(NixOptionSchema {
            name,
            option_type,
            default,
            default_value,
            example: entry.example,
            read_only: entry.read_only,
            location: entry.location,
            description,
            related_packages,
            related_packages_markdown: entry.related_packages_markdown,
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

/// Origin of an option catalog loaded by the controller.
///
/// Neither origin is a substitute for an option set evaluated from a remote
/// target flake. BundledReference is a discovery hint only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptionSchemaSource {
    /// Observed from nixos-option on the controller's own current machine.
    ControllerLocalLive,
    /// Compiled into NixWard; not authoritative for any particular host.
    BundledReference,
}

/// Load options and retain whether they came from the local live host or a
/// bundled reference set. This is deliberately not called TargetBound.
fn load_options_with_provenance(
) -> (Vec<NixOptionSchema>, OptionSchemaSource, Option<Instant>) {
    let observed_at = Instant::now();
    match load_nixos_options() {
        Ok(schemas) if !schemas.is_empty() => {
            tracing::info!(count = schemas.len(), "Loaded controller-local NixOS options");
            (
                schemas,
                OptionSchemaSource::ControllerLocalLive,
                Some(observed_at),
            )
        }
        Ok(_) => {
            tracing::info!("Live NixOS option command returned an empty set; using bundled references");
            (
                load_bundled_options().unwrap_or_default(),
                OptionSchemaSource::BundledReference,
                None,
            )
        }
        Err(error) => {
            tracing::info!(%error, "Live NixOS option lookup failed; using bundled references");
            (
                load_bundled_options().unwrap_or_default(),
                OptionSchemaSource::BundledReference,
                None,
            )
        }
    }
}

/// Load NixOS options with compatibility fallback.
///
/// Prefer OptionSchemaCache when source provenance matters. This compatibility
/// helper intentionally returns only the schema vector and therefore cannot
/// establish whether a result was live or bundled.
pub fn load_options_with_fallback() -> Vec<NixOptionSchema> {
    load_options_with_provenance().0
}

/// Auto-refreshing local option cache with explicit source provenance.
///
/// The live command observes the controller's current machine only. The cache
/// is never target-bound to an arbitrary remote flake or host.
pub struct OptionSchemaCache {
    schemas: Vec<NixOptionSchema>,
    source: OptionSchemaSource,
    last_attempt: Instant,
    last_live_refresh: Option<Instant>,
    refresh_interval: Duration,
}

impl OptionSchemaCache {
    /// Create a new cache, recording whether the source was live or bundled.
    pub fn new() -> Self {
        let (schemas, source, last_live_refresh) = load_options_with_provenance();
        Self {
            schemas,
            source,
            last_attempt: Instant::now(),
            last_live_refresh,
            refresh_interval: Duration::from_secs(3600),
        }
    }

    /// Set the refresh interval.
    pub fn with_refresh_interval(mut self, interval: Duration) -> Self {
        self.refresh_interval = interval;
        self
    }

    /// Get the cached schemas, retrying live discovery after the refresh interval.
    pub fn schemas(&mut self) -> &[NixOptionSchema] {
        if self.last_attempt.elapsed() >= self.refresh_interval {
            self.refresh();
        }
        &self.schemas
    }

    /// Force a live refresh. A failed refresh preserves the existing schema set
    /// and never advances the last successful live-refresh timestamp.
    pub fn refresh(&mut self) {
        self.apply_live_result(load_nixos_options(), Instant::now());
    }

    fn apply_live_result(
        &mut self,
        result: Result<Vec<NixOptionSchema>, String>,
        attempted_at: Instant,
    ) {
        self.last_attempt = attempted_at;
        if let Ok(schemas) = result
            && !schemas.is_empty()
        {
            self.schemas = schemas;
            self.source = OptionSchemaSource::ControllerLocalLive;
            self.last_live_refresh = Some(attempted_at);
            return;
        }

        // Do not replace a previously live catalog with bundled data after a
        // transient failure. If the cache has no entries at all, use references.
        if self.schemas.is_empty()
            && let Ok(bundled) = load_bundled_options()
            && !bundled.is_empty()
        {
            self.schemas = bundled;
            self.source = OptionSchemaSource::BundledReference;
        }
    }

    /// Source of the current cached schema vector.
    pub fn source(&self) -> OptionSchemaSource {
        self.source
    }

    /// True only when the current vector came from a recent controller-local
    /// lookup. This does not mean it applies to a different target host.
    pub fn has_fresh_local_live_data(&self) -> bool {
        self.source == OptionSchemaSource::ControllerLocalLive
            && self
                .last_live_refresh
                .is_some_and(|time| time.elapsed() < self.refresh_interval)
    }

    /// Age of the last successful live lookup, or None if none has succeeded.
    pub fn last_live_refresh_age(&self) -> Option<Duration> {
        self.last_live_refresh.map(|time| time.elapsed())
    }

    /// Local and bundled catalogs are not bound to a caller-selected target flake.
    pub fn is_target_bound(&self) -> bool {
        false
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
    fn test_option_documentation_metadata_is_preserved() {
        let json = r#"{
            "services.example.readOnly": {
                "type": "bool",
                "default": {"_type": "literalExpression", "text": "true"},
                "example": {"_type": "literalExpression", "text": "false"},
                "description": "A synthetic read-only option.",
                "declarations": ["nixos/modules/services/example.nix"],
                "loc": ["services", "example", "readOnly"],
                "relatedPackages": "pkgs.example",
                "readOnly": true
            }
        }"#;

        let schemas = parse_nixos_options(json).unwrap();
        let schema = &schemas[0];

        assert_eq!(schema.read_only, Some(true));
        assert_eq!(
            schema.location.as_deref(),
            Some(&["services".to_string(), "example".to_string(), "readOnly".to_string()][..])
        );
        assert_eq!(
            schema.default_value.as_ref().and_then(|value| value["_type"].as_str()),
            Some("literalExpression")
        );
        assert_eq!(
            schema.example.as_ref().and_then(|value| value["_type"].as_str()),
            Some("literalExpression")
        );
        assert_eq!(
            schema.related_packages_markdown.as_deref(),
            Some("pkgs.example")
        );
    }

    #[test]
    fn failed_live_refresh_does_not_refresh_bundled_reference_age() {
        let attempted_at = Instant::now();
        let mut cache = OptionSchemaCache {
            schemas: parse_nixos_options(sample_json()).unwrap(),
            source: OptionSchemaSource::BundledReference,
            last_attempt: attempted_at,
            last_live_refresh: None,
            refresh_interval: Duration::from_secs(60),
        };

        cache.apply_live_result(Err("synthetic live lookup failure".to_string()), attempted_at);

        assert_eq!(cache.source(), OptionSchemaSource::BundledReference);
        assert!(cache.last_live_refresh_age().is_none());
        assert!(!cache.has_fresh_local_live_data());
        assert!(!cache.is_target_bound());
    }

    #[test]
    fn failed_live_refresh_preserves_previous_live_timestamp() {
        let previous_success = Instant::now() - Duration::from_secs(120);
        let attempted_at = Instant::now();
        let mut cache = OptionSchemaCache {
            schemas: parse_nixos_options(sample_json()).unwrap(),
            source: OptionSchemaSource::ControllerLocalLive,
            last_attempt: previous_success,
            last_live_refresh: Some(previous_success),
            refresh_interval: Duration::from_secs(60),
        };

        cache.apply_live_result(Err("synthetic live lookup failure".to_string()), attempted_at);

        assert_eq!(cache.source(), OptionSchemaSource::ControllerLocalLive);
        assert_eq!(cache.last_live_refresh, Some(previous_success));
        assert!(!cache.has_fresh_local_live_data());
    }

    #[test]
    fn successful_live_refresh_replaces_fallback_and_advances_live_timestamp() {
        let attempted_at = Instant::now();
        let mut cache = OptionSchemaCache {
            schemas: parse_nixos_options(sample_json()).unwrap(),
            source: OptionSchemaSource::BundledReference,
            last_attempt: attempted_at,
            last_live_refresh: None,
            refresh_interval: Duration::from_secs(60),
        };

        cache.apply_live_result(Ok(parse_nixos_options(sample_json()).unwrap()), attempted_at);

        assert_eq!(cache.source(), OptionSchemaSource::ControllerLocalLive);
        assert!(cache.last_live_refresh_age().is_some());
        assert!(cache.has_fresh_local_live_data());
        assert!(!cache.is_target_bound());
    }

    #[test]
    fn empty_live_result_does_not_promote_fallback_to_live() {
        let attempted_at = Instant::now();
        let mut cache = OptionSchemaCache {
            schemas: parse_nixos_options(sample_json()).unwrap(),
            source: OptionSchemaSource::BundledReference,
            last_attempt: attempted_at,
            last_live_refresh: None,
            refresh_interval: Duration::from_secs(60),
        };

        cache.apply_live_result(Ok(Vec::new()), attempted_at);

        assert_eq!(cache.source(), OptionSchemaSource::BundledReference);
        assert!(cache.last_live_refresh_age().is_none());
        assert!(!cache.has_fresh_local_live_data());
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
            ..NixOptionSchema::default()
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
            ..NixOptionSchema::default()
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
            ..NixOptionSchema::default()
        };
        assert!(pkg_list.is_package_list());

        let non_list = NixOptionSchema {
            name: "services.nginx.enable".to_string(),
            option_type: "bool".to_string(),
            default: None,
            description: "".to_string(),
            related_packages: vec![],
            declarations: vec![],
            ..NixOptionSchema::default()
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
