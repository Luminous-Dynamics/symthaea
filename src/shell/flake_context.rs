// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Live Flake.nix Parsing for Contextual Completions
//!
//! Provides real-time parsing of flake.nix to enhance IntelliSense with:
//! - Installed packages (from environment.systemPackages)
//! - Enabled services (services.*.enable)
//! - User programs (users.users.*.packages)
//! - Import paths and module structure
//!
//! Updates automatically on file change or explicit reload.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

use crate::language::nix_parser::{NixOption, NixParser, NixValue};

/// Context extracted from flake.nix for shell completions
#[derive(Debug, Clone, Default)]
pub struct FlakeContext {
    /// Packages currently in environment.systemPackages
    pub installed_packages: HashSet<String>,

    /// Services that are enabled (services.*.enable = true)
    pub enabled_services: HashSet<String>,

    /// All service paths mentioned (for completion suggestions)
    pub known_services: HashSet<String>,

    /// User-specific packages
    pub user_packages: HashSet<String>,

    /// Import paths
    pub imports: Vec<String>,

    /// Module arguments available (config, pkgs, lib, etc.)
    pub module_args: Vec<String>,

    /// All option paths found (for path completion)
    pub option_paths: Vec<String>,

    /// Path to the parsed flake.nix
    pub flake_path: Option<PathBuf>,

    /// Last modification time of flake.nix
    pub last_modified: Option<SystemTime>,

    /// Last parse timestamp
    pub last_parse: Option<Instant>,

    /// Parse errors from last attempt
    pub parse_errors: Vec<String>,

    /// Whether the context is stale and needs reload
    pub is_stale: bool,
}

impl FlakeContext {
    /// Create empty context
    pub fn new() -> Self {
        Self::default()
    }

    /// Try to discover and load flake.nix from common locations
    pub fn discover_and_load() -> Self {
        let mut ctx = Self::new();

        // Try common locations for flake.nix
        let candidates = [
            "/etc/nixos/flake.nix",
            "/etc/nixos/configuration.nix",
            "flake.nix",
            "./flake.nix",
        ];

        for path in candidates {
            let path = PathBuf::from(path);
            if path.exists() {
                if let Err(e) = ctx.load_from_path(&path) {
                    ctx.parse_errors
                        .push(format!("Failed to load {}: {}", path.display(), e));
                } else {
                    break;
                }
            }
        }

        ctx
    }

    /// Load and parse flake.nix from a specific path
    pub fn load_from_path(&mut self, path: &Path) -> Result<(), crate::errors::SymthaeaError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            crate::errors::SymthaeaError::Runtime(format!(
                "Failed to read {}: {}",
                path.display(),
                e
            ))
        })?;

        let modified = std::fs::metadata(path).ok().and_then(|m| m.modified().ok());

        self.parse_nix_content(&content)?;

        self.flake_path = Some(path.to_path_buf());
        self.last_modified = modified;
        self.last_parse = Some(Instant::now());
        self.is_stale = false;

        Ok(())
    }

    /// Reload if file has changed
    pub fn reload_if_changed(&mut self) -> bool {
        let Some(ref path) = self.flake_path.clone() else {
            return false;
        };

        let current_modified = std::fs::metadata(path).ok().and_then(|m| m.modified().ok());

        if current_modified != self.last_modified {
            if let Err(e) = self.load_from_path(path) {
                self.parse_errors.push(e.to_string());
                return false;
            }
            return true;
        }

        false
    }

    /// Parse Nix content and extract context
    fn parse_nix_content(&mut self, content: &str) -> Result<(), crate::errors::SymthaeaError> {
        let mut parser = NixParser::new();

        // Set up tree-sitter language
        parser
            .parser
            .set_language(&tree_sitter_nix::LANGUAGE.into())
            .map_err(|e| {
                crate::errors::SymthaeaError::Runtime(format!("Failed to set language: {e}"))
            })?;

        let config = parser.parse(content).map_err(|e| {
            crate::errors::SymthaeaError::Runtime(format!(
                "Parse error at {}:{}: {}",
                e.line, e.column, e.message
            ))
        })?;

        // Clear previous context
        self.installed_packages.clear();
        self.enabled_services.clear();
        self.known_services.clear();
        self.user_packages.clear();
        self.option_paths.clear();
        self.parse_errors.clear();

        // Extract data from parsed config
        self.imports = config.imports.clone();
        self.module_args = config.module_args.clone();

        // Process options
        for opt in &config.options {
            self.option_paths.push(opt.path.clone());
            self.process_option(opt);
        }

        // Record parse errors
        for err in &config.errors {
            self.parse_errors
                .push(format!("{}:{}: {}", err.line, err.column, err.message));
        }

        Ok(())
    }

    /// Process a single option and categorize it
    fn process_option(&mut self, opt: &NixOption) {
        // Check for environment.systemPackages
        if opt.path == "environment.systemPackages" {
            Self::extract_packages_from_value(&opt.value, &mut self.installed_packages);
        }

        // Check for user packages
        if opt.path.starts_with("users.users.") && opt.path.ends_with(".packages") {
            Self::extract_packages_from_value(&opt.value, &mut self.user_packages);
        }

        // Check for services
        if opt.path.starts_with("services.") {
            // Extract service name
            let parts: Vec<&str> = opt.path.split('.').collect();
            if parts.len() >= 2 {
                let service_name = parts[1].to_string();
                self.known_services.insert(service_name.clone());

                // Check if enabled
                if opt.path.ends_with(".enable") {
                    if let NixValue::Bool(true) = opt.value {
                        self.enabled_services.insert(service_name);
                    }
                }
            }
        }

        // Check for programs
        if opt.path.starts_with("programs.") && opt.path.ends_with(".enable") {
            let parts: Vec<&str> = opt.path.split('.').collect();
            if parts.len() >= 2 {
                if let NixValue::Bool(true) = opt.value {
                    self.enabled_services.insert(parts[1].to_string());
                }
            }
        }
    }

    /// Extract package names from a value (static method to avoid borrow issues)
    fn extract_packages_from_value(value: &NixValue, packages: &mut HashSet<String>) {
        match value {
            NixValue::List(items) => {
                for item in items {
                    Self::extract_packages_from_value(item, packages);
                }
            }
            NixValue::Apply(name) => {
                // e.g., pkgs.firefox -> firefox
                if let Some(pkg) = name.split('.').next_back() {
                    packages.insert(pkg.to_string());
                }
            }
            NixValue::With { body, .. } => {
                Self::extract_packages_from_value(body, packages);
            }
            NixValue::Expression(expr) => {
                // Try to extract simple identifiers
                for word in expr.split_whitespace() {
                    if word
                        .chars()
                        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
                    {
                        packages.insert(word.to_string());
                    }
                }
            }
            _ => {}
        }
    }

    /// Get completions relevant to context
    pub fn get_contextual_suggestions(&self, input: &str) -> Vec<ContextualSuggestion> {
        let mut suggestions = Vec::new();
        let input_lower = input.to_lowercase();

        // Suggest removing installed packages
        if input_lower.starts_with("remove ") || input_lower.starts_with("uninstall ") {
            let prefix = if input_lower.starts_with("remove ") {
                "remove "
            } else {
                "uninstall "
            };
            let query = input.trim_start_matches(prefix);

            for pkg in &self.installed_packages {
                if pkg.to_lowercase().contains(&query.to_lowercase()) {
                    suggestions.push(ContextualSuggestion {
                        text: format!("{prefix}{pkg}"),
                        description: format!("Remove {pkg} (currently installed)"),
                        source: SuggestionSource::InstalledPackage,
                        confidence: 0.9,
                    });
                }
            }
        }

        // Suggest disabling enabled services
        if input_lower.contains("disable") || input_lower.contains("stop") {
            for service in &self.enabled_services {
                if service.to_lowercase().contains(&input_lower) {
                    suggestions.push(ContextualSuggestion {
                        text: format!("disable {service}"),
                        description: format!("Disable {service} (currently enabled)"),
                        source: SuggestionSource::EnabledService,
                        confidence: 0.85,
                    });
                }
            }
        }

        // Suggest enabling known services
        if input_lower.contains("enable") || input_lower.starts_with("service") {
            for service in &self.known_services {
                if !self.enabled_services.contains(service)
                    && service.to_lowercase().contains(&input_lower)
                {
                    suggestions.push(ContextualSuggestion {
                        text: format!("enable {service}"),
                        description: format!("Enable {service} service"),
                        source: SuggestionSource::KnownService,
                        confidence: 0.8,
                    });
                }
            }
        }

        // Option path suggestions
        if input_lower.contains("set ") || input_lower.contains("config") {
            for path in &self.option_paths {
                if path.to_lowercase().contains(&input_lower) {
                    suggestions.push(ContextualSuggestion {
                        text: format!("set {path}"),
                        description: format!("Configure {path}"),
                        source: SuggestionSource::OptionPath,
                        confidence: 0.7,
                    });
                }
            }
        }

        suggestions
    }

    /// Check if a package is installed
    pub fn is_package_installed(&self, package: &str) -> bool {
        self.installed_packages.contains(package)
    }

    /// Check if a service is enabled
    pub fn is_service_enabled(&self, service: &str) -> bool {
        self.enabled_services.contains(service)
    }

    /// Get status summary
    pub fn status_summary(&self) -> String {
        let pkg_count = self.installed_packages.len();
        let svc_count = self.enabled_services.len();
        let path = self
            .flake_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".to_string());

        format!("Flake: {path} | {pkg_count} packages | {svc_count} services enabled")
    }
}

/// A contextual suggestion based on flake.nix analysis
#[derive(Debug, Clone)]
pub struct ContextualSuggestion {
    /// The suggestion text
    pub text: String,

    /// Description/explanation
    pub description: String,

    /// Source of the suggestion
    pub source: SuggestionSource,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
}

/// Source of a contextual suggestion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionSource {
    /// From installed packages list
    InstalledPackage,

    /// From enabled services
    EnabledService,

    /// From known but not enabled services
    KnownService,

    /// From option paths
    OptionPath,

    /// From imports
    Import,

    /// From user configuration
    UserConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_context() {
        let ctx = FlakeContext::new();
        assert!(ctx.installed_packages.is_empty());
        assert!(ctx.enabled_services.is_empty());
    }

    #[test]
    fn test_is_package_installed() {
        let mut ctx = FlakeContext::new();
        ctx.installed_packages.insert("firefox".to_string());
        ctx.installed_packages.insert("git".to_string());

        assert!(ctx.is_package_installed("firefox"));
        assert!(!ctx.is_package_installed("chrome"));
    }

    #[test]
    fn test_contextual_suggestions() {
        let mut ctx = FlakeContext::new();
        ctx.installed_packages.insert("firefox".to_string());
        ctx.installed_packages.insert("git".to_string());

        let suggestions = ctx.get_contextual_suggestions("remove fire");
        assert!(!suggestions.is_empty());
        assert!(suggestions[0].text.contains("firefox"));
    }
}
