// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! H3: Conflict Detector
//!
//! Pre-rebuild conflict warnings for NixOS configurations.
//! Detects option collisions, dependency conflicts, and potential issues.

use std::collections::{HashMap, HashSet};

/// Conflict severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConflictSeverity {
    /// Informational - might want to review
    Info,
    /// Warning - potential issue
    Warning,
    /// Error - will likely cause build failure
    Error,
    /// Critical - definitely will fail
    Critical,
}

impl ConflictSeverity {
    /// Get icon for severity
    pub fn icon(&self) -> &'static str {
        match self {
            ConflictSeverity::Info => "ℹ️",
            ConflictSeverity::Warning => "⚠️",
            ConflictSeverity::Error => "❌",
            ConflictSeverity::Critical => "🚨",
        }
    }

    /// Get ANSI color code
    pub fn color(&self) -> &'static str {
        match self {
            ConflictSeverity::Info => "\x1b[34m",     // Blue
            ConflictSeverity::Warning => "\x1b[33m",  // Yellow
            ConflictSeverity::Error => "\x1b[31m",    // Red
            ConflictSeverity::Critical => "\x1b[91m", // Bright red
        }
    }
}

/// Type of conflict
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictType {
    /// Same option set to different values
    OptionCollision { path: String, values: Vec<String> },
    /// Mutually exclusive services
    ServiceExclusion { services: Vec<String> },
    /// Package version conflict
    VersionConflict {
        package: String,
        versions: Vec<String>,
    },
    /// Missing dependency
    MissingDependency { package: String, dependency: String },
    /// Port collision
    PortConflict { port: u16, services: Vec<String> },
    /// File path collision
    PathConflict { path: String, sources: Vec<String> },
    /// Circular dependency
    CircularDependency { chain: Vec<String> },
    /// Deprecated option usage
    DeprecatedOption {
        old_path: String,
        new_path: Option<String>,
    },
    /// Type mismatch
    TypeMismatch {
        path: String,
        expected: String,
        actual: String,
    },
}

/// A detected conflict
#[derive(Debug, Clone)]
pub struct Conflict {
    /// Conflict type
    pub conflict_type: ConflictType,
    /// Severity
    pub severity: ConflictSeverity,
    /// Description
    pub description: String,
    /// Affected files/modules
    pub sources: Vec<String>,
    /// Suggested resolution
    pub suggestion: Option<String>,
}

impl Conflict {
    /// Create a new conflict
    pub fn new(conflict_type: ConflictType, severity: ConflictSeverity) -> Self {
        let description = Self::describe(&conflict_type);
        let suggestion = Self::suggest(&conflict_type);

        Self {
            conflict_type,
            severity,
            description,
            sources: Vec::new(),
            suggestion,
        }
    }

    /// Add source file/module
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.sources.push(source.into());
        self
    }

    fn describe(conflict_type: &ConflictType) -> String {
        match conflict_type {
            ConflictType::OptionCollision { path, values } => {
                format!(
                    "Option '{}' is set to conflicting values: {}",
                    path,
                    values.join(", ")
                )
            }
            ConflictType::ServiceExclusion { services } => {
                format!("Services {} are mutually exclusive", services.join(" and "))
            }
            ConflictType::VersionConflict { package, versions } => {
                format!(
                    "Package '{}' requested with conflicting versions: {}",
                    package,
                    versions.join(", ")
                )
            }
            ConflictType::MissingDependency {
                package,
                dependency,
            } => {
                format!("Package '{package}' requires '{dependency}' which is not installed")
            }
            ConflictType::PortConflict { port, services } => {
                format!(
                    "Port {} is used by multiple services: {}",
                    port,
                    services.join(", ")
                )
            }
            ConflictType::PathConflict { path, sources } => {
                format!(
                    "Path '{}' is managed by multiple sources: {}",
                    path,
                    sources.join(", ")
                )
            }
            ConflictType::CircularDependency { chain } => {
                format!("Circular dependency detected: {}", chain.join(" -> "))
            }
            ConflictType::DeprecatedOption { old_path, new_path } => {
                if let Some(new) = new_path {
                    format!("Option '{old_path}' is deprecated, use '{new}' instead")
                } else {
                    format!("Option '{old_path}' is deprecated and should be removed")
                }
            }
            ConflictType::TypeMismatch {
                path,
                expected,
                actual,
            } => {
                format!("Option '{path}' expects {expected} but got {actual}")
            }
        }
    }

    fn suggest(conflict_type: &ConflictType) -> Option<String> {
        match conflict_type {
            ConflictType::OptionCollision { .. } => Some(
                "Use mkForce to override or mkDefault to set a lower-priority value".to_string(),
            ),
            ConflictType::ServiceExclusion { .. } => {
                Some("Disable one of the conflicting services".to_string())
            }
            ConflictType::VersionConflict { .. } => {
                Some("Pin to a specific version or use an overlay".to_string())
            }
            ConflictType::MissingDependency { dependency, .. } => {
                Some(format!("Add '{dependency}' to environment.systemPackages"))
            }
            ConflictType::PortConflict { .. } => {
                Some("Change the port for one of the services".to_string())
            }
            ConflictType::PathConflict { .. } => {
                Some("Ensure only one module manages this path".to_string())
            }
            ConflictType::CircularDependency { .. } => {
                Some("Break the cycle by restructuring module dependencies".to_string())
            }
            ConflictType::DeprecatedOption { new_path, .. } => {
                new_path.as_ref().map(|p| format!("Replace with '{p}'"))
            }
            ConflictType::TypeMismatch { expected, .. } => {
                Some(format!("Convert the value to {expected}"))
            }
        }
    }
}

/// Conflict detector
pub struct ConflictDetector {
    /// Known mutually exclusive services
    exclusions: HashMap<String, HashSet<String>>,
    /// Deprecated options
    deprecated: HashMap<String, Option<String>>,
    /// Detected conflicts
    conflicts: Vec<Conflict>,
}

impl ConflictDetector {
    /// Create new conflict detector
    pub fn new() -> Self {
        let mut detector = Self {
            exclusions: HashMap::new(),
            deprecated: HashMap::new(),
            conflicts: Vec::new(),
        };
        detector.init_known_conflicts();
        detector
    }

    fn init_known_conflicts(&mut self) {
        // Known mutually exclusive services
        self.add_exclusion("services.apache2", "services.nginx");
        self.add_exclusion("services.lightdm", "services.gdm");
        self.add_exclusion("services.lightdm", "services.sddm");
        self.add_exclusion("services.gdm", "services.sddm");
        self.add_exclusion("services.nftables", "services.iptables");
        self.add_exclusion("services.networkmanager", "services.connman");
        self.add_exclusion("services.pulseaudio", "services.pipewire.pulse");

        // Deprecated options
        self.deprecated.insert(
            "services.xserver.displayManager.auto".to_string(),
            Some("services.displayManager.autoLogin".to_string()),
        );
        self.deprecated.insert(
            "services.xserver.layout".to_string(),
            Some("services.xserver.xkb.layout".to_string()),
        );
        self.deprecated.insert(
            "networking.useDHCP".to_string(),
            Some("networking.interfaces.<name>.useDHCP".to_string()),
        );
    }

    fn add_exclusion(&mut self, a: &str, b: &str) {
        self.exclusions
            .entry(a.to_string())
            .or_default()
            .insert(b.to_string());
        self.exclusions
            .entry(b.to_string())
            .or_default()
            .insert(a.to_string());
    }

    /// Check for option collision
    pub fn check_option_collision(&mut self, path: &str, values: &[&str]) {
        if values.len() > 1 {
            let unique: HashSet<_> = values.iter().collect();
            if unique.len() > 1 {
                self.conflicts.push(Conflict::new(
                    ConflictType::OptionCollision {
                        path: path.to_string(),
                        values: values.iter().map(|s| s.to_string()).collect(),
                    },
                    ConflictSeverity::Error,
                ));
            }
        }
    }

    /// Check for service exclusions
    pub fn check_service_exclusions(&mut self, enabled_services: &[&str]) {
        for service in enabled_services {
            if let Some(excluded) = self.exclusions.get(*service) {
                for other in enabled_services {
                    if excluded.contains(*other) {
                        self.conflicts.push(Conflict::new(
                            ConflictType::ServiceExclusion {
                                services: vec![service.to_string(), other.to_string()],
                            },
                            ConflictSeverity::Error,
                        ));
                        break;
                    }
                }
            }
        }
    }

    /// Check for port conflicts
    pub fn check_port_conflicts(&mut self, port_mappings: &HashMap<u16, Vec<String>>) {
        for (port, services) in port_mappings {
            if services.len() > 1 {
                self.conflicts.push(Conflict::new(
                    ConflictType::PortConflict {
                        port: *port,
                        services: services.clone(),
                    },
                    ConflictSeverity::Error,
                ));
            }
        }
    }

    /// Check for deprecated options
    pub fn check_deprecated(&mut self, options: &[&str]) {
        for option in options {
            if let Some(replacement) = self.deprecated.get(*option) {
                self.conflicts.push(Conflict::new(
                    ConflictType::DeprecatedOption {
                        old_path: option.to_string(),
                        new_path: replacement.clone(),
                    },
                    ConflictSeverity::Warning,
                ));
            }
        }
    }

    /// Check for circular dependencies
    pub fn check_circular_dependencies(&mut self, dependencies: &HashMap<String, Vec<String>>) {
        fn find_cycle(
            node: &str,
            deps: &HashMap<String, Vec<String>>,
            visiting: &mut HashSet<String>,
            path: &mut Vec<String>,
        ) -> Option<Vec<String>> {
            if visiting.contains(node) {
                // Found cycle
                let cycle_start = path.iter().position(|n| n == node).unwrap_or(0);
                let mut cycle: Vec<String> = path[cycle_start..].to_vec();
                cycle.push(node.to_string());
                return Some(cycle);
            }

            if let Some(node_deps) = deps.get(node) {
                visiting.insert(node.to_string());
                path.push(node.to_string());

                for dep in node_deps {
                    if let Some(cycle) = find_cycle(dep, deps, visiting, path) {
                        return Some(cycle);
                    }
                }

                path.pop();
                visiting.remove(node);
            }

            None
        }

        let mut visited_starts = HashSet::new();
        for start in dependencies.keys() {
            if visited_starts.contains(start) {
                continue;
            }

            let mut visiting = HashSet::new();
            let mut path = Vec::new();

            if let Some(cycle) = find_cycle(start, dependencies, &mut visiting, &mut path) {
                // Add all nodes in cycle to visited to avoid duplicate reports
                for node in &cycle {
                    visited_starts.insert(node.clone());
                }

                self.conflicts.push(Conflict::new(
                    ConflictType::CircularDependency { chain: cycle },
                    ConflictSeverity::Critical,
                ));
            }
        }
    }

    /// Get all detected conflicts
    pub fn conflicts(&self) -> &[Conflict] {
        &self.conflicts
    }

    /// Get conflicts by severity
    pub fn by_severity(&self, severity: ConflictSeverity) -> Vec<&Conflict> {
        self.conflicts
            .iter()
            .filter(|c| c.severity == severity)
            .collect()
    }

    /// Clear detected conflicts
    pub fn clear(&mut self) {
        self.conflicts.clear();
    }

    /// Has any conflicts
    pub fn has_conflicts(&self) -> bool {
        !self.conflicts.is_empty()
    }

    /// Has critical or error conflicts
    pub fn has_blocking_conflicts(&self) -> bool {
        self.conflicts.iter().any(|c| {
            c.severity == ConflictSeverity::Critical || c.severity == ConflictSeverity::Error
        })
    }

    /// Format conflicts for display
    pub fn format(&self) -> String {
        if self.conflicts.is_empty() {
            return "No conflicts detected.".to_string();
        }

        let mut output = String::new();
        let mut sorted: Vec<_> = self.conflicts.iter().collect();
        sorted.sort_by(|a, b| b.severity.cmp(&a.severity));

        for conflict in sorted {
            output.push_str(&format!(
                "{} {}\n",
                conflict.severity.icon(),
                conflict.description
            ));

            if !conflict.sources.is_empty() {
                output.push_str(&format!("   Sources: {}\n", conflict.sources.join(", ")));
            }

            if let Some(suggestion) = &conflict.suggestion {
                output.push_str(&format!("   Suggestion: {suggestion}\n"));
            }

            output.push('\n');
        }

        output
    }
}

impl Default for ConflictDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option_collision() {
        let mut detector = ConflictDetector::new();

        detector.check_option_collision("services.nginx.enable", &["true", "false"]);

        assert!(detector.has_conflicts());
        assert_eq!(detector.conflicts().len(), 1);
    }

    #[test]
    fn test_service_exclusion() {
        let mut detector = ConflictDetector::new();

        detector.check_service_exclusions(&["services.gdm", "services.sddm"]);

        assert!(detector.has_conflicts());
    }

    #[test]
    fn test_port_conflict() {
        let mut detector = ConflictDetector::new();

        let mut ports = HashMap::new();
        ports.insert(80, vec!["nginx".to_string(), "apache".to_string()]);

        detector.check_port_conflicts(&ports);

        assert!(detector.has_conflicts());
    }

    #[test]
    fn test_circular_dependency() {
        let mut detector = ConflictDetector::new();

        let mut deps = HashMap::new();
        deps.insert("a".to_string(), vec!["b".to_string()]);
        deps.insert("b".to_string(), vec!["c".to_string()]);
        deps.insert("c".to_string(), vec!["a".to_string()]);

        detector.check_circular_dependencies(&deps);

        assert!(detector.has_blocking_conflicts());
    }

    #[test]
    fn test_deprecated_option() {
        let mut detector = ConflictDetector::new();

        detector.check_deprecated(&["services.xserver.layout"]);

        assert!(detector.has_conflicts());
        assert!(!detector.has_blocking_conflicts()); // Warnings aren't blocking
    }
}