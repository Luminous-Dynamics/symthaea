// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Constraint Validator - Knowledge Graph Validation
//!
//! Validates NixOS configuration changes against:
//! - Type constraints (boolean, string, int, etc.)
//! - Dependency constraints (service X requires service Y)
//! - Conflict constraints (option A conflicts with option B)
//! - Value constraints (port ranges, path validity, etc.)

use super::widget_mapper::NixPath;
use std::collections::{HashMap, HashSet};

/// Constraint types
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintKind {
    /// Type constraint (expected type)
    Type(NixType),

    /// Dependency constraint (requires another option)
    Requires(NixPath),

    /// Conflict constraint (conflicts with another option)
    ConflictsWith(NixPath),

    /// Value range constraint
    Range { min: Option<i64>, max: Option<i64> },

    /// Regex pattern constraint
    Pattern(String),

    /// Must be one of these values
    OneOf(Vec<String>),

    /// Path must exist or be valid
    ValidPath,

    /// Port number constraint
    ValidPort,

    /// Custom validation function name
    Custom(String),
}

/// NixOS value types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NixType {
    Bool,
    Int,
    Float,
    String,
    Path,
    List,
    AttrSet,
    Null,
    Any,
}

impl NixType {
    /// Infer type from Nix expression
    pub fn from_expr(expr: &str) -> Self {
        let trimmed = expr.trim();

        if trimmed == "true" || trimmed == "false" {
            Self::Bool
        } else if trimmed == "null" {
            Self::Null
        } else if trimmed.parse::<i64>().is_ok() {
            Self::Int
        } else if trimmed.parse::<f64>().is_ok() {
            Self::Float
        } else if trimmed.starts_with('[') && trimmed.ends_with(']') {
            Self::List
        } else if trimmed.starts_with('{') && trimmed.ends_with('}') {
            Self::AttrSet
        } else if trimmed.starts_with('/') || trimmed.starts_with("./") {
            Self::Path
        } else {
            Self::String
        }
    }

    /// Check if a type is compatible with another
    pub fn is_compatible(&self, other: &Self) -> bool {
        if *self == Self::Any || *other == Self::Any {
            return true;
        }
        self == other
    }
}

/// A constraint definition
#[derive(Debug, Clone)]
pub struct Constraint {
    /// The constraint kind
    pub kind: ConstraintKind,

    /// Human-readable description
    pub description: String,

    /// Error message if violated
    pub error_message: String,

    /// Whether this constraint is fatal
    pub is_fatal: bool,
}

impl Constraint {
    /// Create a type constraint
    pub fn type_of(expected: NixType) -> Self {
        Self {
            kind: ConstraintKind::Type(expected),
            description: format!("Must be of type {expected:?}"),
            error_message: format!("Expected type {expected:?}"),
            is_fatal: true,
        }
    }

    /// Create a requires constraint
    pub fn requires(path: impl Into<String>) -> Self {
        let path = path.into();
        Self {
            kind: ConstraintKind::Requires(NixPath::new(&path)),
            description: format!("Requires {path} to be enabled"),
            error_message: format!("Dependency {path} is not enabled"),
            is_fatal: true,
        }
    }

    /// Create a conflicts constraint
    pub fn conflicts_with(path: impl Into<String>) -> Self {
        let path = path.into();
        Self {
            kind: ConstraintKind::ConflictsWith(NixPath::new(&path)),
            description: format!("Conflicts with {path}"),
            error_message: format!("Cannot be used with {path}"),
            is_fatal: true,
        }
    }

    /// Create a range constraint
    pub fn range(min: Option<i64>, max: Option<i64>) -> Self {
        let desc = match (min, max) {
            (Some(min), Some(max)) => format!("Must be between {min} and {max}"),
            (Some(min), None) => format!("Must be at least {min}"),
            (None, Some(max)) => format!("Must be at most {max}"),
            (None, None) => "Unconstrained".to_string(),
        };
        Self {
            kind: ConstraintKind::Range { min, max },
            description: desc.clone(),
            error_message: desc,
            is_fatal: true,
        }
    }

    /// Create a valid port constraint
    pub fn valid_port() -> Self {
        Self {
            kind: ConstraintKind::ValidPort,
            description: "Must be a valid port number (1-65535)".to_string(),
            error_message: "Invalid port number".to_string(),
            is_fatal: true,
        }
    }

    /// Create a valid path constraint
    pub fn valid_path() -> Self {
        Self {
            kind: ConstraintKind::ValidPath,
            description: "Must be a valid file path".to_string(),
            error_message: "Invalid file path".to_string(),
            is_fatal: false,
        }
    }
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Path that caused the error
    pub path: NixPath,

    /// Constraint that was violated
    pub constraint: Constraint,

    /// The offending value
    pub value: String,

    /// Severity (fatal vs warning)
    pub is_fatal: bool,
}

/// Suggestion for fixing validation error
#[derive(Debug, Clone)]
pub struct Suggestion {
    /// Human-readable suggestion
    pub message: String,

    /// Suggested Nix expression
    pub suggested_value: Option<String>,

    /// Related path to check
    pub related_path: Option<NixPath>,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,

    /// Validation errors (if any)
    pub errors: Vec<ValidationError>,

    /// Suggestions for fixing errors
    pub suggestions: Vec<Suggestion>,

    /// Confidence in the validation (0.0-1.0)
    pub confidence: f64,
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            suggestions: Vec::new(),
            confidence: 1.0,
        }
    }
}

/// Constraint validator using knowledge graph
pub struct ConstraintValidator {
    /// Constraints per NixOS path
    constraints: HashMap<String, Vec<Constraint>>,

    /// Currently enabled options (for dependency checking)
    enabled_options: HashSet<String>,

    /// Known NixOS option types
    known_types: HashMap<String, NixType>,
}

impl ConstraintValidator {
    /// Create new constraint validator
    pub fn new() -> Self {
        let mut validator = Self {
            constraints: HashMap::new(),
            enabled_options: HashSet::new(),
            known_types: HashMap::new(),
        };

        // Initialize with common NixOS constraints
        validator.init_common_constraints();

        validator
    }

    /// Initialize common NixOS constraints
    fn init_common_constraints(&mut self) {
        // Service enable patterns are boolean
        self.add_constraint_pattern(".enable", Constraint::type_of(NixType::Bool));

        // Port patterns must be valid ports
        self.add_constraint_pattern(".port", Constraint::valid_port());
        self.add_constraint_pattern(".ports", Constraint::type_of(NixType::List));

        // Path patterns should be valid paths
        self.add_constraint_pattern("Path", Constraint::valid_path());
        self.add_constraint_pattern(".path", Constraint::valid_path());

        // Common service dependencies
        self.add_constraint(
            "services.nginx.virtualHosts",
            Constraint::requires("services.nginx.enable"),
        );
        self.add_constraint(
            "services.postgresql.ensureDatabases",
            Constraint::requires("services.postgresql.enable"),
        );

        // Common type hints
        self.known_types
            .insert("services.*.enable".to_string(), NixType::Bool);
        self.known_types
            .insert("networking.firewall.enable".to_string(), NixType::Bool);
        self.known_types
            .insert("boot.loader.grub.enable".to_string(), NixType::Bool);
        self.known_types.insert("*.port".to_string(), NixType::Int);
        self.known_types
            .insert("*.ports".to_string(), NixType::List);
    }

    /// Add constraint for a specific path
    pub fn add_constraint(&mut self, path: impl Into<String>, constraint: Constraint) {
        let path = path.into();
        self.constraints.entry(path).or_default().push(constraint);
    }

    /// Add constraint for paths matching a pattern
    fn add_constraint_pattern(&mut self, pattern: &str, constraint: Constraint) {
        // Store pattern-based constraints with wildcard prefix
        let key = format!("*{pattern}");
        self.constraints.entry(key).or_default().push(constraint);
    }

    /// Mark an option as enabled
    pub fn set_enabled(&mut self, path: impl Into<String>, enabled: bool) {
        let path = path.into();
        if enabled {
            self.enabled_options.insert(path);
        } else {
            self.enabled_options.remove(&path);
        }
    }

    /// Validate a change to a NixOS option
    pub fn validate_change(&self, path: &NixPath, value: &str) -> ValidationResult {
        let mut result = ValidationResult::default();
        let mut confidence_sum = 0.0;
        let mut constraint_count = 0;

        // Get constraints for this exact path
        let exact_constraints = self.constraints.get(&path.0).cloned().unwrap_or_default();

        // Get pattern-matched constraints
        let pattern_constraints = self.get_pattern_constraints(&path.0);

        let all_constraints: Vec<Constraint> = exact_constraints
            .into_iter()
            .chain(pattern_constraints)
            .collect();

        for constraint in all_constraints {
            let (valid, suggestion) = self.check_constraint(&constraint, path, value);

            if !valid {
                result.errors.push(ValidationError {
                    path: path.clone(),
                    constraint: constraint.clone(),
                    value: value.to_string(),
                    is_fatal: constraint.is_fatal,
                });

                if let Some(sug) = suggestion {
                    result.suggestions.push(sug);
                }

                if constraint.is_fatal {
                    result.is_valid = false;
                }

                confidence_sum += 0.0;
            } else {
                confidence_sum += 1.0;
            }

            constraint_count += 1;
        }

        // Infer type and validate
        if let Some(expected_type) = self.infer_type(&path.0) {
            let actual_type = NixType::from_expr(value);
            if !expected_type.is_compatible(&actual_type) {
                result.errors.push(ValidationError {
                    path: path.clone(),
                    constraint: Constraint::type_of(expected_type),
                    value: value.to_string(),
                    is_fatal: true,
                });
                result.is_valid = false;
            } else {
                confidence_sum += 1.0;
            }
            constraint_count += 1;
        }

        // Calculate confidence
        result.confidence = if constraint_count > 0 {
            confidence_sum / constraint_count as f64
        } else {
            0.8 // Default confidence when no constraints
        };

        result
    }

    /// Get constraints matching patterns
    fn get_pattern_constraints(&self, path: &str) -> Vec<Constraint> {
        let mut matched = Vec::new();

        for (pattern, constraints) in &self.constraints {
            if let Some(suffix) = pattern.strip_prefix('*') {
                if path.ends_with(suffix) || path.contains(suffix) {
                    matched.extend(constraints.clone());
                }
            }
        }

        matched
    }

    /// Check a single constraint
    fn check_constraint(
        &self,
        constraint: &Constraint,
        _path: &NixPath,
        value: &str,
    ) -> (bool, Option<Suggestion>) {
        match &constraint.kind {
            ConstraintKind::Type(expected) => {
                let actual = NixType::from_expr(value);
                (expected.is_compatible(&actual), None)
            }

            ConstraintKind::Requires(dep_path) => {
                let enabled = self.enabled_options.contains(&dep_path.0);
                let suggestion = if !enabled {
                    Some(Suggestion {
                        message: format!("Enable {} first", dep_path.0),
                        suggested_value: Some("true".to_string()),
                        related_path: Some(dep_path.clone()),
                    })
                } else {
                    None
                };
                (enabled, suggestion)
            }

            ConstraintKind::ConflictsWith(conflict_path) => {
                let conflicting = self.enabled_options.contains(&conflict_path.0);
                let suggestion = if conflicting {
                    Some(Suggestion {
                        message: format!("Disable {} to use this option", conflict_path.0),
                        suggested_value: Some("false".to_string()),
                        related_path: Some(conflict_path.clone()),
                    })
                } else {
                    None
                };
                (!conflicting, suggestion)
            }

            ConstraintKind::Range { min, max } => {
                if let Ok(num) = value.trim().parse::<i64>() {
                    let in_range = min.is_none_or(|m| num >= m) && max.is_none_or(|m| num <= m);
                    (in_range, None)
                } else {
                    (
                        false,
                        Some(Suggestion {
                            message: "Expected a number".to_string(),
                            suggested_value: min.map(|m| m.to_string()),
                            related_path: None,
                        }),
                    )
                }
            }

            ConstraintKind::Pattern(regex) => {
                match regex::Regex::new(regex) {
                    Ok(re) => (re.is_match(value), None),
                    Err(_) => (true, None), // Invalid regex = skip check
                }
            }

            ConstraintKind::OneOf(options) => {
                let trimmed = value.trim().trim_matches('"');
                let valid = options.iter().any(|opt| opt == trimmed);
                let suggestion = if !valid {
                    Some(Suggestion {
                        message: format!("Must be one of: {options:?}"),
                        suggested_value: options.first().cloned(),
                        related_path: None,
                    })
                } else {
                    None
                };
                (valid, suggestion)
            }

            ConstraintKind::ValidPath => {
                let path_str = value.trim().trim_matches('"');
                // Basic path validation
                let valid = !path_str.contains("..")
                    && (path_str.starts_with('/') || path_str.starts_with("./"));
                (valid, None)
            }

            ConstraintKind::ValidPort => {
                if let Ok(port) = value.trim().parse::<u16>() {
                    (port >= 1, None)
                } else {
                    (
                        false,
                        Some(Suggestion {
                            message: "Port must be 1-65535".to_string(),
                            suggested_value: Some("80".to_string()),
                            related_path: None,
                        }),
                    )
                }
            }

            ConstraintKind::Custom(_) => {
                // Custom validators would be looked up from a registry
                (true, None)
            }
        }
    }

    /// Infer expected type from path
    fn infer_type(&self, path: &str) -> Option<NixType> {
        // Check exact matches first
        if let Some(t) = self.known_types.get(path) {
            return Some(*t);
        }

        // Check wildcard patterns
        for (pattern, t) in &self.known_types {
            if pattern.contains('*') {
                let parts: Vec<&str> = pattern.split('*').collect();
                if parts.len() == 2 {
                    let (prefix, suffix) = (parts[0], parts[1]);
                    if (prefix.is_empty() || path.starts_with(prefix))
                        && (suffix.is_empty() || path.ends_with(suffix))
                    {
                        return Some(*t);
                    }
                }
            }
        }

        // Infer from path patterns
        if path.ends_with(".enable") {
            Some(NixType::Bool)
        } else if path.ends_with(".port") {
            Some(NixType::Int)
        } else if path.ends_with(".ports") || path.ends_with("Packages") {
            Some(NixType::List)
        } else if path.contains("Path") || path.ends_with(".path") {
            Some(NixType::Path)
        } else {
            None
        }
    }
}

impl Default for ConstraintValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nix_type_from_expr() {
        assert_eq!(NixType::from_expr("true"), NixType::Bool);
        assert_eq!(NixType::from_expr("false"), NixType::Bool);
        assert_eq!(NixType::from_expr("42"), NixType::Int);
        assert_eq!(NixType::from_expr("3.14"), NixType::Float);
        assert_eq!(NixType::from_expr("[ 1 2 3 ]"), NixType::List);
        assert_eq!(NixType::from_expr("{ foo = bar; }"), NixType::AttrSet);
        assert_eq!(NixType::from_expr("/etc/nixos"), NixType::Path);
    }

    #[test]
    fn test_constraint_validator_creation() {
        let validator = ConstraintValidator::new();
        assert!(!validator.constraints.is_empty());
    }

    #[test]
    fn test_validate_boolean() {
        let validator = ConstraintValidator::new();
        let path = NixPath::new("services.nginx.enable");

        let result = validator.validate_change(&path, "true");
        assert!(result.is_valid);

        let result = validator.validate_change(&path, "\"not a bool\"");
        assert!(!result.is_valid);
    }

    #[test]
    fn test_validate_port() {
        let validator = ConstraintValidator::new();
        let path = NixPath::new("services.nginx.port");

        let result = validator.validate_change(&path, "8080");
        assert!(result.is_valid);

        let result = validator.validate_change(&path, "99999");
        assert!(!result.is_valid);
    }

    #[test]
    fn test_dependency_constraint() {
        let mut validator = ConstraintValidator::new();

        // Without nginx enabled
        let path = NixPath::new("services.nginx.virtualHosts");
        let result = validator.validate_change(&path, "{ }");
        assert!(!result.is_valid);
        assert!(!result.suggestions.is_empty());

        // With nginx enabled
        validator.set_enabled("services.nginx.enable", true);
        let result = validator.validate_change(&path, "{ }");
        assert!(result.is_valid);
    }

    #[test]
    fn test_type_inference() {
        let validator = ConstraintValidator::new();

        assert_eq!(
            validator.infer_type("services.nginx.enable"),
            Some(NixType::Bool)
        );
        assert_eq!(
            validator.infer_type("services.nginx.port"),
            Some(NixType::Int)
        );
        assert_eq!(
            validator.infer_type("environment.systemPackages"),
            Some(NixType::List)
        );
    }
}
