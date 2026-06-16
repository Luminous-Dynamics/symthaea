// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Widget Mapper - Bidirectional Widget↔Nix Translation
//!
//! Maps between GUI widget values and NixOS configuration expressions.
//! Supports various widget types and value transformers.

use std::collections::HashMap;
use symthaea_core::hdc::binary_hv::BinaryHV;

/// Unique widget identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WidgetId(pub String);

impl WidgetId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

/// NixOS option path (e.g., "services.nginx.enable")
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NixPath(pub String);

impl NixPath {
    pub fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }

    /// Get the last component of the path
    pub fn leaf(&self) -> &str {
        self.0.rsplit('.').next().unwrap_or(&self.0)
    }

    /// Get parent path
    pub fn parent(&self) -> Option<NixPath> {
        let parts: Vec<&str> = self.0.rsplitn(2, '.').collect();
        if parts.len() > 1 {
            Some(NixPath(parts[1].to_string()))
        } else {
            None
        }
    }
}

/// Widget value types
#[derive(Debug, Clone, PartialEq)]
pub enum WidgetValue {
    /// Boolean toggle
    Bool(bool),

    /// Text input
    String(String),

    /// Integer value
    Int(i64),

    /// Float value
    Float(f64),

    /// Selection from options (index)
    Selection(usize),

    /// Multi-selection (indices)
    MultiSelect(Vec<usize>),

    /// List of strings (e.g., packages)
    StringList(Vec<String>),

    /// Key-value pairs
    Map(HashMap<String, String>),

    /// Null/unset
    Null,
}

impl WidgetValue {
    /// Convert to Nix expression string
    pub fn to_nix(&self) -> String {
        match self {
            Self::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            Self::String(s) => format!("\"{}\"", s.replace('\"', "\\\"")),
            Self::Int(i) => i.to_string(),
            Self::Float(f) => format!("{f:.6}"),
            Self::Selection(i) => i.to_string(),
            Self::MultiSelect(indices) => {
                let items: Vec<String> = indices.iter().map(|i| i.to_string()).collect();
                format!("[ {} ]", items.join(" "))
            }
            Self::StringList(items) => {
                let quoted: Vec<String> = items
                    .iter()
                    .map(|s| format!("\"{}\"", s.replace('\"', "\\\"")))
                    .collect();
                format!("[ {} ]", quoted.join(" "))
            }
            Self::Map(map) => {
                let entries: Vec<String> = map
                    .iter()
                    .map(|(k, v)| format!("{} = \"{}\";", k, v.replace('\"', "\\\"")))
                    .collect();
                format!("{{ {} }}", entries.join(" "))
            }
            Self::Null => "null".to_string(),
        }
    }

    /// Parse from Nix expression string
    pub fn from_nix(nix_expr: &str, expected_type: WidgetValueType) -> Option<Self> {
        let trimmed = nix_expr.trim();

        match expected_type {
            WidgetValueType::Bool => match trimmed {
                "true" => Some(Self::Bool(true)),
                "false" => Some(Self::Bool(false)),
                _ => None,
            },
            WidgetValueType::String => {
                if trimmed.starts_with('"') && trimmed.ends_with('"') {
                    let inner = &trimmed[1..trimmed.len() - 1];
                    Some(Self::String(inner.replace("\\\"", "\"")))
                } else {
                    Some(Self::String(trimmed.to_string()))
                }
            }
            WidgetValueType::Int => trimmed.parse::<i64>().ok().map(Self::Int),
            WidgetValueType::Float => trimmed.parse::<f64>().ok().map(Self::Float),
            WidgetValueType::StringList => {
                if trimmed.starts_with('[') && trimmed.ends_with(']') {
                    let inner = &trimmed[1..trimmed.len() - 1];
                    let items: Vec<String> = inner
                        .split_whitespace()
                        .map(|s| {
                            if s.starts_with('"') && s.ends_with('"') {
                                s[1..s.len() - 1].to_string()
                            } else {
                                s.to_string()
                            }
                        })
                        .collect();
                    Some(Self::StringList(items))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Expected widget value type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WidgetValueType {
    Bool,
    String,
    Int,
    Float,
    Selection,
    MultiSelect,
    StringList,
    Map,
}

/// Value transformer for custom conversions
#[derive(Debug, Clone)]
pub enum ValueTransformer {
    /// Direct pass-through
    Identity,

    /// Boolean enable/disable for service
    ServiceEnable,

    /// Package list with pkgs prefix
    PackageList,

    /// Port number with validation
    Port,

    /// File path with validation
    FilePath,

    /// Custom transformation function name
    Custom(String),
}

impl ValueTransformer {
    /// Transform widget value to Nix expression
    pub fn to_nix(&self, value: &WidgetValue, options: &[String]) -> String {
        match self {
            Self::Identity => value.to_nix(),

            Self::ServiceEnable => match value {
                WidgetValue::Bool(b) => if *b { "true" } else { "false" }.to_string(),
                _ => "false".to_string(),
            },

            Self::PackageList => match value {
                WidgetValue::StringList(packages) => {
                    let with_pkgs: Vec<String> = packages
                        .iter()
                        .map(|p| {
                            if p.starts_with("pkgs.") {
                                p.clone()
                            } else {
                                format!("pkgs.{p}")
                            }
                        })
                        .collect();
                    format!("with pkgs; [ {} ]", with_pkgs.join(" "))
                }
                WidgetValue::Selection(idx) if *idx < options.len() => {
                    format!("pkgs.{}", options[*idx])
                }
                _ => "[ ]".to_string(),
            },

            Self::Port => match value {
                WidgetValue::Int(port) if *port >= 1 && *port <= 65535 => port.to_string(),
                WidgetValue::String(s) => s
                    .parse::<u16>()
                    .map(|p| p.to_string())
                    .unwrap_or_else(|_| "80".to_string()),
                _ => "80".to_string(),
            },

            Self::FilePath => {
                match value {
                    WidgetValue::String(path) => {
                        // Validate path doesn't contain dangerous characters
                        let sanitized = path.replace(['\'', '"', '\\', '$', '`'], "");
                        format!("\"{sanitized}\"")
                    }
                    _ => "\"/tmp\"".to_string(),
                }
            }

            Self::Custom(_name) => {
                // Custom transformers would be looked up from a registry
                value.to_nix()
            }
        }
    }

    /// Transform Nix expression to widget value
    pub fn from_nix(&self, nix_expr: &str, _options: &[String]) -> WidgetValue {
        let trimmed = nix_expr.trim();

        match self {
            Self::Identity => {
                // Try to infer type
                if trimmed == "true" {
                    WidgetValue::Bool(true)
                } else if trimmed == "false" {
                    WidgetValue::Bool(false)
                } else if let Ok(i) = trimmed.parse::<i64>() {
                    WidgetValue::Int(i)
                } else if trimmed.starts_with('"') && trimmed.ends_with('"') {
                    WidgetValue::String(trimmed[1..trimmed.len() - 1].to_string())
                } else {
                    WidgetValue::String(trimmed.to_string())
                }
            }

            Self::ServiceEnable => WidgetValue::Bool(trimmed == "true"),

            Self::PackageList => {
                // Parse "with pkgs; [ ... ]" or "[ pkgs.foo pkgs.bar ]"
                let cleaned = trimmed.replace("with pkgs;", "").trim().to_string();

                if cleaned.starts_with('[') && cleaned.ends_with(']') {
                    let inner = &cleaned[1..cleaned.len() - 1];
                    let packages: Vec<String> = inner
                        .split_whitespace()
                        .map(|s| s.replace("pkgs.", "").to_string())
                        .collect();
                    WidgetValue::StringList(packages)
                } else {
                    WidgetValue::StringList(Vec::new())
                }
            }

            Self::Port => trimmed
                .parse::<i64>()
                .map(WidgetValue::Int)
                .unwrap_or(WidgetValue::Int(80)),

            Self::FilePath => {
                if trimmed.starts_with('"') && trimmed.ends_with('"') {
                    WidgetValue::String(trimmed[1..trimmed.len() - 1].to_string())
                } else {
                    WidgetValue::String(trimmed.to_string())
                }
            }

            Self::Custom(_) => WidgetValue::String(trimmed.to_string()),
        }
    }
}

/// Widget binding configuration
#[derive(Debug, Clone)]
pub struct WidgetBinding {
    /// Unique widget identifier
    pub widget_id: WidgetId,

    /// NixOS option path
    pub nix_path: NixPath,

    /// HDC semantic encoding of the binding
    pub semantic_hv: BinaryHV,

    /// Value transformer
    pub transformer: ValueTransformer,

    /// Expected value type
    pub value_type: WidgetValueType,

    /// Available options (for selection widgets)
    pub options: Vec<String>,

    /// Default value
    pub default: WidgetValue,

    /// Human-readable label
    pub label: String,

    /// Description/help text
    pub description: String,

    /// Whether this is a required field
    pub required: bool,
}

impl WidgetBinding {
    /// Create a new binding builder
    pub fn builder(
        widget_id: impl Into<String>,
        nix_path: impl Into<String>,
    ) -> WidgetBindingBuilder {
        WidgetBindingBuilder::new(widget_id, nix_path)
    }
}

/// Builder for WidgetBinding
pub struct WidgetBindingBuilder {
    widget_id: WidgetId,
    nix_path: NixPath,
    transformer: ValueTransformer,
    value_type: WidgetValueType,
    options: Vec<String>,
    default: WidgetValue,
    label: String,
    description: String,
    required: bool,
}

impl WidgetBindingBuilder {
    pub fn new(widget_id: impl Into<String>, nix_path: impl Into<String>) -> Self {
        let widget_id_str = widget_id.into();
        let nix_path_str = nix_path.into();

        Self {
            label: nix_path_str.rsplit('.').next().unwrap_or("").to_string(),
            widget_id: WidgetId(widget_id_str),
            nix_path: NixPath(nix_path_str),
            transformer: ValueTransformer::Identity,
            value_type: WidgetValueType::String,
            options: Vec::new(),
            default: WidgetValue::Null,
            description: String::new(),
            required: false,
        }
    }

    pub fn transformer(mut self, t: ValueTransformer) -> Self {
        self.transformer = t;
        self
    }

    pub fn value_type(mut self, t: WidgetValueType) -> Self {
        self.value_type = t;
        self
    }

    pub fn options(mut self, opts: Vec<String>) -> Self {
        self.options = opts;
        self
    }

    pub fn default(mut self, d: WidgetValue) -> Self {
        self.default = d;
        self
    }

    pub fn label(mut self, l: impl Into<String>) -> Self {
        self.label = l.into();
        self
    }

    pub fn description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }

    pub fn required(mut self, r: bool) -> Self {
        self.required = r;
        self
    }

    pub fn build(self) -> WidgetBinding {
        WidgetBinding {
            widget_id: self.widget_id,
            nix_path: self.nix_path,
            semantic_hv: BinaryHV::zero(), // Will be set by GuiBridge
            transformer: self.transformer,
            value_type: self.value_type,
            options: self.options,
            default: self.default,
            label: self.label,
            description: self.description,
            required: self.required,
        }
    }
}

/// Result of widget-to-nix mapping
#[derive(Debug, Clone)]
pub enum MappingResult {
    /// Successfully mapped
    Success { nix_expr: String, confidence: f64 },

    /// Validation failed
    ValidationFailed {
        errors: Vec<super::ValidationError>,
        suggestions: Vec<super::Suggestion>,
    },

    /// Unknown widget ID
    UnknownWidget { widget_id: WidgetId },
}

/// Widget mapper for bidirectional translation
pub struct WidgetMapper {
    /// Value type inference hints
    type_hints: HashMap<String, WidgetValueType>,
}

impl WidgetMapper {
    /// Create new widget mapper
    pub fn new() -> Self {
        let mut type_hints = HashMap::new();

        // Common NixOS option type hints
        type_hints.insert("enable".to_string(), WidgetValueType::Bool);
        type_hints.insert("package".to_string(), WidgetValueType::String);
        type_hints.insert("packages".to_string(), WidgetValueType::StringList);
        type_hints.insert("port".to_string(), WidgetValueType::Int);
        type_hints.insert("extraConfig".to_string(), WidgetValueType::String);
        type_hints.insert("user".to_string(), WidgetValueType::String);
        type_hints.insert("group".to_string(), WidgetValueType::String);

        Self { type_hints }
    }

    /// Transform widget value to Nix expression
    pub fn widget_to_nix(&self, binding: &WidgetBinding, value: &WidgetValue) -> String {
        binding.transformer.to_nix(value, &binding.options)
    }

    /// Transform Nix expression to widget value
    pub fn nix_to_widget(&self, binding: &WidgetBinding, nix_expr: &str) -> WidgetValue {
        binding.transformer.from_nix(nix_expr, &binding.options)
    }

    /// Parse Nix content and extract path->value pairs
    pub fn parse_nix_content(&self, content: &str) -> Vec<(NixPath, String)> {
        let mut results = Vec::new();

        // Simple regex-based extraction
        // In a real implementation, this would use a proper Nix parser
        for line in content.lines() {
            let trimmed = line.trim();

            // Skip comments and empty lines
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Match "path.to.option = value;"
            if let Some(eq_pos) = trimmed.find('=') {
                let path = trimmed[..eq_pos].trim();
                let value_part = trimmed[eq_pos + 1..].trim();

                // Remove trailing semicolon
                let value = value_part.strip_suffix(';').unwrap_or(value_part);

                if !path.is_empty() && !value.is_empty() {
                    results.push((NixPath::new(path), value.trim().to_string()));
                }
            }
        }

        results
    }

    /// Infer value type from path leaf
    pub fn infer_type(&self, path: &NixPath) -> WidgetValueType {
        let leaf = path.leaf();
        self.type_hints
            .get(leaf)
            .copied()
            .unwrap_or(WidgetValueType::String)
    }
}

impl Default for WidgetMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_widget_value_to_nix() {
        assert_eq!(WidgetValue::Bool(true).to_nix(), "true");
        assert_eq!(WidgetValue::Bool(false).to_nix(), "false");
        assert_eq!(
            WidgetValue::String("hello".to_string()).to_nix(),
            "\"hello\""
        );
        assert_eq!(WidgetValue::Int(8080).to_nix(), "8080");

        let list = WidgetValue::StringList(vec!["foo".to_string(), "bar".to_string()]);
        assert_eq!(list.to_nix(), "[ \"foo\" \"bar\" ]");
    }

    #[test]
    fn test_widget_value_from_nix() {
        assert_eq!(
            WidgetValue::from_nix("true", WidgetValueType::Bool),
            Some(WidgetValue::Bool(true))
        );
        assert_eq!(
            WidgetValue::from_nix("\"hello\"", WidgetValueType::String),
            Some(WidgetValue::String("hello".to_string()))
        );
        assert_eq!(
            WidgetValue::from_nix("8080", WidgetValueType::Int),
            Some(WidgetValue::Int(8080))
        );
    }

    #[test]
    fn test_nix_path() {
        let path = NixPath::new("services.nginx.enable");
        assert_eq!(path.leaf(), "enable");
        assert_eq!(path.parent(), Some(NixPath::new("services.nginx")));
    }

    #[test]
    fn test_transformer_service_enable() {
        let transformer = ValueTransformer::ServiceEnable;

        assert_eq!(transformer.to_nix(&WidgetValue::Bool(true), &[]), "true");

        assert_eq!(transformer.from_nix("true", &[]), WidgetValue::Bool(true));
    }

    #[test]
    fn test_transformer_package_list() {
        let transformer = ValueTransformer::PackageList;

        let packages = WidgetValue::StringList(vec!["firefox".to_string(), "git".to_string()]);

        let nix_expr = transformer.to_nix(&packages, &[]);
        assert!(nix_expr.contains("pkgs.firefox"));
        assert!(nix_expr.contains("pkgs.git"));
    }

    #[test]
    fn test_binding_builder() {
        let binding = WidgetBinding::builder("nginx_enable", "services.nginx.enable")
            .transformer(ValueTransformer::ServiceEnable)
            .value_type(WidgetValueType::Bool)
            .default(WidgetValue::Bool(false))
            .label("Enable Nginx")
            .description("Enable the Nginx web server")
            .required(false)
            .build();

        assert_eq!(binding.widget_id, WidgetId::new("nginx_enable"));
        assert_eq!(binding.nix_path, NixPath::new("services.nginx.enable"));
        assert_eq!(binding.label, "Enable Nginx");
    }
}
