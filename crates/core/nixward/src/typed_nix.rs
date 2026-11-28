// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed, injection-resistant construction of a small NixOS module subset.
//!
//! This module intentionally does not accept raw Nix expressions. Callers can
//! construct only the expression variants represented by NixExpr. Every
//! identifier, attribute path and package path is validated before rendering.
//! The rendered module is parsed with rnix before it is returned.
//!
//! This is a syntax-construction boundary, not a NixOS option resolver or
//! evaluator. A syntactically valid option can still be unknown, read-only,
//! unavailable in the target release, or semantically invalid.

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Errors from constructing or rendering typed Nix values.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TypedNixError {
    /// A Nix identifier failed the restricted identifier grammar.
    #[error("invalid Nix identifier: {0}")]
    InvalidIdentifier(String),
    /// An attribute path was empty or contained an invalid segment.
    #[error("invalid Nix attribute path: {0}")]
    InvalidAttributePath(String),
    /// A package attribute path was empty or invalid.
    #[error("invalid Nix package attribute path: {0}")]
    InvalidPackagePath(String),
    /// The same option path was assigned more than once.
    #[error("duplicate Nix option assignment: {0}")]
    DuplicateAssignment(String),
    /// A parent and child option path were assigned in the same module.
    #[error("overlapping Nix option assignments: {0} conflicts with {1}")]
    OverlappingAssignment(String, String),
    /// The value contains a control character without a supported Nix escape.
    #[error("unsupported control character U+{0:04X} in Nix string")]
    UnsupportedControlCharacter(u32),
    /// rnix rejected the rendered source.
    #[error("rendered Nix module did not parse: {0}")]
    Syntax(String),
}

/// A restricted Nix identifier suitable for a static attribute name.
///
/// The grammar is deliberately narrower than all syntax accepted by Nix.
/// Dynamic attribute names and arbitrary quoted expressions are not exposed.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NixIdentifier(String);

impl NixIdentifier {
    /// Validate and construct a static Nix identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, TypedNixError> {
        let value = value.into();
        let mut chars = value.chars();

        let valid_start = chars
            .next()
            .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_');
        let valid_tail =
            chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '\''));
        let reserved = matches!(
            value.as_str(),
            "assert"
                | "else"
                | "if"
                | "in"
                | "inherit"
                | "let"
                | "or"
                | "rec"
                | "then"
                | "with"
                | "true"
                | "false"
                | "null"
        );

        if !valid_start || !valid_tail || reserved {
            return Err(TypedNixError::InvalidIdentifier(value));
        }

        Ok(Self(value))
    }

    /// The validated identifier text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A dotted sequence of static Nix attribute identifiers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NixAttributePath(Vec<NixIdentifier>);

impl NixAttributePath {
    /// Parse and validate a dotted attribute path.
    pub fn parse(value: &str) -> Result<Self, TypedNixError> {
        if value.is_empty() {
            return Err(TypedNixError::InvalidAttributePath(value.to_string()));
        }

        let segments = value
            .split('.')
            .map(NixIdentifier::new)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| TypedNixError::InvalidAttributePath(value.to_string()))?;

        Ok(Self(segments))
    }

    /// Render this validated path without introducing interpolation.
    pub fn as_nix(&self) -> String {
        self.0
            .iter()
            .map(NixIdentifier::as_str)
            .collect::<Vec<_>>()
            .join(".")
    }

    fn is_prefix_of(&self, other: &Self) -> bool {
        self.0.len() < other.0.len()
            && self
                .0
                .iter()
                .zip(other.0.iter())
                .all(|(left, right)| left == right)
    }
}

/// A package reference resolved to a static attribute below the target pkgs.
///
/// Construction validates syntax only. The package must still be resolved
/// against the exact target nixpkgs package set before this value is used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NixPackageRef {
    attribute: NixAttributePath,
}

impl NixPackageRef {
    /// Construct a package reference such as ripgrep or
    /// python3Packages.numpy.
    pub fn new(attribute: &str) -> Result<Self, TypedNixError> {
        let parsed = NixAttributePath::parse(attribute)
            .map_err(|_| TypedNixError::InvalidPackagePath(attribute.to_string()))?;
        if parsed
            .0
            .first()
            .is_some_and(|segment| segment.as_str() == "pkgs")
        {
            return Err(TypedNixError::InvalidPackagePath(attribute.to_string()));
        }
        Ok(Self { attribute: parsed })
    }

    /// The package attribute below pkgs.
    pub fn attribute(&self) -> &NixAttributePath {
        &self.attribute
    }
}

/// The subset of Nix expressions that the safe builder can emit.
///
/// There is intentionally no Raw or Source variant. Untrusted strings can
/// become only escaped Nix string literals, never executable expressions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NixExpr {
    /// Nix boolean.
    Bool(bool),
    /// Nix signed integer.
    Int(i64),
    /// Escaped Nix double-quoted string.
    String(String),
    /// Nix null.
    Null,
    /// Nix list.
    List(Vec<NixExpr>),
    /// Nix attribute set with static validated keys.
    AttributeSet(BTreeMap<NixIdentifier, NixExpr>),
    /// A package reference below the target pkgs argument.
    Package(NixPackageRef),
}

impl NixExpr {
    /// Render the expression using only validated syntax nodes.
    pub fn as_nix(&self) -> Result<String, TypedNixError> {
        match self {
            Self::Bool(value) => Ok(value.to_string()),
            Self::Int(value) => Ok(value.to_string()),
            Self::String(value) => render_string_literal(value),
            Self::Null => Ok("null".to_string()),
            Self::List(items) => {
                let rendered = items
                    .iter()
                    .map(Self::as_nix)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(format!("[ {} ]", rendered.join(" ")))
            }
            Self::AttributeSet(attributes) => {
                let mut rendered = String::from("{ ");
                for (key, value) in attributes {
                    write!(rendered, "{} = {}; ", key.as_str(), value.as_nix()?)
                        .expect("writing to a String cannot fail");
                }
                rendered.push('}');
                Ok(rendered)
            }
            Self::Package(package) => Ok(format!("pkgs.{}", package.attribute.as_nix())),
        }
    }
}

/// A deterministic NixOS module plan containing static option assignments.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NixModulePlan {
    assignments: BTreeMap<NixAttributePath, NixExpr>,
}

impl NixModulePlan {
    /// Create an empty plan.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an assignment after validating its path and overlap constraints.
    pub fn insert(
        &mut self,
        path: NixAttributePath,
        value: NixExpr,
    ) -> Result<(), TypedNixError> {
        let rendered_path = path.as_nix();

        if self.assignments.contains_key(&path) {
            return Err(TypedNixError::DuplicateAssignment(rendered_path));
        }

        for existing in self.assignments.keys() {
            if existing.is_prefix_of(&path) || path.is_prefix_of(existing) {
                return Err(TypedNixError::OverlappingAssignment(
                    existing.as_nix(),
                    rendered_path,
                ));
            }
        }

        self.assignments.insert(path, value);
        Ok(())
    }

    /// Number of option assignments in this plan.
    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    /// Whether the plan has no assignments.
    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }

    /// Render a NixOS module and verify its syntax with rnix.
    ///
    /// Passing this method proves parseability only. It does not prove option
    /// existence, type compatibility, read-only status, evaluation, build,
    /// successful activation or runtime behavior.
    pub fn render_and_parse(&self) -> Result<String, TypedNixError> {
        let mut source = String::from("{ config, lib, pkgs, ... }:\n{\n");

        for (path, value) in &self.assignments {
            writeln!(source, "  {} = {}; ", path.as_nix(), value.as_nix()?)
                .expect("writing to a String cannot fail");
        }

        source.push_str("}\n");

        let parse = rnix::Root::parse(&source);
        let errors = parse.errors();
        if !errors.is_empty() {
            return Err(TypedNixError::Syntax(format!("{errors:?}")));
        }

        Ok(source)
    }
}

/// Escape untrusted text as a Nix double-quoted string literal.
///
/// Nix requires escaping double quotes, backslashes and the interpolation
/// opener formed by a dollar sign followed by a left brace. Newlines,
/// carriage returns and tabs use their named escapes. Other control
/// characters are rejected instead of being emitted ambiguously.
pub fn render_string_literal(value: &str) -> Result<String, TypedNixError> {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');

    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '$' if chars.peek() == Some(&'{') => output.push_str("\\$"),
            character if character.is_control() => {
                return Err(TypedNixError::UnsupportedControlCharacter(character as u32));
            }
            character => output.push(character),
        }
    }

    output.push('"');
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_literal_escapes_quotes_slashes_interpolation_and_whitespace() {
        let interpolation = concat!("$", "{builtins.abort \"x\"}");
        let value = format!("quote \" slash \\ literal {interpolation}\n\t");
        let rendered = render_string_literal(&value).unwrap();

        assert!(rendered.starts_with('"'));
        assert!(rendered.ends_with('"'));
        assert!(rendered.contains("\\\""));
        assert!(rendered.contains("\\\\"));
        assert!(rendered.contains(&concat!("\\", "$", "{")));
        assert!(rendered.contains("\\n"));
        assert!(rendered.contains("\\t"));
        let without_escaped_interpolation = rendered.replace(&concat!("\\", "$", "{"), "");
        assert!(!without_escaped_interpolation.contains(&concat!("$", "{")));
    }

    #[test]
    fn unsupported_control_characters_fail_closed() {
        assert_eq!(
            render_string_literal("before\0after"),
            Err(TypedNixError::UnsupportedControlCharacter(0))
        );
    }

    #[test]
    fn invalid_attribute_and_package_paths_are_rejected() {
        assert!(NixAttributePath::parse("services.nginx.enable; builtins.abort").is_err());
        assert!(NixAttributePath::parse("services..nginx").is_err());
        let injected_path = concat!("services.", "$", "{name}.enable");
        assert!(NixAttributePath::parse(&injected_path).is_err());
        assert!(NixPackageRef::new("ripgrep; builtins.abort").is_err());
        assert!(NixPackageRef::new("pkgs.ripgrep").is_err());
    }

    #[test]
    fn module_builder_emits_deterministic_parseable_nix() {
        let mut plan = NixModulePlan::new();
        let interpolation = concat!("$", "{builtins.abort \"not code\"}");
        plan.insert(
            NixAttributePath::parse("networking.hostName").unwrap(),
            NixExpr::String(format!("host-{interpolation}")),
        )
        .unwrap();
        plan.insert(
            NixAttributePath::parse("networking.firewall.enable").unwrap(),
            NixExpr::Bool(true),
        )
        .unwrap();
        plan.insert(
            NixAttributePath::parse("environment.systemPackages").unwrap(),
            NixExpr::List(vec![
                NixExpr::Package(NixPackageRef::new("ripgrep").unwrap()),
                NixExpr::Package(NixPackageRef::new("python3Packages.numpy").unwrap()),
            ]),
        )
        .unwrap();

        let source = plan.render_and_parse().unwrap();

        assert!(source.contains("networking.firewall.enable = true;"));
        assert!(source.contains("environment.systemPackages = [ pkgs.ripgrep pkgs.python3Packages.numpy ];"));
        assert!(source.contains(&concat!("\\", "$", "{builtins.abort")));
    }

    #[test]
    fn duplicate_and_parent_child_assignments_are_rejected() {
        let mut plan = NixModulePlan::new();
        let path = NixAttributePath::parse("services.nginx").unwrap();
        plan.insert(path.clone(), NixExpr::AttributeSet(BTreeMap::new()))
            .unwrap();
        assert_eq!(
            plan.insert(path, NixExpr::Bool(true)),
            Err(TypedNixError::DuplicateAssignment("services.nginx".to_string()))
        );

        let mut other = NixModulePlan::new();
        other
            .insert(
                NixAttributePath::parse("services.nginx").unwrap(),
                NixExpr::AttributeSet(BTreeMap::new()),
            )
            .unwrap();
        assert!(matches!(
            other.insert(
                NixAttributePath::parse("services.nginx.enable").unwrap(),
                NixExpr::Bool(true)
            ),
            Err(TypedNixError::OverlappingAssignment(_, _))
        ));
    }

    #[test]
    fn expression_variants_cover_only_typed_values() {
        let mut nested = BTreeMap::new();
        nested.insert(NixIdentifier::new("enabled").unwrap(), NixExpr::Bool(true));
        nested.insert(NixIdentifier::new("port").unwrap(), NixExpr::Int(443));
        nested.insert(
            NixIdentifier::new("description").unwrap(),
            NixExpr::String("public HTTPS".to_string()),
        );

        let value = NixExpr::AttributeSet(nested);
        assert_eq!(
            value.as_nix().unwrap(),
            "{ description = \"public HTTPS\"; enabled = true; port = 443; }"
        );
    }
}
