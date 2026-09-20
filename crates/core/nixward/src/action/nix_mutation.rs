// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed, conservative Nix configuration mutation semantics.
//!
//! This module is intentionally narrower than `ConfigWriter::set_option(&str, &str)`.
//! It provides a governed source-generation substrate in which option identity,
//! value semantics, and provenance are explicit before any Nix source is rendered.
//!
//! A valid mutation is not authorization and does not imply that the target file
//! is current. State binding, authorization, application, and post-state evidence
//! remain separate boundaries.

use blake3::Hasher;
use thiserror::Error;

const CONFIG_MUTATION_DOMAIN_V1: &[u8] = b"nixward-config-mutation-v1";

/// Conservative v1 representation of an unquoted Nix attribute path.
///
/// Quoted/dynamic attribute names are deliberately unsupported. If they become
/// necessary they should receive an AST-backed representation rather than an
/// arbitrary source-text escape hatch.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NixOptionPathV1 {
    segments: Vec<String>,
}

impl NixOptionPathV1 {
    pub fn parse(path: &str) -> Result<Self, NixMutationErrorV1> {
        if path.is_empty() || path.trim() != path {
            return Err(NixMutationErrorV1::InvalidOptionPath);
        }

        let segments: Vec<String> = path.split('.').map(str::to_string).collect();

        if segments.is_empty() || segments.iter().any(|segment| !valid_identifier(segment)) {
            return Err(NixMutationErrorV1::InvalidOptionPath);
        }

        Ok(Self { segments })
    }

    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    pub fn canonical_path(&self) -> String {
        self.segments.join(".")
    }
}

fn valid_identifier(segment: &str) -> bool {
    let mut chars = segment.chars();
    let Some(first) = chars.next() else {
        return false;
    };

    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }

    chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
}

/// Closed primitive value vocabulary for governed v1 mutations.
///
/// There is deliberately no `RawExpression(String)` variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NixPrimitiveValueV1 {
    Bool(bool),
    Unsigned(u64),
    String(String),
    DurationSeconds(u64),
    MemorySizeMiB(u64),
}

impl NixPrimitiveValueV1 {
    pub fn render(&self) -> String {
        match self {
            Self::Bool(value) => value.to_string(),
            Self::Unsigned(value) => value.to_string(),
            Self::String(value) => render_nix_string(value),
            Self::DurationSeconds(value) => value.to_string(),
            // PostgreSQL and the current Nixward rule family express binary-sized
            // configuration quantities using the conventional `MB` spelling.
            Self::MemorySizeMiB(value) => render_nix_string(&format!("{value}MB")),
        }
    }
}

fn render_nix_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '$' if chars.peek() == Some(&'{') => {
                chars.next();
                out.push_str("\\${");
            }
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

/// Provenance of a typed mutation proposal.
///
/// This is proposal provenance, not execution authority. `origin_ref` should bind
/// the upstream evidence/realization product that justified invoking the rule.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NixMutationProvenanceV1 {
    rule_id: String,
    origin_ref: String,
}

impl NixMutationProvenanceV1 {
    pub fn new(
        rule_id: impl Into<String>,
        origin_ref: impl Into<String>,
    ) -> Result<Self, NixMutationErrorV1> {
        let rule_id = require_nonempty(rule_id)?;
        let origin_ref = require_nonempty(origin_ref)?;
        Ok(Self {
            rule_id,
            origin_ref,
        })
    }

    pub fn rule_id(&self) -> &str {
        &self.rule_id
    }

    pub fn origin_ref(&self) -> &str {
        &self.origin_ref
    }
}

/// One exact typed option mutation proposal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NixConfigMutationV1 {
    path: NixOptionPathV1,
    value: NixPrimitiveValueV1,
    provenance: NixMutationProvenanceV1,
}

impl NixConfigMutationV1 {
    pub fn new(
        path: NixOptionPathV1,
        value: NixPrimitiveValueV1,
        provenance: NixMutationProvenanceV1,
    ) -> Self {
        Self {
            path,
            value,
            provenance,
        }
    }

    pub fn path(&self) -> &NixOptionPathV1 {
        &self.path
    }

    pub fn value(&self) -> &NixPrimitiveValueV1 {
        &self.value
    }

    pub fn provenance(&self) -> &NixMutationProvenanceV1 {
        &self.provenance
    }

    /// Deterministically compile the typed semantic mutation to one Nix assignment.
    pub fn render_assignment(&self) -> String {
        format!("{} = {};", self.path.canonical_path(), self.value.render())
    }

    /// Deterministic semantic identity independent of serde/Debug/source layout.
    pub fn digest(&self) -> String {
        let mut h = Hasher::new();
        h.update(CONFIG_MUTATION_DOMAIN_V1);
        put_str(&mut h, &self.path.canonical_path());
        put_value(&mut h, &self.value);
        put_str(&mut h, self.provenance.rule_id());
        put_str(&mut h, self.provenance.origin_ref());
        h.finalize().to_hex().to_string()
    }
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum NixMutationErrorV1 {
    #[error("invalid governed Nix option path")]
    InvalidOptionPath,
    #[error("empty required mutation provenance field")]
    EmptyProvenance,
}

fn require_nonempty(value: impl Into<String>) -> Result<String, NixMutationErrorV1> {
    let value = value.into();
    if value.trim().is_empty() {
        Err(NixMutationErrorV1::EmptyProvenance)
    } else {
        Ok(value)
    }
}

fn put_u8(h: &mut Hasher, value: u8) {
    h.update(&[value]);
}

fn put_u64(h: &mut Hasher, value: u64) {
    h.update(&value.to_be_bytes());
}

fn put_str(h: &mut Hasher, value: &str) {
    put_u64(h, value.len() as u64);
    h.update(value.as_bytes());
}

fn put_value(h: &mut Hasher, value: &NixPrimitiveValueV1) {
    match value {
        NixPrimitiveValueV1::Bool(value) => {
            put_u8(h, 0);
            put_u8(h, u8::from(*value));
        }
        NixPrimitiveValueV1::Unsigned(value) => {
            put_u8(h, 1);
            put_u64(h, *value);
        }
        NixPrimitiveValueV1::String(value) => {
            put_u8(h, 2);
            put_str(h, value);
        }
        NixPrimitiveValueV1::DurationSeconds(value) => {
            put_u8(h, 3);
            put_u64(h, *value);
        }
        NixPrimitiveValueV1::MemorySizeMiB(value) => {
            put_u8(h, 4);
            put_u64(h, *value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> NixMutationProvenanceV1 {
        NixMutationProvenanceV1::new(
            "hardening:postgres-shared-buffers:v1",
            "realization:fixture:1",
        )
        .unwrap()
    }

    #[test]
    fn accepts_conservative_existing_option_paths() {
        for path in [
            "services.postgresql.settings.shared_buffers",
            "systemd.services.nginx.serviceConfig.RestartSec",
        ] {
            let parsed = NixOptionPathV1::parse(path).unwrap();
            assert_eq!(parsed.canonical_path(), path);
        }
    }

    #[test]
    fn rejects_source_structure_in_option_path() {
        for path in [
            "",
            " services.nginx.enable",
            "services..nginx",
            "services.nginx.enable;systemd.services.x.enable",
            "services.nginx.enable = true",
            "services.nginx.{enable}",
            "services.nginx#comment",
            "services.nginx\nenable",
            "services.${name}.enable",
            "services/nginx/enable",
        ] {
            assert_eq!(
                NixOptionPathV1::parse(path).unwrap_err(),
                NixMutationErrorV1::InvalidOptionPath,
                "unexpectedly accepted {path:?}",
            );
        }
    }

    #[test]
    fn primitive_values_render_deterministically() {
        assert_eq!(NixPrimitiveValueV1::Bool(true).render(), "true");
        assert_eq!(NixPrimitiveValueV1::Unsigned(42).render(), "42");
        assert_eq!(NixPrimitiveValueV1::DurationSeconds(5).render(), "5");
        assert_eq!(NixPrimitiveValueV1::MemorySizeMiB(512).render(), "\"512MB\"");
        assert_eq!(
            NixPrimitiveValueV1::String("a\"b\\c\n${danger}".into()).render(),
            "\"a\\\"b\\\\c\\n\\${danger}\""
        );
    }

    #[test]
    fn string_value_cannot_become_second_assignment() {
        let mutation = NixConfigMutationV1::new(
            NixOptionPathV1::parse("services.demo.label").unwrap(),
            NixPrimitiveValueV1::String("ok\"; services.evil.enable = true; #".into()),
            provenance(),
        );
        assert_eq!(
            mutation.render_assignment(),
            "services.demo.label = \"ok\\\"; services.evil.enable = true; #\";"
        );
    }

    #[test]
    fn mutation_identity_binds_path_value_rule_and_origin() {
        let base = NixConfigMutationV1::new(
            NixOptionPathV1::parse("services.postgresql.settings.shared_buffers").unwrap(),
            NixPrimitiveValueV1::MemorySizeMiB(512),
            provenance(),
        );
        assert_eq!(base.digest(), base.digest());

        let changed_value = NixConfigMutationV1::new(
            base.path().clone(),
            NixPrimitiveValueV1::MemorySizeMiB(1024),
            provenance(),
        );
        assert_ne!(base.digest(), changed_value.digest());

        let changed_path = NixConfigMutationV1::new(
            NixOptionPathV1::parse("services.postgresql.settings.work_mem").unwrap(),
            base.value().clone(),
            provenance(),
        );
        assert_ne!(base.digest(), changed_path.digest());

        let changed_origin = NixConfigMutationV1::new(
            base.path().clone(),
            base.value().clone(),
            NixMutationProvenanceV1::new(
                "hardening:postgres-shared-buffers:v1",
                "realization:fixture:2",
            )
            .unwrap(),
        );
        assert_ne!(base.digest(), changed_origin.digest());
    }

    #[test]
    fn mutation_provenance_is_not_authority() {
        assert_eq!(provenance().rule_id(), "hardening:postgres-shared-buffers:v1");
        assert_eq!(provenance().origin_ref(), "realization:fixture:1");
    }

    #[test]
    fn provenance_fields_must_be_nonempty() {
        assert_eq!(
            NixMutationProvenanceV1::new("", "realization:1").unwrap_err(),
            NixMutationErrorV1::EmptyProvenance
        );
        assert_eq!(
            NixMutationProvenanceV1::new("rule:1", "  ").unwrap_err(),
            NixMutationErrorV1::EmptyProvenance
        );
    }
}
