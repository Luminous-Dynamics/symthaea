// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Closed typed domain for the first future Nixward effect-binding slice.
//! Domain data only: no command conversion, executor handle, or authority.

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const SERVICE_OPERATION_DOMAIN_V1: &[u8] = b"nixward-service-operation-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NixServiceOperationKindV1 { Enable, Disable, Start, Stop, Restart, Reload }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixServiceOperationV1 { unit: String, operation: NixServiceOperationKindV1 }

impl NixServiceOperationV1 {
    pub fn new(unit: impl Into<String>, operation: NixServiceOperationKindV1) -> Result<Self, NixServiceOperationErrorV1> {
        let unit = canonical_service_unit_v1(&unit.into())?;
        let value = Self { unit, operation };
        value.validate_shape()?;
        Ok(value)
    }
    pub fn unit(&self) -> &str { &self.unit }
    pub fn operation(&self) -> NixServiceOperationKindV1 { self.operation }
    pub fn validate_shape(&self) -> Result<(), NixServiceOperationErrorV1> {
        if self.unit.is_empty() { return Err(NixServiceOperationErrorV1::EmptyUnit); }
        if !self.unit.ends_with(".service") { return Err(NixServiceOperationErrorV1::NonServiceUnit); }
        if self.unit.contains('/') || self.unit.contains('\\') { return Err(NixServiceOperationErrorV1::PathLikeUnit); }
        if self.unit.chars().any(|c| c.is_whitespace() || c.is_control()) { return Err(NixServiceOperationErrorV1::WhitespaceOrControl); }
        if self.unit.starts_with('.') || self.unit.ends_with('.') { return Err(NixServiceOperationErrorV1::AmbiguousUnit); }
        Ok(())
    }
    pub fn digest(&self) -> Result<String, NixServiceOperationErrorV1> {
        self.validate_shape()?;
        let mut h = Hasher::new();
        h.update(SERVICE_OPERATION_DOMAIN_V1);
        h.update(&(self.unit.len() as u64).to_be_bytes());
        h.update(self.unit.as_bytes());
        h.update(&[match self.operation {
            NixServiceOperationKindV1::Enable => 0, NixServiceOperationKindV1::Disable => 1,
            NixServiceOperationKindV1::Start => 2, NixServiceOperationKindV1::Stop => 3,
            NixServiceOperationKindV1::Restart => 4, NixServiceOperationKindV1::Reload => 5,
        }]);
        Ok(h.finalize().to_hex().to_string())
    }
}

fn canonical_service_unit_v1(unit: &str) -> Result<String, NixServiceOperationErrorV1> {
    if unit.is_empty() { return Err(NixServiceOperationErrorV1::EmptyUnit); }
    if unit.chars().any(|c| c.is_whitespace() || c.is_control()) { return Err(NixServiceOperationErrorV1::WhitespaceOrControl); }
    if unit.contains('/') || unit.contains('\\') { return Err(NixServiceOperationErrorV1::PathLikeUnit); }
    if unit.ends_with(".service") { Ok(unit.to_string()) }
    else if unit.contains('.') { Err(NixServiceOperationErrorV1::NonServiceUnit) }
    else { Ok(format!("{unit}.service")) }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NixServiceOperationErrorV1 {
    #[error("service unit is empty")] EmptyUnit,
    #[error("service operation requires a .service unit")] NonServiceUnit,
    #[error("service unit must not contain path separators")] PathLikeUnit,
    #[error("service unit must not contain whitespace or control characters")] WhitespaceOrControl,
    #[error("service unit has an ambiguous leading or trailing dot")] AmbiguousUnit,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonicalizes_bare_service_name() {
        let op = NixServiceOperationV1::new("nginx", NixServiceOperationKindV1::Enable).unwrap();
        assert_eq!(op.unit(), "nginx.service");
    }
    #[test]
    fn preserves_explicit_service_suffix() {
        let op = NixServiceOperationV1::new("nginx.service", NixServiceOperationKindV1::Restart).unwrap();
        assert_eq!(op.unit(), "nginx.service");
    }
    #[test]
    fn rejects_non_service_suffixes() {
        for unit in ["nginx.socket", "nginx.timer", "nginx.mount"] {
            assert_eq!(NixServiceOperationV1::new(unit, NixServiceOperationKindV1::Start).unwrap_err(), NixServiceOperationErrorV1::NonServiceUnit);
        }
    }
    #[test]
    fn rejects_path_and_shell_like_unit_spelling() {
        for unit in ["/tmp/nginx", "foo/bar", "foo\\bar", "foo bar", "foo\tbar"] {
            assert!(NixServiceOperationV1::new(unit, NixServiceOperationKindV1::Start).is_err());
        }
    }
    #[test]
    fn operation_mutation_changes_digest() {
        let enable = NixServiceOperationV1::new("nginx", NixServiceOperationKindV1::Enable).unwrap();
        let disable = NixServiceOperationV1::new("nginx", NixServiceOperationKindV1::Disable).unwrap();
        assert_ne!(enable.digest().unwrap(), disable.digest().unwrap());
    }
    #[test]
    fn canonical_spelling_has_same_digest() {
        let bare = NixServiceOperationV1::new("nginx", NixServiceOperationKindV1::Start).unwrap();
        let explicit = NixServiceOperationV1::new("nginx.service", NixServiceOperationKindV1::Start).unwrap();
        assert_eq!(bare.digest().unwrap(), explicit.digest().unwrap());
    }
    #[test]
    fn operation_is_domain_data_not_a_command() {
        let op = NixServiceOperationV1::new("nginx", NixServiceOperationKindV1::Start).unwrap();
        let encoded = serde_json::to_string(&op).unwrap();
        assert!(encoded.contains("nginx.service"));
        assert!(!encoded.contains("systemctl"));
    }
}
