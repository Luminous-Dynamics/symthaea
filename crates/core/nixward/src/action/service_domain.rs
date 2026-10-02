// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Closed typed domain for the first future Nixward effect-binding slice.
//! Domain data only: no command conversion, executor handle, or authority.

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const SERVICE_OPERATION_DOMAIN_V1: &[u8] = b"nixward-service-operation-v1";
const MAX_SERVICE_UNIT_BYTES_V1: usize = 255;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum NixServiceOperationKindV1 { Enable, Disable, Start, Stop, Restart, Reload }

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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
        validate_service_unit_shape_v1(&self.unit)
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
    if unit.is_empty() {
        return Err(NixServiceOperationErrorV1::EmptyUnit);
    }
    if unit.ends_with(".service") {
        validate_service_unit_shape_v1(unit)?;
        return Ok(unit.to_string());
    }
    if unit.contains('.') {
        return Err(NixServiceOperationErrorV1::NonServiceUnit);
    }
    let canonical = format!("{unit}.service");
    validate_service_unit_shape_v1(&canonical)?;
    Ok(canonical)
}

pub(crate) fn validate_canonical_service_operation_v1(
    unit: &str,
    operation: NixServiceOperationKindV1,
) -> Result<(), NixServiceOperationErrorV1> {
    let typed = NixServiceOperationV1::new(unit.to_string(), operation)?;
    if typed.unit() != unit {
        return Err(NixServiceOperationErrorV1::NonCanonicalUnit);
    }
    Ok(())
}

fn validate_service_unit_shape_v1(unit: &str) -> Result<(), NixServiceOperationErrorV1> {
    if unit.is_empty() {
        return Err(NixServiceOperationErrorV1::EmptyUnit);
    }
    if unit.len() > MAX_SERVICE_UNIT_BYTES_V1 {
        return Err(NixServiceOperationErrorV1::TooLong);
    }
    if !unit.is_ascii() {
        return Err(NixServiceOperationErrorV1::InvalidCharacter);
    }
    if unit.starts_with('-') {
        return Err(NixServiceOperationErrorV1::OptionLikeUnit);
    }
    if unit.starts_with('.') || unit.ends_with('.') {
        return Err(NixServiceOperationErrorV1::AmbiguousUnit);
    }
    if !unit.ends_with(".service") {
        return Err(NixServiceOperationErrorV1::NonServiceUnit);
    }
    let name = unit.strip_suffix(".service").unwrap_or(unit);
    if let Some(at) = name.find('@') {
        if at == 0 || name[at + 1..].contains('@') {
            return Err(NixServiceOperationErrorV1::InvalidCharacter);
        }
    }
    if unit.contains('/') || unit.contains('\\') {
        return Err(NixServiceOperationErrorV1::PathLikeUnit);
    }
    if unit.bytes().any(|byte| byte.is_ascii_whitespace() || byte.is_ascii_control()) {
        return Err(NixServiceOperationErrorV1::WhitespaceOrControl);
    }
    if !unit.bytes().all(|byte| {
        byte.is_ascii_alphanumeric()
            || matches!(byte, b':' | b'-' | b'_' | b'.' | b'@')
    }) {
        return Err(NixServiceOperationErrorV1::InvalidCharacter);
    }
    Ok(())
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NixServiceOperationErrorV1 {
    #[error("service unit is empty")] EmptyUnit,
    #[error("service operation requires a .service unit")] NonServiceUnit,
    #[error("service unit must not contain path separators")] PathLikeUnit,
    #[error("service unit must not contain whitespace or control characters")] WhitespaceOrControl,
    #[error("service unit has an ambiguous leading or trailing dot")] AmbiguousUnit,
    #[error("service unit exceeds the systemd 255-byte maximum")] TooLong,
    #[error("service unit must not be option-like")] OptionLikeUnit,
    #[error("service unit is not in canonical normalized form")] NonCanonicalUnit,
    #[error("service unit contains a character outside the conservative v1 allowlist")] InvalidCharacter,
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
        for unit in [
            "/tmp/nginx",
            "foo/bar",
            "foo\\bar",
            "foo bar",
            "foo\tbar",
            "nginx*.service",
            "nginx.service;reboot",
            "$(reboot).service",
        ] {
            assert!(NixServiceOperationV1::new(unit, NixServiceOperationKindV1::Start).is_err());
        }
    }

    #[test]
    fn accepts_and_rejects_exact_service_unit_length_boundary() {
        let max_name = format!("{}.service", "a".repeat(247));
        assert_eq!(max_name.len(), 255);
        assert!(NixServiceOperationV1::new(max_name, NixServiceOperationKindV1::Start).is_ok());

        let oversized = format!("{}.service", "a".repeat(248));
        assert_eq!(oversized.len(), 256);
        assert_eq!(
            NixServiceOperationV1::new(oversized, NixServiceOperationKindV1::Start).unwrap_err(),
            NixServiceOperationErrorV1::TooLong
        );
    }

    #[test]
    fn rejects_option_like_and_oversized_units() {
        assert_eq!(
            NixServiceOperationV1::new("--now.service", NixServiceOperationKindV1::Start)
                .unwrap_err(),
            NixServiceOperationErrorV1::OptionLikeUnit
        );

        let oversized = format!("{}.service", "a".repeat(248));
        assert_eq!(
            NixServiceOperationV1::new(oversized, NixServiceOperationKindV1::Start).unwrap_err(),
            NixServiceOperationErrorV1::TooLong
        );
    }

    #[test]
    fn accepts_conservative_systemd_service_names() {
        for unit in [
            "nginx.service",
            "foo-bar_2.service",
            "dbus-org.example.service",
            "worker@instance.service",
            "worker@.service",
            "foo:bar.service",
        ] {
            assert!(NixServiceOperationV1::new(unit, NixServiceOperationKindV1::Start).is_ok());
        }
    }
    #[test]
    fn command_boundary_requires_canonical_spelling() {
        assert_eq!(
            validate_canonical_service_operation_v1("nginx", NixServiceOperationKindV1::Start)
                .unwrap_err(),
            NixServiceOperationErrorV1::NonCanonicalUnit
        );
        assert!(
            validate_canonical_service_operation_v1(
                "nginx.service",
                NixServiceOperationKindV1::Start
            )
            .is_ok()
        );
    }

    #[test]
    fn rejects_malformed_instance_markers() {
        for unit in ["@worker.service", "worker@@instance.service", "worker@instance@2.service"] {
            assert_eq!(
                NixServiceOperationV1::new(unit, NixServiceOperationKindV1::Start).unwrap_err(),
                NixServiceOperationErrorV1::InvalidCharacter
            );
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
    fn digest_rejects_invalid_internal_domain_state() {
        let mut op = NixServiceOperationV1::new("nginx", NixServiceOperationKindV1::Start).unwrap();
        op.unit = "../nginx.service".to_string();
        assert_eq!(op.validate_shape().unwrap_err(), NixServiceOperationErrorV1::PathLikeUnit);
        assert_eq!(op.digest().unwrap_err(), NixServiceOperationErrorV1::PathLikeUnit);
    }

    #[test]
    fn operation_is_domain_data_not_a_command() {
        let op = NixServiceOperationV1::new("nginx", NixServiceOperationKindV1::Start).unwrap();
        let encoded = serde_json::to_string(&op).unwrap();
        assert!(encoded.contains("nginx.service"));
        assert!(!encoded.contains("systemctl"));
    }
}
