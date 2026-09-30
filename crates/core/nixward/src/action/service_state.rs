// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Closed typed evidence for a systemd service pre-state.
//!
//! This module observes and commits to state; it does not execute commands,
//! query the host, authorize an effect, or mint execution authority.

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const SERVICE_STATE_DOMAIN_V1: &[u8] = b"nixward-service-observed-state-v1";
const MAX_SUB_STATE_BYTES: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceActiveStateV1 {
    Active,
    Reloading,
    Inactive,
    Failed,
    Activating,
    Deactivating,
}

impl ServiceActiveStateV1 {
    pub fn parse(value: &str) -> Result<Self, NixServiceStateErrorV1> {
        match value {
            "active" => Ok(Self::Active),
            "reloading" => Ok(Self::Reloading),
            "inactive" => Ok(Self::Inactive),
            "failed" => Ok(Self::Failed),
            "activating" => Ok(Self::Activating),
            "deactivating" => Ok(Self::Deactivating),
            _ => Err(NixServiceStateErrorV1::UnknownActiveState),
        }
    }

    fn discriminant(self) -> u8 {
        match self {
            Self::Active => 0,
            Self::Reloading => 1,
            Self::Inactive => 2,
            Self::Failed => 3,
            Self::Activating => 4,
            Self::Deactivating => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceUnitFileStateV1 {
    Enabled,
    EnabledRuntime,
    Linked,
    LinkedRuntime,
    Masked,
    MaskedRuntime,
    Static,
    Disabled,
    Invalid,
}

impl ServiceUnitFileStateV1 {
    pub fn parse(value: &str) -> Result<Self, NixServiceStateErrorV1> {
        match value {
            "enabled" => Ok(Self::Enabled),
            "enabled-runtime" => Ok(Self::EnabledRuntime),
            "linked" => Ok(Self::Linked),
            "linked-runtime" => Ok(Self::LinkedRuntime),
            "masked" => Ok(Self::Masked),
            "masked-runtime" => Ok(Self::MaskedRuntime),
            "static" => Ok(Self::Static),
            "disabled" => Ok(Self::Disabled),
            "invalid" => Ok(Self::Invalid),
            _ => Err(NixServiceStateErrorV1::UnknownUnitFileState),
        }
    }

    fn discriminant(self) -> u8 {
        match self {
            Self::Enabled => 0,
            Self::EnabledRuntime => 1,
            Self::Linked => 2,
            Self::LinkedRuntime => 3,
            Self::Masked => 4,
            Self::MaskedRuntime => 5,
            Self::Static => 6,
            Self::Disabled => 7,
            Self::Invalid => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixServiceObservedStateV1 {
    unit: String,
    active_state: ServiceActiveStateV1,
    unit_file_state: ServiceUnitFileStateV1,
    sub_state: String,
}

impl NixServiceObservedStateV1 {
    pub fn new(
        unit: impl Into<String>,
        active_state: ServiceActiveStateV1,
        unit_file_state: ServiceUnitFileStateV1,
        sub_state: impl Into<String>,
    ) -> Result<Self, NixServiceStateErrorV1> {
        let unit = crate::action::service_domain::NixServiceOperationV1::new(
            unit.into(),
            crate::action::service_domain::NixServiceOperationKindV1::Start,
        )
        .map_err(|_| NixServiceStateErrorV1::InvalidServiceUnit)?
        .unit()
        .to_string();

        let sub_state = sub_state.into();
        validate_sub_state(&sub_state)?;

        Ok(Self {
            unit,
            active_state,
            unit_file_state,
            sub_state,
        })
    }

    pub fn unit(&self) -> &str {
        &self.unit
    }

    pub fn active_state(&self) -> ServiceActiveStateV1 {
        self.active_state
    }

    pub fn unit_file_state(&self) -> ServiceUnitFileStateV1 {
        self.unit_file_state
    }

    pub fn sub_state(&self) -> &str {
        &self.sub_state
    }

    pub fn validate_shape(&self) -> Result<(), NixServiceStateErrorV1> {
        let canonical = crate::action::service_domain::NixServiceOperationV1::new(
            self.unit.clone(),
            crate::action::service_domain::NixServiceOperationKindV1::Start,
        )
        .map_err(|_| NixServiceStateErrorV1::InvalidServiceUnit)?
        .unit()
        .to_string();

        if canonical != self.unit {
            return Err(NixServiceStateErrorV1::NonCanonicalServiceUnit);
        }

        validate_sub_state(&self.sub_state)
    }

    pub fn digest(&self) -> Result<String, NixServiceStateErrorV1> {
        self.validate_shape()?;

        let mut hasher = Hasher::new();
        hasher.update(SERVICE_STATE_DOMAIN_V1);
        write_len_prefixed(&mut hasher, self.unit.as_bytes());
        hasher.update(&[self.active_state.discriminant()]);
        hasher.update(&[self.unit_file_state.discriminant()]);
        write_len_prefixed(&mut hasher, self.sub_state.as_bytes());
        Ok(hasher.finalize().to_hex().to_string())
    }
}

fn validate_sub_state(value: &str) -> Result<(), NixServiceStateErrorV1> {
    let bytes = value.as_bytes();
    if bytes.is_empty() {
        return Err(NixServiceStateErrorV1::EmptySubState);
    }
    if bytes.len() > MAX_SUB_STATE_BYTES {
        return Err(NixServiceStateErrorV1::SubStateTooLarge);
    }
    if value.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err(NixServiceStateErrorV1::InvalidSubState);
    }
    Ok(())
}

fn write_len_prefixed(hasher: &mut Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_be_bytes());
    hasher.update(value);
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NixServiceStateErrorV1 {
    #[error("unknown systemd ActiveState")]
    UnknownActiveState,
    #[error("unknown systemd UnitFileState")]
    UnknownUnitFileState,
    #[error("invalid service unit")]
    InvalidServiceUnit,
    #[error("service unit is not in canonical form")]
    NonCanonicalServiceUnit,
    #[error("service SubState is empty")]
    EmptySubState,
    #[error("service SubState is too large")]
    SubStateTooLarge,
    #[error("service SubState contains whitespace or control characters")]
    InvalidSubState,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(active: &str, file: &str, sub: &str) -> NixServiceObservedStateV1 {
        NixServiceObservedStateV1::new(
            "nginx",
            ServiceActiveStateV1::parse(active).unwrap(),
            ServiceUnitFileStateV1::parse(file).unwrap(),
            sub,
        )
        .unwrap()
    }

    #[test]
    fn accepts_documented_active_states() {
        for value in ["active", "reloading", "inactive", "failed", "activating", "deactivating"] {
            assert!(ServiceActiveStateV1::parse(value).is_ok());
        }
    }

    #[test]
    fn accepts_documented_unit_file_states() {
        for value in [
            "enabled", "enabled-runtime", "linked", "linked-runtime",
            "masked", "masked-runtime", "static", "disabled", "invalid",
        ] {
            assert!(ServiceUnitFileStateV1::parse(value).is_ok());
        }
    }

    #[test]
    fn unknown_states_fail_closed() {
        assert_eq!(
            ServiceActiveStateV1::parse("future-state").unwrap_err(),
            NixServiceStateErrorV1::UnknownActiveState
        );
        assert_eq!(
            ServiceUnitFileStateV1::parse("future-state").unwrap_err(),
            NixServiceStateErrorV1::UnknownUnitFileState
        );
    }

    #[test]
    fn sub_state_is_bounded_and_strict() {
        assert_eq!(
            NixServiceObservedStateV1::new(
                "nginx",
                ServiceActiveStateV1::Active,
                ServiceUnitFileStateV1::Enabled,
                "",
            )
            .unwrap_err(),
            NixServiceStateErrorV1::EmptySubState
        );

        assert_eq!(
            NixServiceObservedStateV1::new(
                "nginx",
                ServiceActiveStateV1::Active,
                ServiceUnitFileStateV1::Enabled,
                "running state",
            )
            .unwrap_err(),
            NixServiceStateErrorV1::InvalidSubState
        );

        let oversized = "x".repeat(MAX_SUB_STATE_BYTES + 1);
        assert_eq!(
            NixServiceObservedStateV1::new(
                "nginx",
                ServiceActiveStateV1::Active,
                ServiceUnitFileStateV1::Enabled,
                oversized,
            )
            .unwrap_err(),
            NixServiceStateErrorV1::SubStateTooLarge
        );
    }

    #[test]
    fn representative_pre_states_are_admissible() {
        for (active, file, sub) in [
            ("active", "enabled", "running"),
            ("inactive", "disabled", "dead"),
            ("failed", "disabled", "failed"),
            ("reloading", "enabled-runtime", "reload"),
            ("activating", "static", "start"),
            ("deactivating", "masked", "stop"),
        ] {
            let value = state(active, file, sub);
            assert!(value.digest().is_ok());
        }
    }

    #[test]
    fn each_semantic_field_changes_digest() {
        let baseline = state("active", "enabled", "running");
        let unit = NixServiceObservedStateV1::new(
            "sshd",
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
            "running",
        ).unwrap();
        let active = state("failed", "enabled", "running");
        let file = state("active", "disabled", "running");
        let sub = state("active", "enabled", "dead");

        assert_ne!(baseline.digest().unwrap(), unit.digest().unwrap());
        assert_ne!(baseline.digest().unwrap(), active.digest().unwrap());
        assert_ne!(baseline.digest().unwrap(), file.digest().unwrap());
        assert_ne!(baseline.digest().unwrap(), sub.digest().unwrap());
    }

    #[test]
    fn evidence_serialization_contains_no_executor_material() {
        let value = state("active", "enabled", "running");
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(encoded.contains("nginx.service"));
        assert!(!encoded.contains("systemctl"));
        assert!(!encoded.contains("command"));
        assert!(!encoded.contains("executor"));
    }
}
