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
    Maintenance,
    Refreshing,
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
            "maintenance" => Ok(Self::Maintenance),
            "refreshing" => Ok(Self::Refreshing),
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
            Self::Maintenance => 6,
            Self::Refreshing => 7,
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
    Indirect,
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
            "indirect" => Ok(Self::Indirect),
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
            Self::Indirect => 9,
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

    /// Parse the exact three properties requested by the read-only systemd
    /// observation path. Missing, duplicate, malformed, or unknown properties
    /// fail closed; no property is silently projected into a boolean.
    pub fn parse_systemd_properties(
        unit: impl Into<String>,
        properties: &str,
    ) -> Result<Self, NixServiceStateErrorV1> {
        let mut active_state = None;
        let mut sub_state = None;
        let mut unit_file_state = None;

        for line in properties.lines() {
            if line.is_empty() {
                continue;
            }

            let (key, value) = line
                .split_once('=')
                .ok_or(NixServiceStateErrorV1::MalformedPropertyLine)?;

            if value.contains('\0') {
                return Err(NixServiceStateErrorV1::MalformedPropertyLine);
            }

            match key {
                "ActiveState" if active_state.is_none() => {
                    active_state = Some(ServiceActiveStateV1::parse(value)?);
                }
                "SubState" if sub_state.is_none() => {
                    sub_state = Some(value.to_string());
                }
                "UnitFileState" if unit_file_state.is_none() => {
                    unit_file_state = Some(ServiceUnitFileStateV1::parse(value)?);
                }
                "ActiveState" | "SubState" | "UnitFileState" => {
                    return Err(NixServiceStateErrorV1::DuplicateProperty);
                }
                _ => return Err(NixServiceStateErrorV1::UnexpectedProperty),
            }
        }

        Self::new(
            unit,
            active_state.ok_or(NixServiceStateErrorV1::MissingActiveState)?,
            unit_file_state.ok_or(NixServiceStateErrorV1::MissingUnitFileState)?,
            sub_state.ok_or(NixServiceStateErrorV1::MissingSubState)?,
        )
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
    #[error("malformed systemd property line")]
    MalformedPropertyLine,
    #[error("duplicate required systemd property")]
    DuplicateProperty,
    #[error("unexpected systemd property")]
    UnexpectedProperty,
    #[error("ActiveState property is missing")]
    MissingActiveState,
    #[error("SubState property is missing")]
    MissingSubState,
    #[error("UnitFileState property is missing")]
    MissingUnitFileState,
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
    fn parses_complete_systemd_observation() {
        let value = NixServiceObservedStateV1::parse_systemd_properties(
            "nginx",
            "ActiveState=active\nSubState=running\nUnitFileState=enabled\n",
        ).unwrap();
        assert_eq!(value.unit(), "nginx.service");
        assert_eq!(value.active_state(), ServiceActiveStateV1::Active);
        assert_eq!(value.unit_file_state(), ServiceUnitFileStateV1::Enabled);
        assert_eq!(value.sub_state(), "running");
    }

    #[test]
    fn rejects_partial_observation() {
        assert_eq!(
            NixServiceObservedStateV1::parse_systemd_properties(
                "nginx",
                "ActiveState=active\nSubState=running\n",
            ).unwrap_err(),
            NixServiceStateErrorV1::MissingUnitFileState
        );
    }

    #[test]
    fn rejects_duplicate_and_unexpected_properties() {
        assert_eq!(
            NixServiceObservedStateV1::parse_systemd_properties(
                "nginx",
                "ActiveState=active\nActiveState=inactive\nSubState=running\nUnitFileState=enabled\n",
            ).unwrap_err(),
            NixServiceStateErrorV1::DuplicateProperty
        );
        assert_eq!(
            NixServiceObservedStateV1::parse_systemd_properties(
                "nginx",
                "ActiveState=active\nSubState=running\nUnitFileState=enabled\nMainPID=42\n",
            ).unwrap_err(),
            NixServiceStateErrorV1::UnexpectedProperty
        );
    }

    #[test]
    fn rejects_unknown_and_malformed_values() {
        assert_eq!(
            NixServiceObservedStateV1::parse_systemd_properties(
                "nginx",
                "ActiveState=unknown\nSubState=running\nUnitFileState=enabled\n",
            ).unwrap_err(),
            NixServiceStateErrorV1::UnknownActiveState
        );
        assert_eq!(
            NixServiceObservedStateV1::parse_systemd_properties(
                "nginx",
                "ActiveState=active\nSubState=running state\nUnitFileState=enabled\n",
            ).unwrap_err(),
            NixServiceStateErrorV1::InvalidSubState
        );
        assert_eq!(
            NixServiceObservedStateV1::parse_systemd_properties(
                "nginx",
                "ActiveState=active\nSubState=running\n",
            ).unwrap_err(),
            NixServiceStateErrorV1::MissingUnitFileState
        );
    }

    #[test]
    fn representative_states_remain_distinct() {
        let active = NixServiceObservedStateV1::parse_systemd_properties(
            "nginx",
            "ActiveState=active\nSubState=running\nUnitFileState=enabled\n",
        ).unwrap();
        let masked = NixServiceObservedStateV1::parse_systemd_properties(
            "nginx",
            "ActiveState=inactive\nSubState=dead\nUnitFileState=masked\n",
        ).unwrap();
        assert_ne!(active.digest().unwrap(), masked.digest().unwrap());
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
    fn evidence_serialization_contains_no_executor_material() {
        let value = state("active", "enabled", "running");
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(encoded.contains("nginx.service"));
        assert!(!encoded.contains("systemctl"));
        assert!(!encoded.contains("command"));
        assert!(!encoded.contains("executor"));
    }
}
