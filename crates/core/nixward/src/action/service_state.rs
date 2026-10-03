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

#[derive(Debug, PartialEq, Eq)]
pub struct NixServiceOperationCapabilitiesV1 {
    can_start: bool,
    can_stop: bool,
    can_reload: bool,
    pre_state_digest: String,
}

impl NixServiceOperationCapabilitiesV1 {
    pub fn from_observed_state(
        state: &NixServiceObservedStateV1,
        can_start: bool,
        can_stop: bool,
        can_reload: bool,
    ) -> Result<Self, NixServiceStateErrorV1> {
        let pre_state_digest = state.digest()?;
        Ok(Self { can_start, can_stop, can_reload, pre_state_digest })
    }

    pub fn can_start(&self) -> bool { self.can_start }
    pub fn can_stop(&self) -> bool { self.can_stop }
    pub fn can_reload(&self) -> bool { self.can_reload }
    pub fn pre_state_digest(&self) -> &str { &self.pre_state_digest }

    pub fn validate_shape(&self) -> Result<(), NixServiceStateErrorV1> {
        if self.pre_state_digest.len() != 64
            || !self.pre_state_digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(NixServiceStateErrorV1::InvalidPreStateDigest);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, NixServiceStateErrorV1> {
        self.validate_shape()?;
        let mut hasher = Hasher::new();
        hasher.update(b"nixward-service-operation-capabilities-v1");
        hasher.update(&[self.can_start as u8, self.can_stop as u8, self.can_reload as u8]);
        write_len_prefixed(&mut hasher, self.pre_state_digest.as_bytes());
        Ok(hasher.finalize().to_hex().to_string())
    }
}


/// Closed evidence for unit-file enablement state.
///
/// This is intentionally separate from lifecycle capability facts: systemd
/// treats enable/disable as unit-file configuration operations, not as
/// start/stop/reload lifecycle capabilities. The value is observational only
/// and remains bound to the exact pre-state from which it was derived.
#[derive(Debug, PartialEq, Eq)]
pub struct NixServiceEnablementEvidenceV1 {
    unit: String,
    unit_file_state: ServiceUnitFileStateV1,
    pre_state_digest: String,
}

impl NixServiceEnablementEvidenceV1 {
    pub fn from_observed_state(
        state: &NixServiceObservedStateV1,
    ) -> Result<Self, NixServiceStateErrorV1> {
        let pre_state_digest = state.digest()?;
        Ok(Self {
            unit: state.unit().to_string(),
            unit_file_state: state.unit_file_state(),
            pre_state_digest,
        })
    }

    pub fn unit(&self) -> &str { &self.unit }
    pub fn unit_file_state(&self) -> ServiceUnitFileStateV1 { self.unit_file_state }
    pub fn pre_state_digest(&self) -> &str { &self.pre_state_digest }

    pub fn validate_shape(&self) -> Result<(), NixServiceStateErrorV1> {
        let canonical = canonical_service_unit(self.unit.clone())?;
        if canonical != self.unit {
            return Err(NixServiceStateErrorV1::NonCanonicalServiceUnit);
        }
        if self.pre_state_digest.len() != 64
            || !self.pre_state_digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(NixServiceStateErrorV1::InvalidPreStateDigest);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, NixServiceStateErrorV1> {
        self.validate_shape()?;
        let mut hasher = Hasher::new();
        hasher.update(b"nixward-service-enablement-evidence-v1");
        write_len_prefixed(&mut hasher, self.unit.as_bytes());
        hasher.update(&[self.unit_file_state.discriminant()]);
        write_len_prefixed(&mut hasher, self.pre_state_digest.as_bytes());
        Ok(hasher.finalize().to_hex().to_string())
    }
}


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
pub enum ServiceLoadStateV1 {
    Stub,
    Loaded,
    NotFound,
    BadSetting,
    Error,
    Merged,
    Masked,
}

impl ServiceLoadStateV1 {
    pub fn parse(value: &str) -> Result<Self, NixServiceStateErrorV1> {
        match value {
            "stub" => Ok(Self::Stub),
            "loaded" => Ok(Self::Loaded),
            "not-found" => Ok(Self::NotFound),
            "bad-setting" => Ok(Self::BadSetting),
            "error" => Ok(Self::Error),
            "merged" => Ok(Self::Merged),
            "masked" => Ok(Self::Masked),
            _ => Err(NixServiceStateErrorV1::UnknownLoadState),
        }
    }

    fn discriminant(self) -> u8 {
        match self {
            Self::Stub => 0,
            Self::Loaded => 1,
            Self::NotFound => 2,
            Self::BadSetting => 3,
            Self::Error => 4,
            Self::Merged => 5,
            Self::Masked => 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceUnitFileStateV1 {
    Enabled,
    EnabledRuntime,
    Linked,
    LinkedRuntime,
    Alias,
    Masked,
    MaskedRuntime,
    Static,
    Disabled,
    Indirect,
    Generated,
    Transient,
    Bad,
}

impl ServiceUnitFileStateV1 {
    pub fn parse(value: &str) -> Result<Self, NixServiceStateErrorV1> {
        match value {
            "enabled" => Ok(Self::Enabled),
            "enabled-runtime" => Ok(Self::EnabledRuntime),
            "linked" => Ok(Self::Linked),
            "linked-runtime" => Ok(Self::LinkedRuntime),
            "alias" => Ok(Self::Alias),
            "masked" => Ok(Self::Masked),
            "masked-runtime" => Ok(Self::MaskedRuntime),
            "static" => Ok(Self::Static),
            "disabled" => Ok(Self::Disabled),
            "indirect" => Ok(Self::Indirect),
            "generated" => Ok(Self::Generated),
            "transient" => Ok(Self::Transient),
            "bad" => Ok(Self::Bad),
            _ => Err(NixServiceStateErrorV1::UnknownUnitFileState),
        }
    }

    fn discriminant(self) -> u8 {
        match self {
            Self::Enabled => 0,
            Self::EnabledRuntime => 1,
            Self::Linked => 2,
            Self::LinkedRuntime => 3,
            Self::Alias => 4,
            Self::Masked => 5,
            Self::MaskedRuntime => 6,
            Self::Static => 7,
            Self::Disabled => 8,
            Self::Indirect => 9,
            Self::Generated => 10,
            Self::Transient => 11,
            Self::Bad => 12,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NixServiceObservedStateV1 {
    unit: String,
    load_state: ServiceLoadStateV1,
    active_state: ServiceActiveStateV1,
    unit_file_state: ServiceUnitFileStateV1,
    sub_state: String,
}

impl NixServiceObservedStateV1 {
    pub fn new(
        unit: impl Into<String>,
        load_state: ServiceLoadStateV1,
        active_state: ServiceActiveStateV1,
        unit_file_state: ServiceUnitFileStateV1,
        sub_state: impl Into<String>,
    ) -> Result<Self, NixServiceStateErrorV1> {
        let unit = canonical_service_unit(unit.into())?;
        let sub_state = sub_state.into();
        validate_sub_state(&sub_state)?;

        Ok(Self {
            unit,
            load_state,
            active_state,
            unit_file_state,
            sub_state,
        })
    }

    /// Parse the exact five properties used for the governed pre-state.
    /// Capability facts are parsed only by the complete-observation parser.
    pub fn parse_systemd_properties(
        requested_unit: impl Into<String>,
        properties: &str,
    ) -> Result<Self, NixServiceStateErrorV1> {
        Self::parse_state_properties(requested_unit, properties)
    }

    fn parse_state_properties(
        requested_unit: impl Into<String>,
        properties: &str,
    ) -> Result<Self, NixServiceStateErrorV1> {
        let requested_unit = canonical_service_unit(requested_unit.into())?;
        let mut observed_id = None;
        let mut load_state = None;
        let mut active_state = None;
        let mut sub_state = None;
        let mut unit_file_state = None;

        for line in properties.lines() {
            if line.is_empty() { continue; }
            let (key, value) = line.split_once('=')
                .ok_or(NixServiceStateErrorV1::MalformedPropertyLine)?;
            if value.chars().any(|c| c == '\0') {
                return Err(NixServiceStateErrorV1::MalformedPropertyLine);
            }
            match key {
                "Id" if observed_id.is_none() => observed_id = Some(value.to_string()),
                "LoadState" if load_state.is_none() => load_state = Some(ServiceLoadStateV1::parse(value)?),
                "ActiveState" if active_state.is_none() => active_state = Some(ServiceActiveStateV1::parse(value)?),
                "SubState" if sub_state.is_none() => sub_state = Some(value.to_string()),
                "UnitFileState" if unit_file_state.is_none() => unit_file_state = Some(ServiceUnitFileStateV1::parse(value)?),
                "Id" | "LoadState" | "ActiveState" | "SubState" | "UnitFileState" => {
                    return Err(NixServiceStateErrorV1::DuplicateProperty);
                }
                _ => return Err(NixServiceStateErrorV1::UnexpectedProperty),
            }
        }

        let observed_id = observed_id.ok_or(NixServiceStateErrorV1::MissingId)?;
        if observed_id != requested_unit {
            return Err(NixServiceStateErrorV1::IdentityMismatch);
        }

        Self::new(
            requested_unit,
            load_state.ok_or(NixServiceStateErrorV1::MissingLoadState)?,
            active_state.ok_or(NixServiceStateErrorV1::MissingActiveState)?,
            unit_file_state.ok_or(NixServiceStateErrorV1::MissingUnitFileState)?,
            sub_state.ok_or(NixServiceStateErrorV1::MissingSubState)?,
        )
    }

    /// Parse the complete governed observation atomically. Capability facts are
    /// bound to the exact pre-state digest so they cannot be replayed against a
    /// different state observation.
    pub fn parse_systemd_observation(
        requested_unit: impl Into<String>,
        properties: &str,
    ) -> Result<(Self, NixServiceOperationCapabilitiesV1), NixServiceStateErrorV1> {
        let requested_unit = canonical_service_unit(requested_unit.into())?;
        let mut observed_id = None;
        let mut load_state = None;
        let mut active_state = None;
        let mut sub_state = None;
        let mut unit_file_state = None;
        let mut can_start = None;
        let mut can_stop = None;
        let mut can_reload = None;

        for line in properties.lines() {
            if line.is_empty() { continue; }
            let (key, value) = line.split_once('=')
                .ok_or(NixServiceStateErrorV1::MalformedPropertyLine)?;
            if value.chars().any(|c| c == '\0') {
                return Err(NixServiceStateErrorV1::MalformedPropertyLine);
            }
            match key {
                "Id" if observed_id.is_none() => observed_id = Some(value.to_string()),
                "LoadState" if load_state.is_none() => load_state = Some(ServiceLoadStateV1::parse(value)?),
                "ActiveState" if active_state.is_none() => active_state = Some(ServiceActiveStateV1::parse(value)?),
                "SubState" if sub_state.is_none() => sub_state = Some(value.to_string()),
                "UnitFileState" if unit_file_state.is_none() => unit_file_state = Some(ServiceUnitFileStateV1::parse(value)?),
                "CanStart" if can_start.is_none() => can_start = Some(parse_yes_no(value)?),
                "CanStop" if can_stop.is_none() => can_stop = Some(parse_yes_no(value)?),
                "CanReload" if can_reload.is_none() => can_reload = Some(parse_yes_no(value)?),
                "Id" | "LoadState" | "ActiveState" | "SubState" | "UnitFileState"
                | "CanStart" | "CanStop" | "CanReload" => return Err(NixServiceStateErrorV1::DuplicateProperty),
                _ => return Err(NixServiceStateErrorV1::UnexpectedProperty),
            }
        }

        let observed_id = observed_id.ok_or(NixServiceStateErrorV1::MissingId)?;
        if observed_id != requested_unit { return Err(NixServiceStateErrorV1::IdentityMismatch); }

        let state = Self::new(
            requested_unit,
            load_state.ok_or(NixServiceStateErrorV1::MissingLoadState)?,
            active_state.ok_or(NixServiceStateErrorV1::MissingActiveState)?,
            unit_file_state.ok_or(NixServiceStateErrorV1::MissingUnitFileState)?,
            sub_state.ok_or(NixServiceStateErrorV1::MissingSubState)?,
        )?;
        let capabilities = NixServiceOperationCapabilitiesV1::from_observed_state(
            &state,
            can_start.ok_or(NixServiceStateErrorV1::MissingCanStart)?,
            can_stop.ok_or(NixServiceStateErrorV1::MissingCanStop)?,
            can_reload.ok_or(NixServiceStateErrorV1::MissingCanReload)?,
        )?;
        Ok((state, capabilities))
    }

    pub fn unit(&self) -> &str {
        &self.unit
    }

    pub fn load_state(&self) -> ServiceLoadStateV1 {
        self.load_state
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
        let canonical = canonical_service_unit(self.unit.clone())?;
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
        hasher.update(&[self.load_state.discriminant()]);
        hasher.update(&[self.active_state.discriminant()]);
        hasher.update(&[self.unit_file_state.discriminant()]);
        write_len_prefixed(&mut hasher, self.sub_state.as_bytes());
        Ok(hasher.finalize().to_hex().to_string())
    }
}

fn canonical_service_unit(unit: String) -> Result<String, NixServiceStateErrorV1> {
    crate::action::service_domain::NixServiceOperationV1::new(
        unit,
        crate::action::service_domain::NixServiceOperationKindV1::Start,
    )
    .map_err(|_| NixServiceStateErrorV1::InvalidServiceUnit)
    .map(|operation| operation.unit().to_string())
}

fn parse_yes_no(value: &str) -> Result<bool, NixServiceStateErrorV1> {
    match value {
        "yes" => Ok(true),
        "no" => Ok(false),
        _ => Err(NixServiceStateErrorV1::InvalidCapabilityValue),
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
    #[error("unknown systemd LoadState")]
    UnknownLoadState,
    #[error("unknown systemd ActiveState")]
    UnknownActiveState,
    #[error("unknown systemd UnitFileState")]
    UnknownUnitFileState,
    #[error("invalid systemd yes/no capability value")]
    InvalidCapabilityValue,
    #[error("systemd CanStart property is missing")]
    MissingCanStart,
    #[error("systemd CanStop property is missing")]
    MissingCanStop,
    #[error("systemd CanReload property is missing")]
    MissingCanReload,
    #[error("invalid pre-state digest")]
    InvalidPreStateDigest,
    #[error("invalid service unit")]
    InvalidServiceUnit,
    #[error("systemd Id does not match the requested canonical service unit")]
    IdentityMismatch,
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
    #[error("required systemd Id property is missing")]
    MissingId,
    #[error("LoadState property is missing")]
    MissingLoadState,
    #[error("ActiveState property is missing")]
    MissingActiveState,
    #[error("SubState property is missing")]
    MissingSubState,
    #[error("UnitFileState property is missing")]
    MissingUnitFileState,
    #[error("duplicate required systemd property")]
    DuplicateProperty,
    #[error("unexpected systemd property")]
    UnexpectedProperty,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(active: &str, file: &str, sub: &str) -> NixServiceObservedStateV1 {
        NixServiceObservedStateV1::new(
            "nginx",
            ServiceLoadStateV1::Loaded,
            ServiceActiveStateV1::parse(active).unwrap(),
            ServiceUnitFileStateV1::parse(file).unwrap(),
            sub,
        )
        .unwrap()
    }

    #[test]
    fn parses_operation_capability_facts_and_binds_pre_state() {
        let (state, capabilities) = NixServiceObservedStateV1::parse_systemd_observation(
            "nginx",
            "Id=nginx.service
LoadState=loaded
ActiveState=active
SubState=running
UnitFileState=enabled
CanStart=yes
CanStop=yes
CanReload=no
",
        )
        .unwrap();
        assert!(capabilities.can_start());
        assert!(capabilities.can_stop());
        assert!(!capabilities.can_reload());
        assert_eq!(capabilities.pre_state_digest(), state.digest().unwrap());
        assert_ne!(capabilities.digest().unwrap(), "");
    }

    #[test]
    fn capability_facts_change_digest_without_changing_pre_state_digest() {
        let state = state("active", "enabled", "running");
        let a = NixServiceOperationCapabilitiesV1::from_observed_state(&state, true, true, true).unwrap();
        let b = NixServiceOperationCapabilitiesV1::from_observed_state(&state, true, false, true).unwrap();
        assert_eq!(a.pre_state_digest(), b.pre_state_digest());
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
    }

    #[test]
    fn capability_binding_changes_when_pre_state_changes() {
        let active = state("active", "enabled", "running");
        let failed = state("failed", "enabled", "running");
        let a = NixServiceOperationCapabilitiesV1::from_observed_state(&active, true, true, true).unwrap();
        let b = NixServiceOperationCapabilitiesV1::from_observed_state(&failed, true, true, true).unwrap();
        assert_ne!(a.pre_state_digest(), b.pre_state_digest());
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
    }

    #[test]
    fn legacy_state_parser_does_not_accept_capability_fields() {
        let error = NixServiceObservedStateV1::parse_systemd_properties(
            "nginx",
            "Id=nginx.service
LoadState=loaded
ActiveState=active
SubState=running
UnitFileState=enabled
CanStart=yes
",
        )
        .unwrap_err();
        assert_eq!(error, NixServiceStateErrorV1::UnexpectedProperty);
    }

    #[test]
    fn rejects_unknown_or_missing_operation_capability_values() {
        assert_eq!(
            NixServiceObservedStateV1::parse_systemd_observation(
                "nginx",
                "Id=nginx.service
LoadState=loaded
ActiveState=active
SubState=running
UnitFileState=enabled
CanStart=maybe
CanStop=yes
CanReload=yes
",
            )
            .unwrap_err(),
            NixServiceStateErrorV1::InvalidCapabilityValue
        );
        assert_eq!(
            NixServiceObservedStateV1::parse_systemd_observation(
                "nginx",
                "Id=nginx.service
LoadState=loaded
ActiveState=active
SubState=running
UnitFileState=enabled
CanStart=yes
CanStop=yes
",
            )
            .unwrap_err(),
            NixServiceStateErrorV1::MissingCanReload
        );
    }

    #[test]
    fn parses_complete_systemd_observation() {
        let (value, capabilities) = NixServiceObservedStateV1::parse_systemd_observation(
            "nginx",
            "Id=nginx.service
LoadState=loaded
ActiveState=active
SubState=running
UnitFileState=enabled
CanStart=yes
CanStop=no
CanReload=yes
",
        )
        .unwrap();
        assert_eq!(value.unit(), "nginx.service");
        assert_eq!(value.load_state(), ServiceLoadStateV1::Loaded);
        assert_eq!(value.active_state(), ServiceActiveStateV1::Active);
        assert_eq!(value.unit_file_state(), ServiceUnitFileStateV1::Enabled);
        assert_eq!(value.sub_state(), "running");
        assert!(capabilities.can_start());
        assert!(!capabilities.can_stop());
        assert!(capabilities.can_reload());
        assert_eq!(capabilities.pre_state_digest(), value.digest().unwrap());
    }

    #[test]
    fn rejects_partial_observation() {
        assert_eq!(
            NixServiceObservedStateV1::parse_systemd_properties(
                "nginx",
                "Id=nginx.service
LoadState=loaded
ActiveState=active
SubState=running
",
            )
            .unwrap_err(),
            NixServiceStateErrorV1::MissingUnitFileState
        );
    }

    #[test]
    fn rejects_duplicate_and_unexpected_properties() {
        assert_eq!(
            NixServiceObservedStateV1::parse_systemd_properties(
                "nginx",
                "Id=nginx.service
LoadState=loaded
ActiveState=active
ActiveState=inactive
SubState=running
UnitFileState=enabled
CanStart=yes
CanStop=yes
CanReload=yes
",
            )
            .unwrap_err(),
            NixServiceStateErrorV1::DuplicateProperty
        );
        assert_eq!(
            NixServiceObservedStateV1::parse_systemd_properties(
                "nginx",
                "Id=nginx.service
LoadState=loaded
ActiveState=active
SubState=running
UnitFileState=enabled
CanStart=yes
CanStop=yes
CanReload=yes
MainPID=42
",
            )
            .unwrap_err(),
            NixServiceStateErrorV1::UnexpectedProperty
        );
    }

    #[test]
    fn rejects_unknown_and_malformed_values() {
        assert_eq!(
            NixServiceObservedStateV1::parse_systemd_properties(
                "nginx",
                "Id=nginx.service
LoadState=loaded
ActiveState=unknown
SubState=running
UnitFileState=enabled
CanStart=yes
CanStop=yes
CanReload=yes
",
            )
            .unwrap_err(),
            NixServiceStateErrorV1::UnknownActiveState
        );
        assert_eq!(
            NixServiceObservedStateV1::parse_systemd_properties(
                "nginx",
                "Id=nginx.service
LoadState=loaded
ActiveState=active
SubState=running state
UnitFileState=enabled
CanStart=yes
CanStop=yes
CanReload=yes
",
            )
            .unwrap_err(),
            NixServiceStateErrorV1::InvalidSubState
        );
        assert_eq!(
            NixServiceObservedStateV1::parse_systemd_properties(
                "nginx",
                "Id=nginx.service
ActiveState=active
SubState=running
UnitFileState=enabled
CanStart=yes
CanStop=yes
CanReload=yes
",
            )
            .unwrap_err(),
            NixServiceStateErrorV1::MissingLoadState
        );
    }

    #[test]
    fn parses_all_documented_systemctl_load_states() {
        let cases = [
            ("stub", ServiceLoadStateV1::Stub),
            ("loaded", ServiceLoadStateV1::Loaded),
            ("not-found", ServiceLoadStateV1::NotFound),
            ("bad-setting", ServiceLoadStateV1::BadSetting),
            ("error", ServiceLoadStateV1::Error),
            ("merged", ServiceLoadStateV1::Merged),
            ("masked", ServiceLoadStateV1::Masked),
        ];

        for (raw, expected) in cases {
            assert_eq!(ServiceLoadStateV1::parse(raw).unwrap(), expected);
        }
    }

    #[test]
    fn rejects_identity_mismatch_and_alias_substitution() {
        assert_eq!(
            NixServiceObservedStateV1::parse_systemd_properties(
                "nginx",
                "Id=nginx-alias.service
LoadState=loaded
ActiveState=active
SubState=running
UnitFileState=enabled
CanStart=yes
CanStop=yes
CanReload=yes
",
            )
            .unwrap_err(),
            NixServiceStateErrorV1::IdentityMismatch
        );
    }

    #[test]
    fn rejects_unknown_load_state_and_missing_identity() {
        assert_eq!(
            NixServiceObservedStateV1::parse_systemd_properties(
                "nginx",
                "Id=nginx.service
LoadState=future
ActiveState=active
SubState=running
UnitFileState=enabled
CanStart=yes
CanStop=yes
CanReload=yes
",
            )
            .unwrap_err(),
            NixServiceStateErrorV1::UnknownLoadState
        );
        assert_eq!(
            NixServiceObservedStateV1::parse_systemd_properties(
                "nginx",
                "LoadState=loaded
ActiveState=active
SubState=running
UnitFileState=enabled
CanStart=yes
CanStop=yes
CanReload=yes
",
            )
            .unwrap_err(),
            NixServiceStateErrorV1::MissingId
        );
    }

    #[test]
    fn parses_full_current_systemd_unit_file_state_vocabulary() {
        let cases = [
            ("enabled", ServiceUnitFileStateV1::Enabled),
            ("enabled-runtime", ServiceUnitFileStateV1::EnabledRuntime),
            ("linked", ServiceUnitFileStateV1::Linked),
            ("linked-runtime", ServiceUnitFileStateV1::LinkedRuntime),
            ("alias", ServiceUnitFileStateV1::Alias),
            ("masked", ServiceUnitFileStateV1::Masked),
            ("masked-runtime", ServiceUnitFileStateV1::MaskedRuntime),
            ("static", ServiceUnitFileStateV1::Static),
            ("disabled", ServiceUnitFileStateV1::Disabled),
            ("indirect", ServiceUnitFileStateV1::Indirect),
            ("generated", ServiceUnitFileStateV1::Generated),
            ("transient", ServiceUnitFileStateV1::Transient),
            ("bad", ServiceUnitFileStateV1::Bad),
        ];
        for (raw, expected) in cases {
            assert_eq!(ServiceUnitFileStateV1::parse(raw).unwrap(), expected);
        }
    }

    #[test]
    fn every_unit_file_state_has_a_distinct_digest_commitment() {
        let states = [
            ServiceUnitFileStateV1::Enabled,
            ServiceUnitFileStateV1::EnabledRuntime,
            ServiceUnitFileStateV1::Linked,
            ServiceUnitFileStateV1::LinkedRuntime,
            ServiceUnitFileStateV1::Alias,
            ServiceUnitFileStateV1::Masked,
            ServiceUnitFileStateV1::MaskedRuntime,
            ServiceUnitFileStateV1::Static,
            ServiceUnitFileStateV1::Disabled,
            ServiceUnitFileStateV1::Indirect,
            ServiceUnitFileStateV1::Generated,
            ServiceUnitFileStateV1::Transient,
            ServiceUnitFileStateV1::Bad,
        ];
        let mut digests = std::collections::BTreeSet::new();
        for file_state in states {
            let value = state("inactive", "disabled", "dead");
            let value = NixServiceObservedStateV1::new(
                value.unit().to_string(),
                value.load_state(),
                value.active_state(),
                file_state,
                value.sub_state().to_string(),
            ).unwrap();
            assert!(digests.insert(value.digest().unwrap()));
        }
        assert_eq!(digests.len(), states.len());
    }

    #[test]
    fn every_load_state_has_a_distinct_digest_commitment() {
        let states = [
            ServiceLoadStateV1::Stub,
            ServiceLoadStateV1::Loaded,
            ServiceLoadStateV1::NotFound,
            ServiceLoadStateV1::BadSetting,
            ServiceLoadStateV1::Error,
            ServiceLoadStateV1::Merged,
            ServiceLoadStateV1::Masked,
        ];

        let mut digests = std::collections::BTreeSet::new();
        for load_state in states {
            let value = NixServiceObservedStateV1::new(
                "nginx",
                load_state,
                ServiceActiveStateV1::Inactive,
                ServiceUnitFileStateV1::Disabled,
                "dead",
            )
            .unwrap();
            assert!(digests.insert(value.digest().unwrap()));
        }
        assert_eq!(digests.len(), states.len());
    }

    #[test]
    fn load_state_changes_digest() {
        let loaded = state("active", "enabled", "running");
        let masked = NixServiceObservedStateV1::new(
            "nginx",
            ServiceLoadStateV1::Masked,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
            "running",
        )
        .unwrap();
        assert_ne!(loaded.digest().unwrap(), masked.digest().unwrap());
    }

    #[test]
    fn representative_states_remain_distinct() {
        let active = state("active", "enabled", "running");
        let masked = NixServiceObservedStateV1::new(
            "nginx",
            ServiceLoadStateV1::Loaded,
            ServiceActiveStateV1::Inactive,
            ServiceUnitFileStateV1::Masked,
            "dead",
        )
        .unwrap();
        assert_ne!(active.digest().unwrap(), masked.digest().unwrap());
    }

    #[test]
    fn each_semantic_field_changes_digest() {
        let baseline = state("active", "enabled", "running");
        let unit = NixServiceObservedStateV1::new(
            "sshd",
            ServiceLoadStateV1::Loaded,
            ServiceActiveStateV1::Active,
            ServiceUnitFileStateV1::Enabled,
            "running",
        )
        .unwrap();
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
                ServiceLoadStateV1::Loaded,
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
                ServiceLoadStateV1::Loaded,
                ServiceActiveStateV1::Active,
                ServiceUnitFileStateV1::Enabled,
                oversized,
            )
            .unwrap_err(),
            NixServiceStateErrorV1::SubStateTooLarge
        );
    }

    #[test]
    fn enablement_evidence_is_distinct_from_lifecycle_capabilities() {
        let observed = state("active", "enabled", "running");
        let enablement = NixServiceEnablementEvidenceV1::from_observed_state(&observed).unwrap();
        let capabilities = NixServiceOperationCapabilitiesV1::from_observed_state(&observed, true, true, true).unwrap();

        assert_eq!(enablement.unit(), "nginx.service");
        assert_eq!(enablement.unit_file_state(), ServiceUnitFileStateV1::Enabled);
        assert_eq!(enablement.pre_state_digest(), observed.digest().unwrap());
        assert_ne!(enablement.digest().unwrap(), capabilities.digest().unwrap());
    }

    #[test]
    fn enablement_state_changes_digest() {
        let disabled = state("inactive", "disabled", "dead");
        let enabled = state("inactive", "enabled", "dead");
        let a = NixServiceEnablementEvidenceV1::from_observed_state(&disabled).unwrap();
        let b = NixServiceEnablementEvidenceV1::from_observed_state(&enabled).unwrap();
        assert_ne!(a.pre_state_digest(), b.pre_state_digest());
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
    }

    #[test]
    fn runtime_and_persistent_enablement_states_remain_distinct() {
        let persistent = state("inactive", "enabled", "dead");
        let runtime = state("inactive", "enabled-runtime", "dead");
        let a = NixServiceEnablementEvidenceV1::from_observed_state(&persistent).unwrap();
        let b = NixServiceEnablementEvidenceV1::from_observed_state(&runtime).unwrap();
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
    }

    #[test]
    fn masked_and_disabled_enablement_states_remain_distinct() {
        let masked = state("inactive", "masked", "dead");
        let disabled = state("inactive", "disabled", "dead");
        let a = NixServiceEnablementEvidenceV1::from_observed_state(&masked).unwrap();
        let b = NixServiceEnablementEvidenceV1::from_observed_state(&disabled).unwrap();
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
    }

    #[test]
    fn enablement_evidence_commits_to_exact_unit_identity() {
        let nginx = state("inactive", "disabled", "dead");
        let sshd = NixServiceObservedStateV1::new(
            "sshd",
            ServiceLoadStateV1::Loaded,
            ServiceActiveStateV1::Inactive,
            ServiceUnitFileStateV1::Disabled,
            "dead",
        )
        .unwrap();
        let a = NixServiceEnablementEvidenceV1::from_observed_state(&nginx).unwrap();
        let b = NixServiceEnablementEvidenceV1::from_observed_state(&sshd).unwrap();
        assert_ne!(a.pre_state_digest(), b.pre_state_digest());
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
    }

    #[test]
    fn enablement_evidence_is_observation_not_operation_authorization() {
        let observed = state("inactive", "disabled", "dead");
        let evidence = NixServiceEnablementEvidenceV1::from_observed_state(&observed).unwrap();
        let encoded = format!("{:?}", evidence);
        assert!(!encoded.contains("executor"));
        assert!(!encoded.contains("authorization"));
        assert!(!encoded.contains("DispatchPermit"));
    }

    #[test]
    fn evidence_serialization_contains_no_executor_material() {
        let value = state("active", "enabled", "running");
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(encoded.contains("nginx.service"));
        assert!(encoded.contains("Loaded"));
        assert!(!encoded.contains("systemctl"));
        assert!(!encoded.contains("command"));
        assert!(!encoded.contains("executor"));
    }
}
