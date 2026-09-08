// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical applicability profiles for safety-configuration manifests.
//!
//! A configuration manifest can explicitly mark a component `NotApplicable`, but
//! that declaration is trustworthy only when the exact policy deciding
//! applicability is itself canonical and hash-bound. This crate defines that
//! policy artifact without depending on runtime authority or autonomy.

#![deny(unsafe_code)]

pub mod admission;
pub mod authorization;
pub mod commit_preconditions;
pub mod revocation;
pub mod revocation_admission;
pub mod revocation_commit_preconditions;
pub mod transition;
pub mod trusted_time;

use serde::{Deserialize, Serialize};
use symthaea_safety_configuration::{
    ConfigurationComponent, ConfigurationDigest, SafetyConfigurationManifest,
    SafetyConfigurationManifestError,
};
use thiserror::Error;

pub const SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1: &str =
    "symthaea-safety-configuration-profile-v1";
const DOMAIN_SEPARATOR: &[u8] = b"symthaea:safety-configuration-profile:v1\0";

/// Stable safety-configuration component slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ConfigurationField {
    HardwareInventory = 1,
    Firmware = 2,
    SoftwareClosure = 3,
    ElectricalTopology = 4,
    ThermalTopology = 5,
    ProtectionSettings = 6,
    SensorMap = 7,
    ActuatorMap = 8,
    Calibration = 9,
    NetworkTopology = 10,
}

impl ConfigurationField {
    pub const ALL: [Self; 10] = [
        Self::HardwareInventory,
        Self::Firmware,
        Self::SoftwareClosure,
        Self::ElectricalTopology,
        Self::ThermalTopology,
        Self::ProtectionSettings,
        Self::SensorMap,
        Self::ActuatorMap,
        Self::Calibration,
        Self::NetworkTopology,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HardwareInventory => "hardware_inventory",
            Self::Firmware => "firmware",
            Self::SoftwareClosure => "software_closure",
            Self::ElectricalTopology => "electrical_topology",
            Self::ThermalTopology => "thermal_topology",
            Self::ProtectionSettings => "protection_settings",
            Self::SensorMap => "sensor_map",
            Self::ActuatorMap => "actuator_map",
            Self::Calibration => "calibration",
            Self::NetworkTopology => "network_topology",
        }
    }
}

/// Exact applicability rule for one component slot.
///
/// V1 intentionally has no ambiguous `Optional` state. If a component can be
/// absent under some circumstance, that circumstance should have its own profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentRequirement {
    Required,
    NotApplicable,
}

impl ComponentRequirement {
    const fn tag(self) -> u8 {
        match self {
            Self::NotApplicable => 0,
            Self::Required => 1,
        }
    }
}

/// Canonical v1 policy defining exactly which configuration components must carry
/// evidence for a particular class of autonomous node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyConfigurationProfile {
    pub schema_version: String,
    pub profile_id: String,
    pub hardware_inventory: ComponentRequirement,
    pub firmware: ComponentRequirement,
    pub software_closure: ComponentRequirement,
    pub electrical_topology: ComponentRequirement,
    pub thermal_topology: ComponentRequirement,
    pub protection_settings: ComponentRequirement,
    pub sensor_map: ComponentRequirement,
    pub actuator_map: ComponentRequirement,
    pub calibration: ComponentRequirement,
    pub network_topology: ComponentRequirement,
}

impl SafetyConfigurationProfile {
    pub fn validate(&self) -> Result<(), SafetyConfigurationProfileError> {
        if self.schema_version != SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1 {
            return Err(SafetyConfigurationProfileError::UnsupportedSchemaVersion(
                self.schema_version.clone(),
            ));
        }
        if self.profile_id.trim().is_empty() {
            return Err(SafetyConfigurationProfileError::EmptyProfileId);
        }
        if !self
            .requirements()
            .iter()
            .any(|(_, requirement)| *requirement == ComponentRequirement::Required)
        {
            return Err(SafetyConfigurationProfileError::NoRequiredComponents);
        }
        Ok(())
    }

    /// Fixed-order, domain-separated bytes defining the immutable profile artifact.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, SafetyConfigurationProfileError> {
        self.validate()?;
        let mut out = Vec::with_capacity(192);
        out.extend_from_slice(DOMAIN_SEPARATOR);
        push_string(&mut out, "schema_version", &self.schema_version)?;
        push_string(&mut out, "profile_id", &self.profile_id)?;
        for (field, requirement) in self.requirements() {
            out.push(field as u8);
            out.push(requirement.tag());
        }
        Ok(out)
    }

    pub fn digest(&self) -> Result<ConfigurationDigest, SafetyConfigurationProfileError> {
        Ok(ConfigurationDigest::blake3_256(&self.canonical_bytes()?))
    }

    /// Prove that a manifest both binds this exact profile artifact and obeys every
    /// applicability rule in it.
    pub fn validate_manifest(
        &self,
        manifest: &SafetyConfigurationManifest,
    ) -> Result<(), SafetyConfigurationProfileError> {
        self.validate()?;
        manifest.validate()?;

        if manifest.profile_id != self.profile_id {
            return Err(SafetyConfigurationProfileError::ProfileIdMismatch {
                expected: self.profile_id.clone(),
                observed: manifest.profile_id.clone(),
            });
        }

        let expected_digest = self.digest()?;
        if manifest.profile_digest != expected_digest {
            return Err(SafetyConfigurationProfileError::ProfileDigestMismatch {
                expected: expected_digest,
                observed: manifest.profile_digest,
            });
        }

        for ((field, requirement), component) in self
            .requirements()
            .into_iter()
            .zip(manifest_components(manifest))
        {
            match (requirement, component) {
                (ComponentRequirement::Required, ConfigurationComponent::NotApplicable) => {
                    return Err(SafetyConfigurationProfileError::MissingRequiredComponent(
                        field,
                    ));
                }
                (
                    ComponentRequirement::NotApplicable,
                    ConfigurationComponent::Digest(_),
                ) => {
                    return Err(SafetyConfigurationProfileError::UnexpectedComponent(field));
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn requirements(&self) -> [(ConfigurationField, ComponentRequirement); 10] {
        [
            (ConfigurationField::HardwareInventory, self.hardware_inventory),
            (ConfigurationField::Firmware, self.firmware),
            (ConfigurationField::SoftwareClosure, self.software_closure),
            (ConfigurationField::ElectricalTopology, self.electrical_topology),
            (ConfigurationField::ThermalTopology, self.thermal_topology),
            (ConfigurationField::ProtectionSettings, self.protection_settings),
            (ConfigurationField::SensorMap, self.sensor_map),
            (ConfigurationField::ActuatorMap, self.actuator_map),
            (ConfigurationField::Calibration, self.calibration),
            (ConfigurationField::NetworkTopology, self.network_topology),
        ]
    }
}

fn manifest_components(manifest: &SafetyConfigurationManifest) -> [ConfigurationComponent; 10] {
    [
        manifest.hardware_inventory,
        manifest.firmware,
        manifest.software_closure,
        manifest.electrical_topology,
        manifest.thermal_topology,
        manifest.protection_settings,
        manifest.sensor_map,
        manifest.actuator_map,
        manifest.calibration,
        manifest.network_topology,
    ]
}

fn push_string(
    out: &mut Vec<u8>,
    field: &'static str,
    value: &str,
) -> Result<(), SafetyConfigurationProfileError> {
    let len = u32::try_from(value.len())
        .map_err(|_| SafetyConfigurationProfileError::StringTooLong(field))?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyConfigurationProfileError {
    #[error(transparent)]
    Manifest(#[from] SafetyConfigurationManifestError),
    #[error("unsupported safety configuration profile schema version {0}")]
    UnsupportedSchemaVersion(String),
    #[error("safety configuration profile id must not be empty")]
    EmptyProfileId,
    #[error("safety configuration profile must require at least one component")]
    NoRequiredComponents,
    #[error("safety configuration profile string field {0} exceeds canonical u32 length")]
    StringTooLong(&'static str),
    #[error("manifest profile id mismatch: expected {expected}, observed {observed}")]
    ProfileIdMismatch { expected: String, observed: String },
    #[error("manifest profile digest mismatch: expected {expected:?}, observed {observed:?}")]
    ProfileDigestMismatch {
        expected: ConfigurationDigest,
        observed: ConfigurationDigest,
    },
    #[error("required safety configuration component {0:?} is marked NotApplicable")]
    MissingRequiredComponent(ConfigurationField),
    #[error("safety configuration component {0:?} is present but profile marks it NotApplicable")]
    UnexpectedComponent(ConfigurationField),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_safety_configuration::SAFETY_CONFIGURATION_SCHEMA_V1;

    fn digest(byte: u8) -> ConfigurationDigest {
        ConfigurationDigest::Blake3_256([byte; 32])
    }

    fn component(byte: u8) -> ConfigurationComponent {
        ConfigurationComponent::Digest(digest(byte))
    }

    fn profile() -> SafetyConfigurationProfile {
        SafetyConfigurationProfile {
            schema_version: SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1.to_owned(),
            profile_id: "compute-commons-autonomous-node-v1".to_owned(),
            hardware_inventory: ComponentRequirement::Required,
            firmware: ComponentRequirement::Required,
            software_closure: ComponentRequirement::Required,
            electrical_topology: ComponentRequirement::Required,
            thermal_topology: ComponentRequirement::Required,
            protection_settings: ComponentRequirement::Required,
            sensor_map: ComponentRequirement::Required,
            actuator_map: ComponentRequirement::Required,
            calibration: ComponentRequirement::Required,
            network_topology: ComponentRequirement::Required,
        }
    }

    fn manifest(profile: &SafetyConfigurationProfile) -> SafetyConfigurationManifest {
        SafetyConfigurationManifest {
            schema_version: SAFETY_CONFIGURATION_SCHEMA_V1.to_owned(),
            node_id: "compute-campus".to_owned(),
            profile_id: profile.profile_id.clone(),
            profile_digest: profile.digest().unwrap(),
            hardware_inventory: component(0x01),
            firmware: component(0x02),
            software_closure: component(0x03),
            electrical_topology: component(0x04),
            thermal_topology: component(0x05),
            protection_settings: component(0x06),
            sensor_map: component(0x07),
            actuator_map: component(0x08),
            calibration: component(0x09),
            network_topology: component(0x0a),
        }
    }

    #[test]
    fn exact_profile_and_manifest_validate_together() {
        let profile = profile();
        let manifest = manifest(&profile);
        profile.validate_manifest(&manifest).unwrap();
    }

    #[test]
    fn changing_one_requirement_changes_profile_identity() {
        let first = profile();
        let mut second = first.clone();
        second.firmware = ComponentRequirement::NotApplicable;
        assert_ne!(first.digest().unwrap(), second.digest().unwrap());
    }

    #[test]
    fn manifest_bound_to_different_profile_digest_fails_closed() {
        let profile = profile();
        let mut manifest = manifest(&profile);
        manifest.profile_digest = digest(0xee);
        assert!(matches!(
            profile.validate_manifest(&manifest),
            Err(SafetyConfigurationProfileError::ProfileDigestMismatch { .. })
        ));
    }

    #[test]
    fn manifest_bound_to_different_profile_id_fails_closed() {
        let profile = profile();
        let mut manifest = manifest(&profile);
        manifest.profile_id = "other-profile".to_owned();
        assert!(matches!(
            profile.validate_manifest(&manifest),
            Err(SafetyConfigurationProfileError::ProfileIdMismatch { .. })
        ));
    }

    #[test]
    fn required_component_cannot_be_not_applicable() {
        let profile = profile();
        let mut manifest = manifest(&profile);
        manifest.firmware = ConfigurationComponent::NotApplicable;
        assert_eq!(
            profile.validate_manifest(&manifest),
            Err(SafetyConfigurationProfileError::MissingRequiredComponent(
                ConfigurationField::Firmware
            ))
        );
    }

    #[test]
    fn component_marked_not_applicable_cannot_carry_hidden_evidence() {
        let mut profile = profile();
        profile.firmware = ComponentRequirement::NotApplicable;
        let mut manifest = manifest(&profile);
        manifest.firmware = component(0x02);
        assert_eq!(
            profile.validate_manifest(&manifest),
            Err(SafetyConfigurationProfileError::UnexpectedComponent(
                ConfigurationField::Firmware
            ))
        );
    }

    #[test]
    fn profile_with_no_required_components_is_rejected() {
        let mut profile = profile();
        profile.hardware_inventory = ComponentRequirement::NotApplicable;
        profile.firmware = ComponentRequirement::NotApplicable;
        profile.software_closure = ComponentRequirement::NotApplicable;
        profile.electrical_topology = ComponentRequirement::NotApplicable;
        profile.thermal_topology = ComponentRequirement::NotApplicable;
        profile.protection_settings = ComponentRequirement::NotApplicable;
        profile.sensor_map = ComponentRequirement::NotApplicable;
        profile.actuator_map = ComponentRequirement::NotApplicable;
        profile.calibration = ComponentRequirement::NotApplicable;
        profile.network_topology = ComponentRequirement::NotApplicable;
        assert_eq!(
            profile.validate(),
            Err(SafetyConfigurationProfileError::NoRequiredComponents)
        );
    }

    #[test]
    fn canonical_profile_bytes_are_stable_and_fixed_order() {
        let first = profile();
        let second = first.clone();
        assert_eq!(first.canonical_bytes().unwrap(), second.canonical_bytes().unwrap());
        assert_eq!(first.digest().unwrap(), second.digest().unwrap());
        assert_eq!(ConfigurationField::ALL.len(), 10);
    }
}
