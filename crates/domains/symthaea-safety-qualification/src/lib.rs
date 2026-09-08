// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Proof-carrying qualification for safety-configuration manifests.
//!
//! Raw manifests and profiles are serializable evidence inputs. Runtime code that
//! requires proof that the two agree should consume [`QualifiedSafetyConfiguration`]
//! instead. The qualified type has private fields and intentionally does not
//! implement `Deserialize`, so it cannot be recreated from untrusted wire bytes
//! without re-running qualification.

#![deny(unsafe_code)]

pub mod observation_admission;
pub mod observation_commit_preconditions;

use symthaea_safety_configuration::{
    ConfigurationDigest, SafetyConfigurationManifest, SafetyConfigurationManifestError,
};
use symthaea_safety_profile::{SafetyConfigurationProfile, SafetyConfigurationProfileError};
use thiserror::Error;

/// Opaque proof that one exact configuration manifest conforms to one exact
/// canonical safety profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedSafetyConfiguration {
    manifest: SafetyConfigurationManifest,
    configuration_digest: ConfigurationDigest,
    profile_digest: ConfigurationDigest,
}

impl QualifiedSafetyConfiguration {
    pub fn node_id(&self) -> &str {
        &self.manifest.node_id
    }

    pub fn profile_id(&self) -> &str {
        &self.manifest.profile_id
    }

    pub fn configuration_digest(&self) -> ConfigurationDigest {
        self.configuration_digest
    }

    pub fn profile_digest(&self) -> ConfigurationDigest {
        self.profile_digest
    }

    pub fn manifest(&self) -> &SafetyConfigurationManifest {
        &self.manifest
    }

    pub fn into_manifest(self) -> SafetyConfigurationManifest {
        self.manifest
    }
}

/// Qualify a raw manifest against the exact canonical profile it claims to bind.
///
/// The profile validator first checks schema, profile ID, profile digest, and every
/// Required/NotApplicable slot. Only then is the canonical configuration digest
/// computed and placed in the opaque qualified wrapper.
pub fn qualify_safety_configuration(
    profile: &SafetyConfigurationProfile,
    manifest: SafetyConfigurationManifest,
) -> Result<QualifiedSafetyConfiguration, SafetyQualificationError> {
    profile.validate_manifest(&manifest)?;
    let profile_digest = profile.digest()?;
    let configuration_digest = manifest.digest()?;

    // `validate_manifest` already checks this equality. Keep the value copied from
    // the canonical profile rather than trusting a caller-supplied field.
    debug_assert_eq!(manifest.profile_digest, profile_digest);

    Ok(QualifiedSafetyConfiguration {
        manifest,
        configuration_digest,
        profile_digest,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyQualificationError {
    #[error(transparent)]
    Profile(#[from] SafetyConfigurationProfileError),
    #[error(transparent)]
    Manifest(#[from] SafetyConfigurationManifestError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_safety_configuration::{
        ConfigurationComponent, SAFETY_CONFIGURATION_SCHEMA_V1,
    };
    use symthaea_safety_profile::{
        ComponentRequirement, SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1,
    };

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
    fn exact_manifest_becomes_opaque_qualified_configuration() {
        let profile = profile();
        let manifest = manifest(&profile);
        let expected_digest = manifest.digest().unwrap();
        let qualified = qualify_safety_configuration(&profile, manifest).unwrap();
        assert_eq!(qualified.node_id(), "compute-campus");
        assert_eq!(qualified.profile_id(), profile.profile_id);
        assert_eq!(qualified.profile_digest(), profile.digest().unwrap());
        assert_eq!(qualified.configuration_digest(), expected_digest);
    }

    #[test]
    fn wrong_profile_digest_never_produces_qualified_value() {
        let profile = profile();
        let mut manifest = manifest(&profile);
        manifest.profile_digest = digest(0xff);
        assert!(qualify_safety_configuration(&profile, manifest).is_err());
    }

    #[test]
    fn missing_required_component_never_produces_qualified_value() {
        let profile = profile();
        let mut manifest = manifest(&profile);
        manifest.firmware = ConfigurationComponent::NotApplicable;
        assert!(qualify_safety_configuration(&profile, manifest).is_err());
    }

    #[test]
    fn configuration_digest_changes_with_manifest_evidence() {
        let profile = profile();
        let first = qualify_safety_configuration(&profile, manifest(&profile)).unwrap();
        let mut changed_manifest = manifest(&profile);
        changed_manifest.calibration = component(0xee);
        let second = qualify_safety_configuration(&profile, changed_manifest).unwrap();
        assert_ne!(first.configuration_digest(), second.configuration_digest());
        assert_eq!(first.profile_digest(), second.profile_digest());
    }

    #[test]
    fn qualified_value_exposes_manifest_read_only_and_can_be_consumed() {
        let profile = profile();
        let qualified = qualify_safety_configuration(&profile, manifest(&profile)).unwrap();
        assert_eq!(qualified.manifest().node_id, "compute-campus");
        let raw = qualified.into_manifest();
        assert_eq!(raw.profile_digest, profile.digest().unwrap());
    }
}
