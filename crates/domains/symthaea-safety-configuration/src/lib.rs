// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical safety-relevant configuration identity.
//!
//! Commissioning, provisioning, and evidence tooling must agree on one exact
//! meaning of "configuration". This crate defines a fixed v1 manifest and
//! canonical byte encoding without depending on operating authority or autonomy.

#![deny(unsafe_code)]

pub mod observation;
pub mod state;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const SAFETY_CONFIGURATION_SCHEMA_V1: &str = "symthaea-safety-configuration-v1";
const DOMAIN_SEPARATOR: &[u8] = b"symthaea:safety-configuration-manifest:v1\0";

/// Exact digest algorithm and bytes used to identify one safety-relevant artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConfigurationDigest {
    Blake3_256([u8; 32]),
}

impl ConfigurationDigest {
    pub fn blake3_256(bytes: &[u8]) -> Self {
        let digest = blake3::hash(bytes);
        Self::Blake3_256(*digest.as_bytes())
    }

    pub fn into_blake3_256(self) -> [u8; 32] {
        match self {
            Self::Blake3_256(bytes) => bytes,
        }
    }
}

/// Explicit state for one safety-relevant configuration component.
///
/// `NotApplicable` is intentionally distinct from absence: every v1 manifest must
/// declare every component field, even when a category genuinely does not apply to
/// that node. Whether `NotApplicable` is permitted for a field is defined by the
/// exact configuration profile artifact bound into the same manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfigurationComponent {
    NotApplicable,
    Digest(ConfigurationDigest),
}

impl ConfigurationComponent {
    pub fn is_applicable(&self) -> bool {
        matches!(self, Self::Digest(_))
    }
}

/// Exact v1 safety configuration surface for an autonomous infrastructure node.
///
/// Each component is the identity of a canonical evidence artifact owned by the
/// corresponding engineering/provisioning subsystem. Keeping those lower-level
/// formats out of this crate lets hardware, Nix/Spore, controls, calibration, and
/// network tooling evolve independently while agreeing on the top-level contract.
/// `profile_id` is human-readable; `profile_digest` binds the exact immutable
/// profile artifact that defines which component slots may be `NotApplicable`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyConfigurationManifest {
    pub schema_version: String,
    pub node_id: String,
    pub profile_id: String,
    pub profile_digest: ConfigurationDigest,
    pub hardware_inventory: ConfigurationComponent,
    pub firmware: ConfigurationComponent,
    pub software_closure: ConfigurationComponent,
    pub electrical_topology: ConfigurationComponent,
    pub thermal_topology: ConfigurationComponent,
    pub protection_settings: ConfigurationComponent,
    pub sensor_map: ConfigurationComponent,
    pub actuator_map: ConfigurationComponent,
    pub calibration: ConfigurationComponent,
    pub network_topology: ConfigurationComponent,
}

impl SafetyConfigurationManifest {
    pub fn validate(&self) -> Result<(), SafetyConfigurationManifestError> {
        if self.schema_version != SAFETY_CONFIGURATION_SCHEMA_V1 {
            return Err(SafetyConfigurationManifestError::UnsupportedSchemaVersion(
                self.schema_version.clone(),
            ));
        }
        if self.node_id.trim().is_empty() {
            return Err(SafetyConfigurationManifestError::EmptyNodeId);
        }
        if self.profile_id.trim().is_empty() {
            return Err(SafetyConfigurationManifestError::EmptyProfileId);
        }
        if !self.components().iter().any(ConfigurationComponent::is_applicable) {
            return Err(SafetyConfigurationManifestError::NoSafetyRelevantComponents);
        }
        Ok(())
    }

    /// Fixed-order, domain-separated canonical bytes for cross-tool hashing.
    ///
    /// Encoding v1:
    /// - fixed ASCII domain separator
    /// - schema, node ID, and configuration profile ID as big-endian u32
    ///   length-prefixed UTF-8
    /// - mandatory profile digest at field ID 0 with explicit digest algorithm tag
    /// - ten fixed component slots in field-ID order 1..=10
    /// - component tag 0 = NotApplicable
    /// - component tag 1 = digest, followed by algorithm tag and digest bytes
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, SafetyConfigurationManifestError> {
        self.validate()?;
        let mut out = Vec::with_capacity(512);
        out.extend_from_slice(DOMAIN_SEPARATOR);
        push_string(&mut out, "schema_version", &self.schema_version)?;
        push_string(&mut out, "node_id", &self.node_id)?;
        push_string(&mut out, "profile_id", &self.profile_id)?;
        push_required_digest(&mut out, 0, self.profile_digest);

        push_component(&mut out, 1, self.hardware_inventory);
        push_component(&mut out, 2, self.firmware);
        push_component(&mut out, 3, self.software_closure);
        push_component(&mut out, 4, self.electrical_topology);
        push_component(&mut out, 5, self.thermal_topology);
        push_component(&mut out, 6, self.protection_settings);
        push_component(&mut out, 7, self.sensor_map);
        push_component(&mut out, 8, self.actuator_map);
        push_component(&mut out, 9, self.calibration);
        push_component(&mut out, 10, self.network_topology);
        Ok(out)
    }

    /// BLAKE3-256 identity of the canonical v1 safety configuration manifest.
    pub fn digest(&self) -> Result<ConfigurationDigest, SafetyConfigurationManifestError> {
        Ok(ConfigurationDigest::blake3_256(&self.canonical_bytes()?))
    }

    fn components(&self) -> [ConfigurationComponent; 10] {
        [
            self.hardware_inventory,
            self.firmware,
            self.software_closure,
            self.electrical_topology,
            self.thermal_topology,
            self.protection_settings,
            self.sensor_map,
            self.actuator_map,
            self.calibration,
            self.network_topology,
        ]
    }
}

fn push_string(
    out: &mut Vec<u8>,
    field: &'static str,
    value: &str,
) -> Result<(), SafetyConfigurationManifestError> {
    let len = u32::try_from(value.len())
        .map_err(|_| SafetyConfigurationManifestError::StringTooLong(field))?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn push_required_digest(out: &mut Vec<u8>, field_id: u8, digest: ConfigurationDigest) {
    out.push(field_id);
    match digest {
        ConfigurationDigest::Blake3_256(bytes) => {
            out.push(1); // digest algorithm tag: BLAKE3-256
            out.extend_from_slice(&bytes);
        }
    }
}

fn push_component(out: &mut Vec<u8>, field_id: u8, component: ConfigurationComponent) {
    out.push(field_id);
    match component {
        ConfigurationComponent::NotApplicable => out.push(0),
        ConfigurationComponent::Digest(ConfigurationDigest::Blake3_256(bytes)) => {
            out.push(1); // component is present
            out.push(1); // digest algorithm tag: BLAKE3-256
            out.extend_from_slice(&bytes);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyConfigurationManifestError {
    #[error("unsupported safety configuration schema version {0}")]
    UnsupportedSchemaVersion(String),
    #[error("safety configuration node id must not be empty")]
    EmptyNodeId,
    #[error("safety configuration profile id must not be empty")]
    EmptyProfileId,
    #[error("safety configuration must identify at least one applicable component")]
    NoSafetyRelevantComponents,
    #[error("safety configuration string field {0} exceeds canonical u32 length")]
    StringTooLong(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> ConfigurationDigest {
        ConfigurationDigest::Blake3_256([byte; 32])
    }

    fn component(byte: u8) -> ConfigurationComponent {
        ConfigurationComponent::Digest(digest(byte))
    }

    fn manifest() -> SafetyConfigurationManifest {
        SafetyConfigurationManifest {
            schema_version: SAFETY_CONFIGURATION_SCHEMA_V1.to_owned(),
            node_id: "compute-campus".to_owned(),
            profile_id: "compute-commons-autonomous-node-v1".to_owned(),
            profile_digest: digest(0xf0),
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
    fn identical_manifests_have_identical_canonical_bytes_and_digest() {
        let first = manifest();
        let second = first.clone();
        assert_eq!(first.canonical_bytes().unwrap(), second.canonical_bytes().unwrap());
        assert_eq!(first.digest().unwrap(), second.digest().unwrap());
    }

    #[test]
    fn changing_any_safety_component_changes_top_level_identity() {
        let first = manifest();
        let mut second = first.clone();
        second.thermal_topology = component(0xee);
        assert_ne!(first.digest().unwrap(), second.digest().unwrap());
    }

    #[test]
    fn node_identity_is_bound_into_configuration_digest() {
        let first = manifest();
        let mut second = first.clone();
        second.node_id = "other-campus".to_owned();
        assert_ne!(first.digest().unwrap(), second.digest().unwrap());
    }

    #[test]
    fn profile_identity_and_exact_profile_artifact_are_both_bound() {
        let first = manifest();
        let mut renamed = first.clone();
        renamed.profile_id = "different-policy-v1".to_owned();
        assert_ne!(first.digest().unwrap(), renamed.digest().unwrap());

        let mut changed_profile = first.clone();
        changed_profile.profile_digest = digest(0xf1);
        assert_ne!(first.digest().unwrap(), changed_profile.digest().unwrap());
    }

    #[test]
    fn not_applicable_is_distinct_from_a_zero_digest() {
        let first = manifest();
        let mut second = first.clone();
        second.firmware = ConfigurationComponent::NotApplicable;
        let mut third = first.clone();
        third.firmware = ConfigurationComponent::Digest(ConfigurationDigest::Blake3_256([0; 32]));
        assert_ne!(second.canonical_bytes().unwrap(), third.canonical_bytes().unwrap());
        assert_ne!(second.digest().unwrap(), third.digest().unwrap());
    }

    #[test]
    fn canonical_bytes_have_a_pinned_cross_tool_v1_vector() {
        let minimal = SafetyConfigurationManifest {
            schema_version: SAFETY_CONFIGURATION_SCHEMA_V1.to_owned(),
            node_id: "node".to_owned(),
            profile_id: "test-profile-v1".to_owned(),
            profile_digest: digest(0x22),
            hardware_inventory: component(0x11),
            firmware: ConfigurationComponent::NotApplicable,
            software_closure: ConfigurationComponent::NotApplicable,
            electrical_topology: ConfigurationComponent::NotApplicable,
            thermal_topology: ConfigurationComponent::NotApplicable,
            protection_settings: ConfigurationComponent::NotApplicable,
            sensor_map: ConfigurationComponent::NotApplicable,
            actuator_map: ConfigurationComponent::NotApplicable,
            calibration: ConfigurationComponent::NotApplicable,
            network_topology: ConfigurationComponent::NotApplicable,
        };
        let expected = hex_bytes(
            "73796d74686165613a7361666574792d636f6e66696775726174696f6e2d6d616e69666573743a7631000000002073796d74686165612d7361666574792d636f6e66696775726174696f6e2d7631000000046e6f64650000000f746573742d70726f66696c652d7631000122222222222222222222222222222222222222222222222222222222222222220101011111111111111111111111111111111111111111111111111111111111111111020003000400050006000700080009000a00",
        );
        assert_eq!(minimal.canonical_bytes().unwrap(), expected);
    }

    #[test]
    fn unsupported_schema_fails_closed() {
        let mut invalid = manifest();
        invalid.schema_version = "symthaea-safety-configuration-v2".to_owned();
        assert!(matches!(
            invalid.validate(),
            Err(SafetyConfigurationManifestError::UnsupportedSchemaVersion(_))
        ));
    }

    #[test]
    fn empty_node_id_fails_closed() {
        let mut invalid = manifest();
        invalid.node_id = "  ".to_owned();
        assert_eq!(
            invalid.validate(),
            Err(SafetyConfigurationManifestError::EmptyNodeId)
        );
    }

    #[test]
    fn empty_profile_id_fails_closed() {
        let mut invalid = manifest();
        invalid.profile_id = "  ".to_owned();
        assert_eq!(
            invalid.validate(),
            Err(SafetyConfigurationManifestError::EmptyProfileId)
        );
    }

    #[test]
    fn all_not_applicable_is_rejected() {
        let mut invalid = manifest();
        invalid.hardware_inventory = ConfigurationComponent::NotApplicable;
        invalid.firmware = ConfigurationComponent::NotApplicable;
        invalid.software_closure = ConfigurationComponent::NotApplicable;
        invalid.electrical_topology = ConfigurationComponent::NotApplicable;
        invalid.thermal_topology = ConfigurationComponent::NotApplicable;
        invalid.protection_settings = ConfigurationComponent::NotApplicable;
        invalid.sensor_map = ConfigurationComponent::NotApplicable;
        invalid.actuator_map = ConfigurationComponent::NotApplicable;
        invalid.calibration = ConfigurationComponent::NotApplicable;
        invalid.network_topology = ConfigurationComponent::NotApplicable;
        assert_eq!(
            invalid.validate(),
            Err(SafetyConfigurationManifestError::NoSafetyRelevantComponents)
        );
    }

    fn hex_bytes(hex: &str) -> Vec<u8> {
        assert_eq!(hex.len() % 2, 0);
        hex.as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let hi = from_hex(pair[0]);
                let lo = from_hex(pair[1]);
                (hi << 4) | lo
            })
            .collect()
    }

    fn from_hex(byte: u8) -> u8 {
        match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            _ => panic!("invalid hex byte"),
        }
    }
}
