// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic validity boundary for sensor-topology attestations.
//!
//! This contract establishes when a topology/dependency declaration is
//! temporally admissible. It does not establish that the attestation issuer
//! or the declared physical topology is truthful; authoritative provenance
//! remains an external trust-layer responsibility.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorTopologyAttestation {
    pub schema_version: String,
    pub asset_id: String,
    pub component_id: String,
    pub topology_id: String,
    pub topology_version: String,
    pub topology_digest: String,
    pub configuration_digest: String,
    pub issued_at_ms: u64,
    pub valid_until_ms: u64,
    pub evidence_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensorTopologyAttestationIssue {
    EmptyIdentity,
    InvalidValidityWindow,
    FutureAttestation,
    StaleAttestation,
    ConfigurationMismatch,
    TopologyIdentityMismatch,
}

impl SensorTopologyAttestation {
    pub fn validate(
        &self,
        expected_asset_id: &str,
        expected_component_id: &str,
        expected_configuration_digest: &str,
        observation_timestamp_ms: u64,
    ) -> Result<(), SensorTopologyAttestationIssue> {
        if self.schema_version.trim().is_empty()
            || self.asset_id.trim().is_empty()
            || self.component_id.trim().is_empty()
            || self.topology_id.trim().is_empty()
            || self.topology_version.trim().is_empty()
            || self.topology_digest.trim().is_empty()
            || self.configuration_digest.trim().is_empty()
            || self.evidence_id.trim().is_empty()
            || expected_asset_id.trim().is_empty()
            || expected_component_id.trim().is_empty()
        {
            return Err(SensorTopologyAttestationIssue::EmptyIdentity);
        }

        if self.issued_at_ms > self.valid_until_ms {
            return Err(SensorTopologyAttestationIssue::InvalidValidityWindow);
        }

        if self.asset_id != expected_asset_id || self.component_id != expected_component_id {
            return Err(SensorTopologyAttestationIssue::TopologyIdentityMismatch);
        }

        if self.configuration_digest != expected_configuration_digest {
            return Err(SensorTopologyAttestationIssue::ConfigurationMismatch);
        }

        if observation_timestamp_ms < self.issued_at_ms {
            return Err(SensorTopologyAttestationIssue::FutureAttestation);
        }

        if observation_timestamp_ms > self.valid_until_ms {
            return Err(SensorTopologyAttestationIssue::StaleAttestation);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attestation() -> SensorTopologyAttestation {
        SensorTopologyAttestation {
            schema_version: "0.1".into(),
            asset_id: "vehicle-1".into(),
            component_id: "wing-root".into(),
            topology_id: "topology-wing-root".into(),
            topology_version: "1".into(),
            topology_digest: "topology-v1".into(),
            configuration_digest: "cfg-1".into(),
            issued_at_ms: 500,
            valid_until_ms: 2_500,
            evidence_id: "topology-e-1".into(),
        }
    }

    #[test]
    fn valid_attestation_is_admitted_at_observation_time() {
        assert!(attestation()
            .validate("vehicle-1", "wing-root", "cfg-1", 1_000)
            .is_ok());
    }

    #[test]
    fn stale_attestation_is_not_reused_after_expiry() {
        assert_eq!(
            attestation().validate("vehicle-1", "wing-root", "cfg-1", 2_501),
            Err(SensorTopologyAttestationIssue::StaleAttestation)
        );
    }

    #[test]
    fn future_attestation_cannot_qualify_past_observation() {
        assert_eq!(
            attestation().validate("vehicle-1", "wing-root", "cfg-1", 499),
            Err(SensorTopologyAttestationIssue::FutureAttestation)
        );
    }

    #[test]
    fn topology_binding_must_match_asset_and_component_identity() {
        assert_eq!(
            attestation().validate("vehicle-2", "wing-root", "cfg-1", 1_000),
            Err(SensorTopologyAttestationIssue::TopologyIdentityMismatch)
        );
        assert_eq!(
            attestation().validate("vehicle-1", "tail-root", "cfg-1", 1_000),
            Err(SensorTopologyAttestationIssue::TopologyIdentityMismatch)
        );
    }

    #[test]
    fn topology_binding_must_match_configuration() {
        assert_eq!(
            attestation().validate("vehicle-1", "wing-root", "cfg-attacker", 1_000),
            Err(SensorTopologyAttestationIssue::ConfigurationMismatch)
        );
    }

}
