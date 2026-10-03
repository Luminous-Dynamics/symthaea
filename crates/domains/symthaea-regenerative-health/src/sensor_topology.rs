// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic validity boundary for sensor-topology attestations.
//!
//! This contract establishes when a topology/dependency declaration is
//! locally admissible. It deliberately separates an attestation reference
//! from authoritative verification: a well-formed reference is not proof
//! that the issuer or declared physical topology is truthful.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthoritativeAttestationReference {
    /// Stable identifier assigned by the authoritative attestation service.
    pub attestation_id: String,
    /// Identity of the authority expected to have issued the attestation.
    pub issuer_id: String,
    /// Digest of the authoritative attestation statement.
    pub attestation_digest: String,
    /// External evidence/result reference used to verify the statement.
    pub verification_reference: String,
}

impl AuthoritativeAttestationReference {
    fn validate(&self, expected_issuer_id: &str) -> Result<(), SensorTopologyAttestationIssue> {
        if self.attestation_id.trim().is_empty()
            || self.issuer_id.trim().is_empty()
            || self.attestation_digest.trim().is_empty()
            || self.verification_reference.trim().is_empty()
            || expected_issuer_id.trim().is_empty()
        {
            return Err(SensorTopologyAttestationIssue::EmptyAttestationReference);
        }
        if self.issuer_id != expected_issuer_id {
            return Err(SensorTopologyAttestationIssue::AttestationIssuerMismatch);
        }
        Ok(())
    }
}

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
    /// Reference into the authoritative provenance/attestation layer.
    pub authoritative_reference: AuthoritativeAttestationReference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensorTopologyAttestationIssue {
    EmptyIdentity,
    EmptyAttestationReference,
    AttestationIssuerMismatch,
    InvalidValidityWindow,
    FutureAttestation,
    StaleAttestation,
    ConfigurationMismatch,
    TopologyIdentityMismatch,
    AttestationReferenceMismatch,
}

impl SensorTopologyAttestation {
    pub fn validate(
        &self,
        expected_asset_id: &str,
        expected_component_id: &str,
        expected_configuration_digest: &str,
        expected_issuer_id: &str,
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

        self.authoritative_reference.validate(expected_issuer_id)?;

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

    pub fn same_authoritative_reference(&self, other: &Self) -> bool {
        self.authoritative_reference == other.authoritative_reference
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
            authoritative_reference: AuthoritativeAttestationReference {
                attestation_id: "att-topology-1".into(),
                issuer_id: "mycelix-topology-authority".into(),
                attestation_digest: "att-digest-1".into(),
                verification_reference: "verify-1".into(),
            },
        }
    }

    #[test]
    fn valid_attestation_is_admitted_at_observation_time() {
        assert!(attestation()
            .validate("vehicle-1", "wing-root", "cfg-1", "mycelix-topology-authority", 1_000)
            .is_ok());
    }

    #[test]
    fn stale_attestation_is_not_reused_after_expiry() {
        assert_eq!(
            attestation().validate("vehicle-1", "wing-root", "cfg-1", "mycelix-topology-authority", 2_501),
            Err(SensorTopologyAttestationIssue::StaleAttestation)
        );
    }

    #[test]
    fn future_attestation_cannot_qualify_past_observation() {
        assert_eq!(
            attestation().validate("vehicle-1", "wing-root", "cfg-1", "mycelix-topology-authority", 499),
            Err(SensorTopologyAttestationIssue::FutureAttestation)
        );
    }

    #[test]
    fn topology_binding_must_match_asset_and_component_identity() {
        assert_eq!(
            attestation().validate("vehicle-2", "wing-root", "cfg-1", "mycelix-topology-authority", 1_000),
            Err(SensorTopologyAttestationIssue::TopologyIdentityMismatch)
        );
        assert_eq!(
            attestation().validate("vehicle-1", "tail-root", "cfg-1", "mycelix-topology-authority", 1_000),
            Err(SensorTopologyAttestationIssue::TopologyIdentityMismatch)
        );
    }

    #[test]
    fn topology_binding_must_match_configuration() {
        assert_eq!(
            attestation().validate("vehicle-1", "wing-root", "cfg-attacker", "mycelix-topology-authority", 1_000),
            Err(SensorTopologyAttestationIssue::ConfigurationMismatch)
        );
    }

    #[test]
    fn issuer_must_match_authoritative_policy() {
        let mut a = attestation();
        a.authoritative_reference.issuer_id = "attacker".into();
        assert_eq!(
            a.validate("vehicle-1", "wing-root", "cfg-1", "mycelix-topology-authority", 1_000),
            Err(SensorTopologyAttestationIssue::AttestationIssuerMismatch)
        );
    }

    #[test]
    fn missing_reference_cannot_be_admitted() {
        let mut a = attestation();
        a.authoritative_reference.attestation_id.clear();
        assert_eq!(
            a.validate("vehicle-1", "wing-root", "cfg-1", "mycelix-topology-authority", 1_000),
            Err(SensorTopologyAttestationIssue::EmptyAttestationReference)
        );
    }

    #[test]
    fn authoritative_reference_identity_is_exactly_comparable() {
        let a = attestation();
        let mut b = attestation();
        assert!(a.same_authoritative_reference(&b));
        b.authoritative_reference.attestation_digest = "att-digest-2".into();
        assert!(!a.same_authoritative_reference(&b));
    }
}
