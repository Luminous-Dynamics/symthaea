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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationVerificationResult {
    /// Stable verifier identity. This is a claim supplied by the authoritative layer.
    pub verifier_id: String,
    /// Exact attestation reference this result evaluated.
    pub attestation_id: String,
    pub attestation_digest: String,
    /// Evidence/result identity produced by the verifier.
    pub verification_reference: String,
    /// Verification time and validity horizon, in the verifier's clock domain.
    pub verified_at_ms: u64,
    pub valid_until_ms: u64,
}

impl AttestationVerificationResult {
    fn validate_against(
        &self,
        reference: &AuthoritativeAttestationReference,
        observation_timestamp_ms: u64,
    ) -> Result<(), SensorTopologyAttestationIssue> {
        if self.verifier_id.trim().is_empty()
            || self.attestation_id.trim().is_empty()
            || self.attestation_digest.trim().is_empty()
            || self.verification_reference.trim().is_empty()
        {
            return Err(SensorTopologyAttestationIssue::EmptyVerificationResult);
        }
        if self.verified_at_ms > self.valid_until_ms {
            return Err(SensorTopologyAttestationIssue::InvalidVerificationWindow);
        }
        if self.attestation_id != reference.attestation_id
            || self.attestation_digest != reference.attestation_digest
            || self.verification_reference != reference.verification_reference
        {
            return Err(SensorTopologyAttestationIssue::VerificationReferenceMismatch);
        }
        if observation_timestamp_ms < self.verified_at_ms {
            return Err(SensorTopologyAttestationIssue::FutureVerificationResult);
        }
        if observation_timestamp_ms > self.valid_until_ms {
            return Err(SensorTopologyAttestationIssue::StaleVerificationResult);
        }
        Ok(())
    }
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
pub struct TopologyEpochBinding {
    /// Monotonic lifecycle epoch for the topology statement.
    pub epoch: u64,
    /// Digest of the immediately preceding topology epoch, when one exists.
    pub predecessor_topology_digest: Option<String>,
    /// Lifecycle time at which this epoch became effective.
    pub effective_from_ms: u64,
    /// Stable lifecycle event that created or superseded this epoch.
    pub lifecycle_event_id: String,
}

impl TopologyEpochBinding {
    fn validate(&self) -> Result<(), SensorTopologyAttestationIssue> {
        if self.epoch == 0
            || self.lifecycle_event_id.trim().is_empty()
            || self.effective_from_ms == 0
        {
            return Err(SensorTopologyAttestationIssue::InvalidEpochBinding);
        }
        if self.epoch == 1 {
            if self.predecessor_topology_digest.is_some() {
                return Err(SensorTopologyAttestationIssue::InvalidEpochBinding);
            }
        } else if self
            .predecessor_topology_digest
            .as_deref()
            .map(str::trim)
            .unwrap_or("")
            .is_empty()
        {
            return Err(SensorTopologyAttestationIssue::MissingEpochPredecessor);
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
    pub epoch_binding: TopologyEpochBinding,
    pub issued_at_ms: u64,
    pub valid_until_ms: u64,
    pub evidence_id: String,
    /// Reference into the authoritative provenance/attestation layer.
    pub authoritative_reference: AuthoritativeAttestationReference,
    /// Result returned by the authoritative verifier for this reference.
    pub verification_result: AttestationVerificationResult,
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
    InvalidEpochBinding,
    MissingEpochPredecessor,
    EpochEffectiveTimeMismatch,
    EmptyVerificationResult,
    InvalidVerificationWindow,
    VerificationReferenceMismatch,
    FutureVerificationResult,
    StaleVerificationResult,
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
        self.epoch_binding.validate()?;
        if self.epoch_binding.effective_from_ms > self.issued_at_ms {
            return Err(SensorTopologyAttestationIssue::EpochEffectiveTimeMismatch);
        }
        if self.epoch_binding.epoch > 1
            && self.epoch_binding.predecessor_topology_digest.as_deref() == Some(self.topology_digest.as_str())
        {
            return Err(SensorTopologyAttestationIssue::InvalidEpochBinding);
        }
        self.verification_result
            .validate_against(&self.authoritative_reference, observation_timestamp_ms)?;

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
            epoch_binding: TopologyEpochBinding {
                epoch: 1,
                predecessor_topology_digest: None,
                effective_from_ms: 500,
                lifecycle_event_id: "topology-created-1".into(),
            },
            issued_at_ms: 500,
            valid_until_ms: 2_500,
            evidence_id: "topology-e-1".into(),
            authoritative_reference: AuthoritativeAttestationReference {
                attestation_id: "att-topology-1".into(),
                issuer_id: "mycelix-topology-authority".into(),
                attestation_digest: "att-digest-1".into(),
                verification_reference: "verify-1".into(),
            },
            verification_result: AttestationVerificationResult {
                verifier_id: "mycelix-topology-verifier".into(),
                attestation_id: "att-topology-1".into(),
                attestation_digest: "att-digest-1".into(),
                verification_reference: "verify-1".into(),
                verified_at_ms: 500,
                valid_until_ms: 2_500,
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
    fn verification_result_must_match_attestation_reference() {
        let mut a = attestation();
        a.verification_result.attestation_digest = "wrong".into();
        assert_eq!(
            a.validate("vehicle-1", "wing-root", "cfg-1", "mycelix-topology-authority", 1_000),
            Err(SensorTopologyAttestationIssue::VerificationReferenceMismatch)
        );
    }

    #[test]
    fn verification_result_cannot_be_stale_or_future() {
        let mut a = attestation();
        a.verification_result.valid_until_ms = 999;
        assert_eq!(
            a.validate("vehicle-1", "wing-root", "cfg-1", "mycelix-topology-authority", 1_000),
            Err(SensorTopologyAttestationIssue::StaleVerificationResult)
        );
        a = attestation();
        a.verification_result.verified_at_ms = 1_001;
        assert_eq!(
            a.validate("vehicle-1", "wing-root", "cfg-1", "mycelix-topology-authority", 1_000),
            Err(SensorTopologyAttestationIssue::FutureVerificationResult)
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
