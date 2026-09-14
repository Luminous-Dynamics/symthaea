// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bearing observations and dependency claims.
//!
//! Core invariant:
//!
//! `Observation != DependencyClaim != ContinuityRequirement != Authority`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

const OBSERVATION_DOMAIN: &[u8] = b"symthaea.continuity.observation.v1\0";
const DEPENDENCY_DOMAIN: &[u8] = b"symthaea.continuity.dependency-claim.v1\0";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct ObservationId([u8; 32]);

impl ObservationId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct DependencyClaimId([u8; 32]);

impl DependencyClaimId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Whether a provider actually inspected the relevant observation domain.
///
/// Absence under `Complete` coverage may support an absence claim. Absence under
/// `Incomplete` or `Unknown` coverage must remain unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationCoverage {
    Complete,
    Incomplete,
    Unknown,
    NotApplicable,
}

impl ObservationCoverage {
    fn tag(self) -> u8 {
        match self {
            Self::Complete => 1,
            Self::Incomplete => 2,
            Self::Unknown => 3,
            Self::NotApplicable => 4,
        }
    }
}

/// Epistemic basis for one piece of evidence.
///
/// This is intentionally not an ordered confidence scale. A declaration may be
/// more decision-relevant than telemetry for some requirements, while an
/// inference can never silently become a direct observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceBasis {
    Observed,
    Declared,
    Reconstructed,
    Inferred,
    Tested,
}

impl EvidenceBasis {
    fn tag(self) -> u8 {
        match self {
            Self::Observed => 1,
            Self::Declared => 2,
            Self::Reconstructed => 3,
            Self::Inferred => 4,
            Self::Tested => 5,
        }
    }
}

/// One normalized observation emitted by a scanner/importer/provider.
///
/// `evidence_digest` names the retained/raw evidence outside this value. This
/// crate commits to that digest but does not authenticate the provider or bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationEnvelopeV1 {
    subject: String,
    observation_kind: String,
    provider_id: String,
    provider_version: String,
    captured_at_unix_ms: u64,
    coverage: ObservationCoverage,
    basis: EvidenceBasis,
    evidence_digest: [u8; 32],
    limitations: Vec<String>,
    observation_id: ObservationId,
}

impl ObservationEnvelopeV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: impl Into<String>,
        observation_kind: impl Into<String>,
        provider_id: impl Into<String>,
        provider_version: impl Into<String>,
        captured_at_unix_ms: u64,
        coverage: ObservationCoverage,
        basis: EvidenceBasis,
        evidence_digest: [u8; 32],
        limitations: Vec<String>,
    ) -> Result<Self, ObservationError> {
        let subject = checked_text("subject", subject.into())?;
        let observation_kind = checked_text("observation_kind", observation_kind.into())?;
        let provider_id = checked_text("provider_id", provider_id.into())?;
        let provider_version = checked_text("provider_version", provider_version.into())?;
        if captured_at_unix_ms == 0 {
            return Err(ObservationError::ZeroCaptureTime);
        }
        if evidence_digest == [0; 32] {
            return Err(ObservationError::ZeroEvidenceDigest);
        }
        let limitations = checked_limitations(limitations)?;
        let observation_id = ObservationId(hash_observation(
            &subject,
            &observation_kind,
            &provider_id,
            &provider_version,
            captured_at_unix_ms,
            coverage,
            basis,
            evidence_digest,
            &limitations,
        ));
        Ok(Self {
            subject,
            observation_kind,
            provider_id,
            provider_version,
            captured_at_unix_ms,
            coverage,
            basis,
            evidence_digest,
            limitations,
            observation_id,
        })
    }

    pub fn id(&self) -> ObservationId {
        self.observation_id
    }

    pub fn subject(&self) -> &str {
        &self.subject
    }

    pub fn observation_kind(&self) -> &str {
        &self.observation_kind
    }

    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    pub fn provider_version(&self) -> &str {
        &self.provider_version
    }

    pub fn captured_at_unix_ms(&self) -> u64 {
        self.captured_at_unix_ms
    }

    pub fn coverage(&self) -> ObservationCoverage {
        self.coverage
    }

    pub fn basis(&self) -> EvidenceBasis {
        self.basis
    }

    pub fn evidence_digest(&self) -> [u8; 32] {
        self.evidence_digest
    }

    pub fn limitations(&self) -> &[String] {
        &self.limitations
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        checked_text("subject", self.subject.clone())?;
        checked_text("observation_kind", self.observation_kind.clone())?;
        checked_text("provider_id", self.provider_id.clone())?;
        checked_text("provider_version", self.provider_version.clone())?;
        if self.captured_at_unix_ms == 0 {
            return Err(ObservationError::ZeroCaptureTime);
        }
        if self.evidence_digest == [0; 32] {
            return Err(ObservationError::ZeroEvidenceDigest);
        }
        let canonical_limitations = checked_limitations(self.limitations.clone())?;
        if canonical_limitations != self.limitations {
            return Err(ObservationError::NonCanonicalLimitations);
        }
        let expected = ObservationId(hash_observation(
            &self.subject,
            &self.observation_kind,
            &self.provider_id,
            &self.provider_version,
            self.captured_at_unix_ms,
            self.coverage,
            self.basis,
            self.evidence_digest,
            &self.limitations,
        ));
        if self.observation_id != expected {
            return Err(ObservationError::IdentityMismatch);
        }
        Ok(())
    }
}

/// Basis used to derive a dependency claim from observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DependencyBasis {
    Observed,
    Declared,
    Reconstructed,
    Inferred,
}

impl DependencyBasis {
    fn tag(self) -> u8 {
        match self {
            Self::Observed => 1,
            Self::Declared => 2,
            Self::Reconstructed => 3,
            Self::Inferred => 4,
        }
    }
}

/// Evidence-backed claim that one subject depends on, uses, requires, or relates
/// to another object.
///
/// A dependency claim is still not a continuity requirement. Promotion into a
/// requirement is owned by an explicit contract/approval layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyClaimV1 {
    subject: String,
    relation: String,
    object: String,
    basis: DependencyBasis,
    source_observations: Vec<ObservationId>,
    limitations: Vec<String>,
    claim_id: DependencyClaimId,
}

impl DependencyClaimV1 {
    pub fn new(
        subject: impl Into<String>,
        relation: impl Into<String>,
        object: impl Into<String>,
        basis: DependencyBasis,
        mut source_observations: Vec<ObservationId>,
        limitations: Vec<String>,
    ) -> Result<Self, ObservationError> {
        let subject = checked_text("dependency subject", subject.into())?;
        let relation = checked_text("dependency relation", relation.into())?;
        let object = checked_text("dependency object", object.into())?;
        if source_observations.is_empty() {
            return Err(ObservationError::NoSourceObservations);
        }
        source_observations.sort_unstable();
        source_observations.dedup();
        let limitations = checked_limitations(limitations)?;
        let claim_id = DependencyClaimId(hash_dependency_claim(
            &subject,
            &relation,
            &object,
            basis,
            &source_observations,
            &limitations,
        ));
        Ok(Self {
            subject,
            relation,
            object,
            basis,
            source_observations,
            limitations,
            claim_id,
        })
    }

    pub fn id(&self) -> DependencyClaimId {
        self.claim_id
    }

    pub fn subject(&self) -> &str {
        &self.subject
    }

    pub fn relation(&self) -> &str {
        &self.relation
    }

    pub fn object(&self) -> &str {
        &self.object
    }

    pub fn basis(&self) -> DependencyBasis {
        self.basis
    }

    pub fn source_observations(&self) -> &[ObservationId] {
        &self.source_observations
    }

    pub fn limitations(&self) -> &[String] {
        &self.limitations
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        checked_text("dependency subject", self.subject.clone())?;
        checked_text("dependency relation", self.relation.clone())?;
        checked_text("dependency object", self.object.clone())?;
        if self.source_observations.is_empty() {
            return Err(ObservationError::NoSourceObservations);
        }
        let mut canonical_sources = self.source_observations.clone();
        canonical_sources.sort_unstable();
        canonical_sources.dedup();
        if canonical_sources != self.source_observations {
            return Err(ObservationError::NonCanonicalSourceObservations);
        }
        let canonical_limitations = checked_limitations(self.limitations.clone())?;
        if canonical_limitations != self.limitations {
            return Err(ObservationError::NonCanonicalLimitations);
        }
        let expected = DependencyClaimId(hash_dependency_claim(
            &self.subject,
            &self.relation,
            &self.object,
            self.basis,
            &self.source_observations,
            &self.limitations,
        ));
        if self.claim_id != expected {
            return Err(ObservationError::IdentityMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ObservationError {
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds 1024 bytes")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("captured_at_unix_ms must be non-zero")]
    ZeroCaptureTime,
    #[error("evidence digest must be non-zero")]
    ZeroEvidenceDigest,
    #[error("dependency claim requires at least one source observation")]
    NoSourceObservations,
    #[error("source observations must be sorted and unique")]
    NonCanonicalSourceObservations,
    #[error("limitations contain a blank or invalid entry")]
    InvalidLimitation,
    #[error("limitations must be sorted and unique")]
    NonCanonicalLimitations,
    #[error("stored content identity does not match canonical fields")]
    IdentityMismatch,
}

fn checked_text(field: &'static str, value: String) -> Result<String, ObservationError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(ObservationError::BlankText { field });
    }
    if trimmed.len() > 1024 {
        return Err(ObservationError::TextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(ObservationError::ControlCharacters { field });
    }
    Ok(trimmed.to_string())
}

fn checked_limitations(limitations: Vec<String>) -> Result<Vec<String>, ObservationError> {
    let mut out = Vec::with_capacity(limitations.len());
    for limitation in limitations {
        let trimmed = limitation.trim();
        if trimmed.is_empty() || trimmed.len() > 2048 || trimmed.chars().any(char::is_control) {
            return Err(ObservationError::InvalidLimitation);
        }
        out.push(trimmed.to_string());
    }
    out.sort();
    out.dedup();
    Ok(out)
}

fn hash_observation(
    subject: &str,
    observation_kind: &str,
    provider_id: &str,
    provider_version: &str,
    captured_at_unix_ms: u64,
    coverage: ObservationCoverage,
    basis: EvidenceBasis,
    evidence_digest: [u8; 32],
    limitations: &[String],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    put_str(&mut bytes, subject);
    put_str(&mut bytes, observation_kind);
    put_str(&mut bytes, provider_id);
    put_str(&mut bytes, provider_version);
    bytes.extend_from_slice(&captured_at_unix_ms.to_le_bytes());
    bytes.push(coverage.tag());
    bytes.push(basis.tag());
    bytes.extend_from_slice(&evidence_digest);
    put_strings(&mut bytes, limitations);
    domain_hash(OBSERVATION_DOMAIN, &bytes)
}

fn hash_dependency_claim(
    subject: &str,
    relation: &str,
    object: &str,
    basis: DependencyBasis,
    source_observations: &[ObservationId],
    limitations: &[String],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    put_str(&mut bytes, subject);
    put_str(&mut bytes, relation);
    put_str(&mut bytes, object);
    bytes.push(basis.tag());
    put_len(&mut bytes, source_observations.len());
    for source in source_observations {
        bytes.extend_from_slice(source.as_bytes());
    }
    put_strings(&mut bytes, limitations);
    domain_hash(DEPENDENCY_DOMAIN, &bytes)
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

fn put_len(bytes: &mut Vec<u8>, len: usize) {
    bytes.extend_from_slice(&(len as u64).to_le_bytes());
}

fn put_str(bytes: &mut Vec<u8>, value: &str) {
    put_len(bytes, value.len());
    bytes.extend_from_slice(value.as_bytes());
}

fn put_strings(bytes: &mut Vec<u8>, values: &[String]) {
    put_len(bytes, values.len());
    for value in values {
        put_str(bytes, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(
        coverage: ObservationCoverage,
        basis: EvidenceBasis,
        digest: u8,
    ) -> ObservationEnvelopeV1 {
        ObservationEnvelopeV1::new(
            "machine-1",
            "software.inventory",
            "fixture-provider",
            "1.0",
            1_700_000_000_000,
            coverage,
            basis,
            [digest; 32],
            vec!["fixture evidence only".into()],
        )
        .unwrap()
    }

    #[test]
    fn observation_identity_binds_coverage_and_basis() {
        let complete = observation(ObservationCoverage::Complete, EvidenceBasis::Observed, 7);
        let incomplete = observation(ObservationCoverage::Incomplete, EvidenceBasis::Observed, 7);
        let inferred = observation(ObservationCoverage::Complete, EvidenceBasis::Inferred, 7);
        assert_ne!(complete.id(), incomplete.id());
        assert_ne!(complete.id(), inferred.id());
        complete.validate().unwrap();
    }

    #[test]
    fn zero_evidence_digest_fails_closed() {
        let result = ObservationEnvelopeV1::new(
            "machine-1",
            "hardware.cpu",
            "provider",
            "1",
            1,
            ObservationCoverage::Complete,
            EvidenceBasis::Observed,
            [0; 32],
            vec![],
        );
        assert_eq!(result.unwrap_err(), ObservationError::ZeroEvidenceDigest);
    }

    #[test]
    fn dependency_sources_are_canonicalized() {
        let a = observation(ObservationCoverage::Complete, EvidenceBasis::Observed, 1);
        let b = observation(ObservationCoverage::Complete, EvidenceBasis::Observed, 2);
        let left = DependencyClaimV1::new(
            "workflow:research",
            "requires",
            "software:cuda",
            DependencyBasis::Inferred,
            vec![a.id(), b.id(), a.id()],
            vec!["requires confirmation".into()],
        )
        .unwrap();
        let right = DependencyClaimV1::new(
            "workflow:research",
            "requires",
            "software:cuda",
            DependencyBasis::Inferred,
            vec![b.id(), a.id()],
            vec!["requires confirmation".into()],
        )
        .unwrap();
        assert_eq!(left.id(), right.id());
        assert_eq!(left.source_observations().len(), 2);
        left.validate().unwrap();
    }

    #[test]
    fn inference_remains_distinct_from_observation() {
        let a = observation(ObservationCoverage::Complete, EvidenceBasis::Observed, 3);
        let observed = DependencyClaimV1::new(
            "role:finance",
            "uses",
            "workflow:payroll",
            DependencyBasis::Observed,
            vec![a.id()],
            vec![],
        )
        .unwrap();
        let inferred = DependencyClaimV1::new(
            "role:finance",
            "uses",
            "workflow:payroll",
            DependencyBasis::Inferred,
            vec![a.id()],
            vec![],
        )
        .unwrap();
        assert_ne!(observed.id(), inferred.id());
    }

    #[test]
    fn dependency_without_evidence_is_rejected() {
        let result = DependencyClaimV1::new(
            "role:finance",
            "uses",
            "workflow:payroll",
            DependencyBasis::Declared,
            vec![],
            vec![],
        );
        assert_eq!(result.unwrap_err(), ObservationError::NoSourceObservations);
    }
}
