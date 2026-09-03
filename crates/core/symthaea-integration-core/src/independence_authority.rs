// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit authority contract for positive evidence-independence attestations.
//!
//! Observation-local metadata may conservatively prove shared origin, but it may
//! not prove positive independence. This module defines the separate contract by
//! which a trusted, qualified authority may attest that two source-qualified
//! measurement lineages are independent under a stated basis.
//!
//! v0.1 deliberately stops at validation/admission. These attestations do **not**
//! yet raise [`crate::IndependenceAssessment::independent_lower_bound`]. Wiring
//! positive attestations into corroboration is deferred to a separately
//! qualified reasoning tranche.

use crate::ObservationId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const INDEPENDENCE_ATTESTATION_SCHEMA_VERSION: u16 = 1;

/// Source-qualified reference to one measurement lineage.
///
/// The integration id prevents two adapters that happen to reuse the same local
/// lineage string from becoming the same attestation subject.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EvidenceLineageRef {
    pub integration_id: String,
    pub lineage_id: String,
}

impl EvidenceLineageRef {
    pub fn new(integration_id: impl Into<String>, lineage_id: impl Into<String>) -> Self {
        Self {
            integration_id: integration_id.into(),
            lineage_id: lineage_id.into(),
        }
    }

    pub fn canonical_key(&self) -> String {
        format!(
            "lineage-v1|{}:{}|{}:{}",
            self.integration_id.len(),
            self.integration_id,
            self.lineage_id.len(),
            self.lineage_id
        )
    }
}

/// Why an authority concluded that two lineages are positively independent.
///
/// A basis is descriptive evidence, not authority by itself. Trust comes from
/// admission under an explicit [`IndependenceAuthorityPolicy`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum IndependenceBasis {
    DistinctPhysicalSensor,
    DistinctMeasurementTechnique,
    DistinctAdministrativeSource,
    ReviewedProvenance,
    CryptographicallyBoundProducer,
    Other(String),
}

/// Exact authority qualification trusted by local policy.
///
/// Matching only `authority_id` is intentionally insufficient: a single
/// authority may hold different qualifications with different scopes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct IndependenceAuthorityQualification {
    pub authority_id: String,
    pub qualification_id: String,
}

impl IndependenceAuthorityQualification {
    pub fn new(authority_id: impl Into<String>, qualification_id: impl Into<String>) -> Self {
        Self {
            authority_id: authority_id.into(),
            qualification_id: qualification_id.into(),
        }
    }
}

/// Positive-independence statement issued by a qualified authority.
///
/// `left` and `right` must be in canonical lexical order so A/B and B/A cannot
/// become distinct logical attestations. `issued_at_unix_ms` is knowledge time;
/// validity describes when the attested independence relation applies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependenceAttestation {
    pub schema_version: u16,
    pub attestation_id: String,
    pub left: EvidenceLineageRef,
    pub right: EvidenceLineageRef,
    pub basis: IndependenceBasis,
    pub authority: IndependenceAuthorityQualification,
    pub issued_at_unix_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_observation_ids: Vec<ObservationId>,
}

impl IndependenceAttestation {
    pub fn validate(&self) -> Result<(), IndependenceAttestationError> {
        if self.schema_version != INDEPENDENCE_ATTESTATION_SCHEMA_VERSION {
            return Err(IndependenceAttestationError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        require_non_empty("attestation_id", &self.attestation_id)?;
        validate_lineage("left", &self.left)?;
        validate_lineage("right", &self.right)?;
        require_non_empty("authority.authority_id", &self.authority.authority_id)?;
        require_non_empty(
            "authority.qualification_id",
            &self.authority.qualification_id,
        )?;
        if let IndependenceBasis::Other(value) = &self.basis {
            require_non_empty("basis.other", value)?;
        }

        let left_key = self.left.canonical_key();
        let right_key = self.right.canonical_key();
        if left_key == right_key {
            return Err(IndependenceAttestationError::SameLineage);
        }
        if left_key > right_key {
            return Err(IndependenceAttestationError::NonCanonicalPair);
        }

        if let (Some(from), Some(until)) = (self.valid_from_unix_ms, self.valid_until_unix_ms) {
            if from > until {
                return Err(IndependenceAttestationError::InvalidValidityWindow {
                    valid_from_unix_ms: from,
                    valid_until_unix_ms: until,
                });
            }
        }

        let mut evidence = BTreeSet::new();
        for observation_id in &self.evidence_observation_ids {
            require_non_empty("evidence_observation_id", observation_id.as_str())?;
            if !evidence.insert(observation_id.clone()) {
                return Err(IndependenceAttestationError::DuplicateEvidenceObservationId(
                    observation_id.clone(),
                ));
            }
        }

        Ok(())
    }

    /// An attestation cannot affect a historical query before it was issued.
    pub fn is_active_at(&self, at_unix_ms: u64) -> bool {
        self.issued_at_unix_ms <= at_unix_ms
            && self
                .valid_from_unix_ms
                .is_none_or(|from| at_unix_ms >= from)
            && self
                .valid_until_unix_ms
                .is_none_or(|until| at_unix_ms <= until)
    }
}

/// Local trust policy for positive-independence authority admission.
///
/// The attestation object is not self-authenticating. A caller must explicitly
/// allow the exact `(authority_id, qualification_id)` pair through trusted local
/// policy before the statement is admitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependenceAuthorityPolicy {
    pub trusted_qualifications: BTreeSet<IndependenceAuthorityQualification>,
    pub max_attestations: usize,
    pub max_evidence_refs_per_attestation: usize,
    pub max_string_bytes_per_attestation: usize,
}

impl Default for IndependenceAuthorityPolicy {
    fn default() -> Self {
        Self {
            trusted_qualifications: BTreeSet::new(),
            max_attestations: 4_096,
            max_evidence_refs_per_attestation: 64,
            max_string_bytes_per_attestation: 16 * 1024,
        }
    }
}

/// Bounded set of attestations admitted together under one explicit policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependenceAttestationSet {
    pub attestations: Vec<IndependenceAttestation>,
}

impl IndependenceAttestationSet {
    pub fn validate_with_policy(
        &self,
        policy: &IndependenceAuthorityPolicy,
    ) -> Result<(), IndependenceAttestationSetError> {
        if self.attestations.len() > policy.max_attestations {
            return Err(IndependenceAttestationSetError::TooManyAttestations {
                actual: self.attestations.len(),
                max: policy.max_attestations,
            });
        }

        let mut ids = BTreeSet::new();
        for (index, attestation) in self.attestations.iter().enumerate() {
            attestation
                .validate()
                .map_err(|reason| IndependenceAttestationSetError::InvalidAttestation {
                    index,
                    reason,
                })?;

            if !policy.trusted_qualifications.contains(&attestation.authority) {
                return Err(IndependenceAttestationSetError::UntrustedQualification {
                    index,
                    authority_id: attestation.authority.authority_id.clone(),
                    qualification_id: attestation.authority.qualification_id.clone(),
                });
            }

            if attestation.evidence_observation_ids.len() > policy.max_evidence_refs_per_attestation {
                return Err(IndependenceAttestationSetError::TooManyEvidenceRefs {
                    index,
                    actual: attestation.evidence_observation_ids.len(),
                    max: policy.max_evidence_refs_per_attestation,
                });
            }

            let string_bytes = attestation_string_bytes(attestation)?;
            if string_bytes > policy.max_string_bytes_per_attestation {
                return Err(IndependenceAttestationSetError::StringBudgetExceeded {
                    index,
                    actual: string_bytes,
                    max: policy.max_string_bytes_per_attestation,
                });
            }

            if !ids.insert(attestation.attestation_id.clone()) {
                return Err(IndependenceAttestationSetError::DuplicateAttestationId(
                    attestation.attestation_id.clone(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IndependenceAttestationError {
    #[error("unsupported independence-attestation schema version {0}")]
    UnsupportedSchemaVersion(u16),
    #[error("required field `{0}` is empty")]
    EmptyField(&'static str),
    #[error("positive independence cannot attest one lineage against itself")]
    SameLineage,
    #[error("independence-attestation lineage pair is not in canonical order")]
    NonCanonicalPair,
    #[error(
        "invalid independence-attestation validity window: {valid_from_unix_ms} > {valid_until_unix_ms}"
    )]
    InvalidValidityWindow {
        valid_from_unix_ms: u64,
        valid_until_unix_ms: u64,
    },
    #[error("duplicate supporting observation id `{0}`")]
    DuplicateEvidenceObservationId(ObservationId),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IndependenceAttestationSetError {
    #[error("attestation {index} is invalid: {reason}")]
    InvalidAttestation {
        index: usize,
        reason: IndependenceAttestationError,
    },
    #[error(
        "attestation {index} uses untrusted independence authority `{authority_id}` qualification `{qualification_id}`"
    )]
    UntrustedQualification {
        index: usize,
        authority_id: String,
        qualification_id: String,
    },
    #[error("attestation set has {actual} entries, limit is {max}")]
    TooManyAttestations { actual: usize, max: usize },
    #[error("attestation {index} has {actual} evidence refs, limit is {max}")]
    TooManyEvidenceRefs {
        index: usize,
        actual: usize,
        max: usize,
    },
    #[error("attestation {index} uses {actual} string bytes, limit is {max}")]
    StringBudgetExceeded {
        index: usize,
        actual: usize,
        max: usize,
    },
    #[error("attestation string-size accounting overflowed")]
    StringBudgetOverflow,
    #[error("duplicate independence attestation id `{0}`")]
    DuplicateAttestationId(String),
}

fn validate_lineage(
    prefix: &'static str,
    lineage: &EvidenceLineageRef,
) -> Result<(), IndependenceAttestationError> {
    if lineage.integration_id.trim().is_empty() {
        return Err(IndependenceAttestationError::EmptyField(match prefix {
            "left" => "left.integration_id",
            _ => "right.integration_id",
        }));
    }
    if lineage.lineage_id.trim().is_empty() {
        return Err(IndependenceAttestationError::EmptyField(match prefix {
            "left" => "left.lineage_id",
            _ => "right.lineage_id",
        }));
    }
    Ok(())
}

fn require_non_empty(
    field: &'static str,
    value: &str,
) -> Result<(), IndependenceAttestationError> {
    if value.trim().is_empty() {
        Err(IndependenceAttestationError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn attestation_string_bytes(
    attestation: &IndependenceAttestation,
) -> Result<usize, IndependenceAttestationSetError> {
    let mut total = 0usize;
    for value in [
        attestation.attestation_id.as_str(),
        attestation.left.integration_id.as_str(),
        attestation.left.lineage_id.as_str(),
        attestation.right.integration_id.as_str(),
        attestation.right.lineage_id.as_str(),
        attestation.authority.authority_id.as_str(),
        attestation.authority.qualification_id.as_str(),
    ] {
        total = total
            .checked_add(value.len())
            .ok_or(IndependenceAttestationSetError::StringBudgetOverflow)?;
    }
    if let IndependenceBasis::Other(value) = &attestation.basis {
        total = total
            .checked_add(value.len())
            .ok_or(IndependenceAttestationSetError::StringBudgetOverflow)?;
    }
    for id in &attestation.evidence_observation_ids {
        total = total
            .checked_add(id.as_str().len())
            .ok_or(IndependenceAttestationSetError::StringBudgetOverflow)?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn authority() -> IndependenceAuthorityQualification {
        IndependenceAuthorityQualification::new("security-review", "independence-v1")
    }

    fn attestation() -> IndependenceAttestation {
        IndependenceAttestation {
            schema_version: INDEPENDENCE_ATTESTATION_SCHEMA_VERSION,
            attestation_id: "attestation-1".into(),
            left: EvidenceLineageRef::new("kubernetes", "kubelet-readiness"),
            right: EvidenceLineageRef::new("prometheus", "blackbox-probe"),
            basis: IndependenceBasis::DistinctMeasurementTechnique,
            authority: authority(),
            issued_at_unix_ms: 100,
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            evidence_observation_ids: vec![ObservationId::new("obs-1")],
        }
    }

    fn policy() -> IndependenceAuthorityPolicy {
        IndependenceAuthorityPolicy {
            trusted_qualifications: BTreeSet::from([authority()]),
            ..IndependenceAuthorityPolicy::default()
        }
    }

    #[test]
    fn trusted_qualified_attestation_is_admitted() {
        let set = IndependenceAttestationSet {
            attestations: vec![attestation()],
        };
        assert!(set.validate_with_policy(&policy()).is_ok());
    }

    #[test]
    fn authority_name_alone_cannot_self_certify_qualification() {
        let mut unqualified = attestation();
        unqualified.authority.qualification_id = "made-up".into();
        let set = IndependenceAttestationSet {
            attestations: vec![unqualified],
        };
        assert!(matches!(
            set.validate_with_policy(&policy()),
            Err(IndependenceAttestationSetError::UntrustedQualification { .. })
        ));
    }

    #[test]
    fn untrusted_authority_is_rejected() {
        let mut untrusted = attestation();
        untrusted.authority.authority_id = "adapter-self-report".into();
        let set = IndependenceAttestationSet {
            attestations: vec![untrusted],
        };
        assert!(matches!(
            set.validate_with_policy(&policy()),
            Err(IndependenceAttestationSetError::UntrustedQualification { .. })
        ));
    }

    #[test]
    fn reversed_pair_is_rejected_to_prevent_duplicate_logical_attestations() {
        let mut reversed = attestation();
        std::mem::swap(&mut reversed.left, &mut reversed.right);
        assert_eq!(
            reversed.validate(),
            Err(IndependenceAttestationError::NonCanonicalPair)
        );
    }

    #[test]
    fn future_issued_attestation_does_not_time_travel() {
        let attestation = attestation();
        assert!(!attestation.is_active_at(99));
        assert!(attestation.is_active_at(100));
    }

    #[test]
    fn duplicate_supporting_observations_fail_closed() {
        let mut duplicate = attestation();
        duplicate.evidence_observation_ids =
            vec![ObservationId::new("same"), ObservationId::new("same")];
        assert!(matches!(
            duplicate.validate(),
            Err(IndependenceAttestationError::DuplicateEvidenceObservationId(_))
        ));
    }

    #[test]
    fn duplicate_attestation_ids_fail_closed() {
        let a = attestation();
        let b = a.clone();
        let set = IndependenceAttestationSet {
            attestations: vec![a, b],
        };
        assert!(matches!(
            set.validate_with_policy(&policy()),
            Err(IndependenceAttestationSetError::DuplicateAttestationId(_))
        ));
    }

    #[test]
    fn validated_attestations_do_not_implicitly_change_independence_assessment() {
        // This module intentionally defines only the authority/admission contract.
        // `assess_independence` has no attestation parameter in v0.1, which keeps
        // positive corroboration disabled until a separately qualified tranche.
        let set = IndependenceAttestationSet {
            attestations: vec![attestation()],
        };
        assert!(set.validate_with_policy(&policy()).is_ok());
    }
}
