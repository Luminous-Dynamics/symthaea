// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::{
    AuthorityFacet, AuthorityLevel, AuthorityProfile, EvidenceState, ResearchId, Sha256Digest,
};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeSet;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum EvidenceKind {
    StructuralDeclaration,
    RawObservation,
    CalibratedObservation,
    DerivedMeasurement,
    DeterministicComputation,
    NumericalSimulation,
    SymbolicDerivation,
    BoundedVerification,
    Counterexample,
    FormalProof,
    FormalProofRejection,
    CausalStudy,
    ReplicationStudy,
    LiteratureEvidence,
    HumanReview,
    FormalizationReview,
    NegativeControl,
    Ablation,
    RobustnessTest,
    OutOfDistributionTest,
}

impl EvidenceKind {
    /// Hard ceiling for an ordinary evidence record.
    ///
    /// `EvidenceRecord` is a structural/bound evidence container, never a
    /// qualification capability. A later trusted qualifier may wrap a validated
    /// record in a separate non-forgeable type; it may not mutate this record
    /// into a qualified one.
    pub fn maximum_record_authority(self) -> AuthorityProfile {
        use AuthorityFacet as F;
        use AuthorityLevel as L;
        match self {
            Self::StructuralDeclaration => {
                AuthorityProfile::empty().with(F::Provenance, L::Declared)
            }
            Self::RawObservation | Self::CalibratedObservation | Self::DerivedMeasurement => {
                AuthorityProfile::empty()
                    .with(F::Provenance, L::Bound)
                    .with(F::Empirical, L::Bound)
            }
            Self::DeterministicComputation
            | Self::NumericalSimulation
            | Self::SymbolicDerivation => AuthorityProfile::empty()
                .with(F::Provenance, L::Bound)
                .with(F::Execution, L::Bound),
            Self::BoundedVerification => AuthorityProfile::empty()
                .with(F::Provenance, L::Bound)
                .with(F::Execution, L::Bound)
                .with(F::Formal, L::Declared),
            Self::Counterexample | Self::FormalProof | Self::FormalProofRejection => {
                AuthorityProfile::empty()
                    .with(F::Provenance, L::Bound)
                    .with(F::Execution, L::Bound)
                    .with(F::Formal, L::Bound)
            }
            Self::CausalStudy => AuthorityProfile::empty()
                .with(F::Provenance, L::Bound)
                .with(F::Execution, L::Bound)
                .with(F::Empirical, L::Bound)
                .with(F::Causal, L::Bound),
            Self::ReplicationStudy => AuthorityProfile::empty()
                .with(F::Provenance, L::Bound)
                .with(F::Execution, L::Bound)
                .with(F::Empirical, L::Bound)
                .with(F::Replication, L::Bound),
            Self::LiteratureEvidence | Self::HumanReview | Self::FormalizationReview => {
                AuthorityProfile::empty().with(F::Provenance, L::Bound)
            }
            Self::NegativeControl
            | Self::Ablation
            | Self::RobustnessTest
            | Self::OutOfDistributionTest => AuthorityProfile::empty()
                .with(F::Provenance, L::Bound)
                .with(F::Execution, L::Bound)
                .with(F::Empirical, L::Bound),
        }
    }
}

/// Validated evidence object. Authority-bearing fields are private so callers
/// cannot construct a valid record and then mutate it into a stronger one.
///
/// An optional `qualification_sha256` is only a *reference* to a qualification
/// lineage. Possessing or inventing such a digest grants no additional
/// authority; qualification itself belongs to a later non-forgeable wrapper.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvidenceRecord {
    evidence_id: ResearchId,
    subject_sha256: Sha256Digest,
    kind: EvidenceKind,
    state: EvidenceState,
    artifact_sha256: Sha256Digest,
    provenance_roots: BTreeSet<Sha256Digest>,
    qualification_sha256: Option<Sha256Digest>,
    authority: AuthorityProfile,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceIssue {
    PositiveAuthorityRequiresPass,
    AuthorityExceedsRecordCeiling,
}

impl EvidenceRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        evidence_id: ResearchId,
        subject_sha256: Sha256Digest,
        kind: EvidenceKind,
        state: EvidenceState,
        artifact_sha256: Sha256Digest,
        provenance_roots: impl IntoIterator<Item = Sha256Digest>,
        qualification_sha256: Option<Sha256Digest>,
        authority: AuthorityProfile,
    ) -> Result<Self, Vec<EvidenceIssue>> {
        let record = Self {
            evidence_id,
            subject_sha256,
            kind,
            state,
            artifact_sha256,
            provenance_roots: provenance_roots.into_iter().collect(),
            qualification_sha256,
            authority,
        };
        let issues = record.validate();
        if issues.is_empty() {
            Ok(record)
        } else {
            Err(issues)
        }
    }

    pub fn evidence_id(&self) -> &ResearchId {
        &self.evidence_id
    }

    pub fn subject_sha256(&self) -> &Sha256Digest {
        &self.subject_sha256
    }

    pub fn kind(&self) -> EvidenceKind {
        self.kind
    }

    pub fn state(&self) -> EvidenceState {
        self.state
    }

    pub fn artifact_sha256(&self) -> &Sha256Digest {
        &self.artifact_sha256
    }

    pub fn provenance_roots(&self) -> &BTreeSet<Sha256Digest> {
        &self.provenance_roots
    }

    pub fn qualification_sha256(&self) -> Option<&Sha256Digest> {
        self.qualification_sha256.as_ref()
    }

    pub fn authority(&self) -> &AuthorityProfile {
        &self.authority
    }

    pub fn validate(&self) -> Vec<EvidenceIssue> {
        let mut issues = Vec::new();
        if !self.state.permits_positive_authority() && !self.authority.is_empty() {
            issues.push(EvidenceIssue::PositiveAuthorityRequiresPass);
        }
        if !self.authority.is_within(&self.kind.maximum_record_authority()) {
            issues.push(EvidenceIssue::AuthorityExceedsRecordCeiling);
        }
        issues
    }
}

#[derive(Deserialize)]
struct EvidenceRecordWire {
    evidence_id: ResearchId,
    subject_sha256: Sha256Digest,
    kind: EvidenceKind,
    state: EvidenceState,
    artifact_sha256: Sha256Digest,
    provenance_roots: BTreeSet<Sha256Digest>,
    qualification_sha256: Option<Sha256Digest>,
    authority: AuthorityProfile,
}

impl<'de> Deserialize<'de> for EvidenceRecord {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = EvidenceRecordWire::deserialize(deserializer)?;
        Self::new(
            wire.evidence_id,
            wire.subject_sha256,
            wire.kind,
            wire.state,
            wire.artifact_sha256,
            wire.provenance_roots,
            wire.qualification_sha256,
            wire.authority,
        )
        .map_err(|issues| serde::de::Error::custom(format!("invalid evidence record: {issues:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: &str) -> ResearchId {
        ResearchId::parse(value).unwrap()
    }

    fn digest(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    #[test]
    fn simulation_cannot_claim_empirical_authority() {
        let requested = AuthorityProfile::empty()
            .with(AuthorityFacet::Execution, AuthorityLevel::Bound)
            .with(AuthorityFacet::Empirical, AuthorityLevel::Bound);
        let result = EvidenceRecord::new(
            id("SIM-001"),
            digest("subject"),
            EvidenceKind::NumericalSimulation,
            EvidenceState::Pass,
            digest("artifact"),
            [],
            None,
            requested,
        );
        assert!(result.is_err());
    }

    #[test]
    fn bounded_verification_cannot_claim_bound_formal_authority() {
        let requested =
            AuthorityProfile::empty().with(AuthorityFacet::Formal, AuthorityLevel::Bound);
        let result = EvidenceRecord::new(
            id("BOUND-001"),
            digest("subject"),
            EvidenceKind::BoundedVerification,
            EvidenceState::Pass,
            digest("artifact"),
            [],
            None,
            requested,
        );
        assert!(result.is_err());
    }

    #[test]
    fn invalid_result_cannot_carry_positive_authority() {
        let requested =
            AuthorityProfile::empty().with(AuthorityFacet::Provenance, AuthorityLevel::Bound);
        let result = EvidenceRecord::new(
            id("INVALID-001"),
            digest("subject"),
            EvidenceKind::DeterministicComputation,
            EvidenceState::Invalid,
            digest("artifact"),
            [],
            None,
            requested,
        );
        assert!(result.is_err());
    }

    #[test]
    fn qualification_digest_is_only_a_reference() {
        let requested = AuthorityProfile::empty()
            .with(AuthorityFacet::Provenance, AuthorityLevel::Bound)
            .with(AuthorityFacet::Execution, AuthorityLevel::Bound)
            .with(AuthorityFacet::Formal, AuthorityLevel::Bound);
        let qualification = digest("qualification-reference");
        let record = EvidenceRecord::new(
            id("PROOF-BOUND-001"),
            digest("subject"),
            EvidenceKind::FormalProof,
            EvidenceState::Pass,
            digest("artifact"),
            [],
            Some(qualification.clone()),
            requested.clone(),
        )
        .unwrap();
        assert_eq!(record.authority(), &requested);
        assert_eq!(record.qualification_sha256(), Some(&qualification));
    }

    #[test]
    fn wire_authority_cannot_exceed_kind_ceiling() {
        let subject = digest("subject");
        let artifact = digest("artifact");
        let forged = format!(
            r#"{{"evidence_id":"SIM-001","subject_sha256":"{subject}","kind":"NumericalSimulation","state":"Pass","artifact_sha256":"{artifact}","provenance_roots":[],"qualification_sha256":null,"authority":{{"levels":{{"Empirical":"Bound"}}}}}}"#
        );
        assert!(serde_json::from_str::<EvidenceRecord>(&forged).is_err());
    }

    #[test]
    fn pass_within_kind_ceiling_is_valid() {
        let requested = AuthorityProfile::empty()
            .with(AuthorityFacet::Provenance, AuthorityLevel::Bound)
            .with(AuthorityFacet::Execution, AuthorityLevel::Bound);
        let record = EvidenceRecord::new(
            id("COMP-001"),
            digest("subject"),
            EvidenceKind::DeterministicComputation,
            EvidenceState::Pass,
            digest("artifact"),
            [],
            None,
            requested.clone(),
        )
        .unwrap();
        assert_eq!(record.authority(), &requested);
    }
}
