use crate::{ProvenanceEnvelope, ProvenanceError, RealityDomain};
use serde::{Deserialize, Serialize};
use std::fmt;

/// What kind of transformation produced a derived epistemic object.
///
/// This is descriptive/auditable metadata, not authority. In particular,
/// `ObservationEncoding` does not make its output physically grounded; grounded
/// domains still require explicit `GroundingEvidence` through `ProvenanceEnvelope`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpistemicTransformKind {
    ObservationEncoding,
    InternalInference,
    CounterfactualSimulation,
    ReplayReconstruction,
    MemoryConsolidation,
    ExternalImport,
    Other,
}

/// Immutable receipt binding a derived epistemic object to the exact parent
/// objects and transform implementation that produced it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpistemicDerivationReceipt {
    pub child_subject_sha256: String,
    pub parent_subject_sha256s: Vec<String>,
    pub transform_kind: EpistemicTransformKind,
    pub transform_id: String,
    pub transform_version: String,
    pub event_time_ns: Option<u64>,
}

impl EpistemicDerivationReceipt {
    pub fn parent_count(&self) -> usize {
        self.parent_subject_sha256s.len()
    }
}

#[derive(Debug)]
pub enum EpistemicDerivationError {
    EmptyTransformId,
    EmptyTransformVersion,
    Provenance(ProvenanceError),
}

impl fmt::Display for EpistemicDerivationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTransformId => write!(f, "epistemic derivation transform_id must be non-empty"),
            Self::EmptyTransformVersion => {
                write!(f, "epistemic derivation transform_version must be non-empty")
            }
            Self::Provenance(error) => write!(f, "epistemic derivation provenance error: {error}"),
        }
    }
}

impl std::error::Error for EpistemicDerivationError {}

impl From<ProvenanceError> for EpistemicDerivationError {
    fn from(value: ProvenanceError) -> Self {
        Self::Provenance(value)
    }
}

/// Derive an ungrounded epistemic object and emit an immutable derivation receipt.
///
/// The child envelope is produced by the same fail-closed taint propagation as
/// `ProvenanceEnvelope::derive`: any active counterfactual taint in a parent is
/// inherited transitively, and a grounded target domain is rejected. The receipt
/// additionally binds the child to exact parent subject digests plus the named
/// transform implementation/version.
///
/// This function deliberately cannot clear taint and cannot create
/// `PhysicalGrounded`/`DigitalCommitted` objects. Those transitions require
/// subject-bound grounding evidence.
pub fn derive_with_receipt(
    child_subject_sha256: impl Into<String>,
    domain: RealityDomain,
    source_ids: Vec<String>,
    event_time_ns: Option<u64>,
    confidence: f32,
    parents: &[ProvenanceEnvelope],
    transform_kind: EpistemicTransformKind,
    transform_id: impl Into<String>,
    transform_version: impl Into<String>,
) -> Result<(ProvenanceEnvelope, EpistemicDerivationReceipt), EpistemicDerivationError> {
    let transform_id = transform_id.into();
    if transform_id.trim().is_empty() {
        return Err(EpistemicDerivationError::EmptyTransformId);
    }
    let transform_version = transform_version.into();
    if transform_version.trim().is_empty() {
        return Err(EpistemicDerivationError::EmptyTransformVersion);
    }

    let child_subject_sha256 = child_subject_sha256.into();
    let envelope = ProvenanceEnvelope::derive(
        child_subject_sha256.clone(),
        domain,
        source_ids,
        event_time_ns,
        confidence,
        parents.iter(),
    )?;

    let receipt = EpistemicDerivationReceipt {
        child_subject_sha256,
        parent_subject_sha256s: parents
            .iter()
            .map(|parent| parent.subject_sha256.clone())
            .collect(),
        transform_kind,
        transform_id,
        transform_version,
        event_time_ns,
    };

    Ok((envelope, receipt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GroundingEvidence, ProvenanceError};

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn observed(subject: &str) -> ProvenanceEnvelope {
        ProvenanceEnvelope::from_grounding(
            GroundingEvidence::direct_observation(subject, "obs", "sensor", Some(1), 0.95)
                .unwrap(),
        )
    }

    #[test]
    fn receipt_binds_exact_parent_subjects_and_transform() {
        let parent = observed(A);
        let (derived, receipt) = derive_with_receipt(
            B,
            RealityDomain::Unknown,
            vec!["reasoner".into()],
            Some(2),
            0.8,
            &[parent],
            EpistemicTransformKind::InternalInference,
            "symthaea.reasoner",
            "v1",
        )
        .unwrap();

        assert_eq!(derived.subject_sha256, B);
        assert_eq!(receipt.child_subject_sha256, B);
        assert_eq!(receipt.parent_subject_sha256s, vec![A]);
        assert_eq!(receipt.parent_count(), 1);
        assert_eq!(receipt.transform_id, "symthaea.reasoner");
        assert_eq!(receipt.transform_version, "v1");
        assert!(!derived.may_enter_grounded_history());
    }

    #[test]
    fn counterfactual_parent_taint_propagates_through_receipted_derivation() {
        let imagined = ProvenanceEnvelope::new(
            A,
            RealityDomain::Counterfactual,
            vec!["planner".into()],
            None,
            0.6,
        )
        .unwrap();
        let grounded = observed(B);

        let (derived, receipt) = derive_with_receipt(
            C,
            RealityDomain::Unknown,
            vec!["synthesizer".into()],
            Some(3),
            0.7,
            &[grounded, imagined],
            EpistemicTransformKind::InternalInference,
            "symthaea.synthesis",
            "v1",
        )
        .unwrap();

        assert!(derived.counterfactual_taint);
        assert!(derived.counterfactual_ancestry);
        assert_eq!(receipt.parent_subject_sha256s, vec![B, A]);
    }

    #[test]
    fn derivation_cannot_create_grounded_history() {
        let parent = observed(A);
        let result = derive_with_receipt(
            B,
            RealityDomain::PhysicalGrounded,
            vec!["encoder".into()],
            None,
            0.9,
            &[parent],
            EpistemicTransformKind::ObservationEncoding,
            "symthaea.encoder",
            "v1",
        );
        assert!(matches!(
            result,
            Err(EpistemicDerivationError::Provenance(
                ProvenanceError::GroundedDomainRequiresEvidence(RealityDomain::PhysicalGrounded)
            ))
        ));
    }

    #[test]
    fn empty_transform_identity_fails_closed_before_derivation() {
        let parent = observed(A);
        assert!(matches!(
            derive_with_receipt(
                B,
                RealityDomain::Unknown,
                vec!["reasoner".into()],
                None,
                0.8,
                &[parent.clone()],
                EpistemicTransformKind::InternalInference,
                " ",
                "v1",
            ),
            Err(EpistemicDerivationError::EmptyTransformId)
        ));
        assert!(matches!(
            derive_with_receipt(
                B,
                RealityDomain::Unknown,
                vec!["reasoner".into()],
                None,
                0.8,
                &[parent],
                EpistemicTransformKind::InternalInference,
                "symthaea.reasoner",
                "",
            ),
            Err(EpistemicDerivationError::EmptyTransformVersion)
        ));
    }
}
