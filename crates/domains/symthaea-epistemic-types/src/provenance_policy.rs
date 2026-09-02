use crate::{GroundingEvidence, ProvenanceEnvelope, ProvenanceError, RealityDomain};
use std::fmt;

/// Fail-closed transition policy for reality-domain provenance.
///
/// This policy never grants action authority. It only controls how epistemic
/// objects may derive new objects or acquire grounded provenance.
pub struct ProvenanceTransitionPolicy;

impl ProvenanceTransitionPolicy {
    /// Ordinary derivation may only create non-grounded epistemic objects.
    /// Grounded domains always require explicit [`GroundingEvidence`].
    pub const fn can_derive_to(_source: RealityDomain, target: RealityDomain) -> bool {
        !target.is_grounded()
    }

    /// Derive a new epistemic object while preserving parent taint/ancestry.
    pub fn derive(
        parent: &ProvenanceEnvelope,
        subject_sha256: impl Into<String>,
        target: RealityDomain,
        source_ids: Vec<String>,
        event_time_ns: Option<u64>,
        confidence: f32,
    ) -> Result<ProvenanceEnvelope, ProvenanceTransitionError> {
        if !Self::can_derive_to(parent.domain, target) {
            return Err(ProvenanceTransitionError::GroundedTargetRequiresEvidence(
                target,
            ));
        }
        ProvenanceEnvelope::derive(
            subject_sha256,
            target,
            source_ids,
            event_time_ns,
            confidence,
            [parent],
        )
        .map_err(ProvenanceTransitionError::Provenance)
    }

    /// Ground an epistemic object using evidence bound to the exact same subject.
    ///
    /// The evidence kind determines the target domain:
    /// direct observation -> PhysicalGrounded, commit receipt -> DigitalCommitted.
    pub fn ground(
        value: &ProvenanceEnvelope,
        evidence: GroundingEvidence,
    ) -> Result<ProvenanceEnvelope, ProvenanceTransitionError> {
        value
            .ground(evidence)
            .map_err(ProvenanceTransitionError::Provenance)
    }
}

#[derive(Debug)]
pub enum ProvenanceTransitionError {
    GroundedTargetRequiresEvidence(RealityDomain),
    Provenance(ProvenanceError),
}

impl fmt::Display for ProvenanceTransitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GroundedTargetRequiresEvidence(domain) => {
                write!(f, "ordinary derivation cannot target grounded domain {domain:?}")
            }
            Self::Provenance(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ProvenanceTransitionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GroundingEvidence;

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn observed() -> ProvenanceEnvelope {
        ProvenanceEnvelope::from_grounding(
            GroundingEvidence::direct_observation(A, "obs-a", "sensor-a", Some(1), 0.95)
                .unwrap(),
        )
    }

    #[test]
    fn grounded_parent_can_spawn_counterfactual_child_without_mutating_parent() {
        let parent = observed();
        let child = ProvenanceTransitionPolicy::derive(
            &parent,
            B,
            RealityDomain::Counterfactual,
            vec!["planner".into()],
            Some(2),
            0.7,
        )
        .unwrap();
        assert_eq!(parent.domain, RealityDomain::PhysicalGrounded);
        assert!(parent.may_enter_grounded_history());
        assert_eq!(child.domain, RealityDomain::Counterfactual);
        assert!(child.counterfactual_taint);
    }

    #[test]
    fn dream_to_counterfactual_remains_tainted() {
        let dream = ProvenanceEnvelope::new(A, RealityDomain::Dream, vec!["dream".into()], None, 0.4)
            .unwrap();
        let child = ProvenanceTransitionPolicy::derive(
            &dream,
            B,
            RealityDomain::Counterfactual,
            vec!["planner".into()],
            None,
            0.5,
        )
        .unwrap();
        assert!(child.counterfactual_taint);
        assert!(child.counterfactual_ancestry);
    }

    #[test]
    fn counterfactual_to_replay_remains_tainted() {
        let imagined = ProvenanceEnvelope::new(
            A,
            RealityDomain::Counterfactual,
            vec!["planner".into()],
            None,
            0.6,
        )
        .unwrap();
        let replay = ProvenanceTransitionPolicy::derive(
            &imagined,
            B,
            RealityDomain::Replay,
            vec!["replay".into()],
            None,
            0.6,
        )
        .unwrap();
        assert!(replay.counterfactual_taint);
    }

    #[test]
    fn ordinary_derivation_cannot_create_grounded_history() {
        let parent = observed();
        let error = ProvenanceTransitionPolicy::derive(
            &parent,
            B,
            RealityDomain::DigitalCommitted,
            vec![],
            None,
            0.8,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ProvenanceTransitionError::GroundedTargetRequiresEvidence(
                RealityDomain::DigitalCommitted
            )
        ));
    }

    #[test]
    fn unrelated_grounding_evidence_is_rejected() {
        let imagined = ProvenanceEnvelope::new(
            A,
            RealityDomain::Counterfactual,
            vec!["planner".into()],
            None,
            0.6,
        )
        .unwrap();
        let evidence = GroundingEvidence::direct_observation(
            B,
            "obs-b",
            "sensor-b",
            Some(5),
            0.9,
        )
        .unwrap();
        assert!(ProvenanceTransitionPolicy::ground(&imagined, evidence).is_err());
    }
}
