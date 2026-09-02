use crate::{ProvenanceEnvelope, RealityDomain};

/// Retrieval view over epistemic objects.
///
/// The default history-safe mode should be chosen explicitly by callers; this
/// type has no implicit global default so a query cannot silently change scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceRetrievalMode {
    /// Only untainted PhysicalGrounded / DigitalCommitted objects.
    GroundedHistory,
    /// Grounded history plus untainted Imported objects.
    GroundedOrImported,
    /// Counterfactual/Dream objects and any object carrying active taint.
    CounterfactualOnly,
    /// No filtering. Provenance remains attached and inspectable.
    AllWithProvenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RetrievalAudit {
    pub considered: usize,
    pub returned: usize,
    pub excluded_taint: usize,
    pub excluded_domain: usize,
}

impl ProvenanceRetrievalMode {
    pub const fn allows(self, envelope: &ProvenanceEnvelope) -> bool {
        match self {
            Self::GroundedHistory => envelope.may_enter_grounded_history(),
            Self::GroundedOrImported => {
                envelope.may_enter_grounded_history()
                    || (matches!(envelope.domain, RealityDomain::Imported)
                        && !envelope.counterfactual_taint)
            }
            Self::CounterfactualOnly => {
                envelope.counterfactual_taint
                    || matches!(envelope.domain, RealityDomain::Counterfactual | RealityDomain::Dream)
            }
            Self::AllWithProvenance => true,
        }
    }

    /// Return indices rather than cloning application-specific memory records.
    /// Callers can apply these indices to their own storage while retaining an
    /// auditable account of exclusions.
    pub fn filter_indices(self, envelopes: &[ProvenanceEnvelope]) -> (Vec<usize>, RetrievalAudit) {
        let mut indices = Vec::new();
        let mut audit = RetrievalAudit::default();

        for (index, envelope) in envelopes.iter().enumerate() {
            audit.considered += 1;
            if self.allows(envelope) {
                indices.push(index);
                audit.returned += 1;
                continue;
            }

            if envelope.counterfactual_taint {
                audit.excluded_taint += 1;
            } else {
                audit.excluded_domain += 1;
            }
        }

        (indices, audit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GroundingEvidence;

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn observed(subject: &str) -> ProvenanceEnvelope {
        ProvenanceEnvelope::from_grounding(
            GroundingEvidence::direct_observation(subject, "obs", "sensor", Some(1), 0.9)
                .unwrap(),
        )
    }

    #[test]
    fn grounded_history_excludes_counterfactual_and_imported_objects() {
        let grounded = observed(A);
        let imagined = ProvenanceEnvelope::new(
            B,
            RealityDomain::Counterfactual,
            vec!["planner".into()],
            None,
            0.6,
        )
        .unwrap();
        let imported = ProvenanceEnvelope::new(
            C,
            RealityDomain::Imported,
            vec!["external".into()],
            None,
            0.7,
        )
        .unwrap();
        let values = vec![grounded, imagined, imported];
        let (indices, audit) = ProvenanceRetrievalMode::GroundedHistory.filter_indices(&values);
        assert_eq!(indices, vec![0]);
        assert_eq!(audit.considered, 3);
        assert_eq!(audit.returned, 1);
        assert_eq!(audit.excluded_taint, 1);
        assert_eq!(audit.excluded_domain, 1);
    }

    #[test]
    fn imported_can_be_requested_explicitly_without_admitting_taint() {
        let imported = ProvenanceEnvelope::new(
            A,
            RealityDomain::Imported,
            vec!["external".into()],
            None,
            0.7,
        )
        .unwrap();
        let imagined = ProvenanceEnvelope::new(
            B,
            RealityDomain::Dream,
            vec!["dream".into()],
            None,
            0.5,
        )
        .unwrap();
        let values = vec![imported, imagined];
        let (indices, _) = ProvenanceRetrievalMode::GroundedOrImported.filter_indices(&values);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn counterfactual_view_selects_active_taint() {
        let grounded = observed(A);
        let imagined = ProvenanceEnvelope::new(
            B,
            RealityDomain::Counterfactual,
            vec!["planner".into()],
            None,
            0.6,
        )
        .unwrap();
        let values = vec![grounded, imagined];
        let (indices, audit) = ProvenanceRetrievalMode::CounterfactualOnly.filter_indices(&values);
        assert_eq!(indices, vec![1]);
        assert_eq!(audit.returned, 1);
    }

    #[test]
    fn all_mode_never_silently_drops_items() {
        let values = vec![
            observed(A),
            ProvenanceEnvelope::new(
                B,
                RealityDomain::Unknown,
                vec![],
                None,
                0.1,
            )
            .unwrap(),
        ];
        let (indices, audit) = ProvenanceRetrievalMode::AllWithProvenance.filter_indices(&values);
        assert_eq!(indices, vec![0, 1]);
        assert_eq!(audit.considered, audit.returned);
        assert_eq!(audit.excluded_taint + audit.excluded_domain, 0);
    }
}
