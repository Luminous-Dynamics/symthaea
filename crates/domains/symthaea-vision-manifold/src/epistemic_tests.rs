// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Additional trust-boundary controls for VIS-000.
//!
//! Kept separate from the implementation module so future migrations can retain this
//! regression surface even if provenance types move to a shared core crate.

#[cfg(test)]
mod tests {
    use crate::epistemic::{VisualEvidence, VisualEvidenceError, VisualObservationRef, VisualOrigin};

    #[test]
    fn confidence_never_upgrades_origin() {
        let evidence = VisualEvidence::simulated(Vec::new(), 1.0).unwrap();
        assert_eq!(evidence.origin(), VisualOrigin::Simulated);
        assert!(!evidence.origin().is_observed());
    }

    #[test]
    fn prediction_grounded_in_observation_remains_prediction() {
        let observation = VisualObservationRef::new(3, 17, 88_000);
        let evidence = VisualEvidence::predicted(vec![observation], 0.99).unwrap();
        assert_eq!(evidence.origin(), VisualOrigin::Predicted);
        assert_eq!(evidence.parent_observations(), &[observation]);
        assert_eq!(evidence.observation(), None);
    }

    #[test]
    fn duplicate_lineage_does_not_inflate_evidence() {
        let observation = VisualObservationRef::new(5, 9, 100);
        let result = VisualEvidence::inferred(vec![observation, observation], 0.5);
        assert_eq!(result, Err(VisualEvidenceError::DuplicateParentObservation));
    }
}
