// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Additional trust-boundary controls for VIS-000.
//!
//! Kept separate from the implementation module so future migrations can retain this
//! regression surface even if provenance types move to a shared core crate.

#[cfg(test)]
mod tests {
    use crate::epistemic::{
        VisualCaptureClock, VisualEvidence, VisualEvidenceError, VisualObservationRef, VisualOrigin,
        VisualStreamRef,
    };

    fn observation(source_id: u64, epoch: u64, frame_id: u64) -> VisualObservationRef {
        VisualObservationRef::new(
            VisualStreamRef::new(source_id, epoch).unwrap(),
            frame_id,
            88_000 + frame_id,
            VisualCaptureClock::StreamMonotonic,
        )
    }

    #[test]
    fn confidence_never_upgrades_origin() {
        let evidence = VisualEvidence::simulated(Vec::new(), 1.0).unwrap();
        assert_eq!(evidence.origin(), VisualOrigin::Simulated);
        assert!(!evidence.origin().is_observed());
    }

    #[test]
    fn prediction_grounded_in_observation_remains_prediction() {
        let observation = observation(3, 1, 17);
        let evidence = VisualEvidence::predicted(vec![observation], 0.99).unwrap();
        assert_eq!(evidence.origin(), VisualOrigin::Predicted);
        assert_eq!(evidence.parent_observations(), &[observation]);
        assert_eq!(evidence.observation(), None);
    }

    #[test]
    fn duplicate_lineage_does_not_inflate_evidence() {
        let observation = observation(5, 1, 9);
        let result = VisualEvidence::inferred(vec![observation, observation], 0.5);
        assert_eq!(result, Err(VisualEvidenceError::DuplicateParentObservation));
    }

    #[test]
    fn same_frame_number_after_restart_is_not_same_observation() {
        let before_restart = observation(9, 1, 42);
        let after_restart = observation(9, 2, 42);
        assert_ne!(before_restart, after_restart);
    }
}
