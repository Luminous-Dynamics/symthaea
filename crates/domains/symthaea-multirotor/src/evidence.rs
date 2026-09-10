// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed Embodiment Evidence v1 for the multirotor bridge.
//!
//! The legacy multirotor bridge stores `1 - similarity(current_hv, previous_hv)`
//! in its `prediction_error` compatibility field. That quantity is temporal HDC
//! novelty, not predicted-vs-observed error. This module publishes only the
//! proposition the current implementation can actually support and leaves stronger
//! evidence channels unavailable until they have qualified producers.

use crate::embodiment::FlightEmbodiment;
use crate::simulator::PhysicsSimulator;
use symthaea_core::embodiment_evidence::{
    ClockDomainId, EmbodimentEvidenceKind, EvidenceMetadataV1, EvidenceSourceClass,
    EvidenceValidationError, NormalizedScalarEvidenceV1, TimestampV1,
};
use symthaea_core::embodiment_evidence_provider::EmbodimentEvidenceProviderV1;

const NOVELTY_PROFILE_ID: &str = "symthaea.multirotor.hdc-temporal-novelty.v1";
const SIM_CLOCK_DOMAIN: &str = "symthaea.multirotor.sim.model-time";

fn simulator_timestamp(seconds: f64) -> Option<TimestampV1> {
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    let nanoseconds = seconds * 1_000_000_000.0;
    if !nanoseconds.is_finite() || nanoseconds > u64::MAX as f64 {
        return None;
    }
    let clock_domain = ClockDomainId::new(SIM_CLOCK_DOMAIN).ok()?;
    Some(TimestampV1::new(clock_domain, nanoseconds.round() as u64))
}

impl EmbodimentEvidenceProviderV1 for FlightEmbodiment {
    fn temporal_state_novelty_evidence(
        &self,
    ) -> Result<NormalizedScalarEvidenceV1, EvidenceValidationError> {
        // The first step has no prior frame, so its legacy zero is a sentinel rather
        // than an observed zero-novelty result. Require two completed steps.
        if self.total_steps() < 2 {
            return NormalizedScalarEvidenceV1::unavailable(
                EmbodimentEvidenceKind::TemporalStateNovelty,
            );
        }

        let state = self.simulator().state();
        let Some(observed_at) = simulator_timestamp(state.timestamp) else {
            return NormalizedScalarEvidenceV1::unavailable(
                EmbodimentEvidenceKind::TemporalStateNovelty,
            );
        };

        let mut metadata = EvidenceMetadataV1::available(
            EvidenceSourceClass::Derived,
            Some(observed_at),
        )?;
        metadata.profile_id = Some(NOVELTY_PROFILE_ID.to_string());

        NormalizedScalarEvidenceV1::new(
            EmbodimentEvidenceKind::TemporalStateNovelty,
            Some(self.telemetry().prediction_error),
            metadata,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::embodiment_evidence::{EvidenceAvailability, EvidenceSourceClass};
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_core::hdc::ContinuousHV;

    #[test]
    fn novelty_is_unavailable_until_two_frames_exist() {
        let mut bridge = FlightEmbodiment::new(&GenesisSeed::from_phrase("evidence-novelty"));
        let hv = ContinuousHV::random(16_384, 17);

        assert_eq!(
            bridge
                .temporal_state_novelty_evidence()
                .unwrap()
                .metadata
                .availability,
            EvidenceAvailability::Unavailable
        );

        bridge.step(&hv, 0.002, 0.7);
        assert_eq!(
            bridge
                .temporal_state_novelty_evidence()
                .unwrap()
                .metadata
                .availability,
            EvidenceAvailability::Unavailable
        );

        bridge.step(&hv, 0.002, 0.7);
        let evidence = bridge.temporal_state_novelty_evidence().unwrap();
        assert_eq!(evidence.metadata.availability, EvidenceAvailability::Available);
        assert_eq!(evidence.metadata.source, Some(EvidenceSourceClass::Derived));
        assert_eq!(evidence.kind, EmbodimentEvidenceKind::TemporalStateNovelty);
        assert_eq!(
            evidence.metadata.profile_id.as_deref(),
            Some(NOVELTY_PROFILE_ID)
        );
        assert_eq!(
            evidence
                .metadata
                .observed_at
                .as_ref()
                .unwrap()
                .clock_domain
                .as_str(),
            SIM_CLOCK_DOMAIN
        );
    }

    #[test]
    fn stronger_unimplemented_evidence_remains_unavailable() {
        let bridge = FlightEmbodiment::new(&GenesisSeed::from_phrase("evidence-nonclaims"));

        assert_eq!(
            bridge
                .predictive_residual_evidence()
                .unwrap()
                .metadata
                .availability,
            EvidenceAvailability::Unavailable
        );
        assert_eq!(
            bridge
                .actuator_health_evidence()
                .unwrap()
                .metadata
                .availability,
            EvidenceAvailability::Unavailable
        );
        assert_eq!(
            bridge
                .physical_reliability_evidence()
                .unwrap()
                .metadata
                .availability,
            EvidenceAvailability::Unavailable
        );
        assert_eq!(
            bridge
                .observation_confidence_evidence()
                .unwrap()
                .metadata
                .availability,
            EvidenceAvailability::Unavailable
        );
    }

    #[test]
    fn invalid_simulator_time_cannot_be_promoted_to_timestamped_evidence() {
        assert!(simulator_timestamp(f64::NAN).is_none());
        assert!(simulator_timestamp(-0.001).is_none());
        assert!(simulator_timestamp(f64::INFINITY).is_none());
    }
}
