// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Temporal-admission wrapper around the existing chemical multimodal bridge.
//!
//! This module deliberately does not reimplement HDC aggregation, confidence,
//! conflict, ordering, or evidence-bundle logic. It first classifies generic
//! timing evidence, then calls [`crate::ChemicalModalBridge::aggregate`] only
//! when same-cycle admission is justified. Ambiguous or definitely-outside
//! evidence is preserved without synthesizing an aggregate vector.

use std::fmt;

use crate::{
    ChemicalModalBridge, ChemicalModalBridgeError, ChemicalModalBridgeInput,
    ChemicalTemporalAdmission, ChemicalTemporalAdmissionStatus, ChemicalTimeAlignmentError,
    TimedChemicalPercept, classify_chemical_temporal_admission,
};

/// Result of combining temporal admission with the unchanged chemical HDC bridge.
#[derive(Debug, Clone, PartialEq)]
pub enum TimedChemicalAggregation {
    /// Timing evidence permits one same-cycle aggregate. `timed_components`
    /// preserve the generic receipts while `input` contains the existing
    /// evidence-preserving HDC aggregate.
    Aggregated {
        admission: ChemicalTemporalAdmission,
        input: ChemicalModalBridgeInput,
        timed_components: Vec<TimedChemicalPercept>,
    },
    /// Timing evidence does not justify one simultaneous aggregate. No HDC
    /// aggregate is created; all original timed percepts remain available.
    Abstained {
        admission: ChemicalTemporalAdmission,
        timed_components: Vec<TimedChemicalPercept>,
    },
}

impl TimedChemicalAggregation {
    pub fn admission(&self) -> &ChemicalTemporalAdmission {
        match self {
            Self::Aggregated { admission, .. } | Self::Abstained { admission, .. } => admission,
        }
    }

    pub fn input(&self) -> Option<&ChemicalModalBridgeInput> {
        match self {
            Self::Aggregated { input, .. } => Some(input),
            Self::Abstained { .. } => None,
        }
    }

    pub fn timed_components(&self) -> &[TimedChemicalPercept] {
        match self {
            Self::Aggregated {
                timed_components, ..
            }
            | Self::Abstained {
                timed_components, ..
            } => timed_components,
        }
    }

    pub fn was_aggregated(&self) -> bool {
        matches!(self, Self::Aggregated { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimedChemicalAggregationError {
    Time(ChemicalTimeAlignmentError),
    Bridge(ChemicalModalBridgeError),
}

impl fmt::Display for TimedChemicalAggregationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Time(error) => write!(f, "chemical temporal admission failed: {error}"),
            Self::Bridge(error) => write!(f, "chemical multimodal aggregation failed: {error:?}"),
        }
    }
}

impl std::error::Error for TimedChemicalAggregationError {}

impl From<ChemicalTimeAlignmentError> for TimedChemicalAggregationError {
    fn from(value: ChemicalTimeAlignmentError) -> Self {
        Self::Time(value)
    }
}

impl From<ChemicalModalBridgeError> for TimedChemicalAggregationError {
    fn from(value: ChemicalModalBridgeError) -> Self {
        Self::Bridge(value)
    }
}

/// Apply generic bounded temporal admission before the existing chemical HDC
/// aggregation path.
///
/// During migration, multi-source raw observations still need their legacy
/// `ChemicalClockDomainId` populated consistently because the legacy bridge
/// independently rechecks nominal skew/domain metadata. The generic sidecar is
/// the stronger temporal evidence path; the duplicate legacy metadata can be
/// removed in a later compatibility-breaking release once all callers migrate.
pub fn aggregate_timed_chemical_percepts(
    bridge: &ChemicalModalBridge,
    timed_components: Vec<TimedChemicalPercept>,
) -> Result<TimedChemicalAggregation, TimedChemicalAggregationError> {
    let admission = classify_chemical_temporal_admission(
        &timed_components,
        bridge.config().max_component_skew_us,
    )?;

    if !admission.permits_same_cycle_aggregation() {
        return Ok(TimedChemicalAggregation::Abstained {
            admission,
            timed_components,
        });
    }

    let percepts: Vec<_> = timed_components
        .iter()
        .map(|component| component.percept().clone())
        .collect();
    let input = bridge.aggregate(&percepts)?;

    Ok(TimedChemicalAggregation::Aggregated {
        admission,
        input,
        timed_components,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ChemicalClockDomainId, ChemicalEncodingSpaceId, ChemicalFingerprint, ChemicalModality,
        ChemicalObservation,
    };
    use symthaea_core::hdc::{HDC_DIMENSION, unified_hv::ContinuousHV};
    use symthaea_time_integrity::{
        ClockDomainId, ClockEpochId, ContinuityStatus, TimeIntegrityReceipt, TimeUncertainty,
    };

    fn generic_domain() -> ClockDomainId {
        ClockDomainId::new("rig-01/monotonic").unwrap()
    }

    fn epoch() -> ClockEpochId {
        ClockEpochId::new("rig-01-boot-7").unwrap()
    }

    fn receipt(error_us: u64) -> TimeIntegrityReceipt {
        TimeIntegrityReceipt::declared(generic_domain())
            .with_epoch(epoch())
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(error_us))
    }

    fn percept(timestamp_us: u64, source: &str, seed: u64) -> crate::ChemicalPercept {
        let mut evidence = ChemicalObservation::new(
            timestamp_us,
            ChemicalModality::Olfactory,
            source,
            vec![],
        );
        evidence.clock_domain = Some(ChemicalClockDomainId::new("rig-01/monotonic").unwrap());
        crate::ChemicalPercept {
            evidence,
            fingerprint: ChemicalFingerprint {
                vector: ContinuousHV::random(HDC_DIMENSION, seed),
                confidence: 0.9,
                used_channels: 1,
                ignored_channels: 0,
                encoding_space_id: ChemicalEncodingSpaceId::from_bytes([7; 32]),
            },
        }
    }

    fn timed(timestamp_us: u64, error_us: u64, source: &str, seed: u64) -> TimedChemicalPercept {
        TimedChemicalPercept::new(percept(timestamp_us, source, seed), receipt(error_us)).unwrap()
    }

    #[test]
    fn definitely_within_uses_existing_bridge_math_and_preserves_evidence_identity() {
        let bridge = ChemicalModalBridge::new(crate::ChemicalModalBridgeConfig {
            max_component_skew_us: 150,
        });
        let timed_components = vec![
            timed(1_000, 10, "nose-a", 1),
            timed(1_100, 10, "nose-b", 2),
        ];
        let raw: Vec<_> = timed_components
            .iter()
            .map(|component| component.percept().clone())
            .collect();
        let expected = bridge.aggregate(&raw).unwrap();

        let result = aggregate_timed_chemical_percepts(&bridge, timed_components).unwrap();
        assert_eq!(
            result.admission().status(),
            ChemicalTemporalAdmissionStatus::DefinitelyWithin
        );
        let input = result.input().expect("definitely-within timing should aggregate");
        assert_eq!(input.evidence_bundle_id, expected.evidence_bundle_id);
        assert_eq!(input.vector, expected.vector);
        assert_eq!(input.confidence, expected.confidence);
        assert_eq!(input.agreement, expected.agreement);
    }

    #[test]
    fn ambiguous_time_preserves_components_and_creates_no_vector() {
        let bridge = ChemicalModalBridge::new(crate::ChemicalModalBridgeConfig {
            max_component_skew_us: 150,
        });
        let components = vec![
            timed(1_000, 40, "nose-a", 1),
            timed(1_100, 40, "nose-b", 2),
        ];
        let ids: Vec<_> = components.iter().map(TimedChemicalPercept::observation_id).collect();

        let result = aggregate_timed_chemical_percepts(&bridge, components).unwrap();
        assert_eq!(
            result.admission().status(),
            ChemicalTemporalAdmissionStatus::Ambiguous
        );
        assert!(result.input().is_none());
        assert!(!result.was_aggregated());
        let retained: Vec<_> = result
            .timed_components()
            .iter()
            .map(TimedChemicalPercept::observation_id)
            .collect();
        assert_eq!(retained, ids);
    }

    #[test]
    fn definitely_outside_time_creates_no_vector() {
        let bridge = ChemicalModalBridge::new(crate::ChemicalModalBridgeConfig {
            max_component_skew_us: 150,
        });
        let result = aggregate_timed_chemical_percepts(
            &bridge,
            vec![
                timed(1_000, 10, "nose-a", 1),
                timed(1_300, 10, "nose-b", 2),
            ],
        )
        .unwrap();
        assert_eq!(
            result.admission().status(),
            ChemicalTemporalAdmissionStatus::DefinitelyOutside
        );
        assert!(result.input().is_none());
    }

    #[test]
    fn single_source_still_aggregates_without_strict_cross_source_time_claim() {
        let bridge = ChemicalModalBridge::default();
        let percept = percept(1_000, "nose-a", 1);
        let weak_time = TimeIntegrityReceipt::declared(generic_domain());
        let timed = TimedChemicalPercept::new(percept, weak_time).unwrap();
        let result = aggregate_timed_chemical_percepts(&bridge, vec![timed]).unwrap();
        assert_eq!(
            result.admission().status(),
            ChemicalTemporalAdmissionStatus::NoComparisonNeeded
        );
        assert!(result.was_aggregated());
    }

    #[test]
    fn epochless_multi_source_timing_fails_before_bridge_aggregation() {
        let bridge = ChemicalModalBridge::default();
        let a = timed(1_000, 10, "nose-a", 1);
        let weak_time = TimeIntegrityReceipt::declared(generic_domain())
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(10));
        let b = TimedChemicalPercept::new(percept(1_050, "nose-b", 2), weak_time).unwrap();
        let error = aggregate_timed_chemical_percepts(&bridge, vec![a, b]).unwrap_err();
        assert!(matches!(error, TimedChemicalAggregationError::Time(_)));
    }
}
