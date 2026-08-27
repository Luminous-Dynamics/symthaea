// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-preserving handoff from temporally admitted chemical sensing into
//! Symthaea's root multimodal input contract.
//!
//! This crate is intentionally additive to `symthaea-chemosensation-bridge`.
//! The legacy bridge remains the compatibility path for raw shared-clock
//! chemical aggregates. This bridge consumes [`TimedChemicalAggregation`], asks
//! the chemical domain to revalidate temporal admission + HDC aggregation during
//! [`ChemicalRootProjector::project_timed_aggregation`], and keeps the resulting
//! [`TimedChemicalRootProjection`] attached to the root input.
//!
//! Root timestamp semantics remain asymmetric:
//!
//! - a comparison timestamp is copied into `ModalInput::timestamp` only when the
//!   *verified comparison domain* is explicitly `unix-epoch`;
//! - device-local or other normalized comparison domains remain projection
//!   provenance, while `ModalInput::new` keeps root ingestion time;
//! - raw acquisition timestamps/domains are never rewritten or relabeled.
//!
//! Root modal lineage carries two independent evidence identities: the chemical
//! evidence bundle and a content-addressed temporal-authorization receipt. Root
//! cognition does not need to import timing types to preserve the exact evidence
//! that justified treating multiple chemical samples as one event.

use std::fmt;
use std::time::Duration;

use symthaea::consciousness::integration::modal_lineage::ModalLineageReceipt;
use symthaea::consciousness::integration::modal_lineage_integration::LineagedModalInput;
use symthaea::consciousness::integration::multi_modal_integration::ModalInput;
use symthaea_chemosensation::{
    ChemicalBridgeTarget, ChemicalContentAddressError, ChemicalRootContentLineage,
    ChemicalRootProjector, ChemicalTemporalAuthorizationError, ChemicalTemporalAuthorizationId,
    TimedChemicalAggregation, TimedChemicalRootProjection, TimedChemicalRootProjectionError,
};
use symthaea_chemosensation_bridge::{ChemicalRootBridgeError, root_modality_for_target};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimedChemicalRootBridgeError {
    TimedProjection(TimedChemicalRootProjectionError),
    TemporalAuthorization(ChemicalTemporalAuthorizationError),
    RootBridge(ChemicalRootBridgeError),
}

impl fmt::Display for TimedChemicalRootBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimedProjection(error) => {
                write!(f, "timed chemical root projection failed: {error}")
            }
            Self::TemporalAuthorization(error) => {
                write!(f, "chemical temporal authorization failed: {error}")
            }
            Self::RootBridge(error) => write!(f, "chemical root handoff failed: {error}"),
        }
    }
}

impl std::error::Error for TimedChemicalRootBridgeError {}

impl From<TimedChemicalRootProjectionError> for TimedChemicalRootBridgeError {
    fn from(value: TimedChemicalRootProjectionError) -> Self {
        Self::TimedProjection(value)
    }
}

impl From<ChemicalTemporalAuthorizationError> for TimedChemicalRootBridgeError {
    fn from(value: ChemicalTemporalAuthorizationError) -> Self {
        Self::TemporalAuthorization(value)
    }
}

impl From<ChemicalRootBridgeError> for TimedChemicalRootBridgeError {
    fn from(value: ChemicalRootBridgeError) -> Self {
        Self::RootBridge(value)
    }
}

/// Self-revalidating timed chemical projection paired with the exact root input
/// produced from it.
///
/// Keeping the full timed projection attached preserves rich timing diagnostics,
/// while the generic modal lineage carries a content address for the exact
/// temporal authorization so root-side evidence tracking does not lose that
/// provenance when the timing structs are not otherwise retained.
pub struct TimedChemicalRootHandoff {
    timed_projection: TimedChemicalRootProjection,
    lineaged_input: LineagedModalInput,
}

impl TimedChemicalRootHandoff {
    pub fn timed_projection(&self) -> &TimedChemicalRootProjection {
        &self.timed_projection
    }

    pub fn lineaged_input(&self) -> &LineagedModalInput {
        &self.lineaged_input
    }

    pub fn into_parts(self) -> (TimedChemicalRootProjection, LineagedModalInput) {
        (self.timed_projection, self.lineaged_input)
    }

    pub fn into_lineaged_input(self) -> LineagedModalInput {
        self.lineaged_input
    }
}

/// Convert one temporally admitted chemical aggregate into a typed root input.
///
/// The caller cannot supply a detached `ChemicalRootProjection`: this function
/// starts from [`TimedChemicalAggregation`] and delegates to
/// [`ChemicalRootProjector::project_timed_aggregation`], which re-runs temporal
/// admission and HDC aggregation from the retained component evidence before
/// BinaryHV projection.
///
/// The generic modal lineage contains independent content addresses for the raw
/// chemical evidence bundle and the exact temporal authorization, plus the
/// representation-space and projection-policy identities. The complete timed
/// projection is still retained for direct diagnostics and revalidation.
pub fn project_timed_to_lineaged_root_input(
    projector: &ChemicalRootProjector,
    aggregation: &TimedChemicalAggregation,
) -> Result<TimedChemicalRootHandoff, TimedChemicalRootBridgeError> {
    let timed_projection = projector.project_timed_aggregation(aggregation)?;
    let projection = timed_projection.projection();
    let modality = root_modality_for_target(projection.target)?;
    let content_lineage = ChemicalRootContentLineage::from_projection(projection)
        .map_err(ChemicalRootBridgeError::from)?;
    let temporal_authorization = ChemicalTemporalAuthorizationId::from_aggregation(aggregation)?;
    let temporal_evidence = temporal_authorization
        .content_address()
        .map_err(ChemicalContentAddressError::from)
        .map_err(ChemicalRootBridgeError::from)?;

    let modal_lineage = ModalLineageReceipt::new(vec![
        content_lineage.evidence_bundle,
        temporal_evidence,
    ])
    .map_err(ChemicalRootBridgeError::from)?
    .with_input_space(content_lineage.input_space)
    .with_transform(content_lineage.projection_policy)
    .map_err(ChemicalRootBridgeError::from)?
    .with_output_space(content_lineage.output_space);

    let mut modal_input = ModalInput::new(
        modality,
        projection.binary_vector,
        f64::from(projection.confidence),
    )
    .with_source(source_label(projection.target));

    // The timed projection has revalidated that this timestamp belongs to the
    // comparison receipt. Only an explicit Unix comparison domain may replace
    // root ingestion time. Raw device-local timestamps remain untouched inside
    // the retained timed projection.
    if timed_projection.comparison_is_unix_epoch() {
        modal_input.timestamp =
            Duration::from_micros(timed_projection.latest_comparison_timestamp_us());
    }

    Ok(TimedChemicalRootHandoff {
        timed_projection,
        lineaged_input: LineagedModalInput::new(modal_input, modal_lineage),
    })
}

fn source_label(target: ChemicalBridgeTarget) -> &'static str {
    match target {
        ChemicalBridgeTarget::Olfactory => "symthaea-chemosensation/olfactory",
        ChemicalBridgeTarget::Gustatory => "symthaea-chemosensation/gustatory",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea::consciousness::integration::cross_modal_binding::Modality;
    use symthaea::consciousness::integration::modal_lineage_history::LineagedMultiModalIntegrator;
    use symthaea::consciousness::integration::multi_modal_integration::IntegrationConfig;
    use symthaea_chemosensation::{
        CalibrationState, ChannelEncodingSpec, ChemicalChannel, ChemicalClockDomainId,
        ChemicalFingerprintEncoder, ChemicalModalBridge, ChemicalModalBridgeConfig,
        ChemicalModality, ChemicalObservation, ChemicalPercept, ChemicalTemporalAdmissionStatus,
        MeasurementUnit, SensorHealth, TimedChemicalPercept, aggregate_timed_chemical_percepts,
    };
    use symthaea_time_integrity::{
        ClockDomainId, ClockEpochId, ContinuityStatus, TimeIntegrityReceipt, TimeUncertainty,
    };
    use symthaea_time_normalization::{ClockTransformReceipt, normalize_timestamp_us};

    fn encoder() -> ChemicalFingerprintEncoder {
        ChemicalFingerprintEncoder::new(vec![ChannelEncodingSpec::new(
            "chemical-signal",
            MeasurementUnit::PartsPerMillion,
            0.0,
            100.0,
            11,
            11,
            101,
        )])
        .unwrap()
    }

    fn percept(
        raw_timestamp_us: u64,
        source: &str,
        raw_domain: &str,
        value: f32,
    ) -> ChemicalPercept {
        let mut observation = ChemicalObservation::new(
            raw_timestamp_us,
            ChemicalModality::Olfactory,
            source,
            vec![ChemicalChannel {
                name: "chemical-signal".into(),
                raw_value: value,
                unit: MeasurementUnit::PartsPerMillion,
                calibration: CalibrationState::identity("cal-v1"),
                health: SensorHealth::default(),
            }],
        );
        observation.clock_domain = Some(ChemicalClockDomainId::new(raw_domain).unwrap());
        let fingerprint = encoder().encode(&observation).unwrap().unwrap();
        ChemicalPercept {
            evidence: observation,
            fingerprint,
        }
    }

    fn normalized(
        raw_timestamp_us: u64,
        comparison_timestamp_us: u64,
        source: &str,
        raw_domain: &str,
        raw_epoch: &str,
        target_domain: ClockDomainId,
        target_epoch: ClockEpochId,
        value: f32,
    ) -> TimedChemicalPercept {
        let source_domain = ClockDomainId::new(raw_domain).unwrap();
        let source_epoch = ClockEpochId::new(raw_epoch).unwrap();
        let source_receipt = TimeIntegrityReceipt::declared(source_domain.clone())
            .with_epoch(source_epoch.clone())
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(5));
        let transform = ClockTransformReceipt::offset(
            source_domain,
            source_epoch,
            target_domain,
            target_epoch,
            raw_timestamp_us,
            comparison_timestamp_us,
            raw_timestamp_us.saturating_sub(100),
            raw_timestamp_us.saturating_add(100),
        )
        .unwrap()
        .with_mapping_continuity(ContinuityStatus::Continuous)
        .with_target_continuity(ContinuityStatus::Continuous)
        .with_uncertainty(TimeUncertainty::bounded(5));
        let normalized = normalize_timestamp_us(raw_timestamp_us, &source_receipt, &transform)
            .unwrap();
        TimedChemicalPercept::from_normalized(
            percept(raw_timestamp_us, source, raw_domain, value),
            normalized,
        )
        .unwrap()
    }

    fn normalized_pair(target_domain: ClockDomainId) -> TimedChemicalAggregation {
        let target_epoch = ClockEpochId::new("capture-session-1").unwrap();
        let bridge = ChemicalModalBridge::new(ChemicalModalBridgeConfig {
            max_component_skew_us: 100,
        });
        aggregate_timed_chemical_percepts(
            &bridge,
            vec![
                normalized(
                    1_000,
                    10_000,
                    "nose-a",
                    "nose-a/monotonic",
                    "nose-a-boot-1",
                    target_domain.clone(),
                    target_epoch.clone(),
                    40.0,
                ),
                normalized(
                    5_000,
                    10_050,
                    "nose-b",
                    "nose-b/monotonic",
                    "nose-b-boot-9",
                    target_domain,
                    target_epoch,
                    42.0,
                ),
            ],
        )
        .unwrap()
    }

    #[test]
    fn unix_comparison_time_crosses_root_boundary_without_relabeling_raw_time() {
        let aggregation = normalized_pair(ClockDomainId::unix_epoch());
        let handoff = project_timed_to_lineaged_root_input(
            &ChemicalRootProjector::default(),
            &aggregation,
        )
        .unwrap();

        let timed = handoff.timed_projection();
        assert!(timed.comparison_is_unix_epoch());
        assert_eq!(timed.latest_comparison_timestamp_us(), 10_050);
        assert_eq!(timed.projection().latest_timestamp_us, 5_000);
        assert!(timed.projection().clock_domain.is_none());
        assert_eq!(
            handoff.lineaged_input().input().timestamp,
            Duration::from_micros(10_050)
        );
        assert_eq!(handoff.lineaged_input().input().modality, Modality::Olfactory);
    }

    #[test]
    fn non_unix_comparison_time_remains_provenance_while_root_keeps_ingestion_time() {
        let aggregation = normalized_pair(ClockDomainId::new("capture-host/monotonic").unwrap());
        let handoff = project_timed_to_lineaged_root_input(
            &ChemicalRootProjector::default(),
            &aggregation,
        )
        .unwrap();

        assert_eq!(
            handoff.timed_projection().latest_comparison_timestamp_us(),
            10_050
        );
        assert_ne!(
            handoff.lineaged_input().input().timestamp,
            Duration::from_micros(10_050)
        );
        assert!(handoff.lineaged_input().input().timestamp > Duration::from_secs(1_000_000_000));
    }

    #[test]
    fn timed_handoff_preserves_chemical_and_temporal_lineage_without_activation() {
        let aggregation = normalized_pair(ClockDomainId::unix_epoch());
        let temporal_evidence = ChemicalTemporalAuthorizationId::from_aggregation(&aggregation)
            .unwrap()
            .content_address()
            .unwrap();
        let handoff = project_timed_to_lineaged_root_input(
            &ChemicalRootProjector::default(),
            &aggregation,
        )
        .unwrap();

        let expected = ChemicalRootContentLineage::from_projection(
            handoff.timed_projection().projection(),
        )
        .unwrap();
        let lineage = handoff.lineaged_input().lineage();
        assert_eq!(lineage.evidence().len(), 2);
        assert!(lineage.evidence().contains(&expected.evidence_bundle));
        assert!(lineage.evidence().contains(&temporal_evidence));
        assert_eq!(lineage.input_space(), Some(&expected.input_space));
        assert_eq!(lineage.transforms(), std::slice::from_ref(&expected.projection_policy));
        assert_eq!(lineage.output_space(), Some(&expected.output_space));

        let mut root = LineagedMultiModalIntegrator::new(IntegrationConfig::default());
        let result = root.integrate(&[handoff.into_lineaged_input()]);
        assert!(result.current.processed_lineage_for(Modality::Olfactory).is_some());
        assert!(result.current.fused_lineage_for(Modality::Olfactory).is_none());
        assert_eq!(result.current.integration.integrated_phi, 0.0);
    }

    #[test]
    fn ambiguous_temporal_evidence_cannot_cross_timed_root_handoff() {
        let domain = ClockDomainId::new("shared/monotonic").unwrap();
        let epoch = ClockEpochId::new("shared-boot-1").unwrap();
        let timed = |timestamp_us: u64, source: &str| {
            TimedChemicalPercept::new(
                percept(timestamp_us, source, "shared/monotonic", 40.0),
                TimeIntegrityReceipt::declared(domain.clone())
                    .with_epoch(epoch.clone())
                    .with_continuity(ContinuityStatus::Continuous)
                    .with_uncertainty(TimeUncertainty::bounded(60)),
            )
            .unwrap()
        };
        let bridge = ChemicalModalBridge::new(ChemicalModalBridgeConfig {
            max_component_skew_us: 100,
        });
        let aggregation = aggregate_timed_chemical_percepts(
            &bridge,
            vec![timed(1_000, "nose-a"), timed(1_100, "nose-b")],
        )
        .unwrap();
        assert_eq!(
            aggregation.admission().status(),
            ChemicalTemporalAdmissionStatus::Ambiguous
        );
        assert!(matches!(
            project_timed_to_lineaged_root_input(&ChemicalRootProjector::default(), &aggregation),
            Err(TimedChemicalRootBridgeError::TimedProjection(_))
        ));
    }
}
