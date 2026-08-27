// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! End-to-end two-nose validation controls for timed chemosensation.
//!
//! This test intentionally uses only public APIs. It exercises the complete
//! current software path from independently clocked chemical observations through
//! normalization, bounded temporal admission, HDC aggregation, content-addressed
//! temporal authorization, BinaryHV projection, modal lineage, and root
//! multimodal processing.
//!
//! These are deterministic software controls, not physical-sensor validation.

use std::time::Duration;

use symthaea::consciousness::integration::cross_modal_binding::Modality;
use symthaea::consciousness::integration::modal_lineage_history::LineagedMultiModalIntegrator;
use symthaea::consciousness::integration::multi_modal_integration::IntegrationConfig;
use symthaea_chemosensation::{
    CalibrationState, ChannelEncodingSpec, ChemicalChannel, ChemicalClockDomainId,
    ChemicalFingerprintEncoder, ChemicalModalBridge, ChemicalModalBridgeConfig,
    ChemicalModality, ChemicalObservation, ChemicalPercept, ChemicalRootContentLineage,
    ChemicalRootProjector, ChemicalTemporalAdmissionStatus,
    ChemicalTemporalAuthorizationError, ChemicalTemporalAuthorizationId, MeasurementUnit,
    SensorHealth, TimedChemicalAggregation, TimedChemicalPercept,
    aggregate_timed_chemical_percepts,
};
use symthaea_chemosensation_timed_bridge::{
    TimedChemicalRootBridgeError, project_timed_to_lineaged_root_input,
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

#[allow(clippy::too_many_arguments)]
fn normalized(
    raw_timestamp_us: u64,
    comparison_timestamp_us: u64,
    source: &str,
    raw_domain: &str,
    raw_epoch: &str,
    target_domain: ClockDomainId,
    target_epoch: ClockEpochId,
    value: f32,
    timestamp_error_us: u64,
) -> TimedChemicalPercept {
    let source_domain = ClockDomainId::new(raw_domain).unwrap();
    let source_epoch = ClockEpochId::new(raw_epoch).unwrap();
    let source_receipt = TimeIntegrityReceipt::declared(source_domain.clone())
        .with_epoch(source_epoch.clone())
        .with_continuity(ContinuityStatus::Continuous)
        .with_uncertainty(TimeUncertainty::bounded(timestamp_error_us));
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
    let normalized = normalize_timestamp_us(raw_timestamp_us, &source_receipt, &transform).unwrap();
    TimedChemicalPercept::from_normalized(
        percept(raw_timestamp_us, source, raw_domain, value),
        normalized,
    )
    .unwrap()
}

fn normalized_pair(
    target_domain: ClockDomainId,
    second_comparison_timestamp_us: u64,
    max_skew_us: u64,
) -> TimedChemicalAggregation {
    let target_epoch = ClockEpochId::new("capture-session-1").unwrap();
    aggregate_timed_chemical_percepts(
        &ChemicalModalBridge::new(ChemicalModalBridgeConfig {
            max_component_skew_us: max_skew_us,
        }),
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
                5,
            ),
            normalized(
                5_000,
                second_comparison_timestamp_us,
                "nose-b",
                "nose-b/monotonic",
                "nose-b-boot-9",
                target_domain,
                target_epoch,
                42.0,
                5,
            ),
        ],
    )
    .unwrap()
}

fn shared_clock_percept(
    timestamp_us: u64,
    source: &str,
    value: f32,
    uncertainty_us: u64,
) -> TimedChemicalPercept {
    let domain = ClockDomainId::new("shared/monotonic").unwrap();
    let epoch = ClockEpochId::new("shared-boot-1").unwrap();
    TimedChemicalPercept::new(
        percept(timestamp_us, source, "shared/monotonic", value),
        TimeIntegrityReceipt::declared(domain)
            .with_epoch(epoch)
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(uncertainty_us)),
    )
    .unwrap()
}

#[test]
fn positive_two_nose_path_reaches_root_with_two_evidence_references_but_no_activation() {
    let aggregation = normalized_pair(ClockDomainId::unix_epoch(), 10_050, 100);
    assert_eq!(
        aggregation.admission().status(),
        ChemicalTemporalAdmissionStatus::DefinitelyWithin
    );

    let authorization = ChemicalTemporalAuthorizationId::from_aggregation(&aggregation).unwrap();
    let temporal_address = authorization.content_address().unwrap();
    let handoff = project_timed_to_lineaged_root_input(
        &ChemicalRootProjector::default(),
        &aggregation,
    )
    .unwrap();
    let chemical_lineage =
        ChemicalRootContentLineage::from_projection(handoff.timed_projection().projection())
            .unwrap();
    let lineage = handoff.lineaged_input().lineage();

    assert_eq!(lineage.evidence().len(), 2);
    assert!(lineage.evidence().contains(&chemical_lineage.evidence_bundle));
    assert!(lineage.evidence().contains(&temporal_address));
    assert_eq!(lineage.input_space(), Some(&chemical_lineage.input_space));
    assert_eq!(
        lineage.transforms(),
        std::slice::from_ref(&chemical_lineage.projection_policy)
    );
    assert_eq!(lineage.output_space(), Some(&chemical_lineage.output_space));
    assert_eq!(
        handoff.lineaged_input().input().timestamp,
        Duration::from_micros(10_050)
    );
    assert_eq!(
        handoff.timed_projection().projection().earliest_timestamp_us,
        1_000
    );
    assert_eq!(
        handoff.timed_projection().projection().latest_timestamp_us,
        5_000
    );

    let mut root = LineagedMultiModalIntegrator::new(IntegrationConfig::default());
    let result = root.integrate(&[handoff.into_lineaged_input()]);
    assert!(
        result
            .current
            .processed_lineage_for(Modality::Olfactory)
            .is_some()
    );
    assert!(
        result
            .current
            .fused_lineage_for(Modality::Olfactory)
            .is_none()
    );
    assert_eq!(result.current.integration.integrated_phi, 0.0);
}

#[test]
fn reordering_same_two_nose_evidence_preserves_authorization_projection_and_root_lineage() {
    let domain = ClockDomainId::unix_epoch();
    let epoch = ClockEpochId::new("capture-session-1").unwrap();
    let a = normalized(
        1_000,
        10_000,
        "nose-a",
        "nose-a/monotonic",
        "nose-a-boot-1",
        domain.clone(),
        epoch.clone(),
        40.0,
        5,
    );
    let b = normalized(
        5_000,
        10_050,
        "nose-b",
        "nose-b/monotonic",
        "nose-b-boot-9",
        domain,
        epoch,
        42.0,
        5,
    );
    let bridge = ChemicalModalBridge::new(ChemicalModalBridgeConfig {
        max_component_skew_us: 100,
    });
    let left = aggregate_timed_chemical_percepts(&bridge, vec![a.clone(), b.clone()]).unwrap();
    let right = aggregate_timed_chemical_percepts(&bridge, vec![b, a]).unwrap();

    assert_eq!(
        ChemicalTemporalAuthorizationId::from_aggregation(&left).unwrap(),
        ChemicalTemporalAuthorizationId::from_aggregation(&right).unwrap()
    );
    assert_eq!(left.input(), right.input());

    let left_handoff =
        project_timed_to_lineaged_root_input(&ChemicalRootProjector::default(), &left).unwrap();
    let right_handoff =
        project_timed_to_lineaged_root_input(&ChemicalRootProjector::default(), &right).unwrap();
    assert_eq!(
        left_handoff.timed_projection().projection().binary_vector,
        right_handoff.timed_projection().projection().binary_vector
    );
    assert_eq!(
        left_handoff.lineaged_input().lineage(),
        right_handoff.lineaged_input().lineage()
    );
    assert_eq!(
        left_handoff.lineaged_input().input().timestamp,
        right_handoff.lineaged_input().input().timestamp
    );
}

#[test]
fn ambiguous_timing_preserves_evidence_but_cannot_mint_authorization_or_reach_root() {
    let bridge = ChemicalModalBridge::new(ChemicalModalBridgeConfig {
        max_component_skew_us: 100,
    });
    let aggregation = aggregate_timed_chemical_percepts(
        &bridge,
        vec![
            shared_clock_percept(1_000, "nose-a", 40.0, 60),
            shared_clock_percept(1_100, "nose-b", 42.0, 60),
        ],
    )
    .unwrap();

    assert_eq!(
        aggregation.admission().status(),
        ChemicalTemporalAdmissionStatus::Ambiguous
    );
    assert!(aggregation.input().is_none());
    assert_eq!(aggregation.timed_components().len(), 2);
    assert!(matches!(
        ChemicalTemporalAuthorizationId::from_aggregation(&aggregation),
        Err(ChemicalTemporalAuthorizationError::NotAuthorized {
            status: ChemicalTemporalAdmissionStatus::Ambiguous
        })
    ));
    assert!(matches!(
        project_timed_to_lineaged_root_input(&ChemicalRootProjector::default(), &aggregation),
        Err(TimedChemicalRootBridgeError::TimedProjection(_))
    ));
}

#[test]
fn definitely_outside_timing_preserves_evidence_but_cannot_reach_root() {
    let bridge = ChemicalModalBridge::new(ChemicalModalBridgeConfig {
        max_component_skew_us: 100,
    });
    let aggregation = aggregate_timed_chemical_percepts(
        &bridge,
        vec![
            shared_clock_percept(1_000, "nose-a", 40.0, 10),
            shared_clock_percept(1_300, "nose-b", 42.0, 10),
        ],
    )
    .unwrap();

    assert_eq!(
        aggregation.admission().status(),
        ChemicalTemporalAdmissionStatus::DefinitelyOutside
    );
    assert!(aggregation.input().is_none());
    assert_eq!(aggregation.timed_components().len(), 2);
    assert!(matches!(
        ChemicalTemporalAuthorizationId::from_aggregation(&aggregation),
        Err(ChemicalTemporalAuthorizationError::NotAuthorized {
            status: ChemicalTemporalAdmissionStatus::DefinitelyOutside
        })
    ));
    assert!(matches!(
        project_timed_to_lineaged_root_input(&ChemicalRootProjector::default(), &aggregation),
        Err(TimedChemicalRootBridgeError::TimedProjection(_))
    ));
}

#[test]
fn tampering_with_hdc_aggregate_does_not_change_timing_identity_but_fails_projection_revalidation() {
    let original = normalized_pair(ClockDomainId::unix_epoch(), 10_050, 100);
    let authorization = ChemicalTemporalAuthorizationId::from_aggregation(&original).unwrap();

    let tampered = match original.clone() {
        TimedChemicalAggregation::Aggregated {
            admission,
            mut input,
            timed_components,
        } => {
            input.confidence *= 0.5;
            TimedChemicalAggregation::Aggregated {
                admission,
                input,
                timed_components,
            }
        }
        TimedChemicalAggregation::Abstained { .. } => panic!("positive fixture must aggregate"),
    };

    assert_eq!(
        ChemicalTemporalAuthorizationId::from_aggregation(&tampered).unwrap(),
        authorization
    );
    assert!(matches!(
        project_timed_to_lineaged_root_input(&ChemicalRootProjector::default(), &tampered),
        Err(TimedChemicalRootBridgeError::TimedProjection(_))
    ));
}
