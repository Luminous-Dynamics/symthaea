// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! End-to-end control beginning at the deterministic MOX transducer simulator.
//!
//! This remains a software fixture, not a physical-sensor claim. The purpose is
//! to prove that two independently clocked simulated noses can traverse the
//! public acquisition/representation/timing/projection/root lineage boundary
//! without inventing shared raw time or activating olfaction in root cognition.

use std::time::Duration;

use symthaea::consciousness::integration::cross_modal_binding::Modality;
use symthaea::consciousness::integration::modal_lineage_history::LineagedMultiModalIntegrator;
use symthaea::consciousness::integration::multi_modal_integration::IntegrationConfig;
use symthaea_chemosensation::{
    ChannelEncodingSpec, ChemicalClockDomainId, ChemicalFingerprintEncoder, ChemicalModalBridge,
    ChemicalModalBridgeConfig, ChemicalPercept, ChemicalRootProjector,
    ChemicalTemporalAdmissionStatus, ChemicalTemporalAuthorizationId, MeasurementUnit,
    MoxArraySimulator, MoxChannelModel, OlfactoryStimulus, TimedChemicalPercept,
    aggregate_timed_chemical_percepts,
};
use symthaea_chemosensation_timed_bridge::project_timed_to_lineaged_root_input;
use symthaea_time_integrity::{
    ClockDomainId, ClockEpochId, ContinuityStatus, TimeIntegrityReceipt, TimeUncertainty,
};
use symthaea_time_normalization::{ClockTransformReceipt, normalize_timestamp_us};

fn mox_encoder() -> ChemicalFingerprintEncoder {
    ChemicalFingerprintEncoder::new(vec![
        ChannelEncodingSpec::new(
            "mox-a",
            MeasurementUnit::Ohms,
            0.0,
            150_000.0,
            21,
            21,
            121,
        ),
        ChannelEncodingSpec::new(
            "mox-b",
            MeasurementUnit::Ohms,
            0.0,
            180_000.0,
            22,
            22,
            122,
        ),
    ])
    .unwrap()
}

fn simulated_percept(raw_timestamp_us: u64, raw_domain: &str) -> ChemicalPercept {
    let mut nose = MoxArraySimulator::new(vec![
        MoxChannelModel::new("mox-a", 100_000.0, 1.0),
        MoxChannelModel::new("mox-b", 120_000.0, 0.5),
    ]);
    let stimulus = OlfactoryStimulus {
        concentration_ppm: 10.0,
        affinities: vec![1.0, 0.25],
        temperature_c: 25.0,
        humidity_rh: 50.0,
    };
    let mut observation = nose.step(&stimulus, 1.0, raw_timestamp_us).unwrap();

    // The simulator correctly does not invent a clock. The acquisition fixture
    // supplies the device-local clock identity explicitly at this boundary.
    assert!(observation.clock_domain.is_none());
    observation.clock_domain = Some(ChemicalClockDomainId::new(raw_domain).unwrap());

    let fingerprint = mox_encoder().encode(&observation).unwrap().unwrap();
    ChemicalPercept {
        evidence: observation,
        fingerprint,
    }
}

#[allow(clippy::too_many_arguments)]
fn normalize_percept(
    percept: ChemicalPercept,
    raw_domain: &str,
    raw_epoch: &str,
    comparison_timestamp_us: u64,
    target_domain: ClockDomainId,
    target_epoch: ClockEpochId,
) -> TimedChemicalPercept {
    let source_domain = ClockDomainId::new(raw_domain).unwrap();
    let source_epoch = ClockEpochId::new(raw_epoch).unwrap();
    let source_timestamp_us = percept.timestamp_us();
    let source_receipt = TimeIntegrityReceipt::declared(source_domain.clone())
        .with_epoch(source_epoch.clone())
        .with_continuity(ContinuityStatus::Continuous)
        .with_uncertainty(TimeUncertainty::bounded(5));
    let transform = ClockTransformReceipt::offset(
        source_domain,
        source_epoch,
        target_domain,
        target_epoch,
        source_timestamp_us,
        comparison_timestamp_us,
        source_timestamp_us.saturating_sub(100),
        source_timestamp_us.saturating_add(100),
    )
    .unwrap()
    .with_mapping_continuity(ContinuityStatus::Continuous)
    .with_target_continuity(ContinuityStatus::Continuous)
    .with_uncertainty(TimeUncertainty::bounded(5));
    let normalized = normalize_timestamp_us(source_timestamp_us, &source_receipt, &transform)
        .unwrap();
    TimedChemicalPercept::from_normalized(percept, normalized).unwrap()
}

#[test]
fn two_independent_mox_simulators_reach_timed_root_lineage_without_activation() {
    let raw_a = simulated_percept(1_000, "nose-a/monotonic");
    let raw_b = simulated_percept(5_000, "nose-b/monotonic");
    assert_ne!(raw_a.observation_id(), raw_b.observation_id());

    // Identical sensor models/stimulus should produce the same chemical geometry;
    // only raw timing provenance differs between these two software noses.
    assert_eq!(raw_a.fingerprint.vector, raw_b.fingerprint.vector);

    let target_domain = ClockDomainId::unix_epoch();
    let target_epoch = ClockEpochId::new("capture-session-1").unwrap();
    let a = normalize_percept(
        raw_a,
        "nose-a/monotonic",
        "nose-a-boot-1",
        10_000,
        target_domain.clone(),
        target_epoch.clone(),
    );
    let b = normalize_percept(
        raw_b,
        "nose-b/monotonic",
        "nose-b-boot-9",
        10_050,
        target_domain,
        target_epoch,
    );

    let aggregation = aggregate_timed_chemical_percepts(
        &ChemicalModalBridge::new(ChemicalModalBridgeConfig {
            max_component_skew_us: 100,
        }),
        vec![a, b],
    )
    .unwrap();
    assert_eq!(
        aggregation.admission().status(),
        ChemicalTemporalAdmissionStatus::DefinitelyWithin
    );
    let input = aggregation.input().expect("simulated noses should aggregate");
    assert!(input.clock_domain.is_none());
    assert!((input.agreement - 1.0).abs() < 1.0e-5);

    let authorization = ChemicalTemporalAuthorizationId::from_aggregation(&aggregation).unwrap();
    let temporal_address = authorization.content_address().unwrap();
    let handoff = project_timed_to_lineaged_root_input(
        &ChemicalRootProjector::default(),
        &aggregation,
    )
    .unwrap();

    assert_eq!(
        handoff.lineaged_input().input().timestamp,
        Duration::from_micros(10_050)
    );
    assert!(
        handoff
            .lineaged_input()
            .lineage()
            .evidence()
            .contains(&temporal_address)
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
