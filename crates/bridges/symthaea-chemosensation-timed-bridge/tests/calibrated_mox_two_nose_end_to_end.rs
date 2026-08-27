// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! End-to-end control for the complete calibration-authority path.
//!
//! This test begins at deterministic MOX simulator output, derives one bounded
//! source->target clock calibration per simulated nose from four timestamp
//! exchanges, evaluates a frozen policy, binds the exact calibration evidence,
//! derives a finite holdover transform, normalizes acquisition time only through
//! the public evidence-bound chemical adapter, and then exercises timed fusion,
//! projection, modal lineage, and root non-activation.
//!
//! It is deterministic software validation, not physical timing or sensor proof.

use std::time::Duration;

use symthaea::consciousness::integration::cross_modal_binding::Modality;
use symthaea::consciousness::integration::modal_lineage_history::LineagedMultiModalIntegrator;
use symthaea::consciousness::integration::multi_modal_integration::IntegrationConfig;
use symthaea_chemosensation::{
    ChannelEncodingSpec, ChemicalClockDomainId, ChemicalFingerprintEncoder, ChemicalModalBridge,
    ChemicalModalBridgeConfig, ChemicalPercept, ChemicalRootProjector,
    ChemicalTemporalAdmissionStatus, ChemicalTemporalAuthorizationId, MeasurementUnit,
    MoxArraySimulator, MoxChannelModel, OlfactoryStimulus, TimedChemicalPercept,
    aggregate_timed_chemical_percepts, bind_evidence_bound_acquisition_time,
};
use symthaea_chemosensation_timed_bridge::project_timed_to_lineaged_root_input;
use symthaea_time_calibration::{
    CalibrationConsensus, ClockCalibrationEvidence, FourTimestampExchange, TimestampEvidence,
};
use symthaea_time_calibration_bundle::CalibrationDecisionBundle;
use symthaea_time_calibration_policy::{
    CalibrationDecision, CalibrationDecisionPolicy, CalibrationPolicyId,
};
use symthaea_time_holdover::{BoundedHoldoverTransform, HoldoverClaim};
use symthaea_time_integrity::{
    ClockDomainId, ClockEpochId, ContinuityStatus, TimeIntegrityReceipt, TimeUncertainty,
};

fn target_domain() -> ClockDomainId {
    ClockDomainId::unix_epoch()
}

fn target_epoch() -> ClockEpochId {
    ClockEpochId::new("capture-session-1").unwrap()
}

fn receipt(
    domain: ClockDomainId,
    epoch: ClockEpochId,
    error_us: u64,
) -> TimeIntegrityReceipt {
    TimeIntegrityReceipt::declared(domain)
        .with_epoch(epoch)
        .with_continuity(ContinuityStatus::Continuous)
        .with_uncertainty(TimeUncertainty::bounded(error_us))
}

fn stamp(
    timestamp_us: u64,
    domain: ClockDomainId,
    epoch: ClockEpochId,
) -> TimestampEvidence {
    TimestampEvidence::new(timestamp_us, receipt(domain, epoch, 0))
}

fn calibration_evidence(
    source_domain: ClockDomainId,
    source_epoch: ClockEpochId,
    source_send_us: u64,
    source_to_target_offset_us: u64,
) -> ClockCalibrationEvidence {
    let one_way_delay_us = 10;
    let target_processing_us = 10;
    let target_receive_us = source_send_us + source_to_target_offset_us + one_way_delay_us;
    let target_send_us = target_receive_us + target_processing_us;
    let source_receive_us = source_send_us + one_way_delay_us + target_processing_us + one_way_delay_us;

    let exchange = FourTimestampExchange::new(
        stamp(source_send_us, source_domain.clone(), source_epoch.clone()),
        stamp(target_receive_us, target_domain(), target_epoch()),
        stamp(target_send_us, target_domain(), target_epoch()),
        stamp(source_receive_us, source_domain, source_epoch),
    )
    .unwrap();
    let evidence = ClockCalibrationEvidence::derive(exchange).unwrap();
    assert_eq!(evidence.offset_interval().symmetric_radius_us(), 10);
    assert_eq!(
        evidence.offset_interval().midpoint_us(),
        i128::from(source_to_target_offset_us)
    );
    evidence
}

fn evidence_bound_holdover(
    source_domain: ClockDomainId,
    source_epoch: ClockEpochId,
    calibration_source_send_us: u64,
    source_to_target_offset_us: u64,
    valid_source_start_us: u64,
    valid_source_end_us: u64,
) -> BoundedHoldoverTransform {
    let evidence = vec![calibration_evidence(
        source_domain,
        source_epoch,
        calibration_source_send_us,
        source_to_target_offset_us,
    )];
    let consensus = CalibrationConsensus::from_evidence(&evidence).unwrap();
    let policy = CalibrationDecisionPolicy::new(
        CalibrationPolicyId::new("two-nose-software-calibration-v1").unwrap(),
        1,
        20,
        Some(100),
    )
    .unwrap();
    let decision = policy.evaluate(&consensus);
    assert_eq!(decision.decision(), CalibrationDecision::Accepted);
    let bundle = CalibrationDecisionBundle::new(decision, evidence).unwrap();
    bundle.verify_self().unwrap();

    let holdover = HoldoverClaim::new(
        valid_source_start_us,
        valid_source_end_us,
        0,
        0,
        ContinuityStatus::Continuous,
        ContinuityStatus::Continuous,
        ContinuityStatus::Continuous,
    )
    .unwrap();
    let value = BoundedHoldoverTransform::new(bundle, holdover).unwrap();
    value.verify_self().unwrap();
    assert_eq!(
        value.transform().uncertainty(),
        TimeUncertainty::Bounded { max_error_us: 10 }
    );
    value
}

fn mox_encoder() -> ChemicalFingerprintEncoder {
    ChemicalFingerprintEncoder::new(vec![ChannelEncodingSpec::new(
        "mox-a",
        MeasurementUnit::Ohms,
        0.0,
        150_000.0,
        31,
        31,
        131,
    )])
    .unwrap()
}

fn simulated_percept(raw_timestamp_us: u64, raw_domain: &str) -> ChemicalPercept {
    let mut nose = MoxArraySimulator::new(vec![MoxChannelModel::new(
        "mox-a",
        100_000.0,
        1.0,
    )]);
    let mut observation = nose
        .step(
            &OlfactoryStimulus {
                concentration_ppm: 10.0,
                affinities: vec![1.0],
                temperature_c: 25.0,
                humidity_rh: 50.0,
            },
            1.0,
            raw_timestamp_us,
        )
        .unwrap();
    assert!(observation.clock_domain.is_none());
    observation.clock_domain = Some(ChemicalClockDomainId::new(raw_domain).unwrap());
    let fingerprint = mox_encoder().encode(&observation).unwrap().unwrap();
    ChemicalPercept {
        evidence: observation,
        fingerprint,
    }
}

fn bind_calibrated_time(
    percept: ChemicalPercept,
    source_domain: ClockDomainId,
    source_epoch: ClockEpochId,
    holdover: &BoundedHoldoverTransform,
) -> TimedChemicalPercept {
    let source_receipt = receipt(source_domain, source_epoch, 2);
    bind_evidence_bound_acquisition_time(percept, source_receipt, holdover).unwrap()
}

#[test]
fn calibration_policy_bundle_and_holdover_authorize_two_nose_root_handoff() {
    let source_a_domain = ClockDomainId::new("nose-a/monotonic").unwrap();
    let source_a_epoch = ClockEpochId::new("nose-a-boot-1").unwrap();
    let source_b_domain = ClockDomainId::new("nose-b/monotonic").unwrap();
    let source_b_epoch = ClockEpochId::new("nose-b-boot-9").unwrap();

    // Nose A maps raw 1_000 -> Unix 10_000 through an evidence-derived +9_000 us offset.
    let holdover_a = evidence_bound_holdover(
        source_a_domain.clone(),
        source_a_epoch.clone(),
        900,
        9_000,
        800,
        1_100,
    );
    // Nose B maps raw 5_000 -> Unix 10_050 through an evidence-derived +5_050 us offset.
    let holdover_b = evidence_bound_holdover(
        source_b_domain.clone(),
        source_b_epoch.clone(),
        4_900,
        5_050,
        4_800,
        5_100,
    );

    let raw_a = simulated_percept(1_000, "nose-a/monotonic");
    let raw_b = simulated_percept(5_000, "nose-b/monotonic");
    let raw_a_id = raw_a.observation_id();
    let raw_b_id = raw_b.observation_id();

    let timed_a = bind_calibrated_time(
        raw_a,
        source_a_domain,
        source_a_epoch,
        &holdover_a,
    );
    let timed_b = bind_calibrated_time(
        raw_b,
        source_b_domain,
        source_b_epoch,
        &holdover_b,
    );

    assert_eq!(timed_a.comparison_timestamp_us(), 10_000);
    assert_eq!(timed_b.comparison_timestamp_us(), 10_050);
    assert_eq!(timed_a.observation_id(), raw_a_id);
    assert_eq!(timed_b.observation_id(), raw_b_id);
    assert_eq!(
        timed_a.time().uncertainty,
        TimeUncertainty::Bounded { max_error_us: 12 }
    );
    assert_eq!(
        timed_b.time().uncertainty,
        TimeUncertainty::Bounded { max_error_us: 12 }
    );
    let authority_a = timed_a
        .acquisition_authorization()
        .expect("evidence-bound nose A must retain acquisition authority");
    let authority_b = timed_b
        .acquisition_authorization()
        .expect("evidence-bound nose B must retain acquisition authority");
    assert_ne!(authority_a, authority_b);

    let aggregation = aggregate_timed_chemical_percepts(
        &ChemicalModalBridge::new(ChemicalModalBridgeConfig {
            max_component_skew_us: 100,
        }),
        vec![timed_a, timed_b],
    )
    .unwrap();
    assert_eq!(
        aggregation.admission().status(),
        ChemicalTemporalAdmissionStatus::DefinitelyWithin
    );
    let pair = &aggregation.admission().pairwise_windows()[0].separation;
    assert_eq!(pair.nominal_us, 50);
    assert_eq!(pair.minimum_us, 26);
    assert_eq!(pair.maximum_us, 74);

    let temporal_authorization =
        ChemicalTemporalAuthorizationId::from_aggregation(&aggregation).unwrap();
    let temporal_address = temporal_authorization.content_address().unwrap();
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

    // The root still only records that this evidence was processed. Olfaction is
    // not configured into the fusion topology by this experiment.
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
