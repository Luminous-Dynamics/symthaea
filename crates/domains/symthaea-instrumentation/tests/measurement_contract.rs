// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_instrumentation::{
    ArtifactReference, AssessmentFailure, CalibrationEvidenceResolver, CalibrationReference,
    ClockDomainId, ContractError, InstrumentIdentity, MeasurementEnvelope, MeasurementInput,
    MeasurementPolicy,
    MeasurementStreamGuard, Quantity, QualityFlag, RawDataEvidenceResolver, RawDataReference,
    ResolvedCalibration, ResolvedRawData, StreamOrderFailure, Unit,
};

const SAMPLE_TIME_NS: u64 = 1_000_000_000;
const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

fn clock_domain() -> ClockDomainId {
    ClockDomainId::new("ultrasound-rig-boot-epoch-01").unwrap()
}

fn instrument_identity() -> InstrumentIdentity {
    InstrumentIdentity::new("ultrasound-research-rig-01", "rf-channel-0").unwrap()
}

fn artifact(id: &str, sha: &str) -> ArtifactReference {
    ArtifactReference::new(id, sha).unwrap()
}

fn calibration_reference() -> CalibrationReference {
    CalibrationReference::new("calibration-17", artifact("calibration-certificate", SHA_A)).unwrap()
}

fn resolved_calibration() -> ResolvedCalibration {
    ResolvedCalibration::new(
        "calibration-17",
        instrument_identity(),
        artifact("calibration-certificate", SHA_A),
        artifact("independent-review-receipt", SHA_B),
        Quantity::Frequency,
        Unit::Hertz,
        4_000_000.0,
        6_000_000.0,
        SAMPLE_TIME_NS - 1,
        SAMPLE_TIME_NS + 2_000_000_000,
        900.0,
    )
    .unwrap()
}

fn measurement(sequence: u64, captured_at_ns: u64) -> MeasurementEnvelope {
    measurement_in_clock_domain(sequence, captured_at_ns, clock_domain())
}

fn measurement_in_clock_domain(
    sequence: u64,
    captured_at_ns: u64,
    clock_domain: ClockDomainId,
) -> MeasurementEnvelope {
    MeasurementEnvelope::new(MeasurementInput {
        identity: instrument_identity(),
        sequence,
        captured_at_ns,
        clock_domain,
        quantity: Quantity::Frequency,
        unit: Unit::Hertz,
        value: 5_000_000.0,
        standard_uncertainty: 1_000.0,
        calibration: Some(calibration_reference()),
        raw_data: Some(RawDataReference::new(artifact("raw-acquisition-1", SHA_C))),
        processing_chain_version: "sha256:reconstruction-pipeline-v1".to_owned(),
        quality_flags: vec![],
    })
    .unwrap()
}

fn policy() -> MeasurementPolicy {
    MeasurementPolicy::new(500_000_000, 2_000.0, 1_000.0).unwrap()
}

struct MockResolver {
    resolved: Result<ResolvedCalibration, String>,
    resolved_raw: Result<ResolvedRawData, String>,
}

impl CalibrationEvidenceResolver for MockResolver {
    fn resolve_calibration(
        &self,
        _reference: &CalibrationReference,
    ) -> Result<ResolvedCalibration, String> {
        self.resolved.clone()
    }
}

impl RawDataEvidenceResolver for MockResolver {
    fn resolve_raw_data(
        &self,
        _reference: &RawDataReference,
    ) -> Result<ResolvedRawData, String> {
        self.resolved_raw.clone()
    }
}

fn resolved_raw_data() -> Result<ResolvedRawData, String> {
    Ok(
        ResolvedRawData::new(
            artifact("raw-acquisition-1", SHA_C),
            4_096,
            "application/octet-stream",
        )
        .unwrap(),
    )
}

fn resolver() -> MockResolver {
    MockResolver {
        resolved: Ok(resolved_calibration()),
        resolved_raw: resolved_raw_data(),
    }
}

#[test]
fn creates_a_unit_consistent_measurement_envelope() {
    let m = measurement(1, SAMPLE_TIME_NS);
    assert_eq!(m.quantity(), Quantity::Frequency);
    assert_eq!(m.unit(), Unit::Hertz);
    assert_eq!(m.value(), 5_000_000.0);
    assert_eq!(m.clock_domain().as_str(), "ultrasound-rig-boot-epoch-01");
    assert_eq!(m.identity().instrument_id(), "ultrasound-research-rig-01");
    assert_eq!(m.processing_chain_version(), "sha256:reconstruction-pipeline-v1");
}

#[test]
fn rejects_mismatched_quantity_and_unit() {
    let result = MeasurementEnvelope::new(MeasurementInput {
        identity: InstrumentIdentity::new("instrument", "channel").unwrap(),
        sequence: 1,
        captured_at_ns: SAMPLE_TIME_NS,
        clock_domain: clock_domain(),
        quantity: Quantity::Frequency,
        unit: Unit::Pascal,
        value: 1.0,
        standard_uncertainty: 0.1,
        calibration: None,
        raw_data: None,
        processing_chain_version: "capture-v1".into(),
        quality_flags: vec![],
    });
    assert!(matches!(
        result,
        Err(ContractError::QuantityUnitMismatch {
            quantity: Quantity::Frequency,
            unit: Unit::Pascal
        })
    ));
}

#[test]
fn rejects_non_finite_values_and_uncertainties() {
    let mut input = MeasurementInput {
        identity: InstrumentIdentity::new("instrument", "channel").unwrap(),
        sequence: 1,
        captured_at_ns: SAMPLE_TIME_NS,
        clock_domain: clock_domain(),
        quantity: Quantity::Frequency,
        unit: Unit::Hertz,
        value: 1.0,
        standard_uncertainty: 0.1,
        calibration: None,
        raw_data: None,
        processing_chain_version: "capture-v1".into(),
        quality_flags: vec![],
    };
    input.value = f64::NAN;
    assert_eq!(
        MeasurementEnvelope::new(input.clone()).unwrap_err(),
        ContractError::NonFiniteValue("value")
    );
    input.value = 1.0;
    input.standard_uncertainty = f64::INFINITY;
    assert_eq!(
        MeasurementEnvelope::new(input.clone()).unwrap_err(),
        ContractError::NonFiniteValue("standard_uncertainty")
    );
    input.standard_uncertainty = -0.1;
    assert_eq!(
        MeasurementEnvelope::new(input).unwrap_err(),
        ContractError::NegativeUncertainty("standard_uncertainty")
    );
}

#[test]
fn rejects_empty_ids_processing_version_and_invalid_digests() {
    assert_eq!(
        InstrumentIdentity::new(" ", "channel").unwrap_err(),
        ContractError::EmptyIdentifier("instrument_id")
    );
    assert_eq!(
        InstrumentIdentity::new("instrument", "\t").unwrap_err(),
        ContractError::EmptyIdentifier("channel_id")
    );
    assert_eq!(
        ArtifactReference::new("calibration", "not-a-digest").unwrap_err(),
        ContractError::InvalidSha256
    );
    assert_eq!(
        ClockDomainId::new(" ").unwrap_err(),
        ContractError::EmptyIdentifier("clock_domain_id")
    );
    let result = MeasurementEnvelope::new(MeasurementInput {
        identity: InstrumentIdentity::new("instrument", "channel").unwrap(),
        sequence: 1,
        captured_at_ns: SAMPLE_TIME_NS,
        clock_domain: clock_domain(),
        quantity: Quantity::Frequency,
        unit: Unit::Hertz,
        value: 1.0,
        standard_uncertainty: 0.0,
        calibration: None,
        raw_data: None,
        processing_chain_version: "   ".into(),
        quality_flags: vec![],
    });
    assert_eq!(result.unwrap_err(), ContractError::EmptyProcessingVersion);
}

#[test]
fn calibration_intervals_must_be_non_empty() {
    assert_eq!(
        ResolvedCalibration::new(
            "calibration",
            instrument_identity(),
            artifact("certificate", SHA_A),
            artifact("review", SHA_B),
            Quantity::Frequency,
            Unit::Hertz,
            4_000_000.0,
            6_000_000.0,
            42,
            42,
            0.0,
        )
        .unwrap_err(),
        ContractError::InvalidValidityInterval
    );
}

#[test]
fn calibration_evidence_and_review_receipt_must_be_distinct_artifacts() {
    let cases = [
        // Exact duplicate artifact.
        (artifact("same-artifact", SHA_A), artifact("same-artifact", SHA_A)),
        // Different aliases for identical content are not independent evidence.
        (artifact("certificate", SHA_A), artifact("review", SHA_A)),
        // Same alias pointing at different content is also suspicious.
        (artifact("same-id", SHA_A), artifact("same-id", SHA_B)),
    ];

    for (evidence, receipt) in cases {
        assert_eq!(
            ResolvedCalibration::new(
                "calibration-17",
                instrument_identity(),
                evidence,
                receipt,
                Quantity::Frequency,
                Unit::Hertz,
                4_000_000.0,
                6_000_000.0,
                SAMPLE_TIME_NS - 1,
                SAMPLE_TIME_NS + 2_000_000_000,
                900.0,
            )
            .unwrap_err(),
            ContractError::CalibrationReviewMustBeDistinct
        );
    }
}

#[test]
fn calibration_range_constructor_rejects_inverted_bounds() {
    assert_eq!(
        ResolvedCalibration::new(
            "bad-range-calibration",
            instrument_identity(),
            artifact("certificate", SHA_A),
            artifact("review", SHA_B),
            Quantity::Frequency,
            Unit::Hertz,
            6_000_000.0,
            4_000_000.0,
            SAMPLE_TIME_NS - 1,
            SAMPLE_TIME_NS + 2_000_000_000,
            900.0,
        )
        .unwrap_err(),
        ContractError::InvalidCalibrationRange
    );
}

#[test]
fn quantitative_gate_rejects_calibration_for_another_instrument_channel() {
    let wrong_instrument = MockResolver {
        resolved: Ok(
            ResolvedCalibration::new(
                "calibration-17",
                InstrumentIdentity::new("another-ultrasound-rig", "rf-channel-0").unwrap(),
                artifact("calibration-certificate", SHA_A),
                artifact("independent-review-receipt", SHA_B),
                Quantity::Frequency,
                Unit::Hertz,
                4_000_000.0,
                6_000_000.0,
                SAMPLE_TIME_NS - 1,
                SAMPLE_TIME_NS + 2_000_000_000,
                900.0,
            )
            .unwrap(),
        ),
        resolved_raw: resolved_raw_data(),
    };

    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &wrong_instrument,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationInstrumentMismatch
    );
}

#[test]
fn quantitative_gate_rejects_values_outside_the_calibrated_range() {
    let narrow_range = MockResolver {
        resolved: Ok(
            ResolvedCalibration::new(
                "calibration-17",
                instrument_identity(),
                artifact("calibration-certificate", SHA_A),
                artifact("independent-review-receipt", SHA_B),
                Quantity::Frequency,
                Unit::Hertz,
                4_000_000.0,
                4_900_000.0,
                SAMPLE_TIME_NS - 1,
                SAMPLE_TIME_NS + 2_000_000_000,
                900.0,
            )
            .unwrap(),
        ),
        resolved_raw: resolved_raw_data(),
    };

    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &narrow_range,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationRangeExceeded {
            value: 5_000_000.0,
            minimum: 4_000_000.0,
            maximum: 4_900_000.0,
        }
    );
}

#[test]
fn assessment_passes_only_with_resolved_applicable_calibration_and_complete_provenance() {
    let assessment = measurement(1, SAMPLE_TIME_NS)
        .assess_for_quantitative_use(
            SAMPLE_TIME_NS + 10,
            &clock_domain(),
            &policy(),
            &resolver(),
            &mut MeasurementStreamGuard::default(),
        )
        .unwrap();
    assert_eq!(assessment.age_ns, 10);
    assert_eq!(assessment.measurement_standard_uncertainty, 1_000.0);
    assert_eq!(assessment.calibration_standard_uncertainty, 900.0);
    assert_eq!(assessment.calibration_record_id, "calibration-17");
    assert_eq!(assessment.calibration_review_receipt.artifact_id(), "independent-review-receipt");
    assert_eq!(assessment.raw_data.artifact().artifact_id(), "raw-acquisition-1");
    assert_eq!(assessment.raw_data.byte_length(), 4_096);
    assert_eq!(assessment.raw_data.media_type(), "application/octet-stream");
}

#[test]
fn rejects_measurements_from_a_future_clock_or_beyond_freshness_limit() {
    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS - 1,
                &clock_domain(),
                &policy(),
                &resolver(),
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::ClockInFuture
    );
    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS + 500_000_001,
                &clock_domain(),
                &policy(),
                &resolver(),
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::Stale {
            age_ns: 500_000_001,
            max_age_ns: 500_000_000
        }
    );
}

#[test]
fn any_quality_flag_blocks_strict_quantitative_use() {
    let mut input = MeasurementInput {
        identity: InstrumentIdentity::new("instrument", "channel").unwrap(),
        sequence: 1,
        captured_at_ns: SAMPLE_TIME_NS,
        clock_domain: clock_domain(),
        quantity: Quantity::Frequency,
        unit: Unit::Hertz,
        value: 1.0,
        standard_uncertainty: 0.1,
        calibration: Some(calibration_reference()),
        raw_data: Some(RawDataReference::new(artifact("raw", SHA_B))),
        processing_chain_version: "capture-v1".into(),
        quality_flags: vec![QualityFlag::Saturated, QualityFlag::MotionArtifact],
    };
    input.quality_flags.dedup();
    let m = MeasurementEnvelope::new(input).unwrap();
    assert_eq!(
        m.assess_for_quantitative_use(
            SAMPLE_TIME_NS,
            &clock_domain(),
            &policy(),
            &resolver(),
            &mut MeasurementStreamGuard::default(),
        )
        .unwrap_err(),
        AssessmentFailure::QualityFlagsPresent(vec![
            QualityFlag::Saturated,
            QualityFlag::MotionArtifact
        ])
    );
}

#[test]
fn missing_raw_data_or_calibration_reference_fails_closed() {
    let mut input = MeasurementInput {
        identity: InstrumentIdentity::new("instrument", "channel").unwrap(),
        sequence: 1,
        captured_at_ns: SAMPLE_TIME_NS,
        clock_domain: clock_domain(),
        quantity: Quantity::Frequency,
        unit: Unit::Hertz,
        value: 1.0,
        standard_uncertainty: 0.1,
        calibration: Some(calibration_reference()),
        raw_data: None,
        processing_chain_version: "capture-v1".into(),
        quality_flags: vec![],
    };
    let no_raw = MeasurementEnvelope::new(input.clone()).unwrap();
    assert_eq!(
        no_raw.assess_for_quantitative_use(
            SAMPLE_TIME_NS,
            &clock_domain(),
            &policy(),
            &resolver(),
            &mut MeasurementStreamGuard::default(),
        )
        .unwrap_err(),
        AssessmentFailure::MissingRawDataReference
    );
    input.raw_data = Some(RawDataReference::new(artifact("raw", SHA_B)));
    input.calibration = None;
    let no_cal = MeasurementEnvelope::new(input).unwrap();
    assert_eq!(
        no_cal.assess_for_quantitative_use(
            SAMPLE_TIME_NS,
            &clock_domain(),
            &policy(),
            &resolver(),
            &mut MeasurementStreamGuard::default(),
        )
        .unwrap_err(),
        AssessmentFailure::MissingCalibrationReference
    );
}

#[test]
fn rejects_empty_or_malformed_resolved_raw_data_metadata() {
    assert_eq!(
        ResolvedRawData::new(
            artifact("empty-raw", SHA_C),
            0,
            "application/octet-stream",
        )
        .unwrap_err(),
        ContractError::EmptyRawDataArtifact
    );
    assert_eq!(
        ResolvedRawData::new(artifact("raw", SHA_C), 10, " ")
            .unwrap_err(),
        ContractError::EmptyIdentifier("raw_data_media_type")
    );
}

#[test]
fn unresolved_raw_acquisition_evidence_fails_closed() {
    let missing_raw = MockResolver {
        resolved: Ok(resolved_calibration()),
        resolved_raw: Err("raw artifact bytes not found".into()),
    };

    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &missing_raw,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::RawDataEvidenceUnresolved("raw artifact bytes not found".into())
    );
}

#[test]
fn resolved_raw_data_must_match_the_envelope_digest_and_artifact_id() {
    let mismatched_raw = MockResolver {
        resolved: Ok(resolved_calibration()),
        resolved_raw: Ok(
            ResolvedRawData::new(
                artifact("other-acquisition", SHA_C),
                4_096,
                "application/octet-stream",
            )
            .unwrap(),
        ),
    };

    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &mismatched_raw,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::RawDataReferenceMismatch
    );
}

#[test]
fn unresolved_calibration_evidence_fails_closed() {
    let missing = MockResolver {
        resolved: Err("calibration artifact not found".into()),
        resolved_raw: Ok(
            ResolvedRawData::new(
                artifact("raw-acquisition-1", SHA_C),
                4_096,
                "application/octet-stream",
            )
            .unwrap(),
        ),
    };
    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &missing,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationEvidenceUnresolved("calibration artifact not found".into())
    );
}

#[test]
fn rejects_a_resolver_result_for_a_different_calibration_artifact() {
    let wrong_artifact = MockResolver {
        resolved: Ok(ResolvedCalibration::new(
            "calibration-17",
            instrument_identity(),
            artifact("different-certificate", SHA_A),
            artifact("review", SHA_B),
            Quantity::Frequency,
            Unit::Hertz,
            4_000_000.0,
            6_000_000.0,
            SAMPLE_TIME_NS - 1,
            SAMPLE_TIME_NS + 2_000_000_000,
            900.0,
        )
        .unwrap()),
        resolved_raw: resolved_raw_data(),
    };
    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &wrong_artifact,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationReferenceMismatch
    );
}

#[test]
fn rejects_wrong_calibration_unit_or_out_of_validity_time() {
    let wrong_unit = MockResolver {
        resolved: Ok(ResolvedCalibration::new(
            "calibration-17",
            instrument_identity(),
            artifact("calibration-certificate", SHA_A),
            artifact("review", SHA_B),
            Quantity::AcousticPressure,
            Unit::Pascal,
            0.0,
            10_000_000.0,
            SAMPLE_TIME_NS - 1,
            SAMPLE_TIME_NS + 2_000_000_000,
            900.0,
        )
        .unwrap()),
        resolved_raw: resolved_raw_data(),
    };
    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &wrong_unit,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationUnitMismatch
    );

    let out_of_date = MockResolver {
        resolved: Ok(ResolvedCalibration::new(
            "calibration-17",
            instrument_identity(),
            artifact("calibration-certificate", SHA_A),
            artifact("review", SHA_B),
            Quantity::Frequency,
            Unit::Hertz,
            4_000_000.0,
            6_000_000.0,
            SAMPLE_TIME_NS + 1,
            SAMPLE_TIME_NS + 2_000_000_000,
            900.0,
        )
        .unwrap()),
        resolved_raw: resolved_raw_data(),
    };
    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &out_of_date,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationNotValidAtCapture
    );
}

#[test]
fn excessive_measurement_or_calibration_uncertainty_fails_closed() {
    let tight_measurement_policy = MeasurementPolicy::new(1_000, 999.0, 1_000.0).unwrap();
    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &tight_measurement_policy,
                &resolver(),
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::MeasurementUncertaintyExceeded
    );

    let tight_calibration_policy = MeasurementPolicy::new(1_000, 2_000.0, 899.0).unwrap();
    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &tight_calibration_policy,
                &resolver(),
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationUncertaintyExceeded
    );
}

#[test]
fn stream_guard_consumes_new_sequence_even_when_timestamp_regresses() {
    let first = measurement(7, SAMPLE_TIME_NS);
    let duplicate = measurement(7, SAMPLE_TIME_NS + 1);
    let backward_time = measurement(8, SAMPLE_TIME_NS - 1);
    let retry_same_sequence = measurement(8, SAMPLE_TIME_NS + 1);
    let next = measurement(9, SAMPLE_TIME_NS + 1);

    let mut guard = MeasurementStreamGuard::default();
    guard.observe(&first).unwrap();
    assert_eq!(
        guard.observe(&duplicate).unwrap_err(),
        StreamOrderFailure::SequenceNotIncreasing { previous: 7, received: 7 }
    );
    assert_eq!(
        guard.observe(&backward_time).unwrap_err(),
        StreamOrderFailure::TimestampMovedBackward {
            previous_ns: SAMPLE_TIME_NS,
            received_ns: SAMPLE_TIME_NS - 1
        }
    );
    assert_eq!(
        guard.observe(&retry_same_sequence).unwrap_err(),
        StreamOrderFailure::SequenceNotIncreasing { previous: 8, received: 8 }
    );
    guard.observe(&next).unwrap();
}

#[test]
fn quantitative_gate_itself_rejects_a_replayed_measurement() {
    let m = measurement(9, SAMPLE_TIME_NS);
    let mut guard = MeasurementStreamGuard::default();
    m.assess_for_quantitative_use(
        SAMPLE_TIME_NS,
        &clock_domain(),
        &policy(),
        &resolver(),
        &mut guard,
    )
    .unwrap();
    assert_eq!(
        m.assess_for_quantitative_use(
            SAMPLE_TIME_NS,
            &clock_domain(),
            &policy(),
            &resolver(),
            &mut guard,
        )
        .unwrap_err(),
        AssessmentFailure::StreamOrder(StreamOrderFailure::SequenceNotIncreasing {
            previous: 9,
            received: 9,
        })
    );
}

#[test]
fn clock_domain_mismatch_fails_before_consuming_the_measurement_sequence() {
    let m = measurement(1, SAMPLE_TIME_NS);
    let wrong_domain = ClockDomainId::new("other-rig-boot-epoch").unwrap();
    let mut guard = MeasurementStreamGuard::default();

    assert_eq!(
        m.assess_for_quantitative_use(
            SAMPLE_TIME_NS,
            &wrong_domain,
            &policy(),
            &resolver(),
            &mut guard,
        )
        .unwrap_err(),
        AssessmentFailure::ClockDomainMismatch
    );

    // A caller using the measurement's actual clock domain may retry because the
    // prior failure was a clock-context mismatch, not a consumed sensor sample.
    m.assess_for_quantitative_use(
        SAMPLE_TIME_NS,
        &clock_domain(),
        &policy(),
        &resolver(),
        &mut guard,
    )
    .unwrap();
}

#[test]
fn clock_epoch_change_is_rejected_by_an_existing_stream_guard() {
    let first = measurement(1, SAMPLE_TIME_NS);
    let next_epoch_id = ClockDomainId::new("ultrasound-rig-boot-epoch-02").unwrap();
    let second_epoch = measurement_in_clock_domain(
        2,
        SAMPLE_TIME_NS + 1,
        next_epoch_id.clone(),
    );
    let mut guard = MeasurementStreamGuard::default();

    first
        .assess_for_quantitative_use(
            SAMPLE_TIME_NS,
            &clock_domain(),
            &policy(),
            &resolver(),
            &mut guard,
        )
        .unwrap();

    assert_eq!(
        second_epoch
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS + 2,
                &next_epoch_id,
                &policy(),
                &resolver(),
                &mut guard,
            )
            .unwrap_err(),
        AssessmentFailure::StreamOrder(StreamOrderFailure::ClockDomainChanged {
            previous: clock_domain(),
            received: next_epoch_id,
        })
    );
}

#[test]
fn future_timestamp_consumes_sequence_without_poisoning_timestamp_high_water_mark() {
    let first = measurement(7, SAMPLE_TIME_NS);
    let future = measurement(8, SAMPLE_TIME_NS + 2_000_000_000);
    let next_valid = measurement(9, SAMPLE_TIME_NS + 10);
    let retry_future_sequence = measurement(8, SAMPLE_TIME_NS + 20);

    let mut guard = MeasurementStreamGuard::default();
    first
        .assess_for_quantitative_use(
            SAMPLE_TIME_NS,
            &clock_domain(),
            &policy(),
            &resolver(),
            &mut guard,
        )
        .unwrap();

    assert_eq!(
        future
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS + 1_000_000_000,
                &clock_domain(),
                &policy(),
                &resolver(),
                &mut guard,
            )
            .unwrap_err(),
        AssessmentFailure::ClockInFuture
    );

    // A later sequence with a timestamp beyond the last accepted sample remains
    // usable: the future-dated timestamp did not poison the timestamp high-water mark.
    next_valid
        .assess_for_quantitative_use(
            SAMPLE_TIME_NS + 20,
            &clock_domain(),
            &policy(),
            &resolver(),
            &mut guard,
        )
        .unwrap();

    // The rejected future sample's sequence was still consumed and cannot be replayed.
    assert_eq!(
        retry_future_sequence
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS + 30,
                &clock_domain(),
                &policy(),
                &resolver(),
                &mut guard,
            )
            .unwrap_err(),
        AssessmentFailure::StreamOrder(StreamOrderFailure::SequenceNotIncreasing {
            previous: 9,
            received: 8,
        })
    );
}

#[test]
fn units_map_to_the_declared_quantity_without_implicit_conversion() {
    assert_eq!(Unit::Megahertz.quantity(), Quantity::Frequency);
    assert_eq!(Unit::Microvolt.quantity(), Quantity::ElectricalPotential);
    assert_eq!(Unit::Percent.quantity(), Quantity::Dimensionless);
    assert_eq!(
        Unit::OxygenSaturationPercent.quantity(),
        Quantity::OxygenSaturation
    );
    assert_eq!(Unit::DecibelRe20Micropascal.quantity(), Quantity::SoundPressureLevel);
    // Same physical quantity, different units remain explicit; this crate does not convert.
    assert_ne!(Unit::Hertz, Unit::Megahertz);
}
