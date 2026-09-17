// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::*;
use crate::canonical::CANONICAL_DOMAIN_V1;

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn observation() -> FieldObservationV1 {
    FieldObservationV1 {
        schema_version: FIELD_OBSERVATION_SCHEMA_V1,
        observation_id: "obs-0001".into(),
        modality: FieldModality::Acoustic,
        quantity: QuantityKind::Pressure,
        payload: MeasurementPayload::Scalar(1.25),
        unit: SiUnit::Pascal,
        source_class: SourceClass::Physical,
        source_id: "microphone-array-1".into(),
        coordinate_frame_id: "lab/bench-a".into(),
        capture_time: CaptureTimeV1::Monotonic {
            epoch_id: "boot-42".into(),
            ticks_ns: 1_000,
        },
        received_at_monotonic: MonotonicInstantV1 {
            epoch_id: "boot-42".into(),
            ticks_ns: 1_050,
        },
        calibration: CalibrationContextV1 {
            calibration_id: Some("cal-2026-09".into()),
            state: CalibrationState::Valid,
        },
        uncertainty: UncertaintyV1::StandardDeviation { sigma: 0.01 },
        provenance: ProvenanceV1 {
            evidence: EvidenceIdentity {
                algorithm: DigestAlgorithm::Sha256,
                digest_hex: DIGEST.into(),
            },
            asserted_source_class: SourceClass::Physical,
        },
        validity: ObservationValidity::Nominal,
        quality: QualityFlagsV1::default(),
    }
}

#[test]
fn valid_observation_passes_and_canonicalizes_deterministically() {
    let value = observation();
    assert_eq!(value.validate(), Ok(()));
    assert_eq!(
        value.canonical_bytes().unwrap(),
        value.canonical_bytes().unwrap()
    );
    assert!(
        value
            .canonical_bytes()
            .unwrap()
            .starts_with(CANONICAL_DOMAIN_V1)
    );
}

#[test]
fn validated_wrapper_requires_validation() {
    let validated = observation().into_validated().unwrap();
    assert_eq!(validated.as_observation().observation_id, "obs-0001");

    let mut invalid = observation();
    invalid.unit = SiUnit::Kelvin;
    assert!(ValidatedFieldObservationV1::try_from(invalid).is_err());
}

#[test]
fn positive_and_negative_zero_have_one_canonical_form() {
    let mut positive = observation();
    positive.payload = MeasurementPayload::Scalar(0.0);
    let mut negative = positive.clone();
    negative.payload = MeasurementPayload::Scalar(-0.0);
    assert_eq!(
        positive.canonical_bytes().unwrap(),
        negative.canonical_bytes().unwrap()
    );
}

#[test]
fn wall_and_monotonic_clocks_cannot_collapse() {
    let mut wall = observation();
    wall.capture_time = CaptureTimeV1::UnixUtc { unix_nanos: 1_000 };

    let mut monotonic = observation();
    monotonic.capture_time = CaptureTimeV1::Monotonic {
        epoch_id: "boot-42".into(),
        ticks_ns: 1_000,
    };

    assert_ne!(
        wall.canonical_bytes().unwrap(),
        monotonic.canonical_bytes().unwrap()
    );
}

#[test]
fn replay_class_and_clock_must_agree() {
    let mut value = observation();
    value.source_class = SourceClass::Replay;
    value.provenance.asserted_source_class = SourceClass::Replay;
    assert_eq!(
        value.validate(),
        Err(ValidationError::SourceClockMismatch)
    );

    value.capture_time = CaptureTimeV1::Replay {
        timeline_id: "trace-1".into(),
        ticks_ns: 1,
    };
    assert_eq!(value.validate(), Ok(()));
}

#[test]
fn malformed_clock_identity_fails_closed() {
    let mut value = observation();
    value.capture_time = CaptureTimeV1::Monotonic {
        epoch_id: String::new(),
        ticks_ns: 1,
    };
    assert_eq!(
        value.validate(),
        Err(ValidationError::InvalidClockIdentity)
    );
}

#[test]
fn receive_cannot_precede_capture_in_same_monotonic_epoch() {
    let mut value = observation();
    value.received_at_monotonic.ticks_ns = 999;
    assert_eq!(
        value.validate(),
        Err(ValidationError::ReceiveBeforeCapture)
    );

    value.received_at_monotonic.epoch_id = "ingress-epoch".into();
    assert_eq!(value.validate(), Ok(()));
}

#[test]
fn nonfinite_empty_and_zero_reference_payloads_fail_closed() {
    let mut value = observation();
    value.payload = MeasurementPayload::Scalar(f64::NAN);
    assert_eq!(
        value.validate(),
        Err(ValidationError::NonFiniteMeasurement)
    );

    value.payload = MeasurementPayload::Vector {
        values: vec![],
        schema_id: "field.vector.xyz.v1".into(),
    };
    assert_eq!(value.validate(), Err(ValidationError::EmptyVector));

    value.payload = MeasurementPayload::StructuredReference {
        evidence: EvidenceIdentity {
            algorithm: DigestAlgorithm::Sha256,
            digest_hex: DIGEST.into(),
        },
        schema_id: "field.waveform-f32-le.v1".into(),
        sample_count: Some(0),
    };
    assert_eq!(
        value.validate(),
        Err(ValidationError::InvalidSampleCount)
    );
}

#[test]
fn inline_vector_requires_and_binds_schema_identity() {
    let mut invalid = observation();
    invalid.payload = MeasurementPayload::Vector {
        values: vec![1.0, 2.0, 3.0],
        schema_id: String::new(),
    };
    assert_eq!(
        invalid.validate(),
        Err(ValidationError::InvalidIdentifier("vector-schema"))
    );

    let mut xyz = observation();
    xyz.payload = MeasurementPayload::Vector {
        values: vec![1.0, 2.0, 3.0],
        schema_id: "field.vector.xyz.v1".into(),
    };
    let mut rgb = xyz.clone();
    rgb.payload = MeasurementPayload::Vector {
        values: vec![1.0, 2.0, 3.0],
        schema_id: "field.vector.rgb.v1".into(),
    };

    assert_ne!(xyz.canonical_bytes().unwrap(), rgb.canonical_bytes().unwrap());
}

#[test]
fn structured_reference_requires_a_schema_identity() {
    let mut value = observation();
    value.payload = MeasurementPayload::StructuredReference {
        evidence: EvidenceIdentity {
            algorithm: DigestAlgorithm::Sha256,
            digest_hex: DIGEST.into(),
        },
        schema_id: String::new(),
        sample_count: Some(32),
    };
    assert_eq!(
        value.validate(),
        Err(ValidationError::InvalidIdentifier("payload-schema"))
    );
}

#[test]
fn missing_measurement_is_not_numeric_zero() {
    let mut missing = observation();
    missing.payload = MeasurementPayload::Unavailable {
        reason: UnavailableReason::NoSample,
    };
    assert_eq!(
        missing.validate(),
        Err(ValidationError::UnavailableUncertaintyMismatch)
    );

    missing.uncertainty = UncertaintyV1::Unknown;
    assert_eq!(
        missing.validate(),
        Err(ValidationError::UnavailableNominal)
    );

    missing.validity = ObservationValidity::Degraded;
    assert_eq!(missing.validate(), Ok(()));

    let mut zero = observation();
    zero.payload = MeasurementPayload::Scalar(0.0);
    assert_ne!(
        missing.canonical_bytes().unwrap(),
        zero.canonical_bytes().unwrap()
    );
}

#[test]
fn unavailable_measurement_cannot_claim_numeric_uncertainty() {
    for uncertainty in [
        UncertaintyV1::StandardDeviation { sigma: 0.1 },
        UncertaintyV1::AbsoluteBound { half_width: 0.2 },
    ] {
        let mut value = observation();
        value.payload = MeasurementPayload::Unavailable {
            reason: UnavailableReason::SensorUnavailable,
        };
        value.validity = ObservationValidity::Degraded;
        value.uncertainty = uncertainty;
        assert_eq!(
            value.validate(),
            Err(ValidationError::UnavailableUncertaintyMismatch)
        );
    }
}

#[test]
fn oversized_inline_vector_fails_closed() {
    let mut value = observation();
    value.payload = MeasurementPayload::Vector {
        values: vec![0.0; MAX_INLINE_VECTOR_VALUES + 1],
        schema_id: "field.vector.generic.v1".into(),
    };
    assert_eq!(
        value.validate(),
        Err(ValidationError::InlineVectorTooLarge {
            len: MAX_INLINE_VECTOR_VALUES + 1,
            max: MAX_INLINE_VECTOR_VALUES,
        })
    );
}

#[test]
fn incompatible_quantity_and_unit_fail_closed() {
    let mut value = observation();
    value.unit = SiUnit::Kelvin;
    assert_eq!(
        value.validate(),
        Err(ValidationError::IncompatibleQuantityUnit)
    );
}

#[test]
fn calibration_semantics_fail_closed() {
    for (state, expected) in [
        (
            CalibrationState::Expired,
            ValidationError::ExpiredCalibration,
        ),
        (
            CalibrationState::Invalid,
            ValidationError::InvalidCalibration,
        ),
    ] {
        let mut value = observation();
        value.calibration.state = state;
        assert_eq!(value.validate(), Err(expected));
    }

    let mut value = observation();
    value.calibration.calibration_id = None;
    assert_eq!(
        value.validate(),
        Err(ValidationError::InvalidCalibrationIdentity)
    );

    value.calibration.state = CalibrationState::NotRequired;
    assert_eq!(value.validate(), Ok(()));
}

#[test]
fn unverified_calibration_cannot_claim_nominal_validity() {
    let mut value = observation();
    value.calibration.state = CalibrationState::Unverified;
    assert_eq!(
        value.validate(),
        Err(ValidationError::CalibrationValidityMismatch)
    );

    value.validity = ObservationValidity::Degraded;
    assert_eq!(value.validate(), Ok(()));
}

#[test]
fn impossible_uncertainty_fails_closed() {
    for uncertainty in [
        UncertaintyV1::StandardDeviation { sigma: -0.1 },
        UncertaintyV1::AbsoluteBound {
            half_width: f64::INFINITY,
        },
    ] {
        let mut value = observation();
        value.uncertainty = uncertainty;
        assert_eq!(
            value.validate(),
            Err(ValidationError::InvalidUncertainty)
        );
    }
}

#[test]
fn malformed_frame_digest_and_provenance_fail_closed() {
    let mut value = observation();
    value.coordinate_frame_id = "frame with spaces".into();
    assert_eq!(
        value.validate(),
        Err(ValidationError::InvalidIdentifier("coordinate-frame"))
    );

    let mut value = observation();
    value.provenance.evidence.digest_hex = "A".repeat(64);
    assert_eq!(value.validate(), Err(ValidationError::InvalidDigest));

    let mut value = observation();
    value.provenance.asserted_source_class = SourceClass::Simulated;
    assert_eq!(
        value.validate(),
        Err(ValidationError::SourceProvenanceMismatch)
    );
}

#[test]
fn nominal_validity_cannot_hide_degraded_quality() {
    let mut value = observation();
    value.quality.clipped = true;
    assert_eq!(
        value.validate(),
        Err(ValidationError::QualityValidityMismatch)
    );

    value.validity = ObservationValidity::Degraded;
    assert_eq!(value.validate(), Ok(()));
}

#[test]
fn explicitly_invalid_observation_fails_closed() {
    let mut value = observation();
    value.validity = ObservationValidity::Invalid;
    assert_eq!(
        value.validate(),
        Err(ValidationError::InvalidObservation)
    );
}

fn json_roundtrip<T>(values: &[T])
where
    T: Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug,
{
    let json = serde_json::to_string(values).unwrap();
    let decoded: Vec<T> = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, values);
}

#[test]
fn all_unit_like_enum_variants_roundtrip() {
    json_roundtrip(&[
        FieldModality::Acoustic,
        FieldModality::Optical,
        FieldModality::Plasma,
        FieldModality::Electromagnetic,
        FieldModality::Thermal,
        FieldModality::Chemical,
        FieldModality::Mechanical,
    ]);
    json_roundtrip(&[
        SourceClass::Physical,
        SourceClass::Simulated,
        SourceClass::Replay,
    ]);
    json_roundtrip(&[
        CalibrationState::Valid,
        CalibrationState::NotRequired,
        CalibrationState::Unverified,
        CalibrationState::Expired,
        CalibrationState::Invalid,
    ]);
    json_roundtrip(&[
        ObservationValidity::Nominal,
        ObservationValidity::Degraded,
        ObservationValidity::Invalid,
    ]);
    json_roundtrip(&[
        UnavailableReason::NoSample,
        UnavailableReason::SensorUnavailable,
        UnavailableReason::BelowDetectionLimit,
        UnavailableReason::AboveDetectionLimit,
        UnavailableReason::Unknown,
    ]);
    json_roundtrip(&[DigestAlgorithm::Sha256]);
}

#[test]
fn all_quantity_and_unit_variants_roundtrip() {
    json_roundtrip(&[
        QuantityKind::Pressure,
        QuantityKind::Frequency,
        QuantityKind::Wavelength,
        QuantityKind::PhaseAngle,
        QuantityKind::Time,
        QuantityKind::Power,
        QuantityKind::Energy,
        QuantityKind::Force,
        QuantityKind::Intensity,
        QuantityKind::Irradiance,
        QuantityKind::Voltage,
        QuantityKind::Current,
        QuantityKind::Temperature,
        QuantityKind::Distance,
        QuantityKind::Displacement,
        QuantityKind::Velocity,
        QuantityKind::Acceleration,
        QuantityKind::AngularVelocity,
        QuantityKind::MagneticFluxDensity,
        QuantityKind::ElectricFieldStrength,
        QuantityKind::MassDensity,
        QuantityKind::NumberDensity,
        QuantityKind::Concentration,
        QuantityKind::SpectralIntensity,
        QuantityKind::Dimensionless,
    ]);
    json_roundtrip(&[
        SiUnit::Pascal,
        SiUnit::Hertz,
        SiUnit::Metre,
        SiUnit::Radian,
        SiUnit::Second,
        SiUnit::Watt,
        SiUnit::Joule,
        SiUnit::Newton,
        SiUnit::WattPerSquareMetre,
        SiUnit::Volt,
        SiUnit::Ampere,
        SiUnit::Kelvin,
        SiUnit::MetrePerSecond,
        SiUnit::MetrePerSecondSquared,
        SiUnit::RadianPerSecond,
        SiUnit::Tesla,
        SiUnit::VoltPerMetre,
        SiUnit::KilogramPerCubicMetre,
        SiUnit::PerCubicMetre,
        SiUnit::MolePerCubicMetre,
        SiUnit::WattPerSquareMetreHertz,
        SiUnit::One,
    ]);
}

#[test]
fn all_data_bearing_enum_variants_roundtrip() {
    json_roundtrip(&[
        CaptureTimeV1::UnixUtc { unix_nanos: 1 },
        CaptureTimeV1::Monotonic {
            epoch_id: "boot-a".into(),
            ticks_ns: 2,
        },
        CaptureTimeV1::Replay {
            timeline_id: "replay-a".into(),
            ticks_ns: 3,
        },
    ]);
    json_roundtrip(&[
        UncertaintyV1::Unknown,
        UncertaintyV1::StandardDeviation { sigma: 0.1 },
        UncertaintyV1::AbsoluteBound { half_width: 0.2 },
    ]);
    json_roundtrip(&[
        MeasurementPayload::Scalar(1.0),
        MeasurementPayload::Vector {
            values: vec![1.0, 2.0],
            schema_id: "field.vector.xy.v1".into(),
        },
        MeasurementPayload::StructuredReference {
            evidence: EvidenceIdentity {
                algorithm: DigestAlgorithm::Sha256,
                digest_hex: DIGEST.into(),
            },
            schema_id: "field.spectrum-f64.v1".into(),
            sample_count: Some(2),
        },
        MeasurementPayload::Unavailable {
            reason: UnavailableReason::SensorUnavailable,
        },
    ]);
}

#[test]
fn complete_observation_json_roundtrip() {
    let value = observation();
    let json = serde_json::to_string(&value).unwrap();
    let decoded: FieldObservationV1 = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, value);
}
