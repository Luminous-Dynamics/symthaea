// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_physical_io_types::{
    CalibrationContextV1, CalibrationState, CaptureTimeV1, DigestAlgorithm, EvidenceIdentity,
    FieldModality, FieldObservationV1, MeasurementPayload, MonotonicInstantV1,
    ObservationValidity, ProvenanceV1, QualityFlagsV1, QuantityKind, SiUnit, SourceClass,
    UncertaintyV1, FIELD_OBSERVATION_SCHEMA_V1,
};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn fixture() -> FieldObservationV1 {
    FieldObservationV1 {
        schema_version: FIELD_OBSERVATION_SCHEMA_V1,
        observation_id: "obs-wire-strict".into(),
        modality: FieldModality::Acoustic,
        quantity: QuantityKind::Pressure,
        payload: MeasurementPayload::Scalar(1.0),
        unit: SiUnit::Pascal,
        source_class: SourceClass::Physical,
        source_id: "sensor-1".into(),
        coordinate_frame_id: "lab/frame-1".into(),
        capture_time: CaptureTimeV1::Monotonic {
            epoch_id: "boot-1".into(),
            ticks_ns: 10,
        },
        received_at_monotonic: MonotonicInstantV1 {
            epoch_id: "boot-1".into(),
            ticks_ns: 11,
        },
        calibration: CalibrationContextV1 {
            calibration_id: Some("cal-1".into()),
            state: CalibrationState::Valid,
        },
        uncertainty: UncertaintyV1::Unknown,
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
fn top_level_unknown_field_is_rejected() {
    let mut json = serde_json::to_value(fixture()).unwrap();
    json.as_object_mut()
        .unwrap()
        .insert("future_authority".into(), serde_json::json!(true));

    assert!(serde_json::from_value::<FieldObservationV1>(json).is_err());
}

#[test]
fn nested_clock_unknown_field_is_rejected() {
    let json = r#"{"Monotonic":{"epoch_id":"boot-1","ticks_ns":10,"wall_time":123}}"#;
    assert!(serde_json::from_str::<CaptureTimeV1>(json).is_err());
}

#[test]
fn vector_unknown_field_is_rejected() {
    let json = r#"{"Vector":{"values":[1.0,2.0],"schema_id":"field.vector.xy.v1","implicit_order":"xy"}}"#;
    assert!(serde_json::from_str::<MeasurementPayload>(json).is_err());
}

#[test]
fn ordinary_valid_wire_form_still_roundtrips() {
    let value = fixture();
    let json = serde_json::to_string(&value).unwrap();
    let decoded: FieldObservationV1 = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, value);
}
