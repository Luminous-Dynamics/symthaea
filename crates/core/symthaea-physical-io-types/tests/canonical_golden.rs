// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// FIELD-001 independent known-answer vector.
//
// The expected bytes below were produced from the published canonical format,
// independently of the Rust encoder. This catches accidental changes to field
// order, enum codes, endianness, string framing, floating-point framing, or the
// domain separator even when encoder and unit test code drift together.

use symthaea_physical_io_types::{
    CalibrationContextV1, CalibrationState, CaptureTimeV1, DigestAlgorithm, EvidenceIdentity,
    FieldModality, FieldObservationV1, MeasurementPayload, MonotonicInstantV1,
    ObservationValidity, ProvenanceV1, QualityFlagsV1, QuantityKind, SiUnit, SourceClass,
    UncertaintyV1, FIELD_OBSERVATION_SCHEMA_V1,
};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const EXPECTED_CANONICAL_HEX: &str = concat!(
    "73796d74686165612d6669656c642d6f62736572766174696f6e2d763100",
    "0001000000086f62732d303030310000003ff4000000000000000000000012",
    "6d6963726f70686f6e652d61727261792d310000000b6c61622f62656e6368",
    "2d610100000007626f6f742d343200000000000003e800000007626f6f742d",
    "3432000000000000041a00010000000b63616c2d323032362d3039013f847a",
    "e147ae147b000000004030313233343536373839616263646566303132333435",
    "3637383961626364656630313233343536373839616263646566303132333435",
    "36373839616263646566000000000000"
);

fn fixture() -> FieldObservationV1 {
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

fn decode_hex(value: &str) -> Vec<u8> {
    assert_eq!(value.len() % 2, 0, "golden vector must contain whole bytes");
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let hi = hex_nibble(pair[0]);
            let lo = hex_nibble(pair[1]);
            (hi << 4) | lo
        })
        .collect()
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => panic!("non-canonical hex digit in golden vector"),
    }
}

#[test]
fn field_observation_v1_matches_independent_canonical_golden_vector() {
    let actual = fixture().canonical_bytes().expect("fixture must validate");
    let expected = decode_hex(EXPECTED_CANONICAL_HEX);

    assert_eq!(expected.len(), 234, "golden vector length changed unexpectedly");
    assert_eq!(actual, expected);
}
