// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::{
    CalibrationContextV1, CalibrationState, CaptureTimeV1, DigestAlgorithm, EvidenceIdentity,
    FieldModality, FieldObservationV1, MeasurementPayload, ObservationValidity, QuantityKind,
    SiUnit, SourceClass, UnavailableReason, UncertaintyV1,
};

pub(crate) const CANONICAL_DOMAIN_V1: &[u8] = b"symthaea-field-observation-v1\0";

pub(crate) fn canonical_bytes(value: &FieldObservationV1) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    out.extend_from_slice(CANONICAL_DOMAIN_V1);
    push_u16(&mut out, value.schema_version);
    push_string(&mut out, &value.observation_id);
    push_u8(&mut out, modality_code(value.modality));
    push_u8(&mut out, quantity_code(value.quantity));
    push_payload(&mut out, &value.payload);
    push_u8(&mut out, unit_code(value.unit));
    push_u8(&mut out, source_code(value.source_class));
    push_string(&mut out, &value.source_id);
    push_string(&mut out, &value.coordinate_frame_id);
    push_capture_time(&mut out, &value.capture_time);
    push_string(&mut out, &value.received_at_monotonic.epoch_id);
    push_u64(&mut out, value.received_at_monotonic.ticks_ns);
    push_calibration(&mut out, &value.calibration);
    push_uncertainty(&mut out, value.uncertainty);
    push_evidence(&mut out, &value.provenance.evidence);
    push_u8(
        &mut out,
        source_code(value.provenance.asserted_source_class),
    );
    push_u8(&mut out, validity_code(value.validity));
    push_bool(&mut out, value.quality.clipped);
    push_bool(&mut out, value.quality.saturated);
    push_bool(&mut out, value.quality.low_snr);
    push_bool(&mut out, value.quality.out_of_range);
    out
}

fn normalized_f64_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    }
}

fn push_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn push_bool(out: &mut Vec<u8>, value: bool) {
    push_u8(out, u8::from(value));
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_i64(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_f64(out: &mut Vec<u8>, value: f64) {
    push_u64(out, normalized_f64_bits(value));
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    let len = u32::try_from(value.len()).expect("validated FIELD strings fit in u32");
    push_u32(out, len);
    out.extend_from_slice(value.as_bytes());
}

fn push_payload(out: &mut Vec<u8>, payload: &MeasurementPayload) {
    match payload {
        MeasurementPayload::Scalar(value) => {
            push_u8(out, 0);
            push_f64(out, *value);
        }
        MeasurementPayload::Vector { values, schema_id } => {
            push_u8(out, 1);
            push_string(out, schema_id);
            let len = u32::try_from(values.len())
                .expect("validated FIELD inline vector length fits in u32");
            push_u32(out, len);
            for value in values {
                push_f64(out, *value);
            }
        }
        MeasurementPayload::StructuredReference {
            evidence,
            schema_id,
            sample_count,
        } => {
            push_u8(out, 2);
            push_evidence(out, evidence);
            push_string(out, schema_id);
            match sample_count {
                Some(count) => {
                    push_u8(out, 1);
                    push_u64(out, *count);
                }
                None => push_u8(out, 0),
            }
        }
        MeasurementPayload::Unavailable { reason } => {
            push_u8(out, 3);
            push_u8(out, unavailable_code(*reason));
        }
    }
}

fn push_capture_time(out: &mut Vec<u8>, time: &CaptureTimeV1) {
    match time {
        CaptureTimeV1::UnixUtc { unix_nanos } => {
            push_u8(out, 0);
            push_i64(out, *unix_nanos);
        }
        CaptureTimeV1::Monotonic { epoch_id, ticks_ns } => {
            push_u8(out, 1);
            push_string(out, epoch_id);
            push_u64(out, *ticks_ns);
        }
        CaptureTimeV1::Replay {
            timeline_id,
            ticks_ns,
        } => {
            push_u8(out, 2);
            push_string(out, timeline_id);
            push_u64(out, *ticks_ns);
        }
    }
}

fn push_calibration(out: &mut Vec<u8>, calibration: &CalibrationContextV1) {
    let state = match calibration.state {
        CalibrationState::Valid => 0,
        CalibrationState::NotRequired => 1,
        CalibrationState::Unverified => 2,
        CalibrationState::Expired => 3,
        CalibrationState::Invalid => 4,
    };
    push_u8(out, state);

    match calibration.calibration_id.as_deref() {
        Some(id) => {
            push_u8(out, 1);
            push_string(out, id);
        }
        None => push_u8(out, 0),
    }
}

fn push_uncertainty(out: &mut Vec<u8>, uncertainty: UncertaintyV1) {
    match uncertainty {
        UncertaintyV1::Unknown => push_u8(out, 0),
        UncertaintyV1::StandardDeviation { sigma } => {
            push_u8(out, 1);
            push_f64(out, sigma);
        }
        UncertaintyV1::AbsoluteBound { half_width } => {
            push_u8(out, 2);
            push_f64(out, half_width);
        }
    }
}

fn push_evidence(out: &mut Vec<u8>, evidence: &EvidenceIdentity) {
    let algorithm = match evidence.algorithm {
        DigestAlgorithm::Sha256 => 0,
    };
    push_u8(out, algorithm);
    push_string(out, &evidence.digest_hex);
}

fn modality_code(value: FieldModality) -> u8 {
    match value {
        FieldModality::Acoustic => 0,
        FieldModality::Optical => 1,
        FieldModality::Plasma => 2,
        FieldModality::Electromagnetic => 3,
        FieldModality::Thermal => 4,
        FieldModality::Chemical => 5,
        FieldModality::Mechanical => 6,
    }
}

fn quantity_code(value: QuantityKind) -> u8 {
    match value {
        QuantityKind::Pressure => 0,
        QuantityKind::Frequency => 1,
        QuantityKind::Wavelength => 2,
        QuantityKind::PhaseAngle => 3,
        QuantityKind::Time => 4,
        QuantityKind::Power => 5,
        QuantityKind::Energy => 6,
        QuantityKind::Force => 7,
        QuantityKind::Intensity => 8,
        QuantityKind::Irradiance => 9,
        QuantityKind::Voltage => 10,
        QuantityKind::Current => 11,
        QuantityKind::Temperature => 12,
        QuantityKind::Distance => 13,
        QuantityKind::Displacement => 14,
        QuantityKind::Velocity => 15,
        QuantityKind::Acceleration => 16,
        QuantityKind::AngularVelocity => 17,
        QuantityKind::MagneticFluxDensity => 18,
        QuantityKind::ElectricFieldStrength => 19,
        QuantityKind::MassDensity => 20,
        QuantityKind::NumberDensity => 21,
        QuantityKind::Concentration => 22,
        QuantityKind::SpectralIntensity => 23,
        QuantityKind::Dimensionless => 24,
    }
}

fn unit_code(value: SiUnit) -> u8 {
    match value {
        SiUnit::Pascal => 0,
        SiUnit::Hertz => 1,
        SiUnit::Metre => 2,
        SiUnit::Radian => 3,
        SiUnit::Second => 4,
        SiUnit::Watt => 5,
        SiUnit::Joule => 6,
        SiUnit::Newton => 7,
        SiUnit::WattPerSquareMetre => 8,
        SiUnit::Volt => 9,
        SiUnit::Ampere => 10,
        SiUnit::Kelvin => 11,
        SiUnit::MetrePerSecond => 12,
        SiUnit::MetrePerSecondSquared => 13,
        SiUnit::RadianPerSecond => 14,
        SiUnit::Tesla => 15,
        SiUnit::VoltPerMetre => 16,
        SiUnit::KilogramPerCubicMetre => 17,
        SiUnit::PerCubicMetre => 18,
        SiUnit::MolePerCubicMetre => 19,
        SiUnit::WattPerSquareMetreHertz => 20,
        SiUnit::One => 21,
    }
}

fn source_code(value: SourceClass) -> u8 {
    match value {
        SourceClass::Physical => 0,
        SourceClass::Simulated => 1,
        SourceClass::Replay => 2,
    }
}

fn unavailable_code(value: UnavailableReason) -> u8 {
    match value {
        UnavailableReason::NoSample => 0,
        UnavailableReason::SensorUnavailable => 1,
        UnavailableReason::BelowDetectionLimit => 2,
        UnavailableReason::AboveDetectionLimit => 3,
        UnavailableReason::Unknown => 4,
    }
}

fn validity_code(value: ObservationValidity) -> u8 {
    match value {
        ObservationValidity::Nominal => 0,
        ObservationValidity::Degraded => 1,
        ObservationValidity::Invalid => 2,
    }
}
