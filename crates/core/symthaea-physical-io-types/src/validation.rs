// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::{
    CalibrationContextV1, CalibrationState, CaptureTimeV1, DigestAlgorithm, EvidenceIdentity,
    FieldObservationV1, MeasurementPayload, ObservationValidity, SourceClass, UncertaintyV1,
    ValidationError, FIELD_OBSERVATION_SCHEMA_V1, MAX_INLINE_VECTOR_VALUES,
};

const MAX_ID_LEN: usize = 160;

pub(crate) fn validate(value: &FieldObservationV1) -> Result<(), ValidationError> {
    if value.schema_version != FIELD_OBSERVATION_SCHEMA_V1 {
        return Err(ValidationError::UnsupportedSchemaVersion(
            value.schema_version,
        ));
    }

    validate_id(&value.observation_id, "observation")?;
    validate_id(&value.source_id, "source")?;
    validate_id(&value.coordinate_frame_id, "coordinate-frame")?;

    if !value.unit.is_compatible_with(value.quantity) {
        return Err(ValidationError::IncompatibleQuantityUnit);
    }

    validate_payload(&value.payload)?;
    validate_capture_time(&value.capture_time)?;
    validate_id(
        &value.received_at_monotonic.epoch_id,
        "receive-monotonic-epoch",
    )?;
    validate_source_clock(value.source_class, &value.capture_time)?;
    validate_receive_order(value)?;
    validate_calibration(&value.calibration)?;
    validate_uncertainty(value.uncertainty)?;
    validate_unavailable_uncertainty(&value.payload, value.uncertainty)?;
    validate_evidence(&value.provenance.evidence)?;

    if value.provenance.asserted_source_class != value.source_class {
        return Err(ValidationError::SourceProvenanceMismatch);
    }

    if value.validity == ObservationValidity::Invalid {
        return Err(ValidationError::InvalidObservation);
    }

    if value.validity == ObservationValidity::Nominal && value.quality.any() {
        return Err(ValidationError::QualityValidityMismatch);
    }

    if value.validity == ObservationValidity::Nominal
        && value.calibration.state == CalibrationState::Unverified
    {
        return Err(ValidationError::CalibrationValidityMismatch);
    }

    if value.validity == ObservationValidity::Nominal
        && matches!(&value.payload, MeasurementPayload::Unavailable { .. })
    {
        return Err(ValidationError::UnavailableNominal);
    }

    Ok(())
}

fn validate_id(value: &str, field: &'static str) -> Result<(), ValidationError> {
    let allowed = value.bytes().all(|byte| {
        byte.is_ascii_alphanumeric()
            || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/' | b'@')
    });

    if value.is_empty() || value.len() > MAX_ID_LEN || !allowed {
        return Err(ValidationError::InvalidIdentifier(field));
    }

    Ok(())
}

fn validate_payload(payload: &MeasurementPayload) -> Result<(), ValidationError> {
    match payload {
        MeasurementPayload::Scalar(value) => validate_finite(*value),
        MeasurementPayload::Vector { values, schema_id } => {
            validate_id(schema_id, "vector-schema")?;
            if values.is_empty() {
                return Err(ValidationError::EmptyVector);
            }
            if values.len() > MAX_INLINE_VECTOR_VALUES {
                return Err(ValidationError::InlineVectorTooLarge {
                    len: values.len(),
                    max: MAX_INLINE_VECTOR_VALUES,
                });
            }
            values
                .iter()
                .try_for_each(|value| validate_finite(*value))
        }
        MeasurementPayload::StructuredReference {
            evidence,
            schema_id,
            sample_count,
        } => {
            validate_evidence(evidence)?;
            validate_id(schema_id, "payload-schema")?;
            if matches!(sample_count, Some(0)) {
                return Err(ValidationError::InvalidSampleCount);
            }
            Ok(())
        }
        MeasurementPayload::Unavailable { .. } => Ok(()),
    }
}

fn validate_finite(value: f64) -> Result<(), ValidationError> {
    value
        .is_finite()
        .then_some(())
        .ok_or(ValidationError::NonFiniteMeasurement)
}

fn validate_capture_time(time: &CaptureTimeV1) -> Result<(), ValidationError> {
    let clock_id = match time {
        CaptureTimeV1::UnixUtc { .. } => return Ok(()),
        CaptureTimeV1::Monotonic { epoch_id, .. } => epoch_id,
        CaptureTimeV1::Replay { timeline_id, .. } => timeline_id,
    };

    validate_id(clock_id, "clock-domain").map_err(|_| ValidationError::InvalidClockIdentity)
}

fn validate_source_clock(
    source_class: SourceClass,
    capture_time: &CaptureTimeV1,
) -> Result<(), ValidationError> {
    let replay_clock = matches!(capture_time, CaptureTimeV1::Replay { .. });
    let replay_source = source_class == SourceClass::Replay;

    if replay_clock == replay_source {
        Ok(())
    } else {
        Err(ValidationError::SourceClockMismatch)
    }
}

fn validate_receive_order(value: &FieldObservationV1) -> Result<(), ValidationError> {
    let CaptureTimeV1::Monotonic { epoch_id, ticks_ns } = &value.capture_time else {
        return Ok(());
    };

    if epoch_id == &value.received_at_monotonic.epoch_id
        && value.received_at_monotonic.ticks_ns < *ticks_ns
    {
        return Err(ValidationError::ReceiveBeforeCapture);
    }

    Ok(())
}

fn validate_calibration(calibration: &CalibrationContextV1) -> Result<(), ValidationError> {
    match calibration.state {
        CalibrationState::Valid => match calibration.calibration_id.as_deref() {
            Some(id) => validate_id(id, "calibration")
                .map_err(|_| ValidationError::InvalidCalibrationIdentity),
            None => Err(ValidationError::InvalidCalibrationIdentity),
        },
        CalibrationState::NotRequired => calibration
            .calibration_id
            .is_none()
            .then_some(())
            .ok_or(ValidationError::InvalidCalibrationIdentity),
        CalibrationState::Unverified => {
            if let Some(id) = calibration.calibration_id.as_deref() {
                validate_id(id, "calibration")
                    .map_err(|_| ValidationError::InvalidCalibrationIdentity)?;
            }
            Ok(())
        }
        CalibrationState::Expired => Err(ValidationError::ExpiredCalibration),
        CalibrationState::Invalid => Err(ValidationError::InvalidCalibration),
    }
}

fn validate_uncertainty(uncertainty: UncertaintyV1) -> Result<(), ValidationError> {
    let value = match uncertainty {
        UncertaintyV1::Unknown => return Ok(()),
        UncertaintyV1::StandardDeviation { sigma } => sigma,
        UncertaintyV1::AbsoluteBound { half_width } => half_width,
    };

    (value.is_finite() && value >= 0.0)
        .then_some(())
        .ok_or(ValidationError::InvalidUncertainty)
}

fn validate_unavailable_uncertainty(
    payload: &MeasurementPayload,
    uncertainty: UncertaintyV1,
) -> Result<(), ValidationError> {
    if matches!(payload, MeasurementPayload::Unavailable { .. })
        && !matches!(uncertainty, UncertaintyV1::Unknown)
    {
        return Err(ValidationError::UnavailableUncertaintyMismatch);
    }
    Ok(())
}

fn validate_evidence(evidence: &EvidenceIdentity) -> Result<(), ValidationError> {
    let valid_hex = evidence
        .digest_hex
        .bytes()
        .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));

    if evidence.algorithm != DigestAlgorithm::Sha256
        || evidence.digest_hex.len() != 64
        || !valid_hex
    {
        return Err(ValidationError::InvalidDigest);
    }

    Ok(())
}
