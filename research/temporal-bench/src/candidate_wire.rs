//! Versioned bounded binary protocol for isolated temporal-benchmark candidates.
//!
//! The protocol transports only public observations to a candidate and one
//! bounded action back. It does not transport scenario seeds, hidden labels,
//! evaluator state, scoring, or receipts. The caller must still enforce
//! process/OS isolation and deadlines.

use std::error::Error;
use std::fmt;

use super::active_sensing_simulator::{
    ActiveSensingSensor, PolicyAction, PublicObservation, PublicSensorMeasurement,
};
use super::digest::CanonicalHasher;

const MAGIC: &[u8; 8] = b"SYMEDGE1";
const VERSION: u16 = 1;
const HEADER_LEN: usize = 48;
const OBSERVATION_PAYLOAD_LEN: usize = 51;
const ACTION_PAYLOAD_LEN: usize = 30;
const MAX_FRAME_LEN: usize = HEADER_LEN + OBSERVATION_PAYLOAD_LEN;
const MAX_WAKE_INTERVAL_STEPS: u32 = 1_024;

const KIND_OBSERVATION: u8 = 1;
const KIND_ACTION: u8 = 2;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WireProtocolError {
    pub field: String,
    pub message: String,
}

impl WireProtocolError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for WireProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl Error for WireProtocolError {}

fn digest(kind: u8, payload: &[u8]) -> [u8; 32] {
    let domain: &'static [u8] = match kind {
        KIND_OBSERVATION => b"symthaea.temporal.candidate-wire.observation.v1",
        KIND_ACTION => b"symthaea.temporal.candidate-wire.action.v1",
        _ => b"symthaea.temporal.candidate-wire.invalid.v1",
    };
    let mut hash = CanonicalHasher::new(domain);
    hash.bytes(payload);
    hash.finish()
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

fn append_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn append_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn append_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn wrap_frame(kind: u8, payload: &[u8]) -> Result<Vec<u8>, WireProtocolError> {
    let payload_len = u32::try_from(payload.len()).map_err(|_| {
        WireProtocolError::new("payload_len", "payload length does not fit u32")
    })?;
    let total_len = HEADER_LEN.checked_add(payload.len()).ok_or_else(|| {
        WireProtocolError::new("frame_len", "frame length overflow")
    })?;
    if total_len > MAX_FRAME_LEN {
        return Err(WireProtocolError::new(
            "frame_len",
            format!("frame exceeds maximum length {MAX_FRAME_LEN}"),
        ));
    }

    let mut frame = Vec::with_capacity(total_len);
    frame.extend_from_slice(MAGIC);
    append_u16(&mut frame, VERSION);
    frame.push(kind);
    frame.push(0); // reserved: must remain zero in v1
    append_u32(&mut frame, payload_len);
    frame.extend_from_slice(&digest(kind, payload));
    frame.extend_from_slice(payload);
    Ok(frame)
}

fn unwrap_frame<'a>(
    frame: &'a [u8],
    expected_kind: u8,
    expected_payload_len: usize,
) -> Result<&'a [u8], WireProtocolError> {
    if frame.len() < HEADER_LEN {
        return Err(WireProtocolError::new("frame", "truncated frame header"));
    }
    if frame.len() > MAX_FRAME_LEN {
        return Err(WireProtocolError::new(
            "frame",
            format!("frame exceeds maximum length {MAX_FRAME_LEN}"),
        ));
    }
    if &frame[..8] != MAGIC {
        return Err(WireProtocolError::new("magic", "invalid protocol magic"));
    }
    if read_u16(frame, 8) != VERSION {
        return Err(WireProtocolError::new("version", "unsupported protocol version"));
    }
    if frame[10] != expected_kind {
        return Err(WireProtocolError::new("kind", "unexpected frame kind"));
    }
    if frame[11] != 0 {
        return Err(WireProtocolError::new("reserved", "reserved header byte must be zero"));
    }

    let payload_len = usize::try_from(read_u32(frame, 12)).map_err(|_| {
        WireProtocolError::new("payload_len", "payload length conversion failed")
    })?;
    if payload_len != expected_payload_len {
        return Err(WireProtocolError::new(
            "payload_len",
            format!("expected {expected_payload_len} bytes, got {payload_len}"),
        ));
    }
    let expected_total = HEADER_LEN.checked_add(payload_len).ok_or_else(|| {
        WireProtocolError::new("frame_len", "frame length overflow")
    })?;
    if frame.len() != expected_total {
        return Err(WireProtocolError::new(
            "frame_len",
            format!("expected exactly {expected_total} bytes, got {}", frame.len()),
        ));
    }

    let payload = &frame[HEADER_LEN..];
    let expected_digest = digest(expected_kind, payload);
    if frame[16..48] != expected_digest {
        return Err(WireProtocolError::new("payload_digest", "payload digest mismatch"));
    }
    Ok(payload)
}

fn sensor_to_wire(sensor: ActiveSensingSensor) -> u8 {
    match sensor {
        ActiveSensingSensor::LowPower => 1,
        ActiveSensingSensor::HighFidelity => 2,
    }
}

fn sensor_from_wire(value: u8) -> Result<Option<ActiveSensingSensor>, WireProtocolError> {
    match value {
        0 => Ok(None),
        1 => Ok(Some(ActiveSensingSensor::LowPower)),
        2 => Ok(Some(ActiveSensingSensor::HighFidelity)),
        _ => Err(WireProtocolError::new("sensor", "sensor code must be 0, 1 or 2")),
    }
}

fn validate_observation(observation: &PublicObservation) -> Result<(), WireProtocolError> {
    let summed_samples = observation.low_power_samples
        .checked_add(observation.high_fidelity_samples)
        .ok_or_else(|| WireProtocolError::new("samples_acquired", "sample counter overflow"))?;
    if summed_samples != observation.samples_acquired {
        return Err(WireProtocolError::new(
            "samples_acquired",
            "total sample count does not equal per-sensor counters",
        ));
    }
    match (
        observation.last_measurement,
        observation.last_measurement_age_steps,
    ) {
        (Some(measurement), Some(age)) => {
            if measurement.sample_step > observation.step
                || observation.step - measurement.sample_step != age
            {
                return Err(WireProtocolError::new(
                    "last_measurement_age_steps",
                    "measurement sample step and age are inconsistent",
                ));
            }
            let acquired = match measurement.sensor {
                ActiveSensingSensor::LowPower => observation.low_power_samples,
                ActiveSensingSensor::HighFidelity => observation.high_fidelity_samples,
            };
            if acquired == 0 {
                return Err(WireProtocolError::new(
                    "last_measurement",
                    "measurement modality has no corresponding acquired samples",
                ));
            }
        }
        (None, None) => {}
        _ => {
            return Err(WireProtocolError::new(
                "last_measurement",
                "measurement and age must either both be present or both be absent",
            ));
        }
    }
    Ok(())
}

/// Encode one public observation as a bounded v1 frame.
pub fn encode_observation(
    observation: &PublicObservation,
) -> Result<Vec<u8>, WireProtocolError> {
    validate_observation(observation)?;
    let mut payload = Vec::with_capacity(OBSERVATION_PAYLOAD_LEN);
    append_u64(&mut payload, observation.step);

    let mut flags = 0_u8;
    if observation.last_measurement.is_some() {
        flags |= 1;
    }
    if observation.last_measurement_age_steps.is_some() {
        flags |= 2;
    }
    payload.push(flags);

    if let Some(measurement) = observation.last_measurement {
        append_u64(&mut payload, measurement.sample_step);
        payload.push(sensor_to_wire(measurement.sensor));
        payload.push(u8::from(measurement.positive));
    } else {
        append_u64(&mut payload, 0);
        payload.push(0);
        payload.push(0);
    }
    append_u64(&mut payload, observation.last_measurement_age_steps.unwrap_or(0));
    append_u64(&mut payload, observation.samples_acquired);
    append_u64(&mut payload, observation.low_power_samples);
    append_u64(&mut payload, observation.high_fidelity_samples);
    debug_assert_eq!(payload.len(), OBSERVATION_PAYLOAD_LEN);
    wrap_frame(KIND_OBSERVATION, &payload)
}

/// Decode a public observation frame, rejecting noncanonical optional fields.
pub fn decode_observation(frame: &[u8]) -> Result<PublicObservation, WireProtocolError> {
    let payload = unwrap_frame(frame, KIND_OBSERVATION, OBSERVATION_PAYLOAD_LEN)?;
    let flags = payload[8];
    if flags & !0b11 != 0 {
        return Err(WireProtocolError::new("flags", "unknown observation flag bits"));
    }

    let measurement_present = flags & 1 != 0;
    let age_present = flags & 2 != 0;
    let sample_step = read_u64(payload, 9);
    let sensor_code = payload[17];
    let positive = payload[18];
    if positive > 1 {
        return Err(WireProtocolError::new("positive", "boolean field must be 0 or 1"));
    }
    let age = read_u64(payload, 19);

    let last_measurement = if measurement_present {
        if !age_present {
            return Err(WireProtocolError::new(
                "flags",
                "measurement requires a measurement-age field",
            ));
        }
        let sensor = sensor_from_wire(sensor_code)?
            .ok_or_else(|| WireProtocolError::new("sensor", "measurement sensor cannot be absent"))?;
        Some(PublicSensorMeasurement {
            sample_step,
            sensor,
            positive: positive == 1,
        })
    } else {
        if age_present || sample_step != 0 || sensor_code != 0 || positive != 0 || age != 0 {
            return Err(WireProtocolError::new(
                "optional_measurement",
                "absent measurement fields must use canonical zero encoding",
            ));
        }
        None
    };

    let last_measurement_age_steps = if age_present { Some(age) } else { None };
    let observation = PublicObservation {
        step: read_u64(payload, 0),
        last_measurement,
        last_measurement_age_steps,
        samples_acquired: read_u64(payload, 27),
        low_power_samples: read_u64(payload, 35),
        high_fidelity_samples: read_u64(payload, 43),
    };
    validate_observation(&observation)?;
    Ok(observation)
}

/// Encode one bounded candidate action as a v1 frame.
pub fn encode_action(action: &PolicyAction) -> Result<Vec<u8>, WireProtocolError> {
    if action.next_wake_after_steps == 0
        || action.next_wake_after_steps > MAX_WAKE_INTERVAL_STEPS
    {
        return Err(WireProtocolError::new(
            "next_wake_after_steps",
            format!("wake interval must be in 1..={MAX_WAKE_INTERVAL_STEPS}"),
        ));
    }
    let mut payload = Vec::with_capacity(ACTION_PAYLOAD_LEN);
    payload.push(u8::from(action.alarm));
    payload.push(action.sensor.map(sensor_to_wire).unwrap_or(0));
    append_u32(&mut payload, action.next_wake_after_steps);
    append_u64(&mut payload, action.compute_cycles);
    append_u64(&mut payload, action.transmit_bytes);
    append_u64(&mut payload, action.memory_write_bytes);
    debug_assert_eq!(payload.len(), ACTION_PAYLOAD_LEN);
    wrap_frame(KIND_ACTION, &payload)
}

/// Decode one candidate action frame with strict boolean, sensor and range checks.
pub fn decode_action(frame: &[u8]) -> Result<PolicyAction, WireProtocolError> {
    let payload = unwrap_frame(frame, KIND_ACTION, ACTION_PAYLOAD_LEN)?;
    if payload[0] > 1 {
        return Err(WireProtocolError::new("alarm", "boolean field must be 0 or 1"));
    }
    let sensor = sensor_from_wire(payload[1])?;
    let next_wake_after_steps = read_u32(payload, 2);
    if next_wake_after_steps == 0 || next_wake_after_steps > MAX_WAKE_INTERVAL_STEPS {
        return Err(WireProtocolError::new(
            "next_wake_after_steps",
            format!("wake interval must be in 1..={MAX_WAKE_INTERVAL_STEPS}"),
        ));
    }
    Ok(PolicyAction {
        alarm: payload[0] == 1,
        sensor,
        next_wake_after_steps,
        compute_cycles: read_u64(payload, 6),
        transmit_bytes: read_u64(payload, 14),
        memory_write_bytes: read_u64(payload, 22),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_without_measurement_round_trips() {
        let observation = PublicObservation {
            step: 100,
            last_measurement: None,
            last_measurement_age_steps: None,
            samples_acquired: 0,
            low_power_samples: 0,
            high_fidelity_samples: 0,
        };
        let frame = encode_observation(&observation).unwrap();
        assert_eq!(frame.len(), HEADER_LEN + OBSERVATION_PAYLOAD_LEN);
        assert_eq!(decode_observation(&frame).unwrap(), observation);
    }

    #[test]
    fn observation_with_measurement_round_trips() {
        let observation = PublicObservation {
            step: 105,
            last_measurement: Some(PublicSensorMeasurement {
                sample_step: 100,
                sensor: ActiveSensingSensor::HighFidelity,
                positive: true,
            }),
            last_measurement_age_steps: Some(5),
            samples_acquired: 9,
            low_power_samples: 4,
            high_fidelity_samples: 5,
        };
        let frame = encode_observation(&observation).unwrap();
        assert_eq!(decode_observation(&frame).unwrap(), observation);
    }

    #[test]
    fn action_with_sensor_and_cost_counters_round_trips() {
        let action = PolicyAction {
            alarm: true,
            sensor: Some(ActiveSensingSensor::LowPower),
            next_wake_after_steps: 12,
            compute_cycles: 44_000,
            transmit_bytes: 128,
            memory_write_bytes: 64,
        };
        let frame = encode_action(&action).unwrap();
        assert_eq!(frame.len(), HEADER_LEN + ACTION_PAYLOAD_LEN);
        assert_eq!(decode_action(&frame).unwrap(), action);
    }

    #[test]
    fn action_without_sensor_round_trips() {
        let action = PolicyAction::sleep(10);
        assert_eq!(decode_action(&encode_action(&action).unwrap()).unwrap(), action);
    }

    #[test]
    fn payload_tampering_is_detected() {
        let action = PolicyAction::sleep(10);
        let mut frame = encode_action(&action).unwrap();
        *frame.last_mut().unwrap() ^= 1;
        assert_eq!(decode_action(&frame).unwrap_err().field, "payload_digest");
    }

    #[test]
    fn wrong_magic_version_kind_and_truncation_are_rejected() {
        let frame = encode_action(&PolicyAction::sleep(2)).unwrap();

        let mut bad_magic = frame.clone();
        bad_magic[0] ^= 1;
        assert_eq!(decode_action(&bad_magic).unwrap_err().field, "magic");

        let mut bad_version = frame.clone();
        bad_version[8] = 2;
        assert_eq!(decode_action(&bad_version).unwrap_err().field, "version");

        let mut bad_kind = frame.clone();
        bad_kind[10] = KIND_OBSERVATION;
        assert_eq!(decode_action(&bad_kind).unwrap_err().field, "kind");

        assert_eq!(decode_action(&frame[..frame.len() - 1]).unwrap_err().field, "frame_len");
    }

    #[test]
    fn invalid_wake_interval_is_rejected_on_encode_and_decode() {
        let invalid = PolicyAction::sleep(0);
        assert_eq!(encode_action(&invalid).unwrap_err().field, "next_wake_after_steps");

        let mut frame = encode_action(&PolicyAction::sleep(1)).unwrap();
        frame[HEADER_LEN + 2..HEADER_LEN + 6].copy_from_slice(&0_u32.to_le_bytes());
        let payload = &frame[HEADER_LEN..];
        frame[16..48].copy_from_slice(&digest(KIND_ACTION, payload));
        assert_eq!(decode_action(&frame).unwrap_err().field, "next_wake_after_steps");
    }

    #[test]
    fn observation_inconsistent_age_and_sample_counts_are_rejected() {
        let invalid = PublicObservation {
            step: 10,
            last_measurement: Some(PublicSensorMeasurement {
                sample_step: 4,
                sensor: ActiveSensingSensor::LowPower,
                positive: false,
            }),
            last_measurement_age_steps: Some(5),
            samples_acquired: 2,
            low_power_samples: 1,
            high_fidelity_samples: 0,
        };
        assert_eq!(encode_observation(&invalid).unwrap_err().field, "last_measurement_age_steps");
    }

    #[test]
    fn observation_noncanonical_optional_fields_are_rejected() {
        let valid = PublicObservation {
            step: 10,
            last_measurement: None,
            last_measurement_age_steps: None,
            samples_acquired: 0,
            low_power_samples: 0,
            high_fidelity_samples: 0,
        };
        let mut frame = encode_observation(&valid).unwrap();
        frame[HEADER_LEN + 9] = 1; // sample_step without the present flag
        let payload = &frame[HEADER_LEN..];
        frame[16..48].copy_from_slice(&digest(KIND_OBSERVATION, payload));
        assert_eq!(
            decode_observation(&frame).unwrap_err().field,
            "optional_measurement",
        );
    }
}
