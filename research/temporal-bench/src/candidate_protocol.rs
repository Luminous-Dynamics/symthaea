//! Bounded binary messages for a separate temporal-benchmark candidate process.
//!
//! This protocol deliberately transports only public observations and policy
//! actions. It does not provide process isolation, authentication or attestation.
//! A confirmatory runner must attach these codecs to a separately sandboxed
//! candidate process and keep evaluator-only material outside its namespace.

use std::error::Error;
use std::fmt;
use std::io::{Read, Write};

use super::active_sensing_simulator::{
    ActiveSensingSensor, PolicyAction, PublicObservation, PublicSensorMeasurement,
};
use super::digest::CanonicalHasher;

const FRAME_MAGIC: [u8; 8] = *b"SYMTEDGE";
const FRAME_VERSION: u16 = 1;
const FRAME_HEADER_BYTES: usize = 56;
const MAX_FRAME_PAYLOAD_BYTES: usize = 128;
const MAX_WAKE_INTERVAL_STEPS: u32 = 1_024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CandidateProtocolError {
    Io(String),
    InvalidMagic,
    UnsupportedVersion(u16),
    UnknownMessageType(u16),
    UnexpectedMessageType { expected: u16, received: u16 },
    PayloadTooLarge { declared: usize, maximum: usize },
    SequenceMismatch { expected: u64, received: u64 },
    PayloadDigestMismatch,
    InvalidPayload(&'static str),
}

impl fmt::Display for CandidateProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "candidate protocol I/O: {message}"),
            Self::InvalidMagic => write!(f, "candidate protocol frame magic is invalid"),
            Self::UnsupportedVersion(version) => {
                write!(f, "candidate protocol version {version} is unsupported")
            }
            Self::UnknownMessageType(kind) => {
                write!(f, "candidate protocol message type {kind} is unknown")
            }
            Self::UnexpectedMessageType { expected, received } => write!(
                f,
                "expected candidate message type {expected}, received {received}"
            ),
            Self::PayloadTooLarge { declared, maximum } => write!(
                f,
                "candidate payload length {declared} exceeds maximum {maximum}"
            ),
            Self::SequenceMismatch { expected, received } => write!(
                f,
                "candidate frame sequence mismatch: expected {expected}, received {received}"
            ),
            Self::PayloadDigestMismatch => {
                write!(f, "candidate protocol payload digest mismatch")
            }
            Self::InvalidPayload(message) => write!(f, "invalid candidate payload: {message}"),
        }
    }
}
impl Error for CandidateProtocolError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
enum MessageType {
    Observation = 1,
    Action = 2,
}

impl MessageType {
    fn from_u16(value: u16) -> Result<Self, CandidateProtocolError> {
        match value {
            1 => Ok(Self::Observation),
            2 => Ok(Self::Action),
            other => Err(CandidateProtocolError::UnknownMessageType(other)),
        }
    }
}

struct Frame {
    payload: Vec<u8>,
}

fn frame_payload_digest(message_type: MessageType, sequence: u64, payload: &[u8]) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.candidate-frame.v1");
    hash.u16(message_type as u16);
    hash.u64(sequence);
    hash.bytes(payload);
    hash.finish()
}

fn write_frame<W: Write>(
    writer: &mut W,
    message_type: MessageType,
    sequence: u64,
    payload: &[u8],
) -> Result<(), CandidateProtocolError> {
    if payload.len() > MAX_FRAME_PAYLOAD_BYTES {
        return Err(CandidateProtocolError::PayloadTooLarge {
            declared: payload.len(),
            maximum: MAX_FRAME_PAYLOAD_BYTES,
        });
    }
    let digest = frame_payload_digest(message_type, sequence, payload);
    let mut header = [0_u8; FRAME_HEADER_BYTES];
    header[0..8].copy_from_slice(&FRAME_MAGIC);
    header[8..10].copy_from_slice(&FRAME_VERSION.to_le_bytes());
    header[10..12].copy_from_slice(&(message_type as u16).to_le_bytes());
    header[12..20].copy_from_slice(&sequence.to_le_bytes());
    header[20..24].copy_from_slice(
        &u32::try_from(payload.len())
            .map_err(|_| CandidateProtocolError::PayloadTooLarge {
                declared: payload.len(),
                maximum: MAX_FRAME_PAYLOAD_BYTES,
            })?
            .to_le_bytes(),
    );
    header[24..56].copy_from_slice(&digest);
    writer
        .write_all(&header)
        .map_err(|error| CandidateProtocolError::Io(error.to_string()))?;
    writer
        .write_all(payload)
        .map_err(|error| CandidateProtocolError::Io(error.to_string()))?;
    writer
        .flush()
        .map_err(|error| CandidateProtocolError::Io(error.to_string()))?;
    Ok(())
}

fn read_frame<R: Read>(
    reader: &mut R,
    expected_type: MessageType,
    expected_sequence: u64,
) -> Result<Frame, CandidateProtocolError> {
    let mut header = [0_u8; FRAME_HEADER_BYTES];
    reader
        .read_exact(&mut header)
        .map_err(|error| CandidateProtocolError::Io(error.to_string()))?;
    if header[0..8] != FRAME_MAGIC[..] {
        return Err(CandidateProtocolError::InvalidMagic);
    }
    let version = u16::from_le_bytes([header[8], header[9]]);
    if version != FRAME_VERSION {
        return Err(CandidateProtocolError::UnsupportedVersion(version));
    }
    let raw_type = u16::from_le_bytes([header[10], header[11]]);
    let message_type = MessageType::from_u16(raw_type)?;
    if message_type != expected_type {
        return Err(CandidateProtocolError::UnexpectedMessageType {
            expected: expected_type as u16,
            received: raw_type,
        });
    }
    let sequence = u64::from_le_bytes([
        header[12], header[13], header[14], header[15],
        header[16], header[17], header[18], header[19],
    ]);
    if sequence != expected_sequence {
        return Err(CandidateProtocolError::SequenceMismatch {
            expected: expected_sequence,
            received: sequence,
        });
    }
    let declared_len_u32 = u32::from_le_bytes([
        header[20], header[21], header[22], header[23],
    ]);
    if declared_len_u32 > MAX_FRAME_PAYLOAD_BYTES as u32 {
        return Err(CandidateProtocolError::PayloadTooLarge {
            declared: usize::try_from(declared_len_u32).unwrap_or(usize::MAX),
            maximum: MAX_FRAME_PAYLOAD_BYTES,
        });
    }
    let declared_len = usize::try_from(declared_len_u32).map_err(|_| {
        CandidateProtocolError::PayloadTooLarge {
            declared: usize::MAX,
            maximum: MAX_FRAME_PAYLOAD_BYTES,
        }
    })?;
    let mut expected_digest = [0_u8; 32];
    expected_digest.copy_from_slice(&header[24..56]);
    let mut payload = vec![0_u8; declared_len];
    reader
        .read_exact(&mut payload)
        .map_err(|error| CandidateProtocolError::Io(error.to_string()))?;
    let actual_digest = frame_payload_digest(message_type, sequence, &payload);
    if actual_digest != expected_digest {
        return Err(CandidateProtocolError::PayloadDigestMismatch);
    }
    Ok(Frame { payload })
}

fn push_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn take_u64(payload: &[u8], offset: &mut usize) -> Result<u64, CandidateProtocolError> {
    let end = (*offset).checked_add(8)
        .ok_or(CandidateProtocolError::InvalidPayload("u64 offset overflow"))?;
    let bytes = payload
        .get(*offset..end)
        .ok_or(CandidateProtocolError::InvalidPayload("truncated u64"))?;
    *offset = end;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

fn take_u32(payload: &[u8], offset: &mut usize) -> Result<u32, CandidateProtocolError> {
    let end = (*offset).checked_add(4)
        .ok_or(CandidateProtocolError::InvalidPayload("u32 offset overflow"))?;
    let bytes = payload
        .get(*offset..end)
        .ok_or(CandidateProtocolError::InvalidPayload("truncated u32"))?;
    *offset = end;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn take_u8(payload: &[u8], offset: &mut usize) -> Result<u8, CandidateProtocolError> {
    let value = *payload
        .get(*offset)
        .ok_or(CandidateProtocolError::InvalidPayload("truncated u8"))?;
    *offset = (*offset).checked_add(1)
        .ok_or(CandidateProtocolError::InvalidPayload("u8 offset overflow"))?;
    Ok(value)
}

fn decode_sensor(value: u8) -> Result<ActiveSensingSensor, CandidateProtocolError> {
    match value {
        1 => Ok(ActiveSensingSensor::LowPower),
        2 => Ok(ActiveSensingSensor::HighFidelity),
        _ => Err(CandidateProtocolError::InvalidPayload("unknown sensor tag")),
    }
}

fn encode_observation(observation: &PublicObservation) -> Result<Vec<u8>, CandidateProtocolError> {
    if observation.low_power_samples.checked_add(observation.high_fidelity_samples)
        != Some(observation.samples_acquired)
    {
        return Err(CandidateProtocolError::InvalidPayload(
            "sample counters do not sum to total samples",
        ));
    }
    if let Some(measurement) = observation.last_measurement {
        let modality_count = match measurement.sensor {
            ActiveSensingSensor::LowPower => observation.low_power_samples,
            ActiveSensingSensor::HighFidelity => observation.high_fidelity_samples,
        };
        if modality_count == 0 {
            return Err(CandidateProtocolError::InvalidPayload(
                "latest measurement modality has no corresponding sample count",
            ));
        }
    }
    let mut payload = Vec::with_capacity(40);
    push_u64(&mut payload, observation.step);
    match observation.last_measurement {
        None => {
            if observation.last_measurement_age_steps.is_some() {
                return Err(CandidateProtocolError::InvalidPayload(
                    "measurement age exists without a measurement",
                ));
            }
            payload.push(0);
        }
        Some(measurement) => {
            if measurement.sample_step > observation.step
                || observation.last_measurement_age_steps
                    != Some(observation.step - measurement.sample_step)
            {
                return Err(CandidateProtocolError::InvalidPayload(
                    "measurement timestamp and age are inconsistent",
                ));
            }
            payload.push(1);
            push_u64(&mut payload, measurement.sample_step);
            payload.push(match measurement.sensor {
                ActiveSensingSensor::LowPower => 1,
                ActiveSensingSensor::HighFidelity => 2,
            });
            payload.push(u8::from(measurement.positive));
        }
    }
    match observation.last_measurement_age_steps {
        None => payload.push(0),
        Some(age) => {
            payload.push(1);
            push_u64(&mut payload, age);
        }
    }
    push_u64(&mut payload, observation.samples_acquired);
    push_u64(&mut payload, observation.low_power_samples);
    push_u64(&mut payload, observation.high_fidelity_samples);
    Ok(payload)
}

fn decode_observation(payload: &[u8]) -> Result<PublicObservation, CandidateProtocolError> {
    let mut offset = 0;
    let step = take_u64(payload, &mut offset)?;
    let last_measurement = match take_u8(payload, &mut offset)? {
        0 => None,
        1 => {
            let sample_step = take_u64(payload, &mut offset)?;
            let sensor = decode_sensor(take_u8(payload, &mut offset)?)?;
            let positive = match take_u8(payload, &mut offset)? {
                0 => false,
                1 => true,
                _ => return Err(CandidateProtocolError::InvalidPayload("invalid boolean tag")),
            };
            if sample_step > step {
                return Err(CandidateProtocolError::InvalidPayload(
                    "measurement sample time is in the future",
                ));
            }
            Some(PublicSensorMeasurement { sample_step, sensor, positive })
        }
        _ => return Err(CandidateProtocolError::InvalidPayload("invalid measurement option tag")),
    };
    let last_measurement_age_steps = match take_u8(payload, &mut offset)? {
        0 => None,
        1 => Some(take_u64(payload, &mut offset)?),
        _ => return Err(CandidateProtocolError::InvalidPayload("invalid age option tag")),
    };
    let samples_acquired = take_u64(payload, &mut offset)?;
    let low_power_samples = take_u64(payload, &mut offset)?;
    let high_fidelity_samples = take_u64(payload, &mut offset)?;
    if offset != payload.len() {
        return Err(CandidateProtocolError::InvalidPayload("trailing observation bytes"));
    }
    if low_power_samples.checked_add(high_fidelity_samples) != Some(samples_acquired) {
        return Err(CandidateProtocolError::InvalidPayload(
            "sample counters do not sum to total samples",
        ));
    }
    if let Some(measurement) = last_measurement {
        let modality_count = match measurement.sensor {
            ActiveSensingSensor::LowPower => low_power_samples,
            ActiveSensingSensor::HighFidelity => high_fidelity_samples,
        };
        if modality_count == 0 {
            return Err(CandidateProtocolError::InvalidPayload(
                "latest measurement modality has no corresponding sample count",
            ));
        }
    }
    match (last_measurement, last_measurement_age_steps) {
        (None, None) => {}
        (Some(measurement), Some(age))
            if step.saturating_sub(measurement.sample_step) == age => {}
        _ => {
            return Err(CandidateProtocolError::InvalidPayload(
                "measurement timestamp and age are inconsistent",
            ));
        }
    }
    Ok(PublicObservation {
        step,
        last_measurement,
        last_measurement_age_steps,
        samples_acquired,
        low_power_samples,
        high_fidelity_samples,
    })
}

fn encode_action(action: &PolicyAction) -> Result<Vec<u8>, CandidateProtocolError> {
    if action.next_wake_after_steps == 0
        || action.next_wake_after_steps > MAX_WAKE_INTERVAL_STEPS
    {
        return Err(CandidateProtocolError::InvalidPayload(
            "wake interval must be in 1..=1024 steps",
        ));
    }
    let mut payload = Vec::with_capacity(30);
    payload.push(u8::from(action.alarm));
    payload.push(match action.sensor {
        None => 0,
        Some(ActiveSensingSensor::LowPower) => 1,
        Some(ActiveSensingSensor::HighFidelity) => 2,
    });
    push_u32(&mut payload, action.next_wake_after_steps);
    push_u64(&mut payload, action.compute_cycles);
    push_u64(&mut payload, action.transmit_bytes);
    push_u64(&mut payload, action.memory_write_bytes);
    Ok(payload)
}

fn decode_action(payload: &[u8]) -> Result<PolicyAction, CandidateProtocolError> {
    let mut offset = 0;
    let alarm = match take_u8(payload, &mut offset)? {
        0 => false,
        1 => true,
        _ => return Err(CandidateProtocolError::InvalidPayload("invalid alarm boolean tag")),
    };
    let sensor = match take_u8(payload, &mut offset)? {
        0 => None,
        1 => Some(ActiveSensingSensor::LowPower),
        2 => Some(ActiveSensingSensor::HighFidelity),
        _ => return Err(CandidateProtocolError::InvalidPayload("unknown sensor tag")),
    };
    let next_wake_after_steps = take_u32(payload, &mut offset)?;
    if next_wake_after_steps == 0 || next_wake_after_steps > MAX_WAKE_INTERVAL_STEPS {
        return Err(CandidateProtocolError::InvalidPayload(
            "wake interval must be in 1..=1024 steps",
        ));
    }
    let compute_cycles = take_u64(payload, &mut offset)?;
    let transmit_bytes = take_u64(payload, &mut offset)?;
    let memory_write_bytes = take_u64(payload, &mut offset)?;
    if offset != payload.len() {
        return Err(CandidateProtocolError::InvalidPayload("trailing action bytes"));
    }
    Ok(PolicyAction {
        alarm,
        sensor,
        next_wake_after_steps,
        compute_cycles,
        transmit_bytes,
        memory_write_bytes,
    })
}

/// Evaluator-side: send one public observation to a separately launched candidate.
pub fn write_observation<W: Write>(
    writer: &mut W,
    sequence: u64,
    observation: &PublicObservation,
) -> Result<(), CandidateProtocolError> {
    let payload = encode_observation(observation)?;
    write_frame(writer, MessageType::Observation, sequence, &payload)
}

/// Candidate-side: receive one public observation. The caller pins expected sequence.
pub fn read_observation<R: Read>(
    reader: &mut R,
    expected_sequence: u64,
) -> Result<PublicObservation, CandidateProtocolError> {
    let frame = read_frame(reader, MessageType::Observation, expected_sequence)?;
    decode_observation(&frame.payload)
}

/// Candidate-side: return one bounded policy action.
pub fn write_action<W: Write>(
    writer: &mut W,
    sequence: u64,
    action: &PolicyAction,
) -> Result<(), CandidateProtocolError> {
    let payload = encode_action(action)?;
    write_frame(writer, MessageType::Action, sequence, &payload)
}

/// Evaluator-side: receive exactly the response for the current observation.
pub fn read_action<R: Read>(
    reader: &mut R,
    expected_sequence: u64,
) -> Result<PolicyAction, CandidateProtocolError> {
    let frame = read_frame(reader, MessageType::Action, expected_sequence)?;
    decode_action(&frame.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn observation() -> PublicObservation {
        PublicObservation {
            step: 21,
            last_measurement: Some(PublicSensorMeasurement {
                sample_step: 20,
                sensor: ActiveSensingSensor::HighFidelity,
                positive: true,
            }),
            last_measurement_age_steps: Some(1),
            samples_acquired: 3,
            low_power_samples: 2,
            high_fidelity_samples: 1,
        }
    }

    fn action() -> PolicyAction {
        PolicyAction {
            alarm: true,
            sensor: Some(ActiveSensingSensor::LowPower),
            next_wake_after_steps: 4,
            compute_cycles: 123,
            transmit_bytes: 64,
            memory_write_bytes: 8,
        }
    }

    fn frame_bytes(kind: MessageType, sequence: u64, payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        write_frame(&mut bytes, kind, sequence, payload).unwrap();
        bytes
    }

    #[test]
    fn observation_round_trip_transports_only_public_fields() {
        let mut bytes = Vec::new();
        write_observation(&mut bytes, 7, &observation()).unwrap();
        let decoded = read_observation(&mut Cursor::new(bytes), 7).unwrap();
        assert_eq!(decoded, observation());
    }

    #[test]
    fn action_round_trip_preserves_bounded_policy_decision() {
        let mut bytes = Vec::new();
        write_action(&mut bytes, 11, &action()).unwrap();
        let decoded = read_action(&mut Cursor::new(bytes), 11).unwrap();
        assert_eq!(decoded, action());
    }

    #[test]
    fn sequence_replay_or_reordering_is_rejected() {
        let bytes = frame_bytes(MessageType::Action, 4, &encode_action(&action()).unwrap());
        let error = read_action(&mut Cursor::new(bytes), 5).unwrap_err();
        assert_eq!(
            error,
            CandidateProtocolError::SequenceMismatch { expected: 5, received: 4 }
        );
    }

    #[test]
    fn payload_tampering_is_detected_before_decode() {
        let mut bytes = frame_bytes(MessageType::Action, 1, &encode_action(&action()).unwrap());
        let last = bytes.len() - 1;
        bytes[last] ^= 0x80;
        assert_eq!(
            read_action(&mut Cursor::new(bytes), 1).unwrap_err(),
            CandidateProtocolError::PayloadDigestMismatch
        );
    }

    #[test]
    fn oversized_declared_payload_fails_before_allocation() {
        let mut header = [0_u8; FRAME_HEADER_BYTES];
        header[0..8].copy_from_slice(&FRAME_MAGIC);
        header[8..10].copy_from_slice(&FRAME_VERSION.to_le_bytes());
        header[10..12].copy_from_slice(&(MessageType::Action as u16).to_le_bytes());
        header[20..24].copy_from_slice(&((MAX_FRAME_PAYLOAD_BYTES + 1) as u32).to_le_bytes());
        let error = read_action(&mut Cursor::new(header), 0).unwrap_err();
        assert_eq!(
            error,
            CandidateProtocolError::PayloadTooLarge {
                declared: MAX_FRAME_PAYLOAD_BYTES + 1,
                maximum: MAX_FRAME_PAYLOAD_BYTES
            }
        );
    }

    #[test]
    fn truncated_header_and_payload_fail_closed() {
        let header_error = read_action(&mut Cursor::new(b"short"), 0).unwrap_err();
        assert!(matches!(header_error, CandidateProtocolError::Io(_)));

        let mut bytes = frame_bytes(MessageType::Action, 1, &encode_action(&action()).unwrap());
        bytes.pop();
        let payload_error = read_action(&mut Cursor::new(bytes), 1).unwrap_err();
        assert!(matches!(payload_error, CandidateProtocolError::Io(_)));
    }

    #[test]
    fn malformed_action_tags_and_wake_interval_are_rejected() {
        let invalid_alarm = [2_u8, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let bytes = frame_bytes(MessageType::Action, 1, &invalid_alarm);
        assert!(matches!(
            read_action(&mut Cursor::new(bytes), 1),
            Err(CandidateProtocolError::InvalidPayload(_))
        ));

        let mut invalid_wake = encode_action(&action()).unwrap();
        invalid_wake[2..6].copy_from_slice(&0_u32.to_le_bytes());
        let bytes = frame_bytes(MessageType::Action, 2, &invalid_wake);
        assert!(matches!(
            read_action(&mut Cursor::new(bytes), 2),
            Err(CandidateProtocolError::InvalidPayload(_))
        ));
    }

    #[test]
    fn inconsistent_observation_age_and_counts_are_rejected() {
        let mut bad_age = observation();
        bad_age.last_measurement_age_steps = Some(2);
        let error = encode_observation(&bad_age).unwrap_err();
        assert!(matches!(error, CandidateProtocolError::InvalidPayload(_)));

        let mut bad_counts = observation();
        bad_counts.low_power_samples = 99;
        let error = encode_observation(&bad_counts).unwrap_err();
        assert!(matches!(error, CandidateProtocolError::InvalidPayload(_)));
    }

    #[test]
    fn wrong_frame_type_and_unknown_sensor_are_rejected() {
        let observation_bytes = frame_bytes(
            MessageType::Observation,
            1,
            &encode_observation(&observation()).unwrap(),
        );
        assert!(matches!(
            read_action(&mut Cursor::new(observation_bytes), 1),
            Err(CandidateProtocolError::UnexpectedMessageType { .. })
        ));

        let mut invalid_sensor = encode_action(&action()).unwrap();
        invalid_sensor[1] = 9;
        let action_bytes = frame_bytes(MessageType::Action, 2, &invalid_sensor);
        assert!(matches!(
            read_action(&mut Cursor::new(action_bytes), 2),
            Err(CandidateProtocolError::InvalidPayload(_))
        ));
    }
}
