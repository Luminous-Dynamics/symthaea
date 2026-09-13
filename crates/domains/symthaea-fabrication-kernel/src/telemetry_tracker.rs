// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Persistent anti-replay tracking for verified telemetry frames.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::telemetry::VerifiedMachineTelemetry;
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;
use std::fmt;

pub const TELEMETRY_TRACKER_SCHEMA: &str = "symthaea.fabrication.telemetry-tracker.v1";
pub const MAX_TRACKED_TELEMETRY_STREAMS: usize = 4096;
const TELEMETRY_STREAM_KEY_PREFIX: &str = "symthaea.fabrication.telemetry-stream-id.v1:";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TelemetryStreamId {
    machine_id: String,
    session_digest: Sha256Digest,
}

impl TelemetryStreamId {
    fn canonical_key(&self) -> String {
        format!(
            "{TELEMETRY_STREAM_KEY_PREFIX}{}:{}:{}",
            self.machine_id.len(),
            self.machine_id,
            self.session_digest.to_hex()
        )
    }

    fn from_canonical_key(value: &str) -> Result<Self, String> {
        let rest = value
            .strip_prefix(TELEMETRY_STREAM_KEY_PREFIX)
            .ok_or_else(|| "unsupported telemetry stream key schema".to_string())?;
        let separator = rest.find(':').ok_or_else(|| {
            "telemetry stream key is missing machine length separator".to_string()
        })?;
        let length_text = &rest[..separator];
        if length_text.is_empty()
            || !length_text.bytes().all(|byte| byte.is_ascii_digit())
            || (length_text.len() > 1 && length_text.starts_with('0'))
        {
            return Err("telemetry stream machine length is not canonical decimal".into());
        }
        let machine_len = length_text
            .parse::<usize>()
            .map_err(|_| "telemetry stream machine length is out of range".to_string())?;
        if machine_len.to_string() != length_text {
            return Err("telemetry stream machine length is not canonical decimal".into());
        }

        let payload = &rest[separator + 1..];
        let machine_id = payload.get(..machine_len).ok_or_else(|| {
            "telemetry stream machine length does not end on a UTF-8 boundary".to_string()
        })?;
        let suffix = payload
            .get(machine_len..)
            .ok_or_else(|| "telemetry stream machine length exceeds key payload".to_string())?;
        let digest_text = suffix
            .strip_prefix(':')
            .ok_or_else(|| "telemetry stream key is missing digest separator".to_string())?;
        if digest_text.len() != 64 {
            return Err(
                "telemetry stream digest must contain exactly 64 lowercase hex characters".into(),
            );
        }
        let session_digest = Sha256Digest::from_hex(digest_text)
            .map_err(|error| format!("invalid telemetry stream digest: {error:?}"))?;
        if session_digest.to_hex() != digest_text {
            return Err("telemetry stream digest encoding is not canonical lowercase hex".into());
        }
        if !canonical(machine_id) {
            return Err("telemetry stream machine identity is not canonical".into());
        }

        Ok(Self {
            machine_id: machine_id.to_string(),
            session_digest,
        })
    }
}

impl Serialize for TelemetryStreamId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.canonical_key())
    }
}

impl<'de> Deserialize<'de> for TelemetryStreamId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::from_canonical_key(&value).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TelemetryStreamState {
    session_sequence: u64,
    printer_job_id: String,
    latest_frame_sequence: u64,
    latest_observed_at_unix_ms: u64,
    latest_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineTelemetryTracker {
    pub schema_version: String,
    #[serde(deserialize_with = "deserialize_telemetry_streams")]
    streams: BTreeMap<TelemetryStreamId, TelemetryStreamState>,
}

fn deserialize_telemetry_streams<'de, D>(
    deserializer: D,
) -> Result<BTreeMap<TelemetryStreamId, TelemetryStreamState>, D::Error>
where
    D: Deserializer<'de>,
{
    struct StreamsVisitor;

    impl<'de> Visitor<'de> for StreamsVisitor {
        type Value = BTreeMap<TelemetryStreamId, TelemetryStreamState>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a bounded map of unique canonical telemetry stream identities")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            if map
                .size_hint()
                .is_some_and(|size| size > MAX_TRACKED_TELEMETRY_STREAMS)
            {
                return Err(serde::de::Error::custom(
                    "telemetry stream map exceeds configured capacity",
                ));
            }

            let mut streams = BTreeMap::new();
            while let Some((stream, state)) = map.next_entry()? {
                if streams.contains_key(&stream) {
                    return Err(serde::de::Error::custom(
                        "duplicate telemetry stream identity",
                    ));
                }
                if streams.len() >= MAX_TRACKED_TELEMETRY_STREAMS {
                    return Err(serde::de::Error::custom(
                        "telemetry stream map exceeds configured capacity",
                    ));
                }
                streams.insert(stream, state);
            }
            Ok(streams)
        }
    }

    deserializer.deserialize_map(StreamsVisitor)
}

impl Default for MachineTelemetryTracker {
    fn default() -> Self {
        Self {
            schema_version: TELEMETRY_TRACKER_SCHEMA.into(),
            streams: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TelemetryTrackingError {
    UnsupportedSchema,
    CapacityExceeded,
    InvalidState(&'static str),
    SessionSequenceChanged { previous: u64, current: u64 },
    PrinterJobChanged,
    FrameRollback { latest: u64, proposed: u64 },
    FrameCollision { sequence: u64 },
    ObservationTimeRegressed { latest: u64, proposed: u64 },
    TrackerRollback(&'static str),
    Encoding(String),
}

impl MachineTelemetryTracker {
    /// Accept one verified frame. Replaying the exact latest frame is
    /// idempotent; any same-sequence substitution or rollback fails closed.
    pub fn accept(
        &mut self,
        telemetry: &VerifiedMachineTelemetry,
    ) -> Result<Sha256Digest, TelemetryTrackingError> {
        self.validate()?;
        let payload = telemetry.payload();
        let stream_id = TelemetryStreamId {
            machine_id: payload.machine_id.clone(),
            session_digest: payload.session_digest,
        };
        let digest = telemetry.telemetry_digest();

        if !self.streams.contains_key(&stream_id)
            && self.streams.len() >= MAX_TRACKED_TELEMETRY_STREAMS
        {
            return Err(TelemetryTrackingError::CapacityExceeded);
        }
        if let Some(state) = self.streams.get(&stream_id) {
            if state.session_sequence != payload.session_sequence {
                return Err(TelemetryTrackingError::SessionSequenceChanged {
                    previous: state.session_sequence,
                    current: payload.session_sequence,
                });
            }
            if state.printer_job_id != payload.printer_job_id {
                return Err(TelemetryTrackingError::PrinterJobChanged);
            }
            if payload.frame_sequence < state.latest_frame_sequence {
                return Err(TelemetryTrackingError::FrameRollback {
                    latest: state.latest_frame_sequence,
                    proposed: payload.frame_sequence,
                });
            }
            if payload.frame_sequence == state.latest_frame_sequence {
                if digest == state.latest_digest
                    && payload.observed_at_unix_ms == state.latest_observed_at_unix_ms
                {
                    return Ok(digest);
                }
                return Err(TelemetryTrackingError::FrameCollision {
                    sequence: payload.frame_sequence,
                });
            }
            if payload.observed_at_unix_ms < state.latest_observed_at_unix_ms {
                return Err(TelemetryTrackingError::ObservationTimeRegressed {
                    latest: state.latest_observed_at_unix_ms,
                    proposed: payload.observed_at_unix_ms,
                });
            }
        }

        self.streams.insert(
            stream_id,
            TelemetryStreamState {
                session_sequence: payload.session_sequence,
                printer_job_id: payload.printer_job_id.clone(),
                latest_frame_sequence: payload.frame_sequence,
                latest_observed_at_unix_ms: payload.observed_at_unix_ms,
                latest_digest: digest,
            },
        );
        Ok(digest)
    }

    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    pub fn latest_frame_sequence(
        &self,
        machine_id: &str,
        session_digest: Sha256Digest,
    ) -> Option<u64> {
        self.streams
            .get(&TelemetryStreamId {
                machine_id: machine_id.to_string(),
                session_digest,
            })
            .map(|state| state.latest_frame_sequence)
    }

    pub fn validate(&self) -> Result<(), TelemetryTrackingError> {
        if self.schema_version != TELEMETRY_TRACKER_SCHEMA {
            return Err(TelemetryTrackingError::UnsupportedSchema);
        }
        if self.streams.len() > MAX_TRACKED_TELEMETRY_STREAMS {
            return Err(TelemetryTrackingError::CapacityExceeded);
        }
        for (stream, state) in &self.streams {
            if !canonical(&stream.machine_id) || !canonical(&state.printer_job_id) {
                return Err(TelemetryTrackingError::InvalidState(
                    "non-canonical stream identity",
                ));
            }
            if state.session_sequence == 0 || state.latest_frame_sequence == 0 {
                return Err(TelemetryTrackingError::InvalidState("zero sequence"));
            }
        }
        Ok(())
    }

    pub fn verify_successor_of(&self, previous: &Self) -> Result<(), TelemetryTrackingError> {
        previous.validate()?;
        self.validate()?;
        for (stream, previous_state) in &previous.streams {
            let Some(current) = self.streams.get(stream) else {
                return Err(TelemetryTrackingError::TrackerRollback(
                    "telemetry stream was removed",
                ));
            };
            if current.session_sequence != previous_state.session_sequence
                || current.printer_job_id != previous_state.printer_job_id
            {
                return Err(TelemetryTrackingError::TrackerRollback(
                    "telemetry stream identity changed",
                ));
            }
            if current.latest_frame_sequence < previous_state.latest_frame_sequence
                || current.latest_observed_at_unix_ms < previous_state.latest_observed_at_unix_ms
            {
                return Err(TelemetryTrackingError::TrackerRollback(
                    "telemetry stream position regressed",
                ));
            }
            if current.latest_frame_sequence == previous_state.latest_frame_sequence
                && (current.latest_digest != previous_state.latest_digest
                    || current.latest_observed_at_unix_ms
                        != previous_state.latest_observed_at_unix_ms)
            {
                return Err(TelemetryTrackingError::TrackerRollback(
                    "same telemetry frame was substituted",
                ));
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Sha256Digest, TelemetryTrackingError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| TelemetryTrackingError::Encoding(error.to_string()))?;
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea.fabrication.telemetry-tracker-digest.v1\0");
        hasher.update(&bytes);
        Ok(hasher.finalize())
    }
}

fn canonical(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= 256
        && !value.chars().any(char::is_control)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::SignatureAlgorithm;
    use crate::crypto_digest::sha256;
    use crate::telemetry::{
        MACHINE_TELEMETRY_SCHEMA, MachineTelemetryPayload, MachineTelemetryPolicy,
        MachineTelemetrySigner, MachineTelemetryVerifier, TelemetryExpectation,
        sign_machine_telemetry, verify_machine_telemetry,
    };
    use crate::trust::{KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot};
    use std::collections::BTreeSet;

    struct Provider;

    impl MachineTelemetrySigner for Provider {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Other("test-telemetry".into())
        }
        fn key_id(&self) -> &str {
            "telemetry-key"
        }
        fn sign_telemetry(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(sha256(message).0.to_vec())
        }
    }

    impl MachineTelemetryVerifier for Provider {
        fn verify_telemetry(
            &self,
            algorithm: &SignatureAlgorithm,
            key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(
                algorithm == &SignatureAlgorithm::Other("test-telemetry".into())
                    && key_id == "telemetry-key"
                    && signature == sha256(message).0.as_slice(),
            )
        }
    }

    fn verified(frame_sequence: u64, observed_at_unix_ms: u64) -> VerifiedMachineTelemetry {
        let payload = MachineTelemetryPayload {
            schema_version: MACHINE_TELEMETRY_SCHEMA.into(),
            manifest_digest: sha256(b"manifest"),
            machine_id: "machine-1".into(),
            session_digest: sha256(b"session"),
            session_sequence: 2,
            printer_job_id: "job-1".into(),
            frame_sequence,
            observed_at_unix_ms,
            elapsed_ms: frame_sequence * 100,
            heartbeat_sequence: frame_sequence,
            progress_ppm: (frame_sequence as u32).min(10) * 10_000,
            nozzle_actual_milli_c: 200_000,
            nozzle_target_milli_c: 200_000,
            bed_actual_milli_c: 60_000,
            bed_target_milli_c: 60_000,
        };
        let provider = Provider;
        let signed = sign_machine_telemetry(payload, &provider).unwrap();
        let trust = TrustSnapshot::new(
            1,
            100,
            1_000,
            vec![KeyTrustRecord {
                algorithm: SignatureAlgorithm::Other("test-telemetry".into()),
                key_id: "telemetry-key".into(),
                not_before_unix_s: 100,
                not_after_unix_s: Some(900),
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::MachineTelemetry]),
            }],
        )
        .unwrap();
        verify_machine_telemetry(
            signed,
            &MachineTelemetryPolicy::default(),
            TelemetryExpectation {
                manifest_digest: sha256(b"manifest"),
                machine_id: "machine-1",
                session_digest: sha256(b"session"),
                session_sequence: 2,
                printer_job_id: "job-1",
            },
            &trust,
            observed_at_unix_ms + 1,
            &provider,
        )
        .unwrap()
    }

    #[test]
    fn exact_latest_replay_is_idempotent() {
        let frame = verified(1, 500_000);
        let mut tracker = MachineTelemetryTracker::default();
        let first = tracker.accept(&frame).unwrap();
        assert_eq!(tracker.accept(&frame).unwrap(), first);
        assert_eq!(tracker.stream_count(), 1);
    }

    #[test]
    fn rollback_and_same_sequence_substitution_fail() {
        let mut tracker = MachineTelemetryTracker::default();
        tracker.accept(&verified(2, 500_000)).unwrap();
        assert!(matches!(
            tracker.accept(&verified(1, 500_001)),
            Err(TelemetryTrackingError::FrameRollback { .. })
        ));
        assert!(matches!(
            tracker.accept(&verified(2, 500_002)),
            Err(TelemetryTrackingError::FrameCollision { sequence: 2 })
        ));
    }

    #[test]
    fn observation_time_cannot_regress() {
        let mut tracker = MachineTelemetryTracker::default();
        tracker.accept(&verified(1, 500_010)).unwrap();
        assert!(matches!(
            tracker.accept(&verified(2, 500_009)),
            Err(TelemetryTrackingError::ObservationTimeRegressed { .. })
        ));
    }

    fn verified_for(
        machine_id: &str,
        session_seed: &[u8],
        frame_sequence: u64,
        observed_at_unix_ms: u64,
    ) -> VerifiedMachineTelemetry {
        let session_digest = sha256(session_seed);
        let printer_job_id = format!("job-{machine_id}");
        let payload = MachineTelemetryPayload {
            schema_version: MACHINE_TELEMETRY_SCHEMA.into(),
            manifest_digest: sha256(b"manifest"),
            machine_id: machine_id.into(),
            session_digest,
            session_sequence: 2,
            printer_job_id: printer_job_id.clone(),
            frame_sequence,
            observed_at_unix_ms,
            elapsed_ms: frame_sequence * 100,
            heartbeat_sequence: frame_sequence,
            progress_ppm: (frame_sequence as u32).min(10) * 10_000,
            nozzle_actual_milli_c: 200_000,
            nozzle_target_milli_c: 200_000,
            bed_actual_milli_c: 60_000,
            bed_target_milli_c: 60_000,
        };
        let provider = Provider;
        let signed = sign_machine_telemetry(payload, &provider).unwrap();
        let trust = TrustSnapshot::new(
            1,
            100,
            1_000,
            vec![KeyTrustRecord {
                algorithm: SignatureAlgorithm::Other("test-telemetry".into()),
                key_id: "telemetry-key".into(),
                not_before_unix_s: 100,
                not_after_unix_s: Some(900),
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::MachineTelemetry]),
            }],
        )
        .unwrap();
        verify_machine_telemetry(
            signed,
            &MachineTelemetryPolicy::default(),
            TelemetryExpectation {
                manifest_digest: sha256(b"manifest"),
                machine_id,
                session_digest,
                session_sequence: 2,
                printer_job_id: &printer_job_id,
            },
            &trust,
            observed_at_unix_ms + 1,
            &provider,
        )
        .unwrap()
    }

    #[test]
    fn populated_tracker_json_round_trip_and_digest_are_stable() {
        let mut tracker = MachineTelemetryTracker::default();
        tracker.accept(&verified(1, 500_000)).unwrap();

        let encoded = serde_json::to_vec(&tracker).unwrap();
        let decoded: MachineTelemetryTracker = serde_json::from_slice(&encoded).unwrap();

        assert_eq!(decoded, tracker);
        assert_eq!(decoded.digest().unwrap(), tracker.digest().unwrap());
    }

    #[test]
    fn telemetry_stream_key_round_trips_colons_and_unicode() {
        let stream = TelemetryStreamId {
            machine_id: "machine:α".into(),
            session_digest: sha256(b"unicode-session"),
        };
        let encoded = stream.canonical_key();
        let decoded = TelemetryStreamId::from_canonical_key(&encoded).unwrap();
        assert_eq!(decoded, stream);
    }

    #[test]
    fn telemetry_stream_key_rejects_noncanonical_encodings() {
        let digest = sha256(b"session").to_hex();
        let uppercase = format!("{TELEMETRY_STREAM_KEY_PREFIX}1:a:{}", digest.to_uppercase());
        let leading_zero = format!("{TELEMETRY_STREAM_KEY_PREFIX}01:a:{digest}");
        let wrong_length = format!("{TELEMETRY_STREAM_KEY_PREFIX}2:a:{digest}");
        let empty_machine = format!("{TELEMETRY_STREAM_KEY_PREFIX}0::{digest}");

        for candidate in [uppercase, leading_zero, wrong_length, empty_machine] {
            assert!(TelemetryStreamId::from_canonical_key(&candidate).is_err());
        }
    }

    #[test]
    fn duplicate_persisted_stream_identity_fails_closed() {
        let mut tracker = MachineTelemetryTracker::default();
        tracker.accept(&verified(1, 500_000)).unwrap();
        let (stream, state) = tracker.streams.iter().next().unwrap();
        let key = serde_json::to_string(&stream.canonical_key()).unwrap();
        let state = serde_json::to_string(state).unwrap();
        let raw = format!(
            "{{\"schema_version\":\"{TELEMETRY_TRACKER_SCHEMA}\",\"streams\":{{{key}:{state},{key}:{state}}}}}"
        );

        let error = serde_json::from_str::<MachineTelemetryTracker>(&raw).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("duplicate telemetry stream identity")
        );
    }

    #[test]
    fn insertion_order_does_not_change_persisted_bytes_or_digest() {
        let a = verified_for("machine-a", b"session-a", 1, 500_000);
        let b = verified_for("machine-b", b"session-b", 1, 500_100);

        let mut first = MachineTelemetryTracker::default();
        first.accept(&a).unwrap();
        first.accept(&b).unwrap();

        let mut second = MachineTelemetryTracker::default();
        second.accept(&b).unwrap();
        second.accept(&a).unwrap();

        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        assert_eq!(first.digest().unwrap(), second.digest().unwrap());
    }
}
