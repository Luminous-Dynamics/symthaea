// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Versioned, self-verifying archives for recorded chemical acquisition traces.
//!
//! Archive identity is computed from semantic evidence fields rather than a
//! transport-specific serialization. JSON/CBOR/postcard formatting may therefore
//! evolve without changing the identity of the underlying acquisition evidence.
//!
//! [`ChemicalTraceArchive::verify`] establishes internal consistency, not author
//! authenticity. Call [`ChemicalTraceArchive::verify_pinned`] when the expected
//! digest was captured independently (for example in a signed receipt, evidence
//! ledger, release manifest, or lab record).

use crate::{
    ChemicalEvidenceLevel, ChemicalModality, ChemicalObservation, ChemicalTrace,
    ChemicalTraceError, MeasurementUnit, SamplingPhase,
};
use blake3::Hasher;
use serde::{Deserialize, Serialize};

pub const TRACE_ARCHIVE_SCHEMA_VERSION: u16 = 1;

/// Content digest of one canonical chemical trace archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TraceArchiveDigest(pub [u8; 32]);

/// Manifest fields that can be inspected without interpreting every observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceArchiveManifest {
    pub schema_version: u16,
    pub acquisition_evidence: ChemicalEvidenceLevel,
    pub protocol_id: String,
    pub run_id: String,
    pub modality: ChemicalModality,
    pub replicate: u32,
    /// Fixed-width count for transport stability across 32/64-bit platforms.
    pub observation_count: u64,
    pub first_timestamp_us: u64,
    pub last_timestamp_us: u64,
    pub digest: TraceArchiveDigest,
}

/// Transport-neutral archive envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChemicalTraceArchive {
    pub manifest: TraceArchiveManifest,
    pub observations: Vec<ChemicalObservation>,
}

/// Verified replay result.
///
/// The original acquisition evidence is retained for provenance, but the replay's
/// own evidence level is always [`ChemicalEvidenceLevel::RecordedReplay`].
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedChemicalReplay {
    trace: ChemicalTrace,
    original_acquisition_evidence: ChemicalEvidenceLevel,
    archive_digest: TraceArchiveDigest,
}

impl VerifiedChemicalReplay {
    pub fn trace(&self) -> &ChemicalTrace {
        &self.trace
    }

    pub fn into_trace(self) -> ChemicalTrace {
        self.trace
    }

    pub fn evidence_level(&self) -> ChemicalEvidenceLevel {
        ChemicalEvidenceLevel::RecordedReplay
    }

    pub fn original_acquisition_evidence(&self) -> ChemicalEvidenceLevel {
        self.original_acquisition_evidence
    }

    pub fn archive_digest(&self) -> TraceArchiveDigest {
        self.archive_digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceArchiveError {
    UnsupportedSchemaVersion { actual: u16 },
    InvalidAcquisitionEvidence(ChemicalEvidenceLevel),
    EmptyArchive,
    Trace(ChemicalTraceError),
    ManifestProtocolMismatch,
    ManifestRunMismatch,
    ManifestModalityMismatch,
    ManifestReplicateMismatch,
    ManifestCountMismatch,
    ManifestTimestampMismatch,
    DigestMismatch,
    PinnedDigestMismatch {
        expected: TraceArchiveDigest,
        actual: TraceArchiveDigest,
    },
}

impl From<ChemicalTraceError> for TraceArchiveError {
    fn from(value: ChemicalTraceError) -> Self {
        Self::Trace(value)
    }
}

impl ChemicalTraceArchive {
    /// Create a version-1 archive from an already validated acquisition trace.
    ///
    /// `RecordedReplay` is intentionally rejected as acquisition evidence. Replays
    /// may be re-archived by an outer provenance system, but this envelope is for
    /// recording the origin trace, not recursively upgrading replayed evidence.
    pub fn from_trace(
        trace: &ChemicalTrace,
        acquisition_evidence: ChemicalEvidenceLevel,
    ) -> Result<Self, TraceArchiveError> {
        validate_origin_evidence(acquisition_evidence)?;

        let observations = trace.observations().to_vec();
        let first_timestamp_us = observations
            .first()
            .map(|observation| observation.timestamp_us)
            .ok_or(TraceArchiveError::EmptyArchive)?;
        let last_timestamp_us = observations
            .last()
            .map(|observation| observation.timestamp_us)
            .ok_or(TraceArchiveError::EmptyArchive)?;
        let observation_count = len_u64(observations.len());

        let digest = digest_trace(
            TRACE_ARCHIVE_SCHEMA_VERSION,
            acquisition_evidence,
            trace.protocol_id(),
            trace.run_id(),
            trace.modality(),
            trace.replicate(),
            &observations,
        );

        Ok(Self {
            manifest: TraceArchiveManifest {
                schema_version: TRACE_ARCHIVE_SCHEMA_VERSION,
                acquisition_evidence,
                protocol_id: trace.protocol_id().to_string(),
                run_id: trace.run_id().to_string(),
                modality: trace.modality(),
                replicate: trace.replicate(),
                observation_count,
                first_timestamp_us,
                last_timestamp_us,
                digest,
            },
            observations,
        })
    }

    /// Digest that should be pinned independently when later authenticity or
    /// tamper-evidence across trust boundaries is required.
    pub fn digest(&self) -> TraceArchiveDigest {
        self.manifest.digest
    }

    /// Validate structural invariants, manifest consistency, and semantic digest.
    ///
    /// This proves only that the archive is internally self-consistent. A party
    /// able to modify both payload and manifest can recompute the digest. Use
    /// [`verify_pinned`](Self::verify_pinned) when an independently captured digest
    /// is available.
    pub fn verify(&self) -> Result<VerifiedChemicalReplay, TraceArchiveError> {
        if self.manifest.schema_version != TRACE_ARCHIVE_SCHEMA_VERSION {
            return Err(TraceArchiveError::UnsupportedSchemaVersion {
                actual: self.manifest.schema_version,
            });
        }
        validate_origin_evidence(self.manifest.acquisition_evidence)?;

        let first = self
            .observations
            .first()
            .cloned()
            .ok_or(TraceArchiveError::EmptyArchive)?;
        let mut trace = ChemicalTrace::new(first)?;
        for observation in self.observations.iter().skip(1).cloned() {
            trace.append(observation)?;
        }

        if trace.protocol_id() != self.manifest.protocol_id {
            return Err(TraceArchiveError::ManifestProtocolMismatch);
        }
        if trace.run_id() != self.manifest.run_id {
            return Err(TraceArchiveError::ManifestRunMismatch);
        }
        if trace.modality() != self.manifest.modality {
            return Err(TraceArchiveError::ManifestModalityMismatch);
        }
        if trace.replicate() != self.manifest.replicate {
            return Err(TraceArchiveError::ManifestReplicateMismatch);
        }
        if len_u64(trace.len()) != self.manifest.observation_count {
            return Err(TraceArchiveError::ManifestCountMismatch);
        }

        let first_timestamp_us = self.observations[0].timestamp_us;
        let last_timestamp_us = self.observations[self.observations.len() - 1].timestamp_us;
        if first_timestamp_us != self.manifest.first_timestamp_us
            || last_timestamp_us != self.manifest.last_timestamp_us
        {
            return Err(TraceArchiveError::ManifestTimestampMismatch);
        }

        let expected_digest = digest_trace(
            self.manifest.schema_version,
            self.manifest.acquisition_evidence,
            &self.manifest.protocol_id,
            &self.manifest.run_id,
            self.manifest.modality,
            self.manifest.replicate,
            &self.observations,
        );
        if expected_digest != self.manifest.digest {
            return Err(TraceArchiveError::DigestMismatch);
        }

        Ok(VerifiedChemicalReplay {
            trace,
            original_acquisition_evidence: self.manifest.acquisition_evidence,
            archive_digest: self.manifest.digest,
        })
    }

    /// Verify against a digest captured outside the archive itself.
    ///
    /// This is the appropriate entry point when an experiment receipt, signed
    /// manifest, content-addressed store, or evidence ledger pins the archive ID.
    pub fn verify_pinned(
        &self,
        expected_digest: TraceArchiveDigest,
    ) -> Result<VerifiedChemicalReplay, TraceArchiveError> {
        if self.manifest.digest != expected_digest {
            return Err(TraceArchiveError::PinnedDigestMismatch {
                expected: expected_digest,
                actual: self.manifest.digest,
            });
        }
        self.verify()
    }
}

fn validate_origin_evidence(evidence: ChemicalEvidenceLevel) -> Result<(), TraceArchiveError> {
    match evidence {
        ChemicalEvidenceLevel::SimulatedFixture
        | ChemicalEvidenceLevel::BenchPhysicalObservation
        | ChemicalEvidenceLevel::HeldOutPhysicalObservation => Ok(()),
        ChemicalEvidenceLevel::RecordedReplay => {
            Err(TraceArchiveError::InvalidAcquisitionEvidence(evidence))
        }
    }
}

fn digest_trace(
    schema_version: u16,
    acquisition_evidence: ChemicalEvidenceLevel,
    protocol_id: &str,
    run_id: &str,
    modality: ChemicalModality,
    replicate: u32,
    observations: &[ChemicalObservation],
) -> TraceArchiveDigest {
    let mut hasher = Hasher::new();
    put_bytes(&mut hasher, b"symthaea-chemosensation-trace-archive");
    put_u16(&mut hasher, schema_version);
    put_u8(&mut hasher, evidence_tag(acquisition_evidence));
    put_str(&mut hasher, protocol_id);
    put_str(&mut hasher, run_id);
    put_u8(&mut hasher, modality_tag(modality));
    put_u32(&mut hasher, replicate);
    put_u64(&mut hasher, len_u64(observations.len()));

    for observation in observations {
        hash_observation(&mut hasher, observation);
    }

    TraceArchiveDigest(*hasher.finalize().as_bytes())
}

fn hash_observation(hasher: &mut Hasher, observation: &ChemicalObservation) {
    put_u64(hasher, observation.timestamp_us);
    put_u8(hasher, modality_tag(observation.modality));
    put_str(hasher, &observation.source);

    put_option_f32(hasher, observation.environment.temperature_c);
    put_option_f32(hasher, observation.environment.humidity_rh);
    put_option_f32(hasher, observation.environment.pressure_pa);

    match &observation.sampling {
        Some(sampling) => {
            put_u8(hasher, 1);
            put_str(hasher, &sampling.protocol_id);
            put_str(hasher, &sampling.run_id);
            match &sampling.sample_id {
                Some(sample_id) => {
                    put_u8(hasher, 1);
                    put_str(hasher, sample_id);
                }
                None => put_u8(hasher, 0),
            }
            put_u8(hasher, phase_tag(sampling.phase));
            put_u32(hasher, sampling.step_index);
            put_u32(hasher, sampling.replicate);
        }
        None => put_u8(hasher, 0),
    }

    put_u64(hasher, len_u64(observation.channels.len()));
    for channel in &observation.channels {
        put_str(hasher, &channel.name);
        put_u8(hasher, unit_tag(channel.unit));
        put_f32(hasher, channel.raw_value);
        put_str(hasher, &channel.calibration.id.0);
        put_f32(hasher, channel.calibration.baseline);
        put_f32(hasher, channel.calibration.gain);
        put_f32(hasher, channel.calibration.drift);
        put_f32(hasher, channel.health.score);
        put_u8(hasher, u8::from(channel.health.saturated));
        put_u8(hasher, u8::from(channel.health.contaminated));
    }
}

fn evidence_tag(value: ChemicalEvidenceLevel) -> u8 {
    match value {
        ChemicalEvidenceLevel::SimulatedFixture => 1,
        ChemicalEvidenceLevel::RecordedReplay => 2,
        ChemicalEvidenceLevel::BenchPhysicalObservation => 3,
        ChemicalEvidenceLevel::HeldOutPhysicalObservation => 4,
    }
}

fn modality_tag(value: ChemicalModality) -> u8 {
    match value {
        ChemicalModality::Olfactory => 1,
        ChemicalModality::Gustatory => 2,
    }
}

fn phase_tag(value: SamplingPhase) -> u8 {
    match value {
        SamplingPhase::Calibration => 1,
        SamplingPhase::Baseline => 2,
        SamplingPhase::Exposure => 3,
        SamplingPhase::Purge => 4,
        SamplingPhase::Rinse => 5,
        SamplingPhase::Recovery => 6,
    }
}

fn unit_tag(value: MeasurementUnit) -> u8 {
    match value {
        MeasurementUnit::Arbitrary => 1,
        MeasurementUnit::PartsPerMillion => 2,
        MeasurementUnit::PartsPerBillion => 3,
        MeasurementUnit::Ohms => 4,
        MeasurementUnit::SiemensPerMeter => 5,
        MeasurementUnit::Millivolts => 6,
        MeasurementUnit::Ph => 7,
    }
}

fn put_option_f32(hasher: &mut Hasher, value: Option<f32>) {
    match value {
        Some(value) => {
            put_u8(hasher, 1);
            put_f32(hasher, value);
        }
        None => put_u8(hasher, 0),
    }
}

fn put_f32(hasher: &mut Hasher, value: f32) {
    put_u32(hasher, value.to_bits());
}

fn put_str(hasher: &mut Hasher, value: &str) {
    put_bytes(hasher, value.as_bytes());
}

fn put_bytes(hasher: &mut Hasher, value: &[u8]) {
    put_u64(hasher, len_u64(value.len()));
    hasher.update(value);
}

fn put_u8(hasher: &mut Hasher, value: u8) {
    hasher.update(&[value]);
}

fn put_u16(hasher: &mut Hasher, value: u16) {
    hasher.update(&value.to_le_bytes());
}

fn put_u32(hasher: &mut Hasher, value: u32) {
    hasher.update(&value.to_le_bytes());
}

fn put_u64(hasher: &mut Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn len_u64(value: usize) -> u64 {
    u64::try_from(value).expect("in-memory archive length must fit in u64")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CalibrationState, ChemicalChannel, EnvironmentReading, SamplingContext, SensorHealth,
    };

    fn observation(timestamp_us: u64, phase: SamplingPhase, step: u32) -> ChemicalObservation {
        let sampling = SamplingContext::new("od001-v1", "run-a", phase, step)
            .unwrap()
            .with_sample_id("sample-a")
            .unwrap();
        ChemicalObservation::new(
            timestamp_us,
            ChemicalModality::Olfactory,
            "nose-a",
            vec![ChemicalChannel {
                name: "mox-0".into(),
                raw_value: 42_000.0 + step as f32,
                unit: MeasurementUnit::Ohms,
                calibration: CalibrationState::identity("session-a"),
                health: SensorHealth::default(),
            }],
        )
        .with_environment(EnvironmentReading {
            temperature_c: Some(25.0),
            humidity_rh: Some(50.0),
            pressure_pa: Some(101_325.0),
        })
        .with_sampling(sampling)
    }

    fn trace() -> ChemicalTrace {
        let mut trace = ChemicalTrace::new(observation(1_000, SamplingPhase::Baseline, 0)).unwrap();
        trace
            .append(observation(2_000, SamplingPhase::Exposure, 1))
            .unwrap();
        trace
    }

    #[test]
    fn verified_archive_replays_as_recorded_evidence_not_fresh_physical_evidence() {
        let archive = ChemicalTraceArchive::from_trace(
            &trace(),
            ChemicalEvidenceLevel::HeldOutPhysicalObservation,
        )
        .unwrap();
        let replay = archive.verify().unwrap();

        assert_eq!(replay.evidence_level(), ChemicalEvidenceLevel::RecordedReplay);
        assert_eq!(
            replay.original_acquisition_evidence(),
            ChemicalEvidenceLevel::HeldOutPhysicalObservation
        );
        assert_eq!(replay.trace().len(), 2);
    }

    #[test]
    fn semantic_payload_tampering_breaks_digest() {
        let mut archive = ChemicalTraceArchive::from_trace(
            &trace(),
            ChemicalEvidenceLevel::BenchPhysicalObservation,
        )
        .unwrap();
        archive.observations[1].channels[0].raw_value += 1.0;
        assert_eq!(archive.verify(), Err(TraceArchiveError::DigestMismatch));
    }

    #[test]
    fn manifest_tampering_is_detected_before_replay() {
        let mut archive = ChemicalTraceArchive::from_trace(
            &trace(),
            ChemicalEvidenceLevel::BenchPhysicalObservation,
        )
        .unwrap();
        archive.manifest.run_id = "other-run".into();
        assert_eq!(
            archive.verify(),
            Err(TraceArchiveError::ManifestRunMismatch)
        );
    }

    #[test]
    fn recorded_replay_cannot_be_claimed_as_origin_acquisition() {
        assert_eq!(
            ChemicalTraceArchive::from_trace(&trace(), ChemicalEvidenceLevel::RecordedReplay),
            Err(TraceArchiveError::InvalidAcquisitionEvidence(
                ChemicalEvidenceLevel::RecordedReplay
            ))
        );
    }

    #[test]
    fn digest_is_deterministic_for_same_semantic_trace() {
        let a = ChemicalTraceArchive::from_trace(
            &trace(),
            ChemicalEvidenceLevel::BenchPhysicalObservation,
        )
        .unwrap();
        let b = ChemicalTraceArchive::from_trace(
            &trace(),
            ChemicalEvidenceLevel::BenchPhysicalObservation,
        )
        .unwrap();
        assert_eq!(a.manifest.digest, b.manifest.digest);
    }

    #[test]
    fn changing_origin_evidence_changes_archive_identity() {
        let bench = ChemicalTraceArchive::from_trace(
            &trace(),
            ChemicalEvidenceLevel::BenchPhysicalObservation,
        )
        .unwrap();
        let holdout = ChemicalTraceArchive::from_trace(
            &trace(),
            ChemicalEvidenceLevel::HeldOutPhysicalObservation,
        )
        .unwrap();
        assert_ne!(bench.manifest.digest, holdout.manifest.digest);
    }

    #[test]
    fn independently_pinned_digest_detects_rehashed_tampering() {
        let mut archive = ChemicalTraceArchive::from_trace(
            &trace(),
            ChemicalEvidenceLevel::BenchPhysicalObservation,
        )
        .unwrap();
        let pinned = archive.digest();

        archive.observations[1].channels[0].raw_value += 10.0;
        archive.manifest.digest = digest_trace(
            archive.manifest.schema_version,
            archive.manifest.acquisition_evidence,
            &archive.manifest.protocol_id,
            &archive.manifest.run_id,
            archive.manifest.modality,
            archive.manifest.replicate,
            &archive.observations,
        );

        assert!(archive.verify().is_ok());
        assert_eq!(
            archive.verify_pinned(pinned),
            Err(TraceArchiveError::PinnedDigestMismatch {
                expected: pinned,
                actual: archive.manifest.digest,
            })
        );
    }

    #[test]
    fn json_transport_roundtrip_preserves_identity_and_verification() {
        let archive = ChemicalTraceArchive::from_trace(
            &trace(),
            ChemicalEvidenceLevel::BenchPhysicalObservation,
        )
        .unwrap();
        let pinned = archive.digest();
        let bytes = serde_json::to_vec(&archive).unwrap();
        let decoded: ChemicalTraceArchive = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(decoded, archive);
        assert_eq!(decoded.verify_pinned(pinned).unwrap().archive_digest(), pinned);
    }

    #[test]
    fn unsupported_archive_schema_is_rejected() {
        let mut archive = ChemicalTraceArchive::from_trace(
            &trace(),
            ChemicalEvidenceLevel::BenchPhysicalObservation,
        )
        .unwrap();
        archive.manifest.schema_version = TRACE_ARCHIVE_SCHEMA_VERSION + 1;
        assert_eq!(
            archive.verify(),
            Err(TraceArchiveError::UnsupportedSchemaVersion {
                actual: TRACE_ARCHIVE_SCHEMA_VERSION + 1,
            })
        );
    }
}
