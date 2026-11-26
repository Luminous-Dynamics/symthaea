// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only, decision-cut-bound trajectory features for RRVR.
//!
//! This first slice deliberately emits only HDC representation similarities
//! and an uncalibrated constant-velocity baseline. It has no action authority.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::HDC_DIMENSION;
use symthaea_manipulator::encoder::ManipulatorHdcEncoder;
use symthaea_manipulator::types::ManipulatorState;

pub const SNAPSHOT_SCHEMA_VERSION: &str = "0.1";
pub const REPORT_SCHEMA_VERSION: &str = "0.1";
pub const OBSERVER_ID: &str = "symthaea-rrvr-shadow-observer-v0.1";
pub const MANIPULATOR_STATE_PROFILE_ID: &str = "manipulator-state-21d-si-v1";
pub const MANIPULATOR_UNITS_PROFILE_ID: &str = "manipulator-si-units-v1";
pub const HDC_REPRESENTATION_PROFILE_ID: &str = "symthaea-manipulator-hdc-v1";
const GENESIS_PHRASE: &str = "rrvr-symthaea-shadow-observer-v0.1";

/// The evidence domain label is carried through, never promoted by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceDomain {
    ProcessModel,
    Simulator,
    HardwareInLoop,
    Physical,
}

/// An explicitly named clock. Raw ticks from different clock domains are
/// intentionally incomparable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClockStamp {
    pub clock_domain_id: String,
    pub ticks_ns: i64,
    pub uncertainty_ns: u64,
}

/// Dispositions are preserved in the input and never silently filtered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationQuality {
    Admitted,
    Stale,
    Unknown,
    Disputed,
}

/// One typed manipulator observation. Ingest sequence is the decision-cut
/// ordering key; event/acquisition timestamps retain their source clocks.
#[derive(Debug, Clone, Serialize)]
pub struct ObservationRecord {
    pub observation_id: String,
    pub source_id: String,
    /// Monotonic sequence assigned by this source when the state was captured.
    /// It may arrive out of order relative to receiver ingest_sequence.
    pub source_sequence: u64,
    pub ingest_sequence: u64,
    pub event_time: ClockStamp,
    pub acquired_time: ClockStamp,
    pub state_profile_id: String,
    pub units_profile_id: String,
    pub coordinate_frame_id: String,
    pub quality: ObservationQuality,
    pub state: ManipulatorState,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManipulatorStateWire {
    joint_angles: [f64; 7],
    joint_velocities: [f64; 7],
    end_effector_position: [f64; 3],
    end_effector_force: [f64; 3],
    gripper_opening: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationRecordWire {
    observation_id: String,
    source_id: String,
    source_sequence: u64,
    ingest_sequence: u64,
    event_time: ClockStamp,
    acquired_time: ClockStamp,
    state_profile_id: String,
    units_profile_id: String,
    coordinate_frame_id: String,
    quality: ObservationQuality,
    state: ManipulatorStateWire,
}

impl<'de> Deserialize<'de> for ObservationRecord {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ObservationRecordWire::deserialize(deserializer)?;
        Ok(Self {
            observation_id: wire.observation_id,
            source_id: wire.source_id,
            source_sequence: wire.source_sequence,
            ingest_sequence: wire.ingest_sequence,
            event_time: wire.event_time,
            acquired_time: wire.acquired_time,
            state_profile_id: wire.state_profile_id,
            units_profile_id: wire.units_profile_id,
            coordinate_frame_id: wire.coordinate_frame_id,
            quality: wire.quality,
            state: ManipulatorState {
                joint_angles: wire.state.joint_angles,
                joint_velocities: wire.state.joint_velocities,
                end_effector_position: wire.state.end_effector_position,
                end_effector_force: wire.state.end_effector_force,
                gripper_opening: wire.state.gripper_opening,
            },
        })
    }
}

/// Receipt metadata supplied by the upstream evidence adapter. This crate
/// checks digest syntax, scope and content binding, but does not verify a
/// signature or decide whether the issuer is trusted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionReceipt {
    pub receipt_id: String,
    pub issuer_id: String,
    pub evidence_ledger_id: String,
    pub frontier_sequence: u64,
    pub snapshot_sha256: String,
    pub receipt_artifact_sha256: String,
}

/// Immutable episode snapshot at one declared receiver-ingest frontier.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EpisodeSnapshot {
    pub schema_version: String,
    pub snapshot_id: String,
    pub benchmark_id: String,
    pub scenario_id: String,
    pub task_id: String,
    pub attempt_id: String,
    pub trajectory_cluster_id: String,
    pub evidence_domain: EvidenceDomain,
    pub decision_cut_sequence: u64,
    pub ledger_frontier_sequence: u64,
    pub admission_receipt: AdmissionReceipt,
    pub observations: Vec<ObservationRecord>,
}

#[derive(Serialize)]
struct ReceiptDigestMaterial<'a> {
    receipt_id: &'a str,
    issuer_id: &'a str,
    evidence_ledger_id: &'a str,
    frontier_sequence: u64,
    receipt_artifact_sha256: &'a str,
}

#[derive(Serialize)]
struct SnapshotDigestMaterial<'a> {
    schema_version: &'a str,
    snapshot_id: &'a str,
    benchmark_id: &'a str,
    scenario_id: &'a str,
    task_id: &'a str,
    attempt_id: &'a str,
    trajectory_cluster_id: &'a str,
    evidence_domain: EvidenceDomain,
    decision_cut_sequence: u64,
    ledger_frontier_sequence: u64,
    admission_receipt: ReceiptDigestMaterial<'a>,
    observations: &'a [ObservationRecord],
}

/// Compute the canonical content digest bound by the upstream admission
/// receipt. The receipt's own snapshot digest is excluded to avoid recursion.
pub fn compute_snapshot_sha256(snapshot: &EpisodeSnapshot) -> Result<String, ObserverError> {
    let material = SnapshotDigestMaterial {
        schema_version: &snapshot.schema_version,
        snapshot_id: &snapshot.snapshot_id,
        benchmark_id: &snapshot.benchmark_id,
        scenario_id: &snapshot.scenario_id,
        task_id: &snapshot.task_id,
        attempt_id: &snapshot.attempt_id,
        trajectory_cluster_id: &snapshot.trajectory_cluster_id,
        evidence_domain: snapshot.evidence_domain,
        decision_cut_sequence: snapshot.decision_cut_sequence,
        ledger_frontier_sequence: snapshot.ledger_frontier_sequence,
        admission_receipt: ReceiptDigestMaterial {
            receipt_id: &snapshot.admission_receipt.receipt_id,
            issuer_id: &snapshot.admission_receipt.issuer_id,
            evidence_ledger_id: &snapshot.admission_receipt.evidence_ledger_id,
            frontier_sequence: snapshot.admission_receipt.frontier_sequence,
            receipt_artifact_sha256: &snapshot.admission_receipt.receipt_artifact_sha256,
        },
        observations: &snapshot.observations,
    };
    let bytes = serde_json::to_vec(&material).map_err(|_| ObserverError::SerializationFailed)?;
    Ok(sha256_hex(&bytes))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObserverError {
    UnsupportedSnapshotSchema,
    MissingIdentity,
    InvalidDigest,
    ReceiptFrontierMismatch,
    SnapshotDigestMismatch,
    EmptyObservationSet,
    ObservationBeyondDecisionCut,
    IngestSequenceNotStrictlyIncreasing,
    DuplicateObservationId,
    ClockDomainMissing,
    AcquisitionBeforeEvent,
    SourceClockRegression,
    DuplicateSourceSequence,
    MixedObservationSources,
    StateProfileMismatch,
    NonFiniteState,
    InvalidConfiguration,
    ForecastHorizonOverflow,
    SerializationFailed,
}

impl std::fmt::Display for ObserverError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ObserverError {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObserverConfig {
    /// Number of scalar levels in the existing manipulator HDC encoder.
    pub hdc_levels: usize,
    /// Forecast horizon for the deliberately simple constant-velocity baseline.
    pub forecast_horizon_ns: u64,
}

impl Default for ObserverConfig {
    fn default() -> Self {
        Self {
            hdc_levels: 32,
            forecast_horizon_ns: 50_000_000,
        }
    }
}

impl ObserverConfig {
    fn validate(&self) -> Result<(), ObserverError> {
        if !(2..=256).contains(&self.hdc_levels) || self.forecast_horizon_ns == 0 {
            return Err(ObserverError::InvalidConfiguration);
        }
        Ok(())
    }

    pub fn profile_sha256(&self) -> Result<String, ObserverError> {
        #[derive(Serialize)]
        struct ProfileMaterial<'a> {
            observer_id: &'a str,
            config: &'a ObserverConfig,
            genesis_seed_profile: &'a str,
            representation_profile_id: &'a str,
            hdc_dimension: usize,
        }
        let material = ProfileMaterial {
            observer_id: OBSERVER_ID,
            config: self,
            genesis_seed_profile: "rrvr-symthaea-shadow-observer-genesis-v0.1",
            representation_profile_id: HDC_REPRESENTATION_PROFILE_ID,
            hdc_dimension: HDC_DIMENSION,
        };
        let bytes = serde_json::to_vec(&material).map_err(|_| ObserverError::SerializationFailed)?;
        Ok(sha256_hex(&bytes))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObserverDisposition {
    BaselineOnly,
    Abstain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbstentionReason {
    StaleObservation,
    UnknownObservation,
    DisputedObservation,
    InsufficientObservations,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Abstention {
    pub observation_id: Option<String>,
    pub reason: AbstentionReason,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdjacentSimilarity {
    pub source_id: String,
    pub earlier_observation_id: String,
    pub later_observation_id: String,
    pub cosine_similarity: f32,
}

/// Descriptive trajectory representation only. Similarity is not an anomaly
/// probability, a fault verdict or a measure of physical safety.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncodedStateDigest {
    pub observation_id: String,
    pub source_sequence: u64,
    pub hdc_vector_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryRepresentation {
    pub profile_id: String,
    pub dimension: usize,
    pub encoded_state_digests: Vec<EncodedStateDigest>,
    pub adjacent_state_similarities: Vec<AdjacentSimilarity>,
    pub mean_adjacent_similarity: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstantVelocityForecast {
    pub model_id: String,
    pub calibration_status: String,
    pub horizon_ns: u64,
    pub origin_observation_id: String,
    pub origin_clock_domain_id: String,
    pub origin_acquired_at_ns: i64,
    pub target_acquired_at_ns: i64,
    pub joint_angles_rad: [f64; 7],
    pub joint_velocities_rad_s: [f64; 7],
    pub gripper_opening: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObserverReport {
    pub schema_version: String,
    pub observer_id: String,
    pub snapshot_id: String,
    pub benchmark_id: String,
    pub scenario_id: String,
    pub task_id: String,
    pub attempt_id: String,
    pub trajectory_cluster_id: String,
    pub evidence_domain: EvidenceDomain,
    pub evidence_ledger_id: String,
    pub decision_cut_sequence: u64,
    pub snapshot_sha256: String,
    pub observer_profile_sha256: String,
    pub scheduled_observation_n: usize,
    pub admitted_observation_n: usize,
    pub ingest_order_reversal_n: usize,
    pub stale_observation_n: usize,
    pub unknown_observation_n: usize,
    pub disputed_observation_n: usize,
    pub disposition: ObserverDisposition,
    pub abstentions: Vec<Abstention>,
    pub trajectory_representation: Option<TrajectoryRepresentation>,
    pub constant_velocity_forecast: Option<ConstantVelocityForecast>,
    pub claim_ceiling: String,
}

/// Analyze one frozen episode snapshot. This function is pure with respect to
/// the input and exposes no command, actuator, retry or ledger mutation API.
pub fn observe(
    snapshot: &EpisodeSnapshot,
    config: &ObserverConfig,
) -> Result<ObserverReport, ObserverError> {
    config.validate()?;
    validate_snapshot(snapshot)?;

    let snapshot_sha256 = compute_snapshot_sha256(snapshot)?;
    let observer_profile_sha256 = config.profile_sha256()?;
    let stale_n = snapshot
        .observations
        .iter()
        .filter(|row| row.quality == ObservationQuality::Stale)
        .count();
    let unknown_n = snapshot
        .observations
        .iter()
        .filter(|row| row.quality == ObservationQuality::Unknown)
        .count();
    let disputed_n = snapshot
        .observations
        .iter()
        .filter(|row| row.quality == ObservationQuality::Disputed)
        .count();
    let ingest_order_reversal_n = snapshot
        .observations
        .windows(2)
        .filter(|pair| pair[1].source_sequence < pair[0].source_sequence)
        .count();
    let admitted_n = snapshot
        .observations
        .iter()
        .filter(|row| row.quality == ObservationQuality::Admitted)
        .count();

    let mut abstentions = Vec::new();
    for row in &snapshot.observations {
        let reason = match row.quality {
            ObservationQuality::Admitted => None,
            ObservationQuality::Stale => Some(AbstentionReason::StaleObservation),
            ObservationQuality::Unknown => Some(AbstentionReason::UnknownObservation),
            ObservationQuality::Disputed => Some(AbstentionReason::DisputedObservation),
        };
        if let Some(reason) = reason {
            abstentions.push(Abstention {
                observation_id: Some(row.observation_id.clone()),
                reason,
            });
        }
    }
    if snapshot.observations.len() < 2 {
        abstentions.push(Abstention {
            observation_id: None,
            reason: AbstentionReason::InsufficientObservations,
        });
    }

    let base = |disposition, abstentions, trajectory_representation, constant_velocity_forecast| {
        ObserverReport {
            schema_version: REPORT_SCHEMA_VERSION.to_owned(),
            observer_id: OBSERVER_ID.to_owned(),
            snapshot_id: snapshot.snapshot_id.clone(),
            benchmark_id: snapshot.benchmark_id.clone(),
            scenario_id: snapshot.scenario_id.clone(),
            task_id: snapshot.task_id.clone(),
            attempt_id: snapshot.attempt_id.clone(),
            trajectory_cluster_id: snapshot.trajectory_cluster_id.clone(),
            evidence_domain: snapshot.evidence_domain,
            evidence_ledger_id: snapshot.admission_receipt.evidence_ledger_id.clone(),
            decision_cut_sequence: snapshot.decision_cut_sequence,
            snapshot_sha256: snapshot_sha256.clone(),
            observer_profile_sha256: observer_profile_sha256.clone(),
            scheduled_observation_n: snapshot.observations.len(),
            admitted_observation_n: admitted_n,
            ingest_order_reversal_n,
            stale_observation_n: stale_n,
            unknown_observation_n: unknown_n,
            disputed_observation_n: disputed_n,
            disposition,
            abstentions,
            trajectory_representation,
            constant_velocity_forecast,
            claim_ceiling: "Read-only process-model observer output. HDC similarity is descriptive, and the constant-velocity forecast is uncalibrated. No physical truth, anomaly probability, causal identification, safety certification or action authority is established.".to_owned(),
        }
    };

    if !abstentions.is_empty() {
        return Ok(base(
            ObserverDisposition::Abstain,
            abstentions,
            None,
            None,
        ));
    }

    let genesis = GenesisSeed::from_phrase(GENESIS_PHRASE);
    let mut encoder = ManipulatorHdcEncoder::new(&genesis, config.hdc_levels);
    let mut source_order = snapshot.observations.iter().collect::<Vec<_>>();
    source_order.sort_by_key(|row| row.source_sequence);
    let mut encoded = Vec::with_capacity(source_order.len());
    let mut encoded_state_digests = Vec::with_capacity(source_order.len());
    for row in &source_order {
        let vector = encoder.encode(&row.state);
        let mut vector_bytes = Vec::with_capacity(vector.as_slice().len() * 4);
        for value in vector.as_slice() {
            vector_bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        encoded_state_digests.push(EncodedStateDigest {
            observation_id: row.observation_id.clone(),
            source_sequence: row.source_sequence,
            hdc_vector_sha256: sha256_hex(&vector_bytes),
        });
        encoded.push(vector);
    }

    let mut adjacent = Vec::with_capacity(encoded.len().saturating_sub(1));
    for index in 1..encoded.len() {
        adjacent.push(AdjacentSimilarity {
            source_id: source_order[index].source_id.clone(),
            earlier_observation_id: source_order[index - 1]
                .observation_id
                .clone(),
            later_observation_id: source_order[index].observation_id.clone(),
            cosine_similarity: encoded[index - 1].similarity(&encoded[index]),
        });
    }
    let mean_similarity = if adjacent.is_empty() {
        None
    } else {
        Some(
            adjacent
                .iter()
                .map(|pair| f64::from(pair.cosine_similarity))
                .sum::<f64>()
                / adjacent.len() as f64,
        )
    };
    let trajectory_representation = TrajectoryRepresentation {
        profile_id: HDC_REPRESENTATION_PROFILE_ID.to_owned(),
        dimension: HDC_DIMENSION,
        encoded_state_digests,
        adjacent_state_similarities: adjacent,
        mean_adjacent_similarity: mean_similarity,
    };

    let last = source_order
        .last()
        .copied()
        .ok_or(ObserverError::EmptyObservationSet)?;
    let horizon_ns =
        i64::try_from(config.forecast_horizon_ns).map_err(|_| ObserverError::ForecastHorizonOverflow)?;
    let target_acquired_at_ns = last
        .acquired_time
        .ticks_ns
        .checked_add(horizon_ns)
        .ok_or(ObserverError::ForecastHorizonOverflow)?;
    let horizon_seconds = config.forecast_horizon_ns as f64 / 1_000_000_000.0;
    let mut joint_angles_rad = last.state.joint_angles;
    for index in 0..joint_angles_rad.len() {
        joint_angles_rad[index] += last.state.joint_velocities[index] * horizon_seconds;
        if !joint_angles_rad[index].is_finite() {
            return Err(ObserverError::NonFiniteState);
        }
    }
    let forecast = ConstantVelocityForecast {
        model_id: "constant-velocity-joint-angle-v0.1".to_owned(),
        calibration_status: "uncalibrated_baseline".to_owned(),
        horizon_ns: config.forecast_horizon_ns,
        origin_observation_id: last.observation_id.clone(),
        origin_clock_domain_id: last.acquired_time.clock_domain_id.clone(),
        origin_acquired_at_ns: last.acquired_time.ticks_ns,
        target_acquired_at_ns,
        joint_angles_rad,
        joint_velocities_rad_s: last.state.joint_velocities,
        gripper_opening: last.state.gripper_opening,
    };

    Ok(base(
        ObserverDisposition::BaselineOnly,
        Vec::new(),
        Some(trajectory_representation),
        Some(forecast),
    ))
}

/// Canonical report bytes for deterministic replay checks.
pub fn canonical_report_json(report: &ObserverReport) -> Result<Vec<u8>, ObserverError> {
    serde_json::to_vec(report).map_err(|_| ObserverError::SerializationFailed)
}

fn validate_snapshot(snapshot: &EpisodeSnapshot) -> Result<(), ObserverError> {
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION {
        return Err(ObserverError::UnsupportedSnapshotSchema);
    }
    let required = [
        snapshot.snapshot_id.as_str(),
        snapshot.benchmark_id.as_str(),
        snapshot.scenario_id.as_str(),
        snapshot.task_id.as_str(),
        snapshot.attempt_id.as_str(),
        snapshot.trajectory_cluster_id.as_str(),
        snapshot.admission_receipt.receipt_id.as_str(),
        snapshot.admission_receipt.issuer_id.as_str(),
        snapshot.admission_receipt.evidence_ledger_id.as_str(),
    ];
    if required.iter().any(|value| value.trim().is_empty()) {
        return Err(ObserverError::MissingIdentity);
    }
    if !is_sha256(&snapshot.admission_receipt.snapshot_sha256)
        || !is_sha256(&snapshot.admission_receipt.receipt_artifact_sha256)
    {
        return Err(ObserverError::InvalidDigest);
    }
    if snapshot.decision_cut_sequence != snapshot.ledger_frontier_sequence
        || snapshot.decision_cut_sequence != snapshot.admission_receipt.frontier_sequence
    {
        return Err(ObserverError::ReceiptFrontierMismatch);
    }
    if snapshot.observations.is_empty() {
        return Err(ObserverError::EmptyObservationSet);
    }

    let mut observation_ids = BTreeSet::new();
    let mut source_sequences = BTreeSet::new();
    let mut previous_ingest_sequence = None;
    let mut source_ids = BTreeSet::new();
    for row in &snapshot.observations {
        if row.observation_id.trim().is_empty()
            || row.source_id.trim().is_empty()
            || !observation_ids.insert(row.observation_id.as_str())
        {
            return Err(ObserverError::DuplicateObservationId);
        }
        source_ids.insert(row.source_id.as_str());
        if !source_sequences.insert(row.source_sequence) {
            return Err(ObserverError::DuplicateSourceSequence);
        }
        if row.ingest_sequence > snapshot.decision_cut_sequence {
            return Err(ObserverError::ObservationBeyondDecisionCut);
        }
        if let Some(previous) = previous_ingest_sequence {
            if row.ingest_sequence <= previous {
                return Err(ObserverError::IngestSequenceNotStrictlyIncreasing);
            }
        }
        previous_ingest_sequence = Some(row.ingest_sequence);
        if row.event_time.clock_domain_id.trim().is_empty()
            || row.acquired_time.clock_domain_id.trim().is_empty()
        {
            return Err(ObserverError::ClockDomainMissing);
        }
        if row.event_time.clock_domain_id == row.acquired_time.clock_domain_id
            && row.acquired_time.ticks_ns < row.event_time.ticks_ns
        {
            return Err(ObserverError::AcquisitionBeforeEvent);
        }
        if row.state_profile_id != MANIPULATOR_STATE_PROFILE_ID
            || row.units_profile_id != MANIPULATOR_UNITS_PROFILE_ID
            || row.coordinate_frame_id.trim().is_empty()
        {
            return Err(ObserverError::StateProfileMismatch);
        }
        if !row.state.is_finite() {
            return Err(ObserverError::NonFiniteState);
        }
    }
    if source_ids.len() != 1 {
        return Err(ObserverError::MixedObservationSources);
    }

    // A late-arriving record is not a clock failure. Validate timestamp
    // monotonicity in source-capture order, not receiver-ingest order.
    let mut source_order = snapshot.observations.iter().collect::<Vec<_>>();
    source_order.sort_by_key(|row| row.source_sequence);
    let mut last_event_by_clock: BTreeMap<&str, i64> = BTreeMap::new();
    let mut last_acquisition_by_clock: BTreeMap<&str, i64> = BTreeMap::new();
    for row in source_order {
        if let Some(previous) = last_event_by_clock.insert(
            row.event_time.clock_domain_id.as_str(),
            row.event_time.ticks_ns,
        ) {
            if row.event_time.ticks_ns < previous {
                return Err(ObserverError::SourceClockRegression);
            }
        }
        if let Some(previous) = last_acquisition_by_clock.insert(
            row.acquired_time.clock_domain_id.as_str(),
            row.acquired_time.ticks_ns,
        ) {
            if row.acquired_time.ticks_ns < previous {
                return Err(ObserverError::SourceClockRegression);
            }
        }
    }

    let calculated = compute_snapshot_sha256(snapshot)?;
    if calculated != snapshot.admission_receipt.snapshot_sha256 {
        return Err(ObserverError::SnapshotDigestMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(
        id: &str,
        ingest_sequence: u64,
        event_ticks: i64,
        acquired_ticks: i64,
    ) -> ObservationRecord {
        let mut state = ManipulatorState::home();
        state.joint_angles[0] = 0.1;
        state.joint_velocities[0] = 0.4;
        ObservationRecord {
            observation_id: id.to_owned(),
            source_id: "simulator-sensor-0".to_owned(),
            source_sequence: ingest_sequence,
            ingest_sequence,
            event_time: ClockStamp {
                clock_domain_id: "robot-clock".to_owned(),
                ticks_ns: event_ticks,
                uncertainty_ns: 100,
            },
            acquired_time: ClockStamp {
                clock_domain_id: "robot-clock".to_owned(),
                ticks_ns: acquired_ticks,
                uncertainty_ns: 100,
            },
            state_profile_id: MANIPULATOR_STATE_PROFILE_ID.to_owned(),
            units_profile_id: MANIPULATOR_UNITS_PROFILE_ID.to_owned(),
            coordinate_frame_id: "robot-base".to_owned(),
            quality: ObservationQuality::Admitted,
            state,
        }
    }

    fn fixture() -> EpisodeSnapshot {
        let mut snapshot = EpisodeSnapshot {
            schema_version: SNAPSHOT_SCHEMA_VERSION.to_owned(),
            snapshot_id: "snapshot-001".to_owned(),
            benchmark_id: "rrvr-process-model-v0.1".to_owned(),
            scenario_id: "scenario-001".to_owned(),
            task_id: "pick-object".to_owned(),
            attempt_id: "attempt-001".to_owned(),
            trajectory_cluster_id: "cluster-001".to_owned(),
            evidence_domain: EvidenceDomain::ProcessModel,
            decision_cut_sequence: 20,
            ledger_frontier_sequence: 20,
            admission_receipt: AdmissionReceipt {
                receipt_id: "receipt-001".to_owned(),
                issuer_id: "test-adapter".to_owned(),
                evidence_ledger_id: "test-ledger".to_owned(),
                frontier_sequence: 20,
                snapshot_sha256: "0".repeat(64),
                receipt_artifact_sha256: "a".repeat(64),
            },
            observations: vec![
                observation("obs-001", 10, 1_000, 1_001),
                observation("obs-002", 20, 1_100, 1_101),
            ],
        };
        snapshot.observations[1].state.joint_angles[0] = 0.2;
        snapshot.observations[1].state.joint_velocities[0] = 0.4;
        snapshot.admission_receipt.snapshot_sha256 =
            compute_snapshot_sha256(&snapshot).expect("fixture digest");
        snapshot
    }

    #[test]
    fn known_answer_emits_hdc_representation_and_uncalibrated_forecast() {
        let report = observe(&fixture(), &ObserverConfig::default()).expect("valid snapshot");
        assert_eq!(report.disposition, ObserverDisposition::BaselineOnly);
        assert_eq!(report.scheduled_observation_n, 2);
        let representation = report.trajectory_representation.expect("HDC features");
        assert_eq!(representation.dimension, HDC_DIMENSION);
        assert_eq!(representation.encoded_state_digests.len(), 2);
        assert!(representation
            .encoded_state_digests
            .iter()
            .all(|digest| is_sha256(&digest.hdc_vector_sha256)));
        assert_eq!(representation.adjacent_state_similarities.len(), 1);
        let forecast = report.constant_velocity_forecast.expect("baseline forecast");
        assert_eq!(forecast.calibration_status, "uncalibrated_baseline");
        assert!((forecast.joint_angles_rad[0] - 0.22).abs() < 1e-12);
    }

    #[test]
    fn frozen_input_produces_identical_canonical_report_bytes() {
        let snapshot = fixture();
        let first = observe(&snapshot, &ObserverConfig::default()).expect("first run");
        let second = observe(&snapshot, &ObserverConfig::default()).expect("second run");
        assert_eq!(
            canonical_report_json(&first).expect("first bytes"),
            canonical_report_json(&second).expect("second bytes")
        );
    }

    #[test]
    fn delayed_ingest_is_retained_but_temporal_features_follow_source_sequence() {
        let mut snapshot = fixture();
        snapshot.observations[0].source_sequence = 20;
        snapshot.observations[1].source_sequence = 10;
        snapshot.observations[0].event_time.ticks_ns = 1_100;
        snapshot.observations[0].acquired_time.ticks_ns = 1_101;
        snapshot.observations[1].event_time.ticks_ns = 1_000;
        snapshot.observations[1].acquired_time.ticks_ns = 1_001;
        snapshot.admission_receipt.snapshot_sha256 =
            compute_snapshot_sha256(&snapshot).expect("re-seal reordered fixture");

        let report = observe(&snapshot, &ObserverConfig::default()).expect("delayed record is valid");
        assert_eq!(report.ingest_order_reversal_n, 1);
        assert_eq!(report.scheduled_observation_n, 2);
        let representation = report.trajectory_representation.expect("source-ordered HDC");
        assert_eq!(
            representation.adjacent_state_similarities[0].earlier_observation_id,
            "obs-002"
        );
        assert_eq!(
            representation.adjacent_state_similarities[0].later_observation_id,
            "obs-001"
        );
        let forecast = report.constant_velocity_forecast.expect("source-latest forecast");
        assert_eq!(forecast.origin_observation_id, "obs-001");
        assert!((forecast.joint_angles_rad[0] - 0.12).abs() < 1e-12);
    }

    #[test]
    fn duplicate_source_sequence_is_rejected() {
        let mut snapshot = fixture();
        snapshot.observations[1].source_sequence = snapshot.observations[0].source_sequence;
        snapshot.admission_receipt.snapshot_sha256 =
            compute_snapshot_sha256(&snapshot).expect("re-seal fixture");
        assert_eq!(
            observe(&snapshot, &ObserverConfig::default()).unwrap_err(),
            ObserverError::DuplicateSourceSequence
        );
    }

    #[test]
    fn mixed_source_streams_are_rejected_in_v0_1() {
        let mut snapshot = fixture();
        snapshot.observations[1].source_id = "independent-camera-1".to_owned();
        snapshot.admission_receipt.snapshot_sha256 =
            compute_snapshot_sha256(&snapshot).expect("re-seal fixture");
        assert_eq!(
            observe(&snapshot, &ObserverConfig::default()).unwrap_err(),
            ObserverError::MixedObservationSources
        );
    }

    #[test]
    fn snapshot_mutation_after_receipt_creation_is_rejected() {
        let mut snapshot = fixture();
        snapshot.observations[0].state.joint_angles[0] += 0.01;
        assert_eq!(
            observe(&snapshot, &ObserverConfig::default()).unwrap_err(),
            ObserverError::SnapshotDigestMismatch
        );
    }

    #[test]
    fn unknown_episode_fields_are_rejected() {
        let mut value = serde_json::to_value(fixture()).expect("serialize snapshot");
        value["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<EpisodeSnapshot>(value).is_err());
    }

    #[test]
    fn unknown_observation_fields_are_rejected() {
        let mut value = serde_json::to_value(fixture()).expect("serialize snapshot");
        value["observations"][0]["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<EpisodeSnapshot>(value).is_err());
    }

    #[test]
    fn unknown_nested_manipulator_state_fields_are_rejected() {
        let mut value = serde_json::to_value(fixture()).expect("serialize snapshot");
        value["observations"][0]["state"]["unexpected_channel"] = serde_json::json!(0.0);
        assert!(serde_json::from_value::<EpisodeSnapshot>(value).is_err());
    }

    #[test]
    fn post_cut_observation_is_rejected() {
        let mut snapshot = fixture();
        snapshot.decision_cut_sequence = 15;
        snapshot.ledger_frontier_sequence = 15;
        snapshot.admission_receipt.frontier_sequence = 15;
        snapshot.admission_receipt.snapshot_sha256 =
            compute_snapshot_sha256(&snapshot).expect("re-seal fixture");
        assert_eq!(
            observe(&snapshot, &ObserverConfig::default()).unwrap_err(),
            ObserverError::ObservationBeyondDecisionCut
        );
    }

    #[test]
    fn stale_unknown_and_disputed_records_cause_abstention_without_filtering() {
        let mut snapshot = fixture();
        snapshot.observations[0].quality = ObservationQuality::Stale;
        snapshot.observations[1].quality = ObservationQuality::Disputed;
        snapshot.admission_receipt.snapshot_sha256 =
            compute_snapshot_sha256(&snapshot).expect("re-seal fixture");
        let report = observe(&snapshot, &ObserverConfig::default()).expect("valid but disputed");
        assert_eq!(report.disposition, ObserverDisposition::Abstain);
        assert_eq!(report.scheduled_observation_n, 2);
        assert_eq!(report.stale_observation_n, 1);
        assert_eq!(report.disputed_observation_n, 1);
        assert!(report.trajectory_representation.is_none());
        assert!(report.constant_velocity_forecast.is_none());
        assert_eq!(report.abstentions.len(), 2);
    }

    #[test]
    fn unknown_records_are_not_silently_dropped() {
        let mut snapshot = fixture();
        snapshot.observations[0].quality = ObservationQuality::Unknown;
        snapshot.admission_receipt.snapshot_sha256 =
            compute_snapshot_sha256(&snapshot).expect("re-seal fixture");
        let report = observe(&snapshot, &ObserverConfig::default()).expect("valid unknown input");
        assert_eq!(report.disposition, ObserverDisposition::Abstain);
        assert_eq!(report.unknown_observation_n, 1);
        assert_eq!(report.scheduled_observation_n, 2);
    }

    #[test]
    fn duplicate_ingest_sequence_is_rejected() {
        let mut snapshot = fixture();
        snapshot.observations[1].ingest_sequence = 10;
        snapshot.admission_receipt.snapshot_sha256 =
            compute_snapshot_sha256(&snapshot).expect("re-seal fixture");
        assert_eq!(
            observe(&snapshot, &ObserverConfig::default()).unwrap_err(),
            ObserverError::IngestSequenceNotStrictlyIncreasing
        );
    }

    #[test]
    fn duplicate_observation_identity_is_rejected() {
        let mut snapshot = fixture();
        snapshot.observations[1].observation_id = "obs-001".to_owned();
        snapshot.admission_receipt.snapshot_sha256 =
            compute_snapshot_sha256(&snapshot).expect("re-seal fixture");
        assert_eq!(
            observe(&snapshot, &ObserverConfig::default()).unwrap_err(),
            ObserverError::DuplicateObservationId
        );
    }

    #[test]
    fn event_clock_regression_within_one_domain_is_rejected() {
        let mut snapshot = fixture();
        snapshot.observations[1].event_time.ticks_ns = 900;
        snapshot.observations[1].acquired_time.ticks_ns = 901;
        snapshot.admission_receipt.snapshot_sha256 =
            compute_snapshot_sha256(&snapshot).expect("re-seal fixture");
        assert_eq!(
            observe(&snapshot, &ObserverConfig::default()).unwrap_err(),
            ObserverError::SourceClockRegression
        );
    }

    #[test]
    fn different_clock_domains_are_never_compared_by_raw_tick_value() {
        let mut snapshot = fixture();
        snapshot.observations[0].acquired_time.clock_domain_id = "host-clock".to_owned();
        snapshot.observations[0].acquired_time.ticks_ns = 1;
        snapshot.observations[1].acquired_time.clock_domain_id = "host-clock".to_owned();
        snapshot.observations[1].acquired_time.ticks_ns = 2;
        snapshot.admission_receipt.snapshot_sha256 =
            compute_snapshot_sha256(&snapshot).expect("re-seal fixture");
        assert!(observe(&snapshot, &ObserverConfig::default()).is_ok());
    }

    #[test]
    fn non_finite_state_is_rejected() {
        let mut snapshot = fixture();
        snapshot.observations[0].state.joint_velocities[0] = f64::NAN;
        assert_eq!(
            observe(&snapshot, &ObserverConfig::default()).unwrap_err(),
            ObserverError::NonFiniteState
        );
    }

    #[test]
    fn mismatched_receipt_frontier_is_rejected() {
        let mut snapshot = fixture();
        snapshot.admission_receipt.frontier_sequence = 19;
        snapshot.admission_receipt.snapshot_sha256 =
            compute_snapshot_sha256(&snapshot).expect("re-seal fixture");
        assert_eq!(
            observe(&snapshot, &ObserverConfig::default()).unwrap_err(),
            ObserverError::ReceiptFrontierMismatch
        );
    }

    #[test]
    fn zero_or_excessive_encoder_levels_are_rejected() {
        let snapshot = fixture();
        let invalid = ObserverConfig {
            hdc_levels: 1,
            ..ObserverConfig::default()
        };
        assert_eq!(
            observe(&snapshot, &invalid).unwrap_err(),
            ObserverError::InvalidConfiguration
        );
    }
}
