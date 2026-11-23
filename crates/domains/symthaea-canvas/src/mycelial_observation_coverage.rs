// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-bound accounting for observations considered by mycelial candidate generation.
//!
//! This ledger makes the denominator explicit: every observation ID in the declared
//! inventory must have exactly one coverage disposition. Residual observations are
//! evidence about reconstruction coverage, not graph edges and not biological
//! acceptance decisions.

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

pub const MYCELIAL_OBSERVATION_COVERAGE_SCHEMA_VERSION: u16 = 1;
pub const MAX_MYCELIAL_OBSERVATION_COVERAGE_RECORDS: usize = 16_384;
pub const MAX_MYCELIAL_COVERAGE_CANDIDATES_PER_OBSERVATION: usize = 128;
const MAX_COVERAGE_ID_BYTES: usize = 128;
const MAX_COVERAGE_VERSION_BYTES: usize = 64;
const DIGEST_HEX_BYTES: usize = 64;
const MAX_CANONICAL_COVERAGE_STRING_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialObservationCoverageState {
    /// The generator emitted one or more candidates; no review outcome is implied.
    CandidateEmitted,
    /// One or more emitted candidates were explicitly rejected for analysis.
    CandidateRejected,
    /// One or more emitted candidates remain unresolved.
    CandidateUnresolved,
    /// The observation was in scope, but candidate generation emitted none.
    NoCandidateGenerated,
    /// The observation could not be assessed because the source evidence was unavailable
    /// or unmeasurable (for example, occluded or outside the usable focal range).
    UnavailableOrUnmeasurable,
    /// The source observation is inventoried but explicitly outside the generator scope.
    OutsideDeclaredScope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialObservationResidualKind {
    ApparentCrossingUnresolvedDepth,
    PossibleContactMissing,
    BranchOrJunctionUnrepresented,
    CenterlineDiscontinuity,
    OccludedOrOutOfFocus,
    FieldOfViewBoundary,
    CalibrationRegistrationConflict,
    TemporalMismatch,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialObservationCoverageRecord {
    /// Stable identity in the source observation inventory.
    pub observation_id: String,
    /// Digest of the exact source observation or bounded observation crop.
    pub observation_digest: String,
    /// Groups records that share acquisition conditions or possible common causes.
    pub acquisition_group_id: String,
    /// Biological unit used for leakage-resistant splitting/resampling.
    pub biological_replicate_id: String,
    /// Timepoint/window identity in the declared acquisition time base.
    pub time_window_id: String,
    pub state: MycelialObservationCoverageState,
    /// Candidate identities emitted from this observation. This is not a set of accepted edges.
    pub candidate_ids: Vec<String>,
    /// Required for a source observation with no candidate or unavailable evidence.
    pub residual_kind: Option<MycelialObservationResidualKind>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialObservationCoverageLedger {
    pub schema_version: u16,
    pub source_dataset_id: String,
    pub source_dataset_digest: String,
    /// Digest of the independently declared observation inventory and scope.
    pub observation_inventory_digest: String,
    /// Complete sorted inventory. Every ID must have exactly one corresponding record.
    pub declared_observation_ids: Vec<String>,
    pub coordinate_frame_id: String,
    pub spatial_transform_digest: String,
    pub calibration_digest: String,
    pub candidate_generator_id: String,
    pub candidate_generator_version: String,
    pub candidate_generator_digest: String,
    /// Records are sorted by observation_id and must exactly cover declared_observation_ids.
    pub records: Vec<MycelialObservationCoverageRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MycelialObservationCoverageError {
    InvalidSchemaVersion(u16),
    EmptyObservationInventory,
    TooManyRecords(usize),
    TooManyCandidates { observation_id: String, count: usize },
    InvalidIdentifier(&'static str),
    InvalidDigest(&'static str),
    DuplicateObservationId(String),
    NonCanonicalObservationOrder,
    ObservationInventoryMismatch,
    DuplicateCandidateId { observation_id: String, candidate_id: String },
    NonCanonicalCandidateOrder { observation_id: String },
    InvalidRecordState { observation_id: String, reason: &'static str },
    DigestResourceLimitExceeded,
}

impl fmt::Display for MycelialObservationCoverageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchemaVersion(version) => {
                write!(f, "unsupported mycelial observation coverage schema version {version}")
            }
            Self::EmptyObservationInventory => {
                write!(f, "mycelial observation coverage inventory must not be empty")
            }
            Self::TooManyRecords(count) => write!(
                f,
                "mycelial observation coverage record count {count} exceeds {MAX_MYCELIAL_OBSERVATION_COVERAGE_RECORDS}"
            ),
            Self::TooManyCandidates {
                observation_id,
                count,
            } => write!(
                f,
                "observation {observation_id} has {count} candidates, exceeding {MAX_MYCELIAL_COVERAGE_CANDIDATES_PER_OBSERVATION}"
            ),
            Self::InvalidIdentifier(field) => {
                write!(f, "invalid or unbounded mycelial observation coverage identifier: {field}")
            }
            Self::InvalidDigest(field) => {
                write!(f, "invalid mycelial observation coverage digest: {field}")
            }
            Self::DuplicateObservationId(id) => {
                write!(f, "duplicate observation ID in coverage inventory: {id}")
            }
            Self::NonCanonicalObservationOrder => {
                write!(f, "observation IDs and coverage records must be strictly sorted")
            }
            Self::ObservationInventoryMismatch => {
                write!(f, "coverage records do not exactly match the declared observation inventory")
            }
            Self::DuplicateCandidateId {
                observation_id,
                candidate_id,
            } => write!(
                f,
                "observation {observation_id} contains duplicate candidate ID {candidate_id}"
            ),
            Self::NonCanonicalCandidateOrder { observation_id } => write!(
                f,
                "candidate IDs for observation {observation_id} must be strictly sorted"
            ),
            Self::InvalidRecordState {
                observation_id,
                reason,
            } => write!(
                f,
                "invalid coverage disposition for observation {observation_id}: {reason}"
            ),
            Self::DigestResourceLimitExceeded => {
                write!(f, "coverage ledger exceeds canonical digest resource limits")
            }
        }
    }
}

impl Error for MycelialObservationCoverageError {}

fn validate_identifier(
    value: &str,
    field: &'static str,
    maximum_bytes: usize,
) -> Result<(), MycelialObservationCoverageError> {
    if value.trim().is_empty()
        || value.len() > maximum_bytes
        || value.chars().any(char::is_control)
    {
        return Err(MycelialObservationCoverageError::InvalidIdentifier(field));
    }
    Ok(())
}

fn validate_digest(
    value: &str,
    field: &'static str,
) -> Result<(), MycelialObservationCoverageError> {
    if value.len() != DIGEST_HEX_BYTES || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(MycelialObservationCoverageError::InvalidDigest(field));
    }
    Ok(())
}

fn strictly_sorted(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

impl MycelialObservationCoverageLedger {
    /// Validate source bindings, canonical inventory order and coverage-state semantics.
    ///
    /// This checks internal completeness against the declared inventory. It does not
    /// prove that the inventory itself contains every observation in the raw acquisition;
    /// that claim must be qualified by the source inventory digest and its producer.
    pub fn validate(&self) -> Result<(), MycelialObservationCoverageError> {
        if self.schema_version != MYCELIAL_OBSERVATION_COVERAGE_SCHEMA_VERSION {
            return Err(MycelialObservationCoverageError::InvalidSchemaVersion(
                self.schema_version,
            ));
        }
        if self.records.is_empty() || self.declared_observation_ids.is_empty() {
            return Err(MycelialObservationCoverageError::EmptyObservationInventory);
        }
        if self.records.len() > MAX_MYCELIAL_OBSERVATION_COVERAGE_RECORDS
            || self.declared_observation_ids.len() > MAX_MYCELIAL_OBSERVATION_COVERAGE_RECORDS
        {
            return Err(MycelialObservationCoverageError::TooManyRecords(
                self.records.len().max(self.declared_observation_ids.len()),
            ));
        }

        validate_identifier(&self.source_dataset_id, "source_dataset_id", MAX_COVERAGE_ID_BYTES)?;
        validate_digest(&self.source_dataset_digest, "source_dataset_digest")?;
        validate_digest(
            &self.observation_inventory_digest,
            "observation_inventory_digest",
        )?;
        validate_identifier(&self.coordinate_frame_id, "coordinate_frame_id", MAX_COVERAGE_ID_BYTES)?;
        validate_digest(&self.spatial_transform_digest, "spatial_transform_digest")?;
        validate_digest(&self.calibration_digest, "calibration_digest")?;
        validate_identifier(
            &self.candidate_generator_id,
            "candidate_generator_id",
            MAX_COVERAGE_ID_BYTES,
        )?;
        validate_identifier(
            &self.candidate_generator_version,
            "candidate_generator_version",
            MAX_COVERAGE_VERSION_BYTES,
        )?;
        validate_digest(&self.candidate_generator_digest, "candidate_generator_digest")?;

        for observation_id in &self.declared_observation_ids {
            validate_identifier(observation_id, "declared_observation_id", MAX_COVERAGE_ID_BYTES)?;
        }
        if !strictly_sorted(&self.declared_observation_ids) {
            return Err(MycelialObservationCoverageError::NonCanonicalObservationOrder);
        }

        if self.records.len() != self.declared_observation_ids.len() {
            return Err(MycelialObservationCoverageError::ObservationInventoryMismatch);
        }

        let mut previous_id: Option<&str> = None;
        for (record, declared_id) in self.records.iter().zip(&self.declared_observation_ids) {
            if record.observation_id.as_str() != declared_id.as_str() {
                return Err(MycelialObservationCoverageError::ObservationInventoryMismatch);
            }
            if let Some(previous) = previous_id {
                if previous == record.observation_id.as_str() {
                    return Err(MycelialObservationCoverageError::DuplicateObservationId(
                        record.observation_id.clone(),
                    ));
                }
                if previous > record.observation_id.as_str() {
                    return Err(MycelialObservationCoverageError::NonCanonicalObservationOrder);
                }
            }
            previous_id = Some(record.observation_id.as_str());

            validate_identifier(
                &record.observation_id,
                "observation_id",
                MAX_COVERAGE_ID_BYTES,
            )?;
            validate_digest(&record.observation_digest, "observation_digest")?;
            validate_identifier(
                &record.acquisition_group_id,
                "acquisition_group_id",
                MAX_COVERAGE_ID_BYTES,
            )?;
            validate_identifier(
                &record.biological_replicate_id,
                "biological_replicate_id",
                MAX_COVERAGE_ID_BYTES,
            )?;
            validate_identifier(
                &record.time_window_id,
                "time_window_id",
                MAX_COVERAGE_ID_BYTES,
            )?;

            if record.candidate_ids.len() > MAX_MYCELIAL_COVERAGE_CANDIDATES_PER_OBSERVATION {
                return Err(MycelialObservationCoverageError::TooManyCandidates {
                    observation_id: record.observation_id.clone(),
                    count: record.candidate_ids.len(),
                });
            }
            for candidate_id in &record.candidate_ids {
                validate_identifier(candidate_id, "candidate_id", MAX_COVERAGE_ID_BYTES)?;
            }
            for pair in record.candidate_ids.windows(2) {
                if pair[0] == pair[1] {
                    return Err(MycelialObservationCoverageError::DuplicateCandidateId {
                        observation_id: record.observation_id.clone(),
                        candidate_id: pair[0].clone(),
                    });
                }
                if pair[0] > pair[1] {
                    return Err(
                        MycelialObservationCoverageError::NonCanonicalCandidateOrder {
                            observation_id: record.observation_id.clone(),
                        },
                    );
                }
            }

            let has_candidates = !record.candidate_ids.is_empty();
            let has_residual = record.residual_kind.is_some();
            let invalid_state = match record.state {
                MycelialObservationCoverageState::CandidateEmitted
                | MycelialObservationCoverageState::CandidateRejected
                | MycelialObservationCoverageState::CandidateUnresolved => {
                    (!has_candidates).then_some("candidate disposition requires at least one candidate")
                }
                MycelialObservationCoverageState::NoCandidateGenerated => {
                    if has_candidates {
                        Some("no-candidate disposition cannot contain candidate IDs")
                    } else if !has_residual {
                        Some("no-candidate disposition requires a residual classification")
                    } else {
                        None
                    }
                }
                MycelialObservationCoverageState::UnavailableOrUnmeasurable => {
                    if has_candidates {
                        Some("unavailable evidence cannot contain candidate IDs")
                    } else if !has_residual {
                        Some("unavailable evidence requires a residual classification")
                    } else {
                        None
                    }
                }
                MycelialObservationCoverageState::OutsideDeclaredScope => {
                    if has_candidates {
                        Some("out-of-scope observation cannot contain candidate IDs")
                    } else if has_residual {
                        Some("out-of-scope observation cannot claim an in-scope residual")
                    } else {
                        None
                    }
                }
            };
            if let Some(reason) = invalid_state {
                return Err(MycelialObservationCoverageError::InvalidRecordState {
                    observation_id: record.observation_id.clone(),
                    reason,
                });
            }
        }

        Ok(())
    }

    /// Hashes exact ledger content. Semantic validation is deliberately separate:
    /// invalid/tampered artifacts still receive a distinct content digest.
    pub fn canonical_digest_hex(
        &self,
    ) -> Result<String, MycelialObservationCoverageError> {
        if self.records.len() > MAX_MYCELIAL_OBSERVATION_COVERAGE_RECORDS
            || self.declared_observation_ids.len() > MAX_MYCELIAL_OBSERVATION_COVERAGE_RECORDS
            || self.records.iter().any(|record| {
                record.candidate_ids.len() > MAX_MYCELIAL_COVERAGE_CANDIDATES_PER_OBSERVATION
            })
        {
            return Err(MycelialObservationCoverageError::DigestResourceLimitExceeded);
        }
        let mut total_string_bytes = 0usize;
        let mut account_string = |value: &str| {
            total_string_bytes = total_string_bytes.saturating_add(value.len());
            total_string_bytes <= MAX_CANONICAL_COVERAGE_STRING_BYTES
        };
        for value in [
            self.source_dataset_id.as_str(),
            self.source_dataset_digest.as_str(),
            self.observation_inventory_digest.as_str(),
            self.coordinate_frame_id.as_str(),
            self.spatial_transform_digest.as_str(),
            self.calibration_digest.as_str(),
            self.candidate_generator_id.as_str(),
            self.candidate_generator_version.as_str(),
            self.candidate_generator_digest.as_str(),
        ] {
            if !account_string(value) {
                return Err(MycelialObservationCoverageError::DigestResourceLimitExceeded);
            }
        }
        for observation_id in &self.declared_observation_ids {
            if !account_string(observation_id) {
                return Err(MycelialObservationCoverageError::DigestResourceLimitExceeded);
            }
        }
        for record in &self.records {
            for value in [
                record.observation_id.as_str(),
                record.observation_digest.as_str(),
                record.acquisition_group_id.as_str(),
                record.biological_replicate_id.as_str(),
                record.time_window_id.as_str(),
            ] {
                if !account_string(value) {
                    return Err(MycelialObservationCoverageError::DigestResourceLimitExceeded);
                }
            }
            for candidate_id in &record.candidate_ids {
                if !account_string(candidate_id) {
                    return Err(MycelialObservationCoverageError::DigestResourceLimitExceeded);
                }
            }
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-observation-coverage-ledger-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hash_string(&mut hasher, &self.source_dataset_id);
        hash_string(&mut hasher, &self.source_dataset_digest);
        hash_string(&mut hasher, &self.observation_inventory_digest);
        hash_string(&mut hasher, &self.coordinate_frame_id);
        hash_string(&mut hasher, &self.spatial_transform_digest);
        hash_string(&mut hasher, &self.calibration_digest);
        hash_string(&mut hasher, &self.candidate_generator_id);
        hash_string(&mut hasher, &self.candidate_generator_version);
        hash_string(&mut hasher, &self.candidate_generator_digest);

        hasher.update(&(self.declared_observation_ids.len() as u64).to_le_bytes());
        for observation_id in &self.declared_observation_ids {
            hash_string(&mut hasher, observation_id);
        }

        hasher.update(&(self.records.len() as u64).to_le_bytes());
        for record in &self.records {
            hash_string(&mut hasher, &record.observation_id);
            hash_string(&mut hasher, &record.observation_digest);
            hash_string(&mut hasher, &record.acquisition_group_id);
            hash_string(&mut hasher, &record.biological_replicate_id);
            hash_string(&mut hasher, &record.time_window_id);
            hasher.update(&[coverage_state_tag(record.state)]);
            hasher.update(&(record.candidate_ids.len() as u64).to_le_bytes());
            for candidate_id in &record.candidate_ids {
                hash_string(&mut hasher, candidate_id);
            }
            match record.residual_kind {
                Some(kind) => {
                    hasher.update(&[1, residual_kind_tag(kind)]);
                }
                None => {
                    hasher.update(&[0]);
                }
            }
        }

        Ok(hasher.finalize().to_hex().to_string())
    }
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn coverage_state_tag(state: MycelialObservationCoverageState) -> u8 {
    match state {
        MycelialObservationCoverageState::CandidateEmitted => 0,
        MycelialObservationCoverageState::CandidateRejected => 1,
        MycelialObservationCoverageState::CandidateUnresolved => 2,
        MycelialObservationCoverageState::NoCandidateGenerated => 3,
        MycelialObservationCoverageState::UnavailableOrUnmeasurable => 4,
        MycelialObservationCoverageState::OutsideDeclaredScope => 5,
    }
}

fn residual_kind_tag(kind: MycelialObservationResidualKind) -> u8 {
    match kind {
        MycelialObservationResidualKind::ApparentCrossingUnresolvedDepth => 0,
        MycelialObservationResidualKind::PossibleContactMissing => 1,
        MycelialObservationResidualKind::BranchOrJunctionUnrepresented => 2,
        MycelialObservationResidualKind::CenterlineDiscontinuity => 3,
        MycelialObservationResidualKind::OccludedOrOutOfFocus => 4,
        MycelialObservationResidualKind::FieldOfViewBoundary => 5,
        MycelialObservationResidualKind::CalibrationRegistrationConflict => 6,
        MycelialObservationResidualKind::TemporalMismatch => 7,
        MycelialObservationResidualKind::Unknown => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        let digit = character.to_digit(16).expect("fixture character is hexadecimal");
        let value = char::from_digit(digit, 16).expect("hex digit");
        std::iter::repeat_n(value, DIGEST_HEX_BYTES).collect()
    }

    fn fixture() -> MycelialObservationCoverageLedger {
        MycelialObservationCoverageLedger {
            schema_version: MYCELIAL_OBSERVATION_COVERAGE_SCHEMA_VERSION,
            source_dataset_id: "dataset-001".to_string(),
            source_dataset_digest: digest('a'),
            observation_inventory_digest: digest('b'),
            declared_observation_ids: vec!["obs-001".to_string(), "obs-002".to_string()],
            coordinate_frame_id: "specimen-frame-v1".to_string(),
            spatial_transform_digest: digest('c'),
            calibration_digest: digest('d'),
            candidate_generator_id: "mycelial-candidate-generator".to_string(),
            candidate_generator_version: "0.1.0".to_string(),
            candidate_generator_digest: digest('e'),
            records: vec![
                MycelialObservationCoverageRecord {
                    observation_id: "obs-001".to_string(),
                    observation_digest: digest('1'),
                    acquisition_group_id: "acquisition-001".to_string(),
                    biological_replicate_id: "replicate-001".to_string(),
                    time_window_id: "time-001".to_string(),
                    state: MycelialObservationCoverageState::CandidateEmitted,
                    candidate_ids: vec!["candidate-001".to_string()],
                    residual_kind: None,
                },
                MycelialObservationCoverageRecord {
                    observation_id: "obs-002".to_string(),
                    observation_digest: digest('2'),
                    acquisition_group_id: "acquisition-001".to_string(),
                    biological_replicate_id: "replicate-001".to_string(),
                    time_window_id: "time-001".to_string(),
                    state: MycelialObservationCoverageState::NoCandidateGenerated,
                    candidate_ids: Vec::new(),
                    residual_kind: Some(
                        MycelialObservationResidualKind::PossibleContactMissing,
                    ),
                },
            ],
        }
    }

    #[test]
    fn complete_inventory_and_candidate_residual_states_validate() {
        let ledger = fixture();
        ledger.validate().unwrap();
        assert_eq!(ledger.canonical_digest_hex().unwrap().len(), 64);
    }

    #[test]
    fn digest_is_deterministic() {
        let ledger = fixture();
        assert_eq!(
            ledger.canonical_digest_hex().unwrap(),
            ledger.canonical_digest_hex().unwrap()
        );
    }

    #[test]
    fn missing_observation_record_fails_inventory_binding() {
        let mut ledger = fixture();
        ledger.records.pop();
        assert_eq!(
            ledger.validate(),
            Err(MycelialObservationCoverageError::ObservationInventoryMismatch)
        );
    }

    #[test]
    fn duplicate_candidate_id_is_rejected() {
        let mut ledger = fixture();
        ledger.records[0].candidate_ids =
            vec!["candidate-001".to_string(), "candidate-001".to_string()];
        assert!(matches!(
            ledger.validate(),
            Err(MycelialObservationCoverageError::DuplicateCandidateId { .. })
        ));
    }

    #[test]
    fn rejected_unresolved_and_unavailable_dispositions_preserve_candidate_semantics() {
        let mut ledger = fixture();
        ledger.records[0].state = MycelialObservationCoverageState::CandidateRejected;
        ledger.validate().unwrap();

        ledger.records[0].state = MycelialObservationCoverageState::CandidateUnresolved;
        ledger.records[0].residual_kind =
            Some(MycelialObservationResidualKind::ApparentCrossingUnresolvedDepth);
        ledger.validate().unwrap();

        ledger.records[1].state = MycelialObservationCoverageState::UnavailableOrUnmeasurable;
        ledger.validate().unwrap();
    }

    #[test]
    fn out_of_scope_disposition_is_valid_only_without_candidates_or_residuals() {
        let mut ledger = fixture();
        ledger.records[1].state = MycelialObservationCoverageState::OutsideDeclaredScope;
        ledger.records[1].residual_kind = None;
        ledger.validate().unwrap();
    }

    #[test]
    fn no_candidate_state_requires_a_residual_classification() {
        let mut ledger = fixture();
        ledger.records[1].residual_kind = None;
        assert!(matches!(
            ledger.validate(),
            Err(MycelialObservationCoverageError::InvalidRecordState { .. })
        ));
    }

    #[test]
    fn out_of_scope_observation_cannot_carry_candidates_or_residuals() {
        let mut ledger = fixture();
        ledger.records[0].state = MycelialObservationCoverageState::OutsideDeclaredScope;
        assert!(matches!(
            ledger.validate(),
            Err(MycelialObservationCoverageError::InvalidRecordState { .. })
        ));
    }

    #[test]
    fn digest_commits_invalid_content_without_validating_it() {
        let original = fixture();
        let original_digest = original.canonical_digest_hex().unwrap();
        let mut tampered = original;
        tampered.records[0].candidate_ids.clear();
        let tampered_digest = tampered.canonical_digest_hex().unwrap();
        assert_ne!(tampered_digest, original_digest);
        assert!(tampered.validate().is_err());
    }

    #[test]
    fn noncanonical_observation_order_is_rejected() {
        let mut ledger = fixture();
        ledger.declared_observation_ids.reverse();
        ledger.records.reverse();
        assert_eq!(
            ledger.validate(),
            Err(MycelialObservationCoverageError::NonCanonicalObservationOrder)
        );
    }
}
