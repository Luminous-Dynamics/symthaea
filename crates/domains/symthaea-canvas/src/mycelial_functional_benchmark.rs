// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Leakage-resistant split manifests for mycelial functional benchmarks.
//!
//! Splits are assigned to biological replicates, never to frames, crops,
//! trajectories or individual observations. Every observation set for one
//! biological replicate must remain in one split, even when the replicate is
//! represented by multiple sessions or assay exports.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_functional_observations::{
    MycelialFunctionalObservationError, MycelialFunctionalObservationSet,
};

pub const MYCELIAL_FUNCTIONAL_BENCHMARK_SCHEMA_VERSION: u16 = 1;
pub const MAX_MYCELIAL_FUNCTIONAL_BENCHMARK_ASSIGNMENTS: usize = 65_536;
const MAX_BENCHMARK_ID_BYTES: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialFunctionalBenchmarkSplit {
    Train,
    Validation,
    Test,
    ExternalHoldout,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialFunctionalBenchmarkAssignment {
    /// Digest of one complete immutable observation set.
    pub observation_set_digest: String,
    /// Stable biological replicate identity shared across sessions and assay exports.
    pub biological_replicate_id: String,
    pub split: MycelialFunctionalBenchmarkSplit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialFunctionalBenchmarkSplitManifest {
    pub schema_version: u16,
    pub benchmark_id: String,
    pub split_policy_id: String,
    pub split_policy_version: String,
    /// Sorted by observation_set_digest. Replicate identity is the grouping key.
    pub assignments: Vec<MycelialFunctionalBenchmarkAssignment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MycelialFunctionalBenchmarkError {
    InvalidSchemaVersion(u16),
    InvalidMetadata,
    TooManyAssignments(usize),
    DuplicateOrUnorderedObservationSet(String),
    ReplicateLeakage {
        biological_replicate_id: String,
        first_split: MycelialFunctionalBenchmarkSplit,
        conflicting_split: MycelialFunctionalBenchmarkSplit,
    },
    ObservationSetMismatch(String),
    ReplicateIdentityMismatch {
        observation_set_digest: String,
        expected: String,
        actual: String,
    },
    MissingAssignment(String),
    UnexpectedAssignment(String),
    InvalidObservationSet(String),
}

impl fmt::Display for MycelialFunctionalBenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchemaVersion(version) => write!(
                f,
                "unsupported mycelial functional benchmark schema version {version}"
            ),
            Self::InvalidMetadata => write!(f, "invalid mycelial functional benchmark split metadata"),
            Self::TooManyAssignments(count) => write!(
                f,
                "functional benchmark assignment count {count} exceeds {MAX_MYCELIAL_FUNCTIONAL_BENCHMARK_ASSIGNMENTS}"
            ),
            Self::DuplicateOrUnorderedObservationSet(digest) => write!(
                f,
                "functional benchmark observation-set digests are duplicated or unordered at {digest}"
            ),
            Self::ReplicateLeakage {
                biological_replicate_id,
                first_split,
                conflicting_split,
            } => write!(
                f,
                "biological replicate {biological_replicate_id} leaks across {first_split:?} and {conflicting_split:?}"
            ),
            Self::ObservationSetMismatch(digest) => write!(
                f,
                "functional benchmark assignment does not match observation set {digest}"
            ),
            Self::ReplicateIdentityMismatch {
                observation_set_digest,
                expected,
                actual,
            } => write!(
                f,
                "observation set {observation_set_digest} replicate identity mismatch: expected {expected}, got {actual}"
            ),
            Self::MissingAssignment(digest) => write!(
                f,
                "functional observation set {digest} has no split assignment"
            ),
            Self::UnexpectedAssignment(digest) => write!(
                f,
                "functional benchmark manifest contains unprovided observation set {digest}"
            ),
            Self::InvalidObservationSet(reason) => {
                write!(f, "invalid functional observation set in benchmark: {reason}")
            }
        }
    }
}

impl Error for MycelialFunctionalBenchmarkError {}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= MAX_BENCHMARK_ID_BYTES
        && value.bytes().all(|byte| !byte.is_ascii_control())
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

impl MycelialFunctionalBenchmarkSplitManifest {
    /// Validate canonical ordering and ensure no biological replicate crosses splits.
    pub fn validate(&self) -> Result<(), MycelialFunctionalBenchmarkError> {
        if self.schema_version != MYCELIAL_FUNCTIONAL_BENCHMARK_SCHEMA_VERSION {
            return Err(MycelialFunctionalBenchmarkError::InvalidSchemaVersion(
                self.schema_version,
            ));
        }
        if !valid_id(&self.benchmark_id)
            || !valid_id(&self.split_policy_id)
            || !valid_id(&self.split_policy_version)
        {
            return Err(MycelialFunctionalBenchmarkError::InvalidMetadata);
        }
        if self.assignments.is_empty() {
            return Err(MycelialFunctionalBenchmarkError::InvalidMetadata);
        }
        if self.assignments.len() > MAX_MYCELIAL_FUNCTIONAL_BENCHMARK_ASSIGNMENTS {
            return Err(MycelialFunctionalBenchmarkError::TooManyAssignments(
                self.assignments.len(),
            ));
        }

        let mut replicate_splits =
            BTreeMap::<&str, MycelialFunctionalBenchmarkSplit>::new();
        let mut previous_digest: Option<&str> = None;
        for assignment in &self.assignments {
            if !valid_digest(&assignment.observation_set_digest)
                || !valid_id(&assignment.biological_replicate_id)
            {
                return Err(MycelialFunctionalBenchmarkError::InvalidMetadata);
            }
            if previous_digest
                .is_some_and(|previous| previous >= assignment.observation_set_digest.as_str())
            {
                return Err(
                    MycelialFunctionalBenchmarkError::DuplicateOrUnorderedObservationSet(
                        assignment.observation_set_digest.clone(),
                    ),
                );
            }
            previous_digest = Some(&assignment.observation_set_digest);
            if let Some(first_split) = replicate_splits.get(assignment.biological_replicate_id.as_str()) {
                if *first_split != assignment.split {
                    return Err(MycelialFunctionalBenchmarkError::ReplicateLeakage {
                        biological_replicate_id: assignment.biological_replicate_id.clone(),
                        first_split: *first_split,
                        conflicting_split: assignment.split,
                    });
                }
            } else {
                replicate_splits.insert(
                    assignment.biological_replicate_id.as_str(),
                    assignment.split,
                );
            }
        }
        Ok(())
    }

    /// Require a one-to-one binding between manifest assignments and exact observation sets.
    pub fn validate_against(
        &self,
        observation_sets: &[MycelialFunctionalObservationSet],
    ) -> Result<(), MycelialFunctionalBenchmarkError> {
        self.validate()?;
        if observation_sets.len() != self.assignments.len() {
            return Err(MycelialFunctionalBenchmarkError::InvalidMetadata);
        }

        let mut provided = BTreeMap::<String, &MycelialFunctionalObservationSet>::new();
        for set in observation_sets {
            set.validate().map_err(|error: MycelialFunctionalObservationError| {
                MycelialFunctionalBenchmarkError::InvalidObservationSet(error.to_string())
            })?;
            let digest = set.canonical_digest_hex().map_err(|error| {
                MycelialFunctionalBenchmarkError::InvalidObservationSet(error.to_string())
            })?;
            if provided.insert(digest.clone(), set).is_some() {
                return Err(MycelialFunctionalBenchmarkError::ObservationSetMismatch(
                    digest,
                ));
            }
        }

        for assignment in &self.assignments {
            let Some(set) = provided.remove(&assignment.observation_set_digest) else {
                return Err(MycelialFunctionalBenchmarkError::UnexpectedAssignment(
                    assignment.observation_set_digest.clone(),
                ));
            };
            if set.biological_replicate_id != assignment.biological_replicate_id {
                return Err(MycelialFunctionalBenchmarkError::ReplicateIdentityMismatch {
                    observation_set_digest: assignment.observation_set_digest.clone(),
                    expected: assignment.biological_replicate_id.clone(),
                    actual: set.biological_replicate_id.clone(),
                });
            }
        }
        if let Some(missing_digest) = provided.keys().next() {
            return Err(MycelialFunctionalBenchmarkError::MissingAssignment(
                missing_digest.clone(),
            ));
        }
        Ok(())
    }

    pub fn canonical_digest_hex(&self) -> Result<String, MycelialFunctionalBenchmarkError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-functional-benchmark-split-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hash_string(&mut hasher, &self.benchmark_id);
        hash_string(&mut hasher, &self.split_policy_id);
        hash_string(&mut hasher, &self.split_policy_version);
        hasher.update(&(self.assignments.len() as u64).to_le_bytes());
        for assignment in &self.assignments {
            hash_string(&mut hasher, &assignment.observation_set_digest);
            hash_string(&mut hasher, &assignment.biological_replicate_id);
            hasher.update(&[match assignment.split {
                MycelialFunctionalBenchmarkSplit::Train => 0,
                MycelialFunctionalBenchmarkSplit::Validation => 1,
                MycelialFunctionalBenchmarkSplit::Test => 2,
                MycelialFunctionalBenchmarkSplit::ExternalHoldout => 3,
            }]);
        }
        let digest = hasher.finalize();
        let mut output = String::with_capacity(64);
        use fmt::Write as _;
        for byte in digest.as_bytes() {
            let _ = write!(&mut output, "{byte:02x}");
        }
        Ok(output)
    }

    pub fn assignment_for(
        &self,
        observation_set_digest: &str,
    ) -> Option<MycelialFunctionalBenchmarkSplit> {
        self.assignments
            .binary_search_by(|assignment| {
                assignment
                    .observation_set_digest
                    .as_str()
                    .cmp(observation_set_digest)
            })
            .ok()
            .map(|index| self.assignments[index].split)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelial_functional_observations::{
        MycelialFunctionalFlowDirection, MycelialFunctionalMeasurement,
        MycelialFunctionalObservation, MycelialFunctionalObservationSet,
    };
    use crate::mycelial_reconstruction_metrics::{
        AnnotatedHyphaSegment, AnnotatedJunction, AnnotatedJunctionKind,
        MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION, AnnotatedMycelialSkeleton,
        ReconstructionDimensionality, ReconstructionLengthUnit, ReconstructionPoint3,
    };

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn set(replicate_id: &str) -> MycelialFunctionalObservationSet {
        let graph = AnnotatedMycelialSkeleton {
            schema_version: MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION,
            frame_id: "frame-1".to_string(),
            coordinate_frame_id: "microscope-stage-1".to_string(),
            calibration_digest: digest('b'),
            source_observation_digest: digest('a'),
            dimensionality: ReconstructionDimensionality::ThreeD,
            length_unit: ReconstructionLengthUnit::Micrometer,
            producer_id: "human-annotation".to_string(),
            producer_version: "v1".to_string(),
            junctions: vec![
                AnnotatedJunction {
                    id: "A".to_string(),
                    kind: AnnotatedJunctionKind::Origin,
                    position: ReconstructionPoint3 { x: 0.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "B".to_string(),
                    kind: AnnotatedJunctionKind::Tip,
                    position: ReconstructionPoint3 { x: 10.0, y: 0.0, z: 0.0 },
                },
            ],
            segments: vec![AnnotatedHyphaSegment {
                id: "edge-AB".to_string(),
                start_junction_id: "A".to_string(),
                end_junction_id: "B".to_string(),
                length: 10.0,
            }],
        };
        MycelialFunctionalObservationSet {
            schema_version: crate::mycelial_functional_observations::MYCELIAL_FUNCTIONAL_OBSERVATION_SCHEMA_VERSION,
            dataset_id: "dataset-1".to_string(),
            session_id: "session-1".to_string(),
            biological_replicate_id: replicate_id.to_string(),
            organism_id: "Rhizophagus-irregularis".to_string(),
            treatment_id: None,
            coordinate_frame_id: graph.coordinate_frame_id.clone(),
            calibration_digest: graph.calibration_digest.clone(),
            source_graph_digest: graph.canonical_digest_hex().unwrap(),
            functional_measurement_digest: digest('c'),
            assay_protocol_id: "kymograph-flow".to_string(),
            assay_protocol_version: "1".to_string(),
            assay_protocol_digest: digest('d'),
            timebase_id: "monotonic-ns".to_string(),
            window_start_ns: 0,
            window_end_ns: 100,
            observations: vec![MycelialFunctionalObservation {
                observation_id: "flow-1".to_string(),
                window_start_ns: 0,
                window_end_ns: 100,
                measurement: MycelialFunctionalMeasurement::CytoplasmicFlow {
                    edge_id: "edge-AB".to_string(),
                    direction: MycelialFunctionalFlowDirection::SourceToTarget,
                    mean_speed_um_per_s: Some(1.0),
                    trajectory_count: 1,
                },
            }],
        }
    }

    fn assignment(set: &MycelialFunctionalObservationSet, split: MycelialFunctionalBenchmarkSplit) -> MycelialFunctionalBenchmarkAssignment {
        MycelialFunctionalBenchmarkAssignment {
            observation_set_digest: set.canonical_digest_hex().unwrap(),
            biological_replicate_id: set.biological_replicate_id.clone(),
            split,
        }
    }

    fn manifest(
        sets: &[MycelialFunctionalObservationSet],
        splits: &[MycelialFunctionalBenchmarkSplit],
    ) -> MycelialFunctionalBenchmarkSplitManifest {
        let mut assignments: Vec<_> = sets
            .iter()
            .zip(splits)
            .map(|(set, split)| assignment(set, *split))
            .collect();
        assignments.sort_by(|left, right| {
            left.observation_set_digest.cmp(&right.observation_set_digest)
        });
        MycelialFunctionalBenchmarkSplitManifest {
            schema_version: MYCELIAL_FUNCTIONAL_BENCHMARK_SCHEMA_VERSION,
            benchmark_id: "mycelial-functional-v0".to_string(),
            split_policy_id: "biological-replicate-disjoint".to_string(),
            split_policy_version: "1".to_string(),
            assignments,
        }
    }

    #[test]
    fn replicate_cannot_leak_across_train_and_test() {
        let first = set("replicate-1");
        let second = set("replicate-1");
        let value = manifest(
            &[first, second],
            &[
                MycelialFunctionalBenchmarkSplit::Train,
                MycelialFunctionalBenchmarkSplit::Test,
            ],
        );
        assert!(matches!(
            value.validate(),
            Err(MycelialFunctionalBenchmarkError::ReplicateLeakage {
                biological_replicate_id,
                ..
            }) if biological_replicate_id == "replicate-1"
        ));
    }

    #[test]
    fn multiple_sessions_from_one_replicate_may_share_one_split() {
        let first = set("replicate-1");
        let mut second = set("replicate-1");
        second.session_id = "session-2".to_string();
        let value = manifest(
            &[first.clone(), second.clone()],
            &[
                MycelialFunctionalBenchmarkSplit::Train,
                MycelialFunctionalBenchmarkSplit::Train,
            ],
        );
        value.validate_against(&[first, second]).unwrap();
    }

    #[test]
    fn exact_observation_sets_must_be_bound_to_manifest() {
        let first = set("replicate-1");
        let second = set("replicate-2");
        let value = manifest(
            &[first.clone(), second.clone()],
            &[
                MycelialFunctionalBenchmarkSplit::Train,
                MycelialFunctionalBenchmarkSplit::Test,
            ],
        );
        value.validate_against(&[first.clone(), second.clone()]).unwrap();

        let mut altered = second;
        altered.treatment_id = Some("different-treatment".to_string());
        assert!(matches!(
            value.validate_against(&[first, altered]),
            Err(MycelialFunctionalBenchmarkError::UnexpectedAssignment(_))
                | Err(MycelialFunctionalBenchmarkError::MissingAssignment(_))
        ));
    }

    #[test]
    fn replicate_identity_must_match_the_bound_observation_set() {
        let original = set("replicate-1");
        let mut altered = original.clone();
        altered.biological_replicate_id = "replicate-2".to_string();
        let mut value = manifest(
            std::slice::from_ref(&original),
            &[MycelialFunctionalBenchmarkSplit::Train],
        );
        value.assignments[0].biological_replicate_id = "replicate-2".to_string();
        assert!(matches!(
            value.validate_against(&[altered]),
            Err(MycelialFunctionalBenchmarkError::ReplicateIdentityMismatch { .. })
        ));
    }

    #[test]
    fn manifest_digest_changes_when_split_changes() {
        let first = set("replicate-1");
        let train = manifest(
            std::slice::from_ref(&first),
            &[MycelialFunctionalBenchmarkSplit::Train],
        );
        let test = manifest(
            std::slice::from_ref(&first),
            &[MycelialFunctionalBenchmarkSplit::Test],
        );
        assert_ne!(
            train.canonical_digest_hex().unwrap(),
            test.canonical_digest_hex().unwrap()
        );
        assert_eq!(
            train.assignment_for(&first.canonical_digest_hex().unwrap()),
            Some(MycelialFunctionalBenchmarkSplit::Train)
        );
    }
}
