// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence contract for measured functions associated with reconstructed mycelial graphs.
//!
//! Functional observations are deliberately separate from graph-derived predictions.
//! This module records measurements; it does not infer conductance, simulate flow,
//! accept candidate fusions, or treat connectivity as biological viability.
//!
//! Flow, tracer transfer and microbial dispersal are distinct assay types. Their
//! values retain their units, time base, replicate identity and independent evidence
//! digest so downstream benchmarks cannot silently conflate different phenomena.

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_reconstruction_metrics::AnnotatedMycelialSkeleton;

pub const MYCELIAL_FUNCTIONAL_OBSERVATION_SCHEMA_VERSION: u16 = 1;
pub const MAX_MYCELIAL_FUNCTIONAL_OBSERVATIONS: usize = 65_536;
pub const MAX_MYCELIAL_FUNCTIONAL_ID_BYTES: usize = 128;
const DIGEST_HEX_BYTES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialFunctionalFlowDirection {
    /// Positive direction follows the annotated edge's source-to-target orientation.
    SourceToTarget,
    /// Negative direction follows the annotated edge's target-to-source orientation.
    TargetToSource,
    /// Both directions were observed within the declared measurement window.
    Bidirectional,
    /// Direction could not be resolved from the measurement.
    Unresolved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialFunctionalQuantityUnit {
    Nanogram,
    Microgram,
    Nanomole,
    Micromole,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MycelialFunctionalQuantity {
    pub value: f64,
    pub unit: MycelialFunctionalQuantityUnit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "measurement_kind", rename_all = "snake_case")]
pub enum MycelialFunctionalMeasurement {
    /// Measured movement of cytoplasm within one observed/accepted hypha segment.
    CytoplasmicFlow {
        edge_id: String,
        direction: MycelialFunctionalFlowDirection,
        /// Arithmetic mean speed in micrometers per second; absent means unmeasured.
        mean_speed_um_per_s: Option<f64>,
        /// Number of independently tracked trajectories contributing to this record.
        trajectory_count: u64,
    },
    /// A source-to-target tracer assay. Arrival time is on the set's monotonic time base.
    TracerTransfer {
        source_node_id: String,
        target_node_id: String,
        tracer_id: String,
        injection_time_ns: u64,
        arrival_time_ns: Option<u64>,
        injected_quantity: Option<MycelialFunctionalQuantity>,
        recovered_quantity: Option<MycelialFunctionalQuantity>,
    },
    /// Microbial dispersal along a hyphal highway or an associated microfluidic region.
    /// Region IDs are assay-local and are not silently treated as graph junction IDs.
    MicrobialDispersal {
        source_region_id: String,
        target_region_id: String,
        transported_cell_count: u64,
        first_arrival_time_ns: Option<u64>,
        /// Optional measured liquid-saturation fraction in [0, 1].
        liquid_saturation_fraction: Option<f64>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialFunctionalObservation {
    pub observation_id: String,
    /// Half-open monotonic observation interval [start_ns, end_ns).
    pub window_start_ns: u64,
    pub window_end_ns: u64,
    pub measurement: MycelialFunctionalMeasurement,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialFunctionalObservationSet {
    pub schema_version: u16,
    pub dataset_id: String,
    pub session_id: String,
    /// Biological replicate, not an image-frame or crop identifier.
    pub biological_replicate_id: String,
    pub organism_id: String,
    pub treatment_id: Option<String>,
    pub coordinate_frame_id: String,
    /// Digest of the calibration manifest used by the graph and measurements.
    pub calibration_digest: String,
    /// Digest of the exact graph snapshot to which edge/node identities refer.
    pub source_graph_digest: String,
    /// Independent digest of the flow/tracer/dispersal source data and annotations.
    pub functional_measurement_digest: String,
    pub assay_protocol_id: String,
    pub assay_protocol_version: String,
    pub assay_protocol_digest: String,
    /// Monotonic clock identity; timestamps are not wall-clock Unix timestamps.
    pub timebase_id: String,
    pub window_start_ns: u64,
    pub window_end_ns: u64,
    /// Strictly sorted by observation ID; exact source order is represented by timestamps.
    pub observations: Vec<MycelialFunctionalObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MycelialFunctionalObservationError {
    InvalidSchemaVersion(u16),
    InvalidMetadata,
    TooManyObservations(usize),
    DuplicateOrUnorderedObservationId(String),
    InvalidObservation(String),
    GraphDigestMismatch,
    CoordinateFrameMismatch,
    CalibrationDigestMismatch,
    UnknownGraphEdge(String),
    UnknownGraphNode(String),
}

impl fmt::Display for MycelialFunctionalObservationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchemaVersion(version) => {
                write!(f, "unsupported mycelial functional observation schema version {version}")
            }
            Self::InvalidMetadata => {
                write!(f, "invalid mycelial functional observation set metadata")
            }
            Self::TooManyObservations(count) => write!(
                f,
                "functional observation count {count} exceeds {MAX_MYCELIAL_FUNCTIONAL_OBSERVATIONS}"
            ),
            Self::DuplicateOrUnorderedObservationId(id) => {
                write!(f, "functional observation IDs are duplicated or unordered at {id}")
            }
            Self::InvalidObservation(id) => {
                write!(f, "invalid functional observation {id}")
            }
            Self::GraphDigestMismatch => {
                write!(f, "functional observations do not bind to the supplied graph digest")
            }
            Self::CoordinateFrameMismatch => {
                write!(f, "functional observations use a different coordinate frame")
            }
            Self::CalibrationDigestMismatch => {
                write!(f, "functional observations use a different calibration digest")
            }
            Self::UnknownGraphEdge(id) => {
                write!(f, "functional flow observation references unknown graph edge {id}")
            }
            Self::UnknownGraphNode(id) => {
                write!(f, "functional tracer observation references unknown graph node {id}")
            }
        }
    }
}

impl Error for MycelialFunctionalObservationError {}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= MAX_MYCELIAL_FUNCTIONAL_ID_BYTES
        && value.bytes().all(|byte| !byte.is_ascii_control())
}

fn valid_digest(value: &str) -> bool {
    value.len() == DIGEST_HEX_BYTES && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_optional_string(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hash_string(hasher, value);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn hash_optional_quantity(
    hasher: &mut blake3::Hasher,
    value: Option<MycelialFunctionalQuantity>,
) {
    match value {
        Some(quantity) => {
            hasher.update(&[1]);
            hasher.update(&quantity.value.to_bits().to_le_bytes());
            hasher.update(&[match quantity.unit {
                MycelialFunctionalQuantityUnit::Nanogram => 0,
                MycelialFunctionalQuantityUnit::Microgram => 1,
                MycelialFunctionalQuantityUnit::Nanomole => 2,
                MycelialFunctionalQuantityUnit::Micromole => 3,
            }]);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

impl MycelialFunctionalObservationSet {
    pub fn validate(&self) -> Result<(), MycelialFunctionalObservationError> {
        if self.schema_version != MYCELIAL_FUNCTIONAL_OBSERVATION_SCHEMA_VERSION {
            return Err(MycelialFunctionalObservationError::InvalidSchemaVersion(
                self.schema_version,
            ));
        }
        if !valid_id(&self.dataset_id)
            || !valid_id(&self.session_id)
            || !valid_id(&self.biological_replicate_id)
            || !valid_id(&self.organism_id)
            || self.treatment_id.as_deref().is_some_and(|id| !valid_id(id))
            || !valid_id(&self.coordinate_frame_id)
            || !valid_digest(&self.calibration_digest)
            || !valid_digest(&self.source_graph_digest)
            || !valid_digest(&self.functional_measurement_digest)
            || !valid_id(&self.assay_protocol_id)
            || !valid_id(&self.assay_protocol_version)
            || !valid_digest(&self.assay_protocol_digest)
            || !valid_id(&self.timebase_id)
            || self.window_end_ns <= self.window_start_ns
            || self.observations.is_empty()
        {
            return Err(MycelialFunctionalObservationError::InvalidMetadata);
        }
        if self.observations.len() > MAX_MYCELIAL_FUNCTIONAL_OBSERVATIONS {
            return Err(MycelialFunctionalObservationError::TooManyObservations(
                self.observations.len(),
            ));
        }

        for pair in self.observations.windows(2) {
            if pair[0].observation_id >= pair[1].observation_id {
                return Err(
                    MycelialFunctionalObservationError::DuplicateOrUnorderedObservationId(
                        pair[1].observation_id.clone(),
                    ),
                );
            }
        }

        for observation in &self.observations {
            if !valid_id(&observation.observation_id)
                || observation.window_start_ns < self.window_start_ns
                || observation.window_end_ns > self.window_end_ns
                || observation.window_end_ns <= observation.window_start_ns
            {
                return Err(MycelialFunctionalObservationError::InvalidObservation(
                    observation.observation_id.clone(),
                ));
            }
            let valid_measurement = match &observation.measurement {
                MycelialFunctionalMeasurement::CytoplasmicFlow {
                    edge_id,
                    mean_speed_um_per_s,
                    trajectory_count,
                    ..
                } => {
                    valid_id(edge_id)
                        && *trajectory_count > 0
                        && mean_speed_um_per_s.is_none_or(|speed| {
                            speed.is_finite() && speed >= 0.0
                        })
                }
                MycelialFunctionalMeasurement::TracerTransfer {
                    source_node_id,
                    target_node_id,
                    tracer_id,
                    injection_time_ns,
                    arrival_time_ns,
                    injected_quantity,
                    recovered_quantity,
                } => {
                    valid_id(source_node_id)
                        && valid_id(target_node_id)
                        && source_node_id != target_node_id
                        && valid_id(tracer_id)
                        && *injection_time_ns >= observation.window_start_ns
                        && *injection_time_ns < observation.window_end_ns
                        && arrival_time_ns.is_none_or(|arrival| {
                            arrival >= *injection_time_ns && arrival < observation.window_end_ns
                        })
                        && [injected_quantity, recovered_quantity]
                            .into_iter()
                            .flatten()
                            .all(|quantity| quantity.value.is_finite() && quantity.value >= 0.0)
                }
                MycelialFunctionalMeasurement::MicrobialDispersal {
                    source_region_id,
                    target_region_id,
                    transported_cell_count,
                    first_arrival_time_ns,
                    liquid_saturation_fraction,
                } => {
                    valid_id(source_region_id)
                        && valid_id(target_region_id)
                        && source_region_id != target_region_id
                        && *transported_cell_count > 0
                        && first_arrival_time_ns.is_none_or(|arrival| {
                            arrival >= observation.window_start_ns
                                && arrival < observation.window_end_ns
                        })
                        && liquid_saturation_fraction.is_none_or(|fraction| {
                            fraction.is_finite() && (0.0..=1.0).contains(&fraction)
                        })
                }
            };
            if !valid_measurement {
                return Err(MycelialFunctionalObservationError::InvalidObservation(
                    observation.observation_id.clone(),
                ));
            }
        }
        Ok(())
    }

    /// Verify that graph-linked measurements refer to this exact annotated graph.
    /// Assay-local microbial regions are intentionally not coerced into graph nodes.
    pub fn validate_against(
        &self,
        skeleton: &AnnotatedMycelialSkeleton,
    ) -> Result<(), MycelialFunctionalObservationError> {
        self.validate()?;
        skeleton.validate().map_err(|_| {
            MycelialFunctionalObservationError::GraphDigestMismatch
        })?;
        if self.source_graph_digest != skeleton.canonical_digest_hex().map_err(|_| {
            MycelialFunctionalObservationError::GraphDigestMismatch
        })? {
            return Err(MycelialFunctionalObservationError::GraphDigestMismatch);
        }
        if self.coordinate_frame_id != skeleton.coordinate_frame_id {
            return Err(MycelialFunctionalObservationError::CoordinateFrameMismatch);
        }
        if self.calibration_digest != skeleton.calibration_digest {
            return Err(MycelialFunctionalObservationError::CalibrationDigestMismatch);
        }
        let edge_ids: std::collections::BTreeSet<_> =
            skeleton.segments.iter().map(|segment| segment.id.as_str()).collect();
        let node_ids: std::collections::BTreeSet<_> =
            skeleton.junctions.iter().map(|junction| junction.id.as_str()).collect();
        for observation in &self.observations {
            match &observation.measurement {
                MycelialFunctionalMeasurement::CytoplasmicFlow { edge_id, .. } => {
                    if !edge_ids.contains(edge_id.as_str()) {
                        return Err(MycelialFunctionalObservationError::UnknownGraphEdge(
                            edge_id.clone(),
                        ));
                    }
                }
                MycelialFunctionalMeasurement::TracerTransfer {
                    source_node_id,
                    target_node_id,
                    ..
                } => {
                    for node_id in [source_node_id, target_node_id] {
                        if !node_ids.contains(node_id.as_str()) {
                            return Err(MycelialFunctionalObservationError::UnknownGraphNode(
                                node_id.clone(),
                            ));
                        }
                    }
                }
                MycelialFunctionalMeasurement::MicrobialDispersal { .. } => {}
            }
        }
        Ok(())
    }

    pub fn canonical_digest_hex(
        &self,
    ) -> Result<String, MycelialFunctionalObservationError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-functional-observation-set-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        for value in [
            &self.dataset_id,
            &self.session_id,
            &self.biological_replicate_id,
            &self.organism_id,
            &self.coordinate_frame_id,
            &self.calibration_digest,
            &self.source_graph_digest,
            &self.functional_measurement_digest,
            &self.assay_protocol_id,
            &self.assay_protocol_version,
            &self.assay_protocol_digest,
            &self.timebase_id,
        ] {
            hash_string(&mut hasher, value);
        }
        hash_optional_string(&mut hasher, self.treatment_id.as_deref());
        hasher.update(&self.window_start_ns.to_le_bytes());
        hasher.update(&self.window_end_ns.to_le_bytes());
        hasher.update(&(self.observations.len() as u64).to_le_bytes());
        for observation in &self.observations {
            hash_string(&mut hasher, &observation.observation_id);
            hasher.update(&observation.window_start_ns.to_le_bytes());
            hasher.update(&observation.window_end_ns.to_le_bytes());
            match &observation.measurement {
                MycelialFunctionalMeasurement::CytoplasmicFlow {
                    edge_id,
                    direction,
                    mean_speed_um_per_s,
                    trajectory_count,
                } => {
                    hasher.update(&[0]);
                    hash_string(&mut hasher, edge_id);
                    hasher.update(&[match direction {
                        MycelialFunctionalFlowDirection::SourceToTarget => 0,
                        MycelialFunctionalFlowDirection::TargetToSource => 1,
                        MycelialFunctionalFlowDirection::Bidirectional => 2,
                        MycelialFunctionalFlowDirection::Unresolved => 3,
                    }]);
                    match mean_speed_um_per_s {
                        Some(speed) => {
                            hasher.update(&[1]);
                            hasher.update(&speed.to_bits().to_le_bytes());
                        }
                        None => {
                            hasher.update(&[0]);
                        }
                    }
                    hasher.update(&trajectory_count.to_le_bytes());
                }
                MycelialFunctionalMeasurement::TracerTransfer {
                    source_node_id,
                    target_node_id,
                    tracer_id,
                    injection_time_ns,
                    arrival_time_ns,
                    injected_quantity,
                    recovered_quantity,
                } => {
                    hasher.update(&[1]);
                    hash_string(&mut hasher, source_node_id);
                    hash_string(&mut hasher, target_node_id);
                    hash_string(&mut hasher, tracer_id);
                    hasher.update(&injection_time_ns.to_le_bytes());
                    match arrival_time_ns {
                        Some(arrival) => {
                            hasher.update(&[1]);
                            hasher.update(&arrival.to_le_bytes());
                        }
                        None => {
                            hasher.update(&[0]);
                        }
                    }
                    hash_optional_quantity(&mut hasher, *injected_quantity);
                    hash_optional_quantity(&mut hasher, *recovered_quantity);
                }
                MycelialFunctionalMeasurement::MicrobialDispersal {
                    source_region_id,
                    target_region_id,
                    transported_cell_count,
                    first_arrival_time_ns,
                    liquid_saturation_fraction,
                } => {
                    hasher.update(&[2]);
                    hash_string(&mut hasher, source_region_id);
                    hash_string(&mut hasher, target_region_id);
                    hasher.update(&transported_cell_count.to_le_bytes());
                    match first_arrival_time_ns {
                        Some(arrival) => {
                            hasher.update(&[1]);
                            hasher.update(&arrival.to_le_bytes());
                        }
                        None => {
                            hasher.update(&[0]);
                        }
                    }
                    match liquid_saturation_fraction {
                        Some(fraction) => {
                            hasher.update(&[1]);
                            hasher.update(&fraction.to_bits().to_le_bytes());
                        }
                        None => {
                            hasher.update(&[0]);
                        }
                    }
                }
            }
        }
        let digest = hasher.finalize();
        let mut output = String::with_capacity(64);
        use fmt::Write as _;
        for byte in digest.as_bytes() {
            let _ = write!(&mut output, "{byte:02x}");
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelial_reconstruction_metrics::{
        AnnotatedHyphaSegment, AnnotatedJunction, AnnotatedJunctionKind,
        MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION, ReconstructionDimensionality,
        ReconstructionLengthUnit, ReconstructionPoint3,
    };

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn skeleton() -> AnnotatedMycelialSkeleton {
        AnnotatedMycelialSkeleton {
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
        }
    }

    fn set() -> MycelialFunctionalObservationSet {
        let graph = skeleton();
        MycelialFunctionalObservationSet {
            schema_version: MYCELIAL_FUNCTIONAL_OBSERVATION_SCHEMA_VERSION,
            dataset_id: "dataset-1".to_string(),
            session_id: "session-1".to_string(),
            biological_replicate_id: "replicate-1".to_string(),
            organism_id: "Rhizophagus-irregularis".to_string(),
            treatment_id: Some("myristic-acid".to_string()),
            coordinate_frame_id: graph.coordinate_frame_id.clone(),
            calibration_digest: graph.calibration_digest.clone(),
            source_graph_digest: graph.canonical_digest_hex().unwrap(),
            functional_measurement_digest: digest('c'),
            assay_protocol_id: "kymograph-flow-tracking".to_string(),
            assay_protocol_version: "1.0".to_string(),
            assay_protocol_digest: digest('d'),
            timebase_id: "session-monotonic-ns".to_string(),
            window_start_ns: 0,
            window_end_ns: 1_000,
            observations: vec![
                MycelialFunctionalObservation {
                    observation_id: "flow-001".to_string(),
                    window_start_ns: 0,
                    window_end_ns: 500,
                    measurement: MycelialFunctionalMeasurement::CytoplasmicFlow {
                        edge_id: "edge-AB".to_string(),
                        direction: MycelialFunctionalFlowDirection::Bidirectional,
                        mean_speed_um_per_s: Some(3.0),
                        trajectory_count: 8,
                    },
                },
                MycelialFunctionalObservation {
                    observation_id: "tracer-001".to_string(),
                    window_start_ns: 500,
                    window_end_ns: 1_000,
                    measurement: MycelialFunctionalMeasurement::TracerTransfer {
                        source_node_id: "A".to_string(),
                        target_node_id: "B".to_string(),
                        tracer_id: "15N-ammonium".to_string(),
                        injection_time_ns: 500,
                        arrival_time_ns: Some(900),
                        injected_quantity: Some(MycelialFunctionalQuantity {
                            value: 10.0,
                            unit: MycelialFunctionalQuantityUnit::Nanomole,
                        }),
                        recovered_quantity: Some(MycelialFunctionalQuantity {
                            value: 2.5,
                            unit: MycelialFunctionalQuantityUnit::Nanomole,
                        }),
                    },
                },
            ],
        }
    }

    #[test]
    fn functional_observation_set_binds_to_exact_graph_and_has_stable_digest() {
        let graph = skeleton();
        let observations = set();
        observations.validate_against(&graph).unwrap();
        assert_eq!(
            observations.canonical_digest_hex().unwrap(),
            observations.canonical_digest_hex().unwrap()
        );
    }

    #[test]
    fn measured_flow_change_changes_functional_evidence_digest() {
        let mut observations = set();
        let before = observations.canonical_digest_hex().unwrap();
        if let MycelialFunctionalMeasurement::CytoplasmicFlow {
            mean_speed_um_per_s,
            ..
        } = &mut observations.observations[0].measurement
        {
            *mean_speed_um_per_s = Some(4.0);
        }
        assert_ne!(before, observations.canonical_digest_hex().unwrap());
    }

    #[test]
    fn graph_linked_flow_rejects_unknown_edge_id() {
        let graph = skeleton();
        let mut observations = set();
        if let MycelialFunctionalMeasurement::CytoplasmicFlow { edge_id, .. } =
            &mut observations.observations[0].measurement
        {
            *edge_id = "not-in-graph".to_string();
        }
        assert!(matches!(
            observations.validate_against(&graph),
            Err(MycelialFunctionalObservationError::UnknownGraphEdge(id))
                if id == "not-in-graph"
        ));
    }

    #[test]
    fn graph_linked_tracer_rejects_unknown_node_id() {
        let graph = skeleton();
        let mut observations = set();
        if let MycelialFunctionalMeasurement::TracerTransfer { target_node_id, .. } =
            &mut observations.observations[1].measurement
        {
            *target_node_id = "unknown-node".to_string();
        }
        assert!(matches!(
            observations.validate_against(&graph),
            Err(MycelialFunctionalObservationError::UnknownGraphNode(id))
                if id == "unknown-node"
        ));
    }

    #[test]
    fn tracer_arrival_before_injection_is_rejected() {
        let mut observations = set();
        if let MycelialFunctionalMeasurement::TracerTransfer {
            arrival_time_ns, ..
        } = &mut observations.observations[1].measurement
        {
            *arrival_time_ns = Some(400);
        }
        assert!(matches!(
            observations.validate(),
            Err(MycelialFunctionalObservationError::InvalidObservation(id))
                if id == "tracer-001"
        ));
    }

    #[test]
    fn quantity_and_saturation_values_are_range_checked() {
        let mut observations = set();
        if let MycelialFunctionalMeasurement::TracerTransfer {
            injected_quantity, ..
        } = &mut observations.observations[1].measurement
        {
            *injected_quantity = Some(MycelialFunctionalQuantity {
                value: f64::NAN,
                unit: MycelialFunctionalQuantityUnit::Nanomole,
            });
        }
        assert!(observations.validate().is_err());

        let mut microbial = set();
        microbial.observations.push(MycelialFunctionalObservation {
            observation_id: "microbe-001".to_string(),
            window_start_ns: 0,
            window_end_ns: 500,
            measurement: MycelialFunctionalMeasurement::MicrobialDispersal {
                source_region_id: "region-A".to_string(),
                target_region_id: "region-B".to_string(),
                transported_cell_count: 12,
                first_arrival_time_ns: Some(300),
                liquid_saturation_fraction: Some(1.2),
            },
        });
        microbial.observations.sort_by(|left, right| {
            left.observation_id.cmp(&right.observation_id)
        });
        assert!(microbial.validate().is_err());
    }

    #[test]
    fn duplicate_observation_ids_are_rejected() {
        let mut observations = set();
        observations.observations[1].observation_id = "flow-001".to_string();
        observations.observations.sort_by(|left, right| {
            left.observation_id.cmp(&right.observation_id)
        });
        assert!(matches!(
            observations.validate(),
            Err(MycelialFunctionalObservationError::DuplicateOrUnorderedObservationId(id))
                if id == "flow-001"
        ));
    }
}
