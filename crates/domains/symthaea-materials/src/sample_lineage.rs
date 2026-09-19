// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Append-only physical sample/process lineage for experimental materials evidence.
//!
//! A physical specimen is not interchangeable with its intended composition string.
//! Precursor lots, process history, splitting/merging, geometry, scale, instruments,
//! calibration, raw files, and analysis lineage are first-class provenance.

use crate::conditioned_property::PropertyArtifactRef;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

const SHA256_HEX_LEN: usize = 64;

/// Scale at which a physical sample/result exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentalScale {
    /// Small coupon/specimen.
    Coupon,
    /// Laboratory batch.
    LabBatch,
    /// Representative component/device.
    Device,
    /// Pilot-process/pilot-line material.
    Pilot,
    /// Industrial/full-scale material/process.
    Industrial,
}

/// One precursor/feedstock lot used to create a physical sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrecursorLot {
    /// Stable material/feedstock identifier.
    pub material_id: String,
    /// Supplier/manufacturer lot identifier.
    pub lot_id: String,
    /// Supplier/manufacturer identifier.
    pub supplier_id: String,
    /// Declared purity fraction in (0,1], when known.
    pub purity_fraction: Option<f64>,
    /// Exact certificate/specification/source artifact.
    pub source_artifact: PropertyArtifactRef,
}

/// Broad transformation represented by a lineage node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SampleNodeKind {
    /// New synthesized/mixed batch from precursor lots.
    SynthesizedBatch,
    /// Heat treatment, anneal, mechanical activation, curing, or other reprocessing.
    Processed,
    /// Child specimen split from one parent.
    SplitSpecimen,
    /// Material merged from multiple physical parents.
    MergedBatch,
    /// Surface coating/deposition/interface-processing step.
    Coated,
    /// Material integrated into a representative device/component.
    DeviceIntegrated,
}

/// One append-only node in the physical sample/process DAG.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SampleLineageNode {
    /// Deterministic lineage-node identity derived from semantic provenance.
    pub node_id: String,
    /// Exact intended MAT-007 material-subject identity.
    pub intended_subject_identity: String,
    /// Human/lab-facing physical sample or batch identifier.
    pub sample_id: String,
    /// Physical scale.
    pub scale: ExperimentalScale,
    /// Transformation/node class.
    pub kind: SampleNodeKind,
    /// Parent physical lineage nodes.
    pub parent_node_ids: Vec<String>,
    /// Precursor lots introduced at this step.
    pub precursor_lots: Vec<PrecursorLot>,
    /// Exact synthesis/process protocol artifact when applicable.
    pub process_artifact: Option<PropertyArtifactRef>,
    /// Measured mass in kilograms, when known.
    pub mass_kg: Option<f64>,
    /// Bound geometry/dimensions identifier.
    pub geometry_id: Option<String>,
    /// Bound orientation/texture/loading-direction identifier.
    pub orientation_id: Option<String>,
    /// Bound environment/chamber/run-condition artifact when needed.
    pub environment_artifact: Option<PropertyArtifactRef>,
}

impl SampleLineageNode {
    /// Construct a node and derive its deterministic lineage identity.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        intended_subject_identity: String,
        sample_id: String,
        scale: ExperimentalScale,
        kind: SampleNodeKind,
        parent_node_ids: Vec<String>,
        precursor_lots: Vec<PrecursorLot>,
        process_artifact: Option<PropertyArtifactRef>,
        mass_kg: Option<f64>,
        geometry_id: Option<String>,
        orientation_id: Option<String>,
        environment_artifact: Option<PropertyArtifactRef>,
    ) -> Result<Self, SampleLineageError> {
        let mut node = Self {
            node_id: String::new(),
            intended_subject_identity,
            sample_id,
            scale,
            kind,
            parent_node_ids,
            precursor_lots,
            process_artifact,
            mass_kg,
            geometry_id,
            orientation_id,
            environment_artifact,
        };
        node.validate_without_id()?;
        node.node_id = node.derived_identity();
        Ok(node)
    }

    /// Validate the stored node including identity consistency.
    pub fn validate(&self) -> Result<(), SampleLineageError> {
        self.validate_without_id()?;
        if self.node_id != self.derived_identity() {
            return Err(SampleLineageError::NodeIdentityMismatch);
        }
        Ok(())
    }

    fn validate_without_id(&self) -> Result<(), SampleLineageError> {
        nonempty("intended_subject_identity", &self.intended_subject_identity)?;
        nonempty("sample_id", &self.sample_id)?;
        unique_nonempty(&self.parent_node_ids, "parent_node_id")?;
        if let Some(value) = self.mass_kg {
            positive("mass_kg", value)?;
        }
        if let Some(value) = &self.geometry_id {
            nonempty("geometry_id", value)?;
        }
        if let Some(value) = &self.orientation_id {
            nonempty("orientation_id", value)?;
        }
        if let Some(value) = &self.process_artifact {
            artifact(value)?;
        }
        if let Some(value) = &self.environment_artifact {
            artifact(value)?;
        }

        let mut lots = HashSet::new();
        for precursor in &self.precursor_lots {
            validate_precursor(precursor)?;
            let key = format!(
                "{}|{}|{}",
                precursor.material_id, precursor.supplier_id, precursor.lot_id
            );
            if !lots.insert(key) {
                return Err(SampleLineageError::DuplicatePrecursorLot);
            }
        }

        match self.kind {
            SampleNodeKind::SynthesizedBatch => {
                if self.precursor_lots.is_empty() {
                    return Err(SampleLineageError::SynthesizedBatchWithoutPrecursors);
                }
                if self.process_artifact.is_none() {
                    return Err(SampleLineageError::ProcessNodeWithoutProtocol);
                }
            }
            SampleNodeKind::Processed | SampleNodeKind::Coated | SampleNodeKind::DeviceIntegrated => {
                if self.parent_node_ids.len() != 1 {
                    return Err(SampleLineageError::SingleParentNodeRequired);
                }
                if self.process_artifact.is_none() {
                    return Err(SampleLineageError::ProcessNodeWithoutProtocol);
                }
            }
            SampleNodeKind::SplitSpecimen => {
                if self.parent_node_ids.len() != 1 {
                    return Err(SampleLineageError::SingleParentNodeRequired);
                }
            }
            SampleNodeKind::MergedBatch => {
                if self.parent_node_ids.len() < 2 {
                    return Err(SampleLineageError::MergeRequiresMultipleParents);
                }
                if self.process_artifact.is_none() {
                    return Err(SampleLineageError::ProcessNodeWithoutProtocol);
                }
            }
        }
        Ok(())
    }

    fn derived_identity(&self) -> String {
        let mut parents = self.parent_node_ids.clone();
        parents.sort();
        let mut precursors = self
            .precursor_lots
            .iter()
            .map(precursor_key)
            .collect::<Vec<_>>();
        precursors.sort();
        format!(
            "sample-lineage:v1|subject={}|sample={}|scale={}|kind={}|parents=[{}]|precursors=[{}]|process={}|mass={}|geom={}|orient={}|env={}",
            token(&self.intended_subject_identity),
            token(&self.sample_id),
            scale_key(self.scale),
            kind_key(self.kind),
            parents.iter().map(|value| token(value)).collect::<Vec<_>>().join(","),
            precursors.join(","),
            artifact_key(self.process_artifact.as_ref()),
            optional_float_key(self.mass_kg),
            optional_token(self.geometry_id.as_deref()),
            optional_token(self.orientation_id.as_deref()),
            artifact_key(self.environment_artifact.as_ref())
        )
    }
}

/// Append-only physical lineage DAG.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExperimentalSampleLineage {
    nodes: Vec<SampleLineageNode>,
}

impl ExperimentalSampleLineage {
    /// Create an empty lineage.
    pub fn new() -> Self {
        Self::default()
    }

    /// Immutable append-order view.
    pub fn nodes(&self) -> &[SampleLineageNode] {
        &self.nodes
    }

    /// Append a node after validating that every parent already exists.
    ///
    /// Reprocessing must create a new node; existing nodes are never overwritten.
    pub fn append(&mut self, node: SampleLineageNode) -> Result<(), SampleLineageError> {
        node.validate()?;
        if self.nodes.iter().any(|existing| existing.node_id == node.node_id) {
            return Err(SampleLineageError::DuplicateNode(node.node_id));
        }
        let known: HashSet<&str> = self.nodes.iter().map(|node| node.node_id.as_str()).collect();
        for parent in &node.parent_node_ids {
            if !known.contains(parent.as_str()) {
                return Err(SampleLineageError::UnknownParent(parent.clone()));
            }
        }
        self.nodes.push(node);
        Ok(())
    }

    /// Find one exact lineage node.
    pub fn node(&self, node_id: &str) -> Option<&SampleLineageNode> {
        self.nodes.iter().find(|node| node.node_id == node_id)
    }

    /// Whether two physical nodes represent independent sample lineages.
    ///
    /// Two nodes are not independent if they are the same node, if one descends from
    /// the other, or if they share any physical ancestor (for example two coupons
    /// split from the same parent specimen).
    pub fn physically_independent(
        &self,
        left_node_id: &str,
        right_node_id: &str,
    ) -> Result<bool, SampleLineageError> {
        if self.node(left_node_id).is_none() {
            return Err(SampleLineageError::UnknownNode(left_node_id.to_string()));
        }
        if self.node(right_node_id).is_none() {
            return Err(SampleLineageError::UnknownNode(right_node_id.to_string()));
        }
        if left_node_id == right_node_id {
            return Ok(false);
        }
        let left = self.ancestor_closure(left_node_id)?;
        let right = self.ancestor_closure(right_node_id)?;
        Ok(left.is_disjoint(&right))
    }

    /// Require all supplied nodes to be at the same experimental scale.
    pub fn require_same_scale(&self, node_ids: &[String]) -> Result<ExperimentalScale, SampleLineageError> {
        if node_ids.is_empty() {
            return Err(SampleLineageError::EmptyScaleComparison);
        }
        let first = self
            .node(&node_ids[0])
            .ok_or_else(|| SampleLineageError::UnknownNode(node_ids[0].clone()))?
            .scale;
        for node_id in &node_ids[1..] {
            let node = self
                .node(node_id)
                .ok_or_else(|| SampleLineageError::UnknownNode(node_id.clone()))?;
            if node.scale != first {
                return Err(SampleLineageError::ScaleMismatch);
            }
        }
        Ok(first)
    }

    fn ancestor_closure(&self, node_id: &str) -> Result<HashSet<String>, SampleLineageError> {
        let index: HashMap<&str, &SampleLineageNode> =
            self.nodes.iter().map(|node| (node.node_id.as_str(), node)).collect();
        let mut seen = HashSet::new();
        let mut stack = vec![node_id.to_string()];
        while let Some(current) = stack.pop() {
            if !seen.insert(current.clone()) {
                continue;
            }
            let node = index
                .get(current.as_str())
                .ok_or_else(|| SampleLineageError::UnknownNode(current.clone()))?;
            for parent in &node.parent_node_ids {
                stack.push(parent.clone());
            }
        }
        Ok(seen)
    }
}

/// Status of one characterization/measurement run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CharacterizationStatus {
    /// Run completed and produced usable data.
    Completed,
    /// Run completed with a scientifically meaningful null/non-detection result.
    ScientificNull,
    /// Run invalid due to protocol/sample/data issue.
    InvalidRun,
    /// Instrument or calibration failure invalidated the run.
    InstrumentFault,
}

/// Provenance-complete characterization run on one physical lineage node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CharacterizationRun {
    /// Deterministic run identity.
    pub run_id: String,
    /// Physical sample lineage node measured.
    pub sample_node_id: String,
    /// Instrument identifier.
    pub instrument_id: String,
    /// Calibration artifact applicable to this run.
    pub calibration_artifact: PropertyArtifactRef,
    /// Exact measurement/test protocol.
    pub protocol_artifact: PropertyArtifactRef,
    /// Immutable raw-data artifact.
    pub raw_data_artifact: PropertyArtifactRef,
    /// Optional derived/processed-data artifact.
    pub derived_data_artifact: Option<PropertyArtifactRef>,
    /// Exact analysis code/notebook/workflow artifact.
    pub analysis_artifact: PropertyArtifactRef,
    /// Run status.
    pub status: CharacterizationStatus,
    /// MAT-008 conditioned-property observation IDs emitted by this run.
    pub produced_observation_ids: Vec<String>,
}

impl CharacterizationRun {
    /// Construct and derive a deterministic run identity.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sample_node_id: String,
        instrument_id: String,
        calibration_artifact: PropertyArtifactRef,
        protocol_artifact: PropertyArtifactRef,
        raw_data_artifact: PropertyArtifactRef,
        derived_data_artifact: Option<PropertyArtifactRef>,
        analysis_artifact: PropertyArtifactRef,
        status: CharacterizationStatus,
        produced_observation_ids: Vec<String>,
    ) -> Result<Self, SampleLineageError> {
        let mut run = Self {
            run_id: String::new(),
            sample_node_id,
            instrument_id,
            calibration_artifact,
            protocol_artifact,
            raw_data_artifact,
            derived_data_artifact,
            analysis_artifact,
            status,
            produced_observation_ids,
        };
        run.validate_without_id()?;
        run.run_id = run.derived_identity();
        Ok(run)
    }

    /// Validate a stored characterization run.
    pub fn validate(&self) -> Result<(), SampleLineageError> {
        self.validate_without_id()?;
        if self.run_id != self.derived_identity() {
            return Err(SampleLineageError::CharacterizationIdentityMismatch);
        }
        Ok(())
    }

    fn validate_without_id(&self) -> Result<(), SampleLineageError> {
        nonempty("sample_node_id", &self.sample_node_id)?;
        nonempty("instrument_id", &self.instrument_id)?;
        artifact(&self.calibration_artifact)?;
        artifact(&self.protocol_artifact)?;
        artifact(&self.raw_data_artifact)?;
        if let Some(value) = &self.derived_data_artifact {
            artifact(value)?;
        }
        artifact(&self.analysis_artifact)?;
        unique_nonempty(&self.produced_observation_ids, "produced_observation_id")?;
        if matches!(self.status, CharacterizationStatus::Completed | CharacterizationStatus::ScientificNull)
            && self.produced_observation_ids.is_empty()
        {
            return Err(SampleLineageError::CompletedCharacterizationWithoutObservation);
        }
        Ok(())
    }

    fn derived_identity(&self) -> String {
        format!(
            "characterization:v1|sample={}|instrument={}|cal={}|protocol={}|raw={}|derived={}|analysis={}",
            token(&self.sample_node_id),
            token(&self.instrument_id),
            self.calibration_artifact.artifact_sha256.to_ascii_lowercase(),
            self.protocol_artifact.artifact_sha256.to_ascii_lowercase(),
            self.raw_data_artifact.artifact_sha256.to_ascii_lowercase(),
            artifact_key(self.derived_data_artifact.as_ref()),
            self.analysis_artifact.artifact_sha256.to_ascii_lowercase()
        )
    }
}

fn validate_precursor(value: &PrecursorLot) -> Result<(), SampleLineageError> {
    nonempty("precursor material_id", &value.material_id)?;
    nonempty("precursor lot_id", &value.lot_id)?;
    nonempty("precursor supplier_id", &value.supplier_id)?;
    if let Some(purity) = value.purity_fraction {
        finite("purity_fraction", purity)?;
        if purity <= 0.0 || purity > 1.0 {
            return Err(SampleLineageError::FractionOutOfRange {
                field: "purity_fraction",
                value: purity,
            });
        }
    }
    artifact(&value.source_artifact)
}

fn precursor_key(value: &PrecursorLot) -> String {
    format!(
        "{}:{}:{}:{}:{}",
        token(&value.material_id),
        token(&value.supplier_id),
        token(&value.lot_id),
        optional_float_key(value.purity_fraction),
        value.source_artifact.artifact_sha256.to_ascii_lowercase()
    )
}

fn artifact(value: &PropertyArtifactRef) -> Result<(), SampleLineageError> {
    nonempty("artifact source_id", &value.source_id)?;
    sha256(&value.artifact_sha256)
}

fn artifact_key(value: Option<&PropertyArtifactRef>) -> String {
    value
        .map(|artifact| format!("{}:{}", token(&artifact.source_id), artifact.artifact_sha256.to_ascii_lowercase()))
        .unwrap_or_else(|| "none".to_string())
}

fn unique_nonempty(values: &[String], field: &'static str) -> Result<(), SampleLineageError> {
    let mut seen = HashSet::new();
    for value in values {
        nonempty(field, value)?;
        if !seen.insert(value.as_str()) {
            return Err(SampleLineageError::DuplicateStringValue {
                field,
                value: value.clone(),
            });
        }
    }
    Ok(())
}

fn sha256(value: &str) -> Result<(), SampleLineageError> {
    if value.len() != SHA256_HEX_LEN || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(SampleLineageError::InvalidSha256)
    } else {
        Ok(())
    }
}

fn nonempty(field: &'static str, value: &str) -> Result<(), SampleLineageError> {
    if value.trim().is_empty() {
        Err(SampleLineageError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn finite(field: &'static str, value: f64) -> Result<(), SampleLineageError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SampleLineageError::NonFiniteValue { field, value })
    }
}

fn positive(field: &'static str, value: f64) -> Result<(), SampleLineageError> {
    finite(field, value)?;
    if value <= 0.0 {
        Err(SampleLineageError::NonPositiveValue { field, value })
    } else {
        Ok(())
    }
}

fn optional_float_key(value: Option<f64>) -> String {
    value
        .map(|value| format!("0x{:016x}", value.to_bits()))
        .unwrap_or_else(|| "none".to_string())
}

fn optional_token(value: Option<&str>) -> String {
    value.map(token).unwrap_or_else(|| "none".to_string())
}

fn scale_key(value: ExperimentalScale) -> &'static str {
    match value {
        ExperimentalScale::Coupon => "coupon",
        ExperimentalScale::LabBatch => "lab-batch",
        ExperimentalScale::Device => "device",
        ExperimentalScale::Pilot => "pilot",
        ExperimentalScale::Industrial => "industrial",
    }
}

fn kind_key(value: SampleNodeKind) -> &'static str {
    match value {
        SampleNodeKind::SynthesizedBatch => "synthesized-batch",
        SampleNodeKind::Processed => "processed",
        SampleNodeKind::SplitSpecimen => "split-specimen",
        SampleNodeKind::MergedBatch => "merged-batch",
        SampleNodeKind::Coated => "coated",
        SampleNodeKind::DeviceIntegrated => "device-integrated",
    }
}

fn token(value: &str) -> String {
    let mut output = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            output.push(byte as char);
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}

/// Sample-lineage or characterization validation failure.
#[derive(Debug, Clone, PartialEq)]
pub enum SampleLineageError {
    /// Required text field was empty.
    EmptyField(&'static str),
    /// Numeric value was NaN/infinite.
    NonFiniteValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Numeric value had to be strictly positive.
    NonPositiveValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Fraction was outside its valid range.
    FractionOutOfRange {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// SHA-256 binding was malformed.
    InvalidSha256,
    /// Duplicate set-like string value.
    DuplicateStringValue {
        /// Field name.
        field: &'static str,
        /// Repeated value.
        value: String,
    },
    /// Same precursor lot was introduced twice in one node.
    DuplicatePrecursorLot,
    /// Stored node identity did not match its semantic provenance.
    NodeIdentityMismatch,
    /// Duplicate lineage node.
    DuplicateNode(String),
    /// Parent node was not already present in the append-only DAG.
    UnknownParent(String),
    /// Requested lineage node did not exist.
    UnknownNode(String),
    /// Synthesized batch omitted precursor lots.
    SynthesizedBatchWithoutPrecursors,
    /// Process-bearing node omitted protocol artifact.
    ProcessNodeWithoutProtocol,
    /// Node class requires exactly one physical parent.
    SingleParentNodeRequired,
    /// Merge operation requires at least two physical parents.
    MergeRequiresMultipleParents,
    /// No nodes were supplied for scale comparison.
    EmptyScaleComparison,
    /// Experimental evidence from different physical scales was mixed without translation.
    ScaleMismatch,
    /// Stored characterization identity was inconsistent.
    CharacterizationIdentityMismatch,
    /// Completed/null characterization did not bind an emitted observation.
    CompletedCharacterizationWithoutObservation,
}

#[cfg(test)]
mod tests {
    use super::*;

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C64: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn artifact_ref(source: &str, hash: &str) -> PropertyArtifactRef {
        PropertyArtifactRef {
            source_id: source.to_string(),
            artifact_sha256: hash.to_string(),
        }
    }

    fn precursor(lot: &str) -> PrecursorLot {
        PrecursorLot {
            material_id: "Ti-feedstock".to_string(),
            lot_id: lot.to_string(),
            supplier_id: "supplier-a".to_string(),
            purity_fraction: Some(0.999),
            source_artifact: artifact_ref("coa", A64),
        }
    }

    fn root(sample: &str, lot: &str) -> SampleLineageNode {
        SampleLineageNode::new(
            "material-subject:v1|fixture".to_string(),
            sample.to_string(),
            ExperimentalScale::LabBatch,
            SampleNodeKind::SynthesizedBatch,
            vec![],
            vec![precursor(lot)],
            Some(artifact_ref("synthesis-protocol", B64)),
            Some(0.01),
            None,
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn split_specimens_are_not_independent_replications() {
        let mut lineage = ExperimentalSampleLineage::new();
        let parent = root("batch-a", "lot-1");
        let parent_id = parent.node_id.clone();
        lineage.append(parent).unwrap();
        let left = SampleLineageNode::new(
            "material-subject:v1|fixture".to_string(),
            "coupon-left".to_string(),
            ExperimentalScale::Coupon,
            SampleNodeKind::SplitSpecimen,
            vec![parent_id.clone()],
            vec![],
            None,
            Some(0.001),
            Some("coupon-geometry".to_string()),
            None,
            None,
        )
        .unwrap();
        let right = SampleLineageNode::new(
            "material-subject:v1|fixture".to_string(),
            "coupon-right".to_string(),
            ExperimentalScale::Coupon,
            SampleNodeKind::SplitSpecimen,
            vec![parent_id],
            vec![],
            None,
            Some(0.001),
            Some("coupon-geometry".to_string()),
            None,
            None,
        )
        .unwrap();
        let left_id = left.node_id.clone();
        let right_id = right.node_id.clone();
        lineage.append(left).unwrap();
        lineage.append(right).unwrap();
        assert!(!lineage.physically_independent(&left_id, &right_id).unwrap());
    }

    #[test]
    fn separate_root_batches_are_independent_physical_lineages() {
        let mut lineage = ExperimentalSampleLineage::new();
        let a = root("batch-a", "lot-1");
        let b = root("batch-b", "lot-1");
        let a_id = a.node_id.clone();
        let b_id = b.node_id.clone();
        lineage.append(a).unwrap();
        lineage.append(b).unwrap();
        assert!(lineage.physically_independent(&a_id, &b_id).unwrap());
    }

    #[test]
    fn precursor_lot_change_creates_distinct_physical_identity() {
        let a = root("batch-same-label", "lot-1");
        let b = root("batch-same-label", "lot-2");
        assert_ne!(a.node_id, b.node_id);
    }

    #[test]
    fn reprocessing_creates_new_node_instead_of_overwriting_parent() {
        let parent = root("batch-a", "lot-1");
        let child = SampleLineageNode::new(
            parent.intended_subject_identity.clone(),
            "batch-a-annealed".to_string(),
            ExperimentalScale::LabBatch,
            SampleNodeKind::Processed,
            vec![parent.node_id.clone()],
            vec![],
            Some(artifact_ref("anneal-protocol", C64)),
            parent.mass_kg,
            None,
            None,
            None,
        )
        .unwrap();
        assert_ne!(parent.node_id, child.node_id);
    }

    #[test]
    fn scale_mismatch_is_explicit() {
        let mut lineage = ExperimentalSampleLineage::new();
        let batch = root("batch-a", "lot-1");
        let batch_id = batch.node_id.clone();
        lineage.append(batch).unwrap();
        let coupon = SampleLineageNode::new(
            "material-subject:v1|fixture".to_string(),
            "coupon-a".to_string(),
            ExperimentalScale::Coupon,
            SampleNodeKind::SplitSpecimen,
            vec![batch_id.clone()],
            vec![],
            None,
            Some(0.001),
            None,
            None,
            None,
        )
        .unwrap();
        let coupon_id = coupon.node_id.clone();
        lineage.append(coupon).unwrap();
        assert_eq!(
            lineage.require_same_scale(&[batch_id, coupon_id]),
            Err(SampleLineageError::ScaleMismatch)
        );
    }

    #[test]
    fn completed_characterization_requires_bound_observation() {
        let result = CharacterizationRun::new(
            "sample-node".to_string(),
            "xrd-1".to_string(),
            artifact_ref("calibration", A64),
            artifact_ref("xrd-protocol", B64),
            artifact_ref("raw-xrd", C64),
            None,
            artifact_ref("analysis-code", A64),
            CharacterizationStatus::Completed,
            vec![],
        );
        assert_eq!(
            result.unwrap_err(),
            SampleLineageError::CompletedCharacterizationWithoutObservation
        );
    }
}
