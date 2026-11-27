// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned, provenance-bound process-subsystem graph manifests.
//!
//! The graph describes declared engineering structure for feature grouping and
//! graph-model inputs. It does not infer physical causality. Material-flow and
//! control-dependency relations are required to be acyclic; thermal/electrical
//! couplings may be cyclic and are validated as symmetric relations.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const PROCESS_SUBSYSTEM_GRAPH_SCHEMA_V1: &str =
    "luminous.semi-bench.process-subsystem-graph.v1";
pub const PROCESS_SUBSYSTEM_NODE_SCHEMA_V1: &str =
    "luminous.semi-bench.process-subsystem-node.v1";
pub const PROCESS_SENSOR_CHANNEL_SCHEMA_V1: &str =
    "luminous.semi-bench.process-sensor-channel.v1";
pub const PROCESS_SENSOR_MEMBERSHIP_SCHEMA_V1: &str =
    "luminous.semi-bench.process-sensor-membership.v1";
pub const PROCESS_SUBSYSTEM_RELATION_SCHEMA_V1: &str =
    "luminous.semi-bench.process-subsystem-relation.v1";
pub const PROCESS_CONTEXT_DIMENSION_SCHEMA_V1: &str =
    "luminous.semi-bench.process-context-dimension.v1";
pub const PROCESS_SUBSYSTEM_GRAPH_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.process-subsystem-graph-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessSubsystemKindV1 {
    GasDelivery,
    PlasmaRf,
    ThermalControl,
    VacuumExhaust,
    MotionHandling,
    Endpoint,
    Material,
    MeasurementMetrology,
    Other,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessSubsystemRelationKindV1 {
    MaterialProcessFlow,
    ControlDependency,
    MeasurementDependency,
    ThermalCoupling,
    ElectricalCoupling,
    Other,
}

impl ProcessSubsystemRelationKindV1 {
    fn requires_acyclic_order(self) -> bool {
        matches!(
            self,
            Self::MaterialProcessFlow | Self::ControlDependency | Self::MeasurementDependency
        )
    }

    fn is_symmetric(self) -> bool {
        matches!(self, Self::ThermalCoupling | Self::ElectricalCoupling)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessContextRoleV1 {
    Tool,
    Chamber,
    RecipeRevision,
    ProductFamily,
    Lot,
    Material,
    ProcessStep,
    Facility,
    Other,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessSubsystemGraphStatusV1 {
    Complete,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessSubsystemNodeV1 {
    pub schema: String,
    pub subsystem_id: String,
    pub process_stage_id: String,
    pub subsystem_kind: ProcessSubsystemKindV1,
    pub source_artifact_sha256: String,
    pub engineering_evidence_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessSensorChannelV1 {
    pub schema: String,
    pub sensor_id: String,
    pub unit: String,
    pub clock_domain_id: String,
    pub feature_schema_sha256: String,
    pub source_artifact_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessSensorMembershipV1 {
    pub schema: String,
    pub membership_id: String,
    pub sensor_id: String,
    pub subsystem_id: String,
    pub transform_sha256: String,
    pub membership_evidence_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessSubsystemRelationV1 {
    pub schema: String,
    pub relation_id: String,
    pub source_subsystem_id: String,
    pub target_subsystem_id: String,
    pub relation_kind: ProcessSubsystemRelationKindV1,
    /// Evidence that this relation is declared in the engineering model.
    /// It is not a causal-effect estimate or proof of a physical intervention.
    pub engineering_evidence_sha256: String,
    pub relation_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessContextDimensionV1 {
    pub schema: String,
    pub context_dimension_id: String,
    pub context_role: ProcessContextRoleV1,
    pub feature_id: String,
    pub vocabulary_sha256: String,
    pub source_artifact_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessSubsystemGraphManifestV1 {
    pub schema: String,
    pub graph_id: String,
    pub feature_schema_sha256: String,
    pub inventory_complete: bool,
    pub subsystems: Vec<ProcessSubsystemNodeV1>,
    pub sensor_channels: Vec<ProcessSensorChannelV1>,
    pub memberships: Vec<ProcessSensorMembershipV1>,
    pub relations: Vec<ProcessSubsystemRelationV1>,
    pub context_dimensions: Vec<ProcessContextDimensionV1>,
    pub graph_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessSubsystemGraphReportV1 {
    pub schema: String,
    pub graph_id: String,
    pub graph_sha256: String,
    pub status: ProcessSubsystemGraphStatusV1,
    pub subsystem_count: usize,
    pub sensor_channel_count: usize,
    pub membership_count: usize,
    pub relation_count: usize,
    pub context_dimension_count: usize,
    pub process_flow_relation_count: usize,
    pub control_dependency_relation_count: usize,
    pub measurement_dependency_relation_count: usize,
    pub coupling_relation_count: usize,
    pub inventory_complete: bool,
    pub sensor_membership_resolves: bool,
    pub relation_endpoints_resolve: bool,
    pub context_features_are_separate_from_sensor_channels: bool,
    pub directed_dependency_graph_is_acyclic: bool,
    pub physical_causality_proven: bool,
    pub equipment_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct ProcessSubsystemGraphErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), ProcessSubsystemGraphErrorV1> {
    if value.trim().is_empty() {
        return Err(ProcessSubsystemGraphErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, ProcessSubsystemGraphErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        ProcessSubsystemGraphErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn relation_digest(
    relation: &ProcessSubsystemRelationV1,
) -> Result<String, ProcessSubsystemGraphErrorV1> {
    let mut canonical = relation.clone();
    canonical.relation_sha256.clear();
    if relation.relation_kind.is_symmetric()
        && canonical.source_subsystem_id > canonical.target_subsystem_id
    {
        std::mem::swap(
            &mut canonical.source_subsystem_id,
            &mut canonical.target_subsystem_id,
        );
    }
    json_sha256(&canonical)
}

fn graph_digest(
    graph: &ProcessSubsystemGraphManifestV1,
) -> Result<String, ProcessSubsystemGraphErrorV1> {
    let mut canonical = graph.clone();
    canonical.graph_sha256.clear();
    canonical
        .subsystems
        .sort_by(|left, right| left.subsystem_id.cmp(&right.subsystem_id));
    canonical
        .sensor_channels
        .sort_by(|left, right| left.sensor_id.cmp(&right.sensor_id));
    canonical
        .memberships
        .sort_by(|left, right| left.membership_id.cmp(&right.membership_id));
    canonical
        .relations
        .sort_by(|left, right| left.relation_id.cmp(&right.relation_id));
    for relation in &mut canonical.relations {
        if relation.relation_kind.is_symmetric()
            && relation.source_subsystem_id > relation.target_subsystem_id
        {
            std::mem::swap(
                &mut relation.source_subsystem_id,
                &mut relation.target_subsystem_id,
            );
        }
    }
    canonical.context_dimensions.sort_by(|left, right| {
        left.context_dimension_id.cmp(&right.context_dimension_id)
    });
    json_sha256(&canonical)
}

fn report_digest(
    report: &ProcessSubsystemGraphReportV1,
) -> Result<String, ProcessSubsystemGraphErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

/// Seals a relation after the endpoints, semantics and evidence are declared.
pub fn seal_process_subsystem_relation_v1(
    relation: &ProcessSubsystemRelationV1,
) -> Result<String, ProcessSubsystemGraphErrorV1> {
    relation_digest(relation)
}

/// Seals a graph with collection-order-independent canonicalization.
pub fn seal_process_subsystem_graph_v1(
    graph: &ProcessSubsystemGraphManifestV1,
) -> Result<String, ProcessSubsystemGraphErrorV1> {
    graph_digest(graph)
}

fn validate_subsystem(node: &ProcessSubsystemNodeV1) -> Result<(), ProcessSubsystemGraphErrorV1> {
    if node.schema != PROCESS_SUBSYSTEM_NODE_SCHEMA_V1 {
        return Err(ProcessSubsystemGraphErrorV1(
            "subsystem node schema is unsupported".into(),
        ));
    }
    require_text(&node.subsystem_id, "subsystem.subsystem_id")?;
    require_text(&node.process_stage_id, "subsystem.process_stage_id")?;
    if !valid_sha256(&node.source_artifact_sha256)
        || !valid_sha256(&node.engineering_evidence_sha256)
    {
        return Err(ProcessSubsystemGraphErrorV1(format!(
            "subsystem {} contains a malformed evidence digest",
            node.subsystem_id
        )));
    }
    Ok(())
}

fn validate_sensor(sensor: &ProcessSensorChannelV1) -> Result<(), ProcessSubsystemGraphErrorV1> {
    if sensor.schema != PROCESS_SENSOR_CHANNEL_SCHEMA_V1 {
        return Err(ProcessSubsystemGraphErrorV1(
            "sensor channel schema is unsupported".into(),
        ));
    }
    require_text(&sensor.sensor_id, "sensor.sensor_id")?;
    require_text(&sensor.unit, "sensor.unit")?;
    require_text(&sensor.clock_domain_id, "sensor.clock_domain_id")?;
    if !valid_sha256(&sensor.feature_schema_sha256)
        || !valid_sha256(&sensor.source_artifact_sha256)
    {
        return Err(ProcessSubsystemGraphErrorV1(format!(
            "sensor {} contains a malformed schema or source digest",
            sensor.sensor_id
        )));
    }
    Ok(())
}

fn validate_membership(
    membership: &ProcessSensorMembershipV1,
) -> Result<(), ProcessSubsystemGraphErrorV1> {
    if membership.schema != PROCESS_SENSOR_MEMBERSHIP_SCHEMA_V1 {
        return Err(ProcessSubsystemGraphErrorV1(
            "sensor membership schema is unsupported".into(),
        ));
    }
    require_text(&membership.membership_id, "membership.membership_id")?;
    require_text(&membership.sensor_id, "membership.sensor_id")?;
    require_text(&membership.subsystem_id, "membership.subsystem_id")?;
    if !valid_sha256(&membership.transform_sha256)
        || !valid_sha256(&membership.membership_evidence_sha256)
    {
        return Err(ProcessSubsystemGraphErrorV1(format!(
            "membership {} contains a malformed transform or evidence digest",
            membership.membership_id
        )));
    }
    Ok(())
}

fn validate_relation(
    relation: &ProcessSubsystemRelationV1,
) -> Result<(), ProcessSubsystemGraphErrorV1> {
    if relation.schema != PROCESS_SUBSYSTEM_RELATION_SCHEMA_V1 {
        return Err(ProcessSubsystemGraphErrorV1(
            "subsystem relation schema is unsupported".into(),
        ));
    }
    require_text(&relation.relation_id, "relation.relation_id")?;
    require_text(&relation.source_subsystem_id, "relation.source_subsystem_id")?;
    require_text(&relation.target_subsystem_id, "relation.target_subsystem_id")?;
    if relation.source_subsystem_id == relation.target_subsystem_id {
        return Err(ProcessSubsystemGraphErrorV1(
            "self relation is not permitted".into(),
        ));
    }
    if !valid_sha256(&relation.engineering_evidence_sha256)
        || !valid_sha256(&relation.relation_sha256)
    {
        return Err(ProcessSubsystemGraphErrorV1(format!(
            "relation {} contains a malformed evidence or relation digest",
            relation.relation_id
        )));
    }
    if relation_digest(relation)? != relation.relation_sha256 {
        return Err(ProcessSubsystemGraphErrorV1(format!(
            "relation {} digest does not match its content",
            relation.relation_id
        )));
    }
    Ok(())
}

fn validate_context(
    context: &ProcessContextDimensionV1,
) -> Result<(), ProcessSubsystemGraphErrorV1> {
    if context.schema != PROCESS_CONTEXT_DIMENSION_SCHEMA_V1 {
        return Err(ProcessSubsystemGraphErrorV1(
            "process context dimension schema is unsupported".into(),
        ));
    }
    require_text(&context.context_dimension_id, "context.context_dimension_id")?;
    require_text(&context.feature_id, "context.feature_id")?;
    if !valid_sha256(&context.vocabulary_sha256)
        || !valid_sha256(&context.source_artifact_sha256)
    {
        return Err(ProcessSubsystemGraphErrorV1(format!(
            "context dimension {} contains a malformed digest",
            context.context_dimension_id
        )));
    }
    Ok(())
}

fn validate_acyclic_dependencies(
    subsystem_ids: &BTreeSet<String>,
    relations: &[ProcessSubsystemRelationV1],
) -> Result<(), ProcessSubsystemGraphErrorV1> {
    let mut indegree: BTreeMap<String, usize> = subsystem_ids
        .iter()
        .map(|id| (id.clone(), 0))
        .collect();
    let mut outgoing: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for relation in relations
        .iter()
        .filter(|relation| relation.relation_kind.requires_acyclic_order())
    {
        outgoing
            .entry(relation.source_subsystem_id.clone())
            .or_default()
            .push(relation.target_subsystem_id.clone());
        *indegree
            .get_mut(&relation.target_subsystem_id)
            .expect("relation endpoints validated") += 1;
    }
    for children in outgoing.values_mut() {
        children.sort();
    }
    let mut ready: BTreeSet<String> = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(id, _)| id.clone())
        .collect();
    let mut visited = 0_usize;
    while let Some(next) = ready.pop_first() {
        visited += 1;
        for child in outgoing.get(&next).into_iter().flatten() {
            let child_degree = indegree.get_mut(child).expect("child endpoint validated");
            *child_degree -= 1;
            if *child_degree == 0 {
                ready.insert(child.clone());
            }
        }
    }
    if visited != subsystem_ids.len() {
        return Err(ProcessSubsystemGraphErrorV1(
            "material/control/measurement dependency relations contain a directed cycle".into(),
        ));
    }
    Ok(())
}

/// Validates a complete declared subsystem graph. An incomplete owner inventory
/// returns UNKNOWN, not an empty/fully qualified graph.
pub fn validate_process_subsystem_graph_v1(
    graph: &ProcessSubsystemGraphManifestV1,
) -> Result<ProcessSubsystemGraphReportV1, ProcessSubsystemGraphErrorV1> {
    if graph.schema != PROCESS_SUBSYSTEM_GRAPH_SCHEMA_V1 {
        return Err(ProcessSubsystemGraphErrorV1(
            "process subsystem graph schema is unsupported".into(),
        ));
    }
    require_text(&graph.graph_id, "graph.graph_id")?;
    if !valid_sha256(&graph.feature_schema_sha256) {
        return Err(ProcessSubsystemGraphErrorV1(
            "graph feature-schema digest is malformed".into(),
        ));
    }
    if !valid_sha256(&graph.graph_sha256) || graph_digest(graph)? != graph.graph_sha256 {
        return Err(ProcessSubsystemGraphErrorV1(
            "graph digest is absent or does not match graph content".into(),
        ));
    }

    let mut subsystem_ids = BTreeSet::new();
    for node in &graph.subsystems {
        validate_subsystem(node)?;
        if !subsystem_ids.insert(node.subsystem_id.as_str()) {
            return Err(ProcessSubsystemGraphErrorV1(
                "duplicate process subsystem ID".into(),
            ));
        }
    }
    let mut sensor_ids = BTreeSet::new();
    for sensor in &graph.sensor_channels {
        validate_sensor(sensor)?;
        if !sensor_ids.insert(sensor.sensor_id.as_str()) {
            return Err(ProcessSubsystemGraphErrorV1(
                "duplicate sensor channel ID".into(),
            ));
        }
        if sensor.feature_schema_sha256 != graph.feature_schema_sha256 {
            return Err(ProcessSubsystemGraphErrorV1(format!(
                "sensor {} feature schema does not match graph schema",
                sensor.sensor_id
            )));
        }
    }

    let mut membership_ids = BTreeSet::new();
    let mut membership_pairs = BTreeSet::new();
    for membership in &graph.memberships {
        validate_membership(membership)?;
        if !membership_ids.insert(membership.membership_id.as_str()) {
            return Err(ProcessSubsystemGraphErrorV1(
                "duplicate sensor membership ID".into(),
            ));
        }
        if !sensor_ids.contains(membership.sensor_id.as_str()) {
            return Err(ProcessSubsystemGraphErrorV1(format!(
                "membership {} references an unknown sensor",
                membership.membership_id
            )));
        }
        if !subsystem_ids.contains(membership.subsystem_id.as_str()) {
            return Err(ProcessSubsystemGraphErrorV1(format!(
                "membership {} references an unknown subsystem",
                membership.membership_id
            )));
        }
        if !membership_pairs.insert((
            membership.sensor_id.as_str(),
            membership.subsystem_id.as_str(),
        )) {
            return Err(ProcessSubsystemGraphErrorV1(
                "duplicate sensor-to-subsystem membership pair".into(),
            ));
        }
    }

    let mut relation_ids = BTreeSet::new();
    let mut relation_pairs = BTreeSet::new();
    for relation in &graph.relations {
        validate_relation(relation)?;
        if !relation_ids.insert(relation.relation_id.as_str()) {
            return Err(ProcessSubsystemGraphErrorV1(
                "duplicate subsystem relation ID".into(),
            ));
        }
        if !subsystem_ids.contains(relation.source_subsystem_id.as_str())
            || !subsystem_ids.contains(relation.target_subsystem_id.as_str())
        {
            return Err(ProcessSubsystemGraphErrorV1(format!(
                "relation {} has an unknown subsystem endpoint",
                relation.relation_id
            )));
        }
        let mut left = relation.source_subsystem_id.as_str();
        let mut right = relation.target_subsystem_id.as_str();
        if relation.relation_kind.is_symmetric() && left > right {
            std::mem::swap(&mut left, &mut right);
        }
        if !relation_pairs.insert((left, right, relation.relation_kind)) {
            return Err(ProcessSubsystemGraphErrorV1(
                "duplicate subsystem relation pair and kind".into(),
            ));
        }
    }

    let mut context_ids = BTreeSet::new();
    let mut context_feature_ids = BTreeSet::new();
    for context in &graph.context_dimensions {
        validate_context(context)?;
        if !context_ids.insert(context.context_dimension_id.as_str()) {
            return Err(ProcessSubsystemGraphErrorV1(
                "duplicate context dimension ID".into(),
            ));
        }
        if !context_feature_ids.insert(context.feature_id.as_str()) {
            return Err(ProcessSubsystemGraphErrorV1(
                "context feature IDs must be unique".into(),
            ));
        }
        if sensor_ids.contains(context.feature_id.as_str()) {
            return Err(ProcessSubsystemGraphErrorV1(
                "context features must remain distinct from sensor channel IDs".into(),
            ));
        }
    }

    validate_acyclic_dependencies(
        &subsystem_ids
            .iter()
            .map(|id| id.to_string())
            .collect(),
        &graph.relations,
    )?;

    let process_flow_relation_count = graph
        .relations
        .iter()
        .filter(|relation| relation.relation_kind == ProcessSubsystemRelationKindV1::MaterialProcessFlow)
        .count();
    let control_dependency_relation_count = graph
        .relations
        .iter()
        .filter(|relation| relation.relation_kind == ProcessSubsystemRelationKindV1::ControlDependency)
        .count();
    let measurement_dependency_relation_count = graph
        .relations
        .iter()
        .filter(|relation| relation.relation_kind == ProcessSubsystemRelationKindV1::MeasurementDependency)
        .count();
    let coupling_relation_count = graph
        .relations
        .iter()
        .filter(|relation| {
            matches!(
                relation.relation_kind,
                ProcessSubsystemRelationKindV1::ThermalCoupling
                    | ProcessSubsystemRelationKindV1::ElectricalCoupling
            )
        })
        .count();

    let mut report = ProcessSubsystemGraphReportV1 {
        schema: PROCESS_SUBSYSTEM_GRAPH_REPORT_SCHEMA_V1.into(),
        graph_id: graph.graph_id.clone(),
        graph_sha256: graph.graph_sha256.clone(),
        status: if graph.inventory_complete {
            ProcessSubsystemGraphStatusV1::Complete
        } else {
            ProcessSubsystemGraphStatusV1::Unknown
        },
        subsystem_count: graph.subsystems.len(),
        sensor_channel_count: graph.sensor_channels.len(),
        membership_count: graph.memberships.len(),
        relation_count: graph.relations.len(),
        context_dimension_count: graph.context_dimensions.len(),
        process_flow_relation_count,
        control_dependency_relation_count,
        measurement_dependency_relation_count,
        coupling_relation_count,
        inventory_complete: graph.inventory_complete,
        sensor_membership_resolves: true,
        relation_endpoints_resolve: true,
        context_features_are_separate_from_sensor_channels: true,
        directed_dependency_graph_is_acyclic: true,
        physical_causality_proven: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn subsystem(id: &str, stage: &str) -> ProcessSubsystemNodeV1 {
        ProcessSubsystemNodeV1 {
            schema: PROCESS_SUBSYSTEM_NODE_SCHEMA_V1.into(),
            subsystem_id: id.into(),
            process_stage_id: stage.into(),
            subsystem_kind: ProcessSubsystemKindV1::GasDelivery,
            source_artifact_sha256: digest('a'),
            engineering_evidence_sha256: digest('b'),
        }
    }

    fn sensor(id: &str, schema: &str) -> ProcessSensorChannelV1 {
        ProcessSensorChannelV1 {
            schema: PROCESS_SENSOR_CHANNEL_SCHEMA_V1.into(),
            sensor_id: id.into(),
            unit: "sccm".into(),
            clock_domain_id: "fab-clock".into(),
            feature_schema_sha256: schema.into(),
            source_artifact_sha256: digest('c'),
        }
    }

    fn membership(id: &str, sensor_id: &str, subsystem_id: &str) -> ProcessSensorMembershipV1 {
        ProcessSensorMembershipV1 {
            schema: PROCESS_SENSOR_MEMBERSHIP_SCHEMA_V1.into(),
            membership_id: id.into(),
            sensor_id: sensor_id.into(),
            subsystem_id: subsystem_id.into(),
            transform_sha256: digest('d'),
            membership_evidence_sha256: digest('e'),
        }
    }

    fn relation(
        id: &str,
        source: &str,
        target: &str,
        kind: ProcessSubsystemRelationKindV1,
    ) -> ProcessSubsystemRelationV1 {
        let mut value = ProcessSubsystemRelationV1 {
            schema: PROCESS_SUBSYSTEM_RELATION_SCHEMA_V1.into(),
            relation_id: id.into(),
            source_subsystem_id: source.into(),
            target_subsystem_id: target.into(),
            relation_kind: kind,
            engineering_evidence_sha256: digest('f'),
            relation_sha256: String::new(),
        };
        value.relation_sha256 = seal_process_subsystem_relation_v1(&value).unwrap();
        value
    }

    fn context(id: &str, feature: &str) -> ProcessContextDimensionV1 {
        ProcessContextDimensionV1 {
            schema: PROCESS_CONTEXT_DIMENSION_SCHEMA_V1.into(),
            context_dimension_id: id.into(),
            context_role: ProcessContextRoleV1::Chamber,
            feature_id: feature.into(),
            vocabulary_sha256: digest('1'),
            source_artifact_sha256: digest('2'),
        }
    }

    fn manifest(
        subsystems: Vec<ProcessSubsystemNodeV1>,
        sensors: Vec<ProcessSensorChannelV1>,
        memberships: Vec<ProcessSensorMembershipV1>,
        relations: Vec<ProcessSubsystemRelationV1>,
        contexts: Vec<ProcessContextDimensionV1>,
        complete: bool,
    ) -> ProcessSubsystemGraphManifestV1 {
        let mut value = ProcessSubsystemGraphManifestV1 {
            schema: PROCESS_SUBSYSTEM_GRAPH_SCHEMA_V1.into(),
            graph_id: "process-subsystem-graph-001".into(),
            feature_schema_sha256: digest('3'),
            inventory_complete: complete,
            subsystems,
            sensor_channels: sensors,
            memberships,
            relations,
            context_dimensions: contexts,
            graph_sha256: String::new(),
        };
        value.graph_sha256 = seal_process_subsystem_graph_v1(&value).unwrap();
        value
    }

    fn valid_manifest() -> ProcessSubsystemGraphManifestV1 {
        manifest(
            vec![subsystem("gas", "etch"), subsystem("plasma", "etch")],
            vec![sensor("gas-flow-01", &digest('3'))],
            vec![membership("m1", "gas-flow-01", "gas")],
            vec![relation(
                "r1",
                "gas",
                "plasma",
                ProcessSubsystemRelationKindV1::MaterialProcessFlow,
            )],
            vec![context("chamber", "ctx-chamber")],
            true,
        )
    }

    #[test]
    fn valid_physics_structured_manifest_is_complete_but_not_causal_proof() {
        let report = validate_process_subsystem_graph_v1(&valid_manifest()).unwrap();
        assert_eq!(report.status, ProcessSubsystemGraphStatusV1::Complete);
        assert_eq!(report.subsystem_count, 2);
        assert_eq!(report.sensor_channel_count, 1);
        assert_eq!(report.membership_count, 1);
        assert_eq!(report.relation_count, 1);
        assert!(report.directed_dependency_graph_is_acyclic);
        assert!(!report.physical_causality_proven);
        assert!(!report.equipment_control_authority);
    }

    #[test]
    fn incomplete_owner_inventory_returns_unknown() {
        let mut graph = valid_manifest();
        graph.inventory_complete = false;
        graph.graph_sha256 = seal_process_subsystem_graph_v1(&graph).unwrap();
        let report = validate_process_subsystem_graph_v1(&graph).unwrap();
        assert_eq!(report.status, ProcessSubsystemGraphStatusV1::Unknown);
    }

    #[test]
    fn duplicate_subsystem_id_is_rejected() {
        let mut graph = valid_manifest();
        graph.subsystems.push(graph.subsystems[0].clone());
        graph.graph_sha256 = seal_process_subsystem_graph_v1(&graph).unwrap();
        assert!(validate_process_subsystem_graph_v1(&graph).is_err());
    }

    #[test]
    fn dangling_sensor_membership_is_rejected() {
        let mut graph = valid_manifest();
        graph.memberships[0].subsystem_id = "missing-subsystem".into();
        graph.graph_sha256 = seal_process_subsystem_graph_v1(&graph).unwrap();
        assert!(validate_process_subsystem_graph_v1(&graph).is_err());
    }

    #[test]
    fn duplicate_sensor_channel_and_duplicate_membership_pair_are_rejected() {
        let mut graph = valid_manifest();
        graph.sensor_channels.push(graph.sensor_channels[0].clone());
        graph.graph_sha256 = seal_process_subsystem_graph_v1(&graph).unwrap();
        assert!(validate_process_subsystem_graph_v1(&graph).is_err());

        let mut graph = valid_manifest();
        graph.memberships.push(membership("m2", "gas-flow-01", "gas"));
        graph.graph_sha256 = seal_process_subsystem_graph_v1(&graph).unwrap();
        assert!(validate_process_subsystem_graph_v1(&graph).is_err());
    }

    #[test]
    fn feature_schema_mismatch_is_rejected() {
        let mut graph = valid_manifest();
        graph.sensor_channels[0].feature_schema_sha256 = digest('4');
        graph.graph_sha256 = seal_process_subsystem_graph_v1(&graph).unwrap();
        assert!(validate_process_subsystem_graph_v1(&graph).is_err());
    }

    #[test]
    fn directed_process_dependency_cycle_is_rejected() {
        let mut graph = valid_manifest();
        graph.relations.push(relation(
            "r2",
            "plasma",
            "gas",
            ProcessSubsystemRelationKindV1::ControlDependency,
        ));
        graph.graph_sha256 = seal_process_subsystem_graph_v1(&graph).unwrap();
        assert!(validate_process_subsystem_graph_v1(&graph).is_err());
    }

    #[test]
    fn symmetric_thermal_coupling_can_be_bidirectional_without_false_cycle() {
        let mut graph = valid_manifest();
        graph.relations = vec![
            relation(
                "r-thermal-1",
                "gas",
                "plasma",
                ProcessSubsystemRelationKindV1::ThermalCoupling,
            ),
            relation(
                "r-thermal-2",
                "plasma",
                "gas",
                ProcessSubsystemRelationKindV1::ThermalCoupling,
            ),
        ];
        graph.graph_sha256 = seal_process_subsystem_graph_v1(&graph).unwrap();
        assert!(validate_process_subsystem_graph_v1(&graph).is_err());
    }

    #[test]
    fn context_dimensions_are_not_sensor_channels() {
        let mut graph = valid_manifest();
        graph.context_dimensions[0].feature_id = "gas-flow-01".into();
        graph.graph_sha256 = seal_process_subsystem_graph_v1(&graph).unwrap();
        assert!(validate_process_subsystem_graph_v1(&graph).is_err());
    }

    #[test]
    fn graph_digest_is_independent_of_collection_order() {
        let mut graph = valid_manifest();
        let first = graph.graph_sha256.clone();
        graph.subsystems.reverse();
        graph.sensor_channels.reverse();
        graph.memberships.reverse();
        graph.relations.reverse();
        graph.context_dimensions.reverse();
        graph.graph_sha256 = seal_process_subsystem_graph_v1(&graph).unwrap();
        assert_eq!(first, graph.graph_sha256);
        assert!(validate_process_subsystem_graph_v1(&graph).is_ok());
    }

    #[test]
    fn symmetric_relation_digest_is_endpoint_order_invariant() {
        let forward = relation(
            "thermal",
            "gas",
            "plasma",
            ProcessSubsystemRelationKindV1::ThermalCoupling,
        );
        let reverse = relation(
            "thermal",
            "plasma",
            "gas",
            ProcessSubsystemRelationKindV1::ThermalCoupling,
        );
        assert_eq!(forward.relation_sha256, reverse.relation_sha256);
    }

    #[test]
    fn directed_process_relation_digest_preserves_direction() {
        let forward = relation(
            "flow",
            "gas",
            "plasma",
            ProcessSubsystemRelationKindV1::MaterialProcessFlow,
        );
        let reverse = relation(
            "flow",
            "plasma",
            "gas",
            ProcessSubsystemRelationKindV1::MaterialProcessFlow,
        );
        assert_ne!(forward.relation_sha256, reverse.relation_sha256);
    }

    #[test]
    fn malformed_feature_schema_digest_is_rejected() {
        let mut graph = valid_manifest();
        graph.feature_schema_sha256 = "not-a-digest".into();
        graph.graph_sha256 = seal_process_subsystem_graph_v1(&graph).unwrap();
        assert!(validate_process_subsystem_graph_v1(&graph).is_err());
    }
}
