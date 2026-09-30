use serde::Deserialize;
use std::collections::BTreeMap;

use symthaea_engineering::{
    Cp04QualificationArtifact, EngineeringObjectId, EngineeringRelation,
    EngineeringRelationKind, ScientificLineageGraph,
};

const SOURCE_FIXTURE: &str =
    include_str!("../../../../docs/engineering/data/cp-04-scientific-lineage-source-snapshot-v1.json");
const ARTIFACT_FIXTURE: &str =
    include_str!("../../../../docs/engineering/data/cp-04-scientific-lineage-adapter-v1.json");

#[derive(Debug, Deserialize)]
struct SourceSnapshot {
    schema: String,
    source_graph_digest: String,
    projection_digest: String,
    nodes: Vec<EngineeringObjectId>,
    relations: Vec<SourceRelation>,
    epistemic_mutation: SourceRelationSpec,
}

#[derive(Debug, Deserialize)]
struct SourceRelation {
    source_identity_digest: String,
    edge_type: String,
    target_identity_digest: String,
    relation_digest: String,
}

#[derive(Debug, Deserialize)]
struct SourceRelationSpec {
    source_identity_digest: String,
    edge_type: String,
    target_identity_digest: String,
}

fn relation_kind(wire: &str) -> EngineeringRelationKind {
    match wire {
        "requires" => EngineeringRelationKind::Requires,
        "implements" => EngineeringRelationKind::Implements,
        "parameterizes" => EngineeringRelationKind::Parameterizes,
        "executes_with" => EngineeringRelationKind::ExecutesWith,
        "compiled_by" => EngineeringRelationKind::CompiledBy,
        "runs_on" => EngineeringRelationKind::RunsOn,
        "deploys" => EngineeringRelationKind::Deploys,
        "executes" => EngineeringRelationKind::Executes,
        "observes" => EngineeringRelationKind::Observes,
        "quantifies" => EngineeringRelationKind::Quantifies,
        "traces_to" => EngineeringRelationKind::TracesTo,
        "currentness_for" => EngineeringRelationKind::CurrentnessFor,
        "applicable_to" => EngineeringRelationKind::ApplicableTo,
        "derives" => EngineeringRelationKind::Derives,
        "supports" => EngineeringRelationKind::Supports,
        other => panic!("unexpected source snapshot relation kind: {other}"),
    }
}

fn reconstruct(snapshot: &SourceSnapshot) -> ScientificLineageGraph {
    let by_digest: BTreeMap<String, EngineeringObjectId> = snapshot
        .nodes
        .iter()
        .map(|node| (node.identity_digest(), node.clone()))
        .collect();

    let mut graph = ScientificLineageGraph::new();
    for node in &snapshot.nodes {
        graph.add_node(node.clone());
    }

    for relation in &snapshot.relations {
        let source = by_digest
            .get(&relation.source_identity_digest)
            .expect("source snapshot relation source exists")
            .clone();
        let target = by_digest
            .get(&relation.target_identity_digest)
            .expect("source snapshot relation target exists")
            .clone();
        let relation_kind_value = relation_kind(&relation.edge_type);
        let relation_digest = relation.relation_digest.clone();
        let relation_source_digest = relation.source_identity_digest.clone();
        let relation_target_digest = relation.target_identity_digest.clone();
        let relation_edge_type = relation.edge_type.clone();
        let relation = EngineeringRelation::new(source, target, relation_kind_value)
            .expect("source snapshot relation is valid");
        let expected_digest = snapshot
            .relations
            .iter()
            .find(|candidate| {
                candidate.source_identity_digest == relation.source_identity_digest()
                    && candidate.target_identity_digest == relation.target_identity_digest()
                    && candidate.edge_type == relation.edge_type
            })
            .expect("source snapshot relation is represented")
            .relation_digest
            .clone();
        assert_eq!(
            relation.relation_digest(),
            expected_digest,
            "source snapshot relation digest must match reconstructed relation"
        );
        graph.add_relation(relation);
    }

    graph
}

#[test]
fn checked_in_source_snapshot_replays_through_projection_and_cp04() {
    let snapshot: SourceSnapshot =
        serde_json::from_str(SOURCE_FIXTURE).expect("source snapshot JSON must parse");
    assert_eq!(
        snapshot.schema,
        "symthaea.cp-04-scientific-lineage-source-snapshot-v1"
    );

    let graph = reconstruct(&snapshot);
    assert_eq!(graph.graph_digest(), snapshot.source_graph_digest);

    let projection = graph
        .qualification_projection()
        .expect("checked-in source snapshot must yield a valid projection");
    assert_eq!(projection.source_graph_digest(), snapshot.source_graph_digest);
    assert_eq!(projection.projection_digest(), snapshot.projection_digest);

    let artifact: Cp04QualificationArtifact =
        serde_json::from_str(ARTIFACT_FIXTURE).expect("CP-04 artifact JSON must parse");
    artifact.validate().expect("checked-in CP-04 artifact must validate");
    artifact
        .validate_against_projection(&projection)
        .expect("CP-04 artifact must bind to the independently reconstructed source snapshot");
}

#[test]
fn epistemic_source_mutation_changes_snapshot_identity_not_qualification_identity() {
    let snapshot: SourceSnapshot =
        serde_json::from_str(SOURCE_FIXTURE).expect("source snapshot JSON must parse");
    let graph = reconstruct(&snapshot);
    let baseline_projection = graph
        .qualification_projection()
        .expect("baseline source snapshot must project");

    let by_digest: BTreeMap<String, EngineeringObjectId> = snapshot
        .nodes
        .iter()
        .map(|node| (node.identity_digest(), node.clone()))
        .collect();
    let mutation = &snapshot.epistemic_mutation;
    let source = by_digest
        .get(&mutation.source_identity_digest)
        .expect("epistemic mutation source exists")
        .clone();
    let target = by_digest
        .get(&mutation.target_identity_digest)
        .expect("epistemic mutation target exists")
        .clone();

    let mut mutated_graph = graph.clone();
    mutated_graph.add_relation(
        EngineeringRelation::new(source, target, relation_kind(&mutation.edge_type))
            .expect("epistemic mutation must be a valid relation"),
    );

    let mutated_projection = mutated_graph
        .qualification_projection()
        .expect("epistemic mutation must preserve qualification acyclicity");

    assert_ne!(mutated_graph.graph_digest(), graph.graph_digest());
    assert_eq!(
        mutated_projection.projection_digest(),
        baseline_projection.projection_digest()
    );
    assert_ne!(
        mutated_projection.source_graph_digest(),
        baseline_projection.source_graph_digest()
    );

    let artifact: Cp04QualificationArtifact =
        serde_json::from_str(ARTIFACT_FIXTURE).expect("CP-04 artifact JSON must parse");
    assert!(artifact.validate_against_projection(&mutated_projection).is_err());
}
