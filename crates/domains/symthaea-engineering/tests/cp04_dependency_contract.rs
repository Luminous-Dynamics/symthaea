use serde::Deserialize;
use symthaea_engineering::{
    Cp04QualificationArtifact, EngineeringObjectId, EngineeringRelation, EngineeringRelationKind,
    ScientificLineageGraph,
};

const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

#[derive(Debug, Deserialize)]
struct ContractVectors {
    dependency_matrix: DependencyMatrix,
}

#[derive(Debug, Deserialize)]
struct DependencyMatrix {
    schema: String,
    rules: Vec<DependencyRule>,
}

#[derive(Debug, Deserialize)]
struct DependencyRule {
    mutation: String,
    relation: String,
    projection: String,
    source_graph: String,
    artifact: String,
}

const CONTRACT: &str = include_str!(
    "../../../../docs/engineering/data/cp-04-scientific-lineage-adapter-v1-contract-vectors.json"
);

fn object(kind: &str, id: &str, version: &str, digest: &str) -> EngineeringObjectId {
    EngineeringObjectId::new("cp04-contract-test", kind, id, version, digest).unwrap()
}

fn relation(
    source: EngineeringObjectId,
    target: EngineeringObjectId,
    kind: EngineeringRelationKind,
) -> EngineeringRelation {
    EngineeringRelation::new(source, target, kind).unwrap()
}

fn artifact(graph: &ScientificLineageGraph) -> Cp04QualificationArtifact {
    let projection = graph.qualification_projection().unwrap();
    Cp04QualificationArtifact::try_from_projection(&projection).unwrap()
}

#[test]
fn positive_dependency_matrix_is_executable_contract() {
    let contract: ContractVectors = serde_json::from_str(CONTRACT).unwrap();
    assert_eq!(
        contract.dependency_matrix.schema,
        "symthaea.cp-04-dependency-matrix-v1"
    );

    let rule = |name: &str| {
        contract
            .dependency_matrix
            .rules
            .iter()
            .find(|rule| rule.mutation == name)
            .unwrap_or_else(|| panic!("missing dependency rule: {name}"))
    };

    let identity = rule("identity");
    assert_eq!(identity.relation, "changed");
    assert_eq!(identity.projection, "changed");
    assert_eq!(identity.source_graph, "changed");
    assert_eq!(identity.artifact, "changed");

    let qualification = rule("qualification_relation");
    assert_eq!(qualification.relation, "changed");
    assert_eq!(qualification.projection, "changed");
    assert_eq!(qualification.source_graph, "changed");
    assert_eq!(qualification.artifact, "changed");

    let epistemic = rule("epistemic_relation");
    assert_eq!(epistemic.relation, "changed");
    assert_eq!(epistemic.projection, "unchanged");
    assert_eq!(epistemic.source_graph, "changed");
    assert_eq!(epistemic.artifact, "changed");

    let requirement = object("requirement", "r", "1", A);
    let representation = object("representation", "rep", "1", B);
    let model = object("model", "m", "1", A);

    let mut base = ScientificLineageGraph::new();
    base.add_node(requirement.clone());
    base.add_node(representation.clone());
    base.add_node(model.clone());
    base.add_relation(relation(
        requirement.clone(),
        representation.clone(),
        EngineeringRelationKind::Requires,
    ));

    let mut identity_mutation = ScientificLineageGraph::new();
    identity_mutation.add_node(object("requirement", "r", "2", A));
    identity_mutation.add_node(representation.clone());
    identity_mutation.add_node(model.clone());
    identity_mutation.add_relation(relation(
        object("requirement", "r", "2", A),
        representation.clone(),
        EngineeringRelationKind::Requires,
    ));

    let mut qualification_mutation = base.clone();
    qualification_mutation.add_relation(relation(
        representation.clone(),
        model.clone(),
        EngineeringRelationKind::Implements,
    ));

    let mut epistemic_mutation = base.clone();
    epistemic_mutation.add_relation(relation(
        requirement,
        model,
        EngineeringRelationKind::Supports,
    ));

    let base_projection = base.qualification_projection().unwrap();
    let identity_projection = identity_mutation.qualification_projection().unwrap();
    let qualification_projection = qualification_mutation.qualification_projection().unwrap();
    let epistemic_projection = epistemic_mutation.qualification_projection().unwrap();

    assert_ne!(base_projection.projection_digest(), identity_projection.projection_digest());
    assert_ne!(base_projection.source_graph_digest(), identity_projection.source_graph_digest());
    assert_ne!(artifact(&base).artifact_digest, artifact(&identity_mutation).artifact_digest);

    assert_ne!(base_projection.projection_digest(), qualification_projection.projection_digest());
    assert_ne!(base_projection.source_graph_digest(), qualification_projection.source_graph_digest());
    assert_ne!(artifact(&base).artifact_digest, artifact(&qualification_mutation).artifact_digest);

    assert_eq!(base_projection.projection_digest(), epistemic_projection.projection_digest());
    assert_ne!(base_projection.source_graph_digest(), epistemic_projection.source_graph_digest());
    assert_ne!(artifact(&base).artifact_digest, artifact(&epistemic_mutation).artifact_digest);
}
