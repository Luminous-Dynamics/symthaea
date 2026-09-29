use symthaea_engineering::{
    Cp04QualificationArtifact, EngineeringObjectId, EngineeringRelation, EngineeringRelationKind,
    ScientificLineageGraph,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn object(kind: &str, id: &str, version: &str, digest: &str) -> EngineeringObjectId {
    EngineeringObjectId::new("cp04-digest-test", kind, id, version, digest).unwrap()
}
fn artifact_for_graph(graph: &ScientificLineageGraph) -> Cp04QualificationArtifact {
    let projection = graph.qualification_projection().unwrap();
    Cp04QualificationArtifact::try_from_projection(&projection).unwrap()
}
fn relation(source: EngineeringObjectId, target: EngineeringObjectId, kind: EngineeringRelationKind) -> EngineeringRelation {
    EngineeringRelation::new(source, target, kind).unwrap()
}
fn three_node_graph(kind: EngineeringRelationKind) -> ScientificLineageGraph {
    let requirement = object("requirement", "r", "1", A);
    let representation = object("representation", "rep", "1", B);
    let model = object("model", "m", "1", A);
    let mut graph = ScientificLineageGraph::new();
    graph.add_node(requirement.clone());
    graph.add_node(representation.clone());
    graph.add_node(model.clone());
    match kind {
        EngineeringRelationKind::Requires => graph.add_relation(relation(requirement, representation, kind)),
        EngineeringRelationKind::Implements => graph.add_relation(relation(representation, model, kind)),
        _ => panic!("test helper only supports the two CP-04 relation variants"),
    }
    graph
}

#[test]
fn mutation_matrix_rejects_tampering_at_the_intended_boundary() {
    let projection = three_node_graph(EngineeringRelationKind::Requires).qualification_projection().unwrap();
    let original = Cp04QualificationArtifact::try_from_projection(&projection).unwrap();

    let cases = [
        ("source_graph_digest", "source-binding", 0usize),
        ("projection_digest", "source-binding", 0usize),
        ("edge.relation_digest", "relation", 0usize),
        ("node.identity.canonical_identifier", "identity", 0usize),
        ("claim_ceiling", "envelope", 0usize),
    ];

    for (name, boundary, _) in cases {
        let mut value = serde_json::to_value(&original).unwrap();
        match name {
            "source_graph_digest" => value["source_graph_digest"] = Value::String("0".repeat(64)),
            "projection_digest" => value["projection_digest"] = Value::String("1".repeat(64)),
            "edge.relation_digest" => value["edges"][0]["relation_digest"] = Value::String("2".repeat(64)),
            "node.identity.canonical_identifier" => value["nodes"][0]["identity"]["canonical_identifier"] = Value::String("tampered".into()),
            "claim_ceiling" => value["claim_ceiling"] = Value::String("tampered".into()),
            _ => unreachable!(),
        }
        let tampered: Cp04QualificationArtifact = serde_json::from_value(value).unwrap();
        assert!(tampered.validate().is_err(), "{name} should fail standalone validation");
        assert!(tampered.validate_against_projection(&projection).is_err(), "{name} should fail source binding");
        assert_eq!(boundary, match name {
            "source_graph_digest" | "projection_digest" => "source-binding",
            "edge.relation_digest" => "relation",
            "node.identity.canonical_identifier" => "identity",
            "claim_ceiling" => "envelope",
            _ => unreachable!(),
        });
    }
}

#[test]
fn recomputed_envelope_digest_does_not_upgrade_tampered_source_data() {
    let projection = three_node_graph(EngineeringRelationKind::Requires).qualification_projection().unwrap();
    let original = Cp04QualificationArtifact::try_from_projection(&projection).unwrap();

    let mut value = serde_json::to_value(&original).unwrap();
    value["source_graph_digest"] = Value::String("0".repeat(64));
    value["projection_digest"] = Value::String("1".repeat(64));
    value["artifact_digest"] = Value::String(String::new());
    let bytes = serde_json::to_vec(&value).unwrap();
    value["artifact_digest"] = Value::String(
        Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
    );

    let tampered: Cp04QualificationArtifact = serde_json::from_value(value).unwrap();
    assert!(tampered.validate().is_ok());
    assert!(tampered.validate_against_projection(&projection).is_err());
}
