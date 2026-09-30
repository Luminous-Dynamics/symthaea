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

#[derive(Debug, serde::Deserialize)]
struct NegativeContract {
    schema: String,
    vectors: Vec<NegativeVector>,
}

#[derive(Debug, serde::Deserialize)]
struct NegativeVector {
    id: String,
    mutation_path: String,
    mutation_value: String,
    digest_strategy: String,
    expected_validate: String,
    expected_source_binding: String,
    first_boundary: String,
}

const NEGATIVE_CONTRACT: &str = include_str!(
    "../../../../docs/engineering/data/cp-04-scientific-lineage-adapter-v1-negative-contract-vectors.json"
);

fn set_mutation(value: &mut Value, path: &str, mutation_value: &str) {
    match path {
        "source_graph_digest" | "projection_digest" | "claim_ceiling" => {
            value[path] = Value::String(if mutation_value.len() == 1 {
                mutation_value.repeat(64)
            } else {
                mutation_value.to_owned()
            });
        }
        "edges[0].relation_digest" => {
            value["edges"][0]["relation_digest"] = Value::String(mutation_value.repeat(64));
        }
        "nodes[0].identity.canonical_identifier" => {
            value["nodes"][0]["identity"]["canonical_identifier"] =
                Value::String(mutation_value.to_owned());
        }
        "nodes[0].identity.content_digest" => {
            value["nodes"][0]["identity"]["content_digest"] =
                Value::String(mutation_value.repeat(64));
        }
        "edges[0].target_identity_digest" => {
            value["edges"][0]["target_identity_digest"] =
                Value::String(mutation_value.repeat(64));
        }
        "edges[0].edge_type" => {
            value["edges"][0]["edge_type"] = Value::String(mutation_value.to_owned());
        }
        _ => panic!("unknown negative contract mutation path: {path}"),
    }
}

fn recompute_artifact_digest(value: &mut Value) {
    value["artifact_digest"] = Value::String(String::new());
    let bytes = serde_json::to_vec(value).unwrap();
    value["artifact_digest"] = Value::String(
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
    );
}

#[test]
fn machine_readable_mutation_matrix_enforces_declared_boundaries() {
    let contract: NegativeContract = serde_json::from_str(NEGATIVE_CONTRACT).unwrap();
    assert_eq!(contract.schema, "symthaea.cp-04-qualification-adapter-negative-contract-v1");

    let projection = three_node_graph(EngineeringRelationKind::Requires)
        .qualification_projection()
        .unwrap();
    let original = Cp04QualificationArtifact::try_from_projection(&projection).unwrap();

    for vector in contract.vectors {
        let mut value = serde_json::to_value(&original).unwrap();
        set_mutation(&mut value, &vector.mutation_path, &vector.mutation_value);
        if vector.digest_strategy == "recompute_artifact_digest" {
            recompute_artifact_digest(&mut value);
        }
        let tampered: Cp04QualificationArtifact = serde_json::from_value(value).unwrap();
        let validate_ok = tampered.validate().is_ok();
        let source_binding_ok = tampered.validate_against_projection(&projection).is_ok();

        assert_eq!(
            validate_ok,
            vector.expected_validate == "accept",
            "{}: standalone validation boundary drifted",
            vector.id
        );
        assert_eq!(
            source_binding_ok,
            vector.expected_source_binding == "accept",
            "{}: source-binding boundary drifted",
            vector.id
        );
        assert_eq!(
            vector.first_boundary,
            if validate_ok { "source-binding" } else { "artifact-integrity" },
            "{}: first rejection boundary drifted",
            vector.id
        );
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
