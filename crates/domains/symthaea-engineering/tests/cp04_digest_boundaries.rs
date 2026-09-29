use symthaea_engineering::{
    Cp04QualificationArtifact, EngineeringObjectId, EngineeringRelation, EngineeringRelationKind,
    ScientificLineageGraph,
};

const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn object(kind: &str, id: &str, version: &str, digest: &str) -> EngineeringObjectId {
    EngineeringObjectId::new("cp04-digest-test", kind, id, version, digest).unwrap()
}

fn artifact_for_graph(graph: &ScientificLineageGraph) -> Cp04QualificationArtifact {
    let projection = graph.qualification_projection().unwrap();
    Cp04QualificationArtifact::try_from_projection(&projection).unwrap()
}

fn relation(
    source: EngineeringObjectId,
    target: EngineeringObjectId,
    kind: EngineeringRelationKind,
) -> EngineeringRelation {
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
        EngineeringRelationKind::Requires => {
            graph.add_relation(relation(requirement, representation, kind));
        }
        EngineeringRelationKind::Implements => {
            graph.add_relation(relation(representation, model, kind));
        }
        _ => panic!("test helper only supports the two CP-04 relation variants"),
    }

    graph
}

#[test]
fn identity_mutation_changes_relation_projection_graph_and_artifact_identities() {
    let source = object("requirement", "r", "1", A);
    let target = object("representation", "rep", "1", B);
    let changed_source = object("requirement", "r", "2", A);

    let base_relation = relation(
        source.clone(),
        target.clone(),
        EngineeringRelationKind::Requires,
    );
    let changed_relation = relation(
        changed_source.clone(),
        target.clone(),
        EngineeringRelationKind::Requires,
    );

    assert_ne!(source.identity_digest(), changed_source.identity_digest());
    assert_ne!(base_relation.relation_digest(), changed_relation.relation_digest());

    let mut base_graph = ScientificLineageGraph::new();
    base_graph.add_relation(base_relation);
    let mut changed_graph = ScientificLineageGraph::new();
    changed_graph.add_relation(changed_relation);

    let base_projection = base_graph.qualification_projection().unwrap();
    let changed_projection = changed_graph.qualification_projection().unwrap();
    assert_ne!(base_projection.projection_digest(), changed_projection.projection_digest());
    assert_ne!(base_projection.source_graph_digest(), changed_projection.source_graph_digest());

    let base_artifact = Cp04QualificationArtifact::try_from_projection(&base_projection).unwrap();
    let changed_artifact = Cp04QualificationArtifact::try_from_projection(&changed_projection).unwrap();
    assert_ne!(base_artifact.artifact_digest, changed_artifact.artifact_digest);
}

#[test]
fn qualification_relation_mutation_changes_relation_projection_graph_and_artifact_identities() {
    let requires_graph = three_node_graph(EngineeringRelationKind::Requires);
    let implements_graph = three_node_graph(EngineeringRelationKind::Implements);

    let requires_projection = requires_graph.qualification_projection().unwrap();
    let implements_projection = implements_graph.qualification_projection().unwrap();

    let requires_relation = requires_projection.relations().next().unwrap().relation_digest();
    let implements_relation = implements_projection.relations().next().unwrap().relation_digest();

    assert_ne!(requires_relation, implements_relation);
    assert_ne!(
        requires_projection.projection_digest(),
        implements_projection.projection_digest()
    );
    assert_ne!(
        requires_projection.source_graph_digest(),
        implements_projection.source_graph_digest()
    );

    let requires_artifact = artifact_for_graph(&requires_graph);
    let implements_artifact = artifact_for_graph(&implements_graph);
    assert_ne!(requires_artifact.artifact_digest, implements_artifact.artifact_digest);
}

#[test]
fn epistemic_only_mutation_changes_source_graph_and_artifact_but_not_projection_identity() {
    let requirement = object("requirement", "r", "1", A);
    let representation = object("representation", "rep", "1", B);
    let model = object("model", "m", "1", A);

    let mut base = ScientificLineageGraph::new();
    base.add_node(requirement.clone());
    base.add_node(representation.clone());
    base.add_node(model.clone());
    base.add_relation(relation(
        requirement.clone(),
        representation,
        EngineeringRelationKind::Requires,
    ));

    let mut with_epistemic = base.clone();
    with_epistemic.add_relation(relation(
        requirement,
        model,
        EngineeringRelationKind::Supports,
    ));

    let base_projection = base.qualification_projection().unwrap();
    let epistemic_projection = with_epistemic.qualification_projection().unwrap();

    assert_eq!(
        base_projection.projection_digest(),
        epistemic_projection.projection_digest()
    );
    assert_ne!(
        base_projection.source_graph_digest(),
        epistemic_projection.source_graph_digest()
    );

    let base_artifact = artifact_for_graph(&base);
    let epistemic_artifact = artifact_for_graph(&with_epistemic);
    assert_ne!(base_artifact.artifact_digest, epistemic_artifact.artifact_digest);
    assert_eq!(base_artifact.edges, epistemic_artifact.edges);
    assert_eq!(base_artifact.nodes, epistemic_artifact.nodes);
}

#[test]
fn source_graph_digest_mutation_cannot_preserve_artifact_integrity() {
    let graph = three_node_graph(EngineeringRelationKind::Requires);
    let original = artifact_for_graph(&graph);

    let mut tampered = original.clone();
    tampered.source_graph_digest = "0".repeat(64);

    assert_eq!(tampered.projection_digest, original.projection_digest);
    assert_eq!(tampered.nodes, original.nodes);
    assert_eq!(tampered.edges, original.edges);
    assert_eq!(tampered.artifact_digest, original.artifact_digest);
    assert!(tampered.validate().is_err());
    assert!(tampered.canonical_bytes().is_err());
}

#[test]
fn recomputed_source_graph_tamper_is_rejected_only_at_source_binding_boundary() {
    let graph = three_node_graph(EngineeringRelationKind::Requires);
    let projection = graph.qualification_projection().unwrap();
    let mut tampered = Cp04QualificationArtifact::try_from_projection(&projection).unwrap();

    tampered.source_graph_digest = "0".repeat(64);
    let mut digest_input = tampered.clone();
    digest_input.artifact_digest.clear();
    let bytes = serde_json::to_vec(&digest_input).unwrap();
    use sha2::{Digest, Sha256};
    tampered.artifact_digest = Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();

    assert!(tampered.validate().is_ok());
    assert!(tampered.validate_against_projection(&projection).is_err());
}

#[test]
fn relation_digest_mutation_cannot_be_hidden_by_an_unchanged_artifact_digest() {
    let graph = three_node_graph(EngineeringRelationKind::Requires);
    let original = artifact_for_graph(&graph);

    let mut tampered = original.clone();
    tampered.edges[0].relation_digest = "0".repeat(64);

    assert_eq!(tampered.artifact_digest, original.artifact_digest);
    assert!(tampered.validate().is_err());
}

#[test]
fn artifact_envelope_mutation_cannot_be_hidden_by_an_unchanged_artifact_digest() {
    let graph = three_node_graph(EngineeringRelationKind::Requires);
    let original = artifact_for_graph(&graph);

    let mut tampered = original.clone();
    tampered.claim_ceiling.push_str(" tampered");

    assert_eq!(tampered.artifact_digest, original.artifact_digest);
    assert!(tampered.validate().is_err());
}
