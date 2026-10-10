use std::collections::{BTreeMap, BTreeSet};

use serde_json::json;

use symthaea_communication::hdc_ontology::{
    HdcConceptIdentityBinding, HdcOntologyCodebook, HdcOntologyDecodePolicy,
    HdcOntologyManifest, HdcRelationIdentityBinding, HDC_ONTOLOGY_ADAPTER_ID,
    HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
};
use symthaea_communication::{ConceptEdge, ConceptKind, ConceptNode, GroundedConceptGraph};

const SCHEME_ID: &str = "scheme:generalization-v1";
const MAPPING_PROVENANCE: &str = "validated-upstream-identity-map-v1";

fn kind(index: usize) -> ConceptKind {
    match index % 6 {
        0 => ConceptKind::Agent,
        1 => ConceptKind::Event,
        2 => ConceptKind::Object,
        3 => ConceptKind::Action,
        4 => ConceptKind::State,
        _ => ConceptKind::Property,
    }
}

fn make_graph(
    prefix: &str,
    concept_indices: &[usize],
    edges: &[(usize, usize, usize)],
) -> (GroundedConceptGraph, HdcOntologyManifest) {
    let concepts = concept_indices
        .iter()
        .map(|index| {
            let node_id = format!("{prefix}-node-{index}");
            ConceptNode {
                id: node_id,
                kind: kind(*index),
                label: Some(format!("{prefix}-label-{index}")),
                grounded_by: vec![format!("{prefix}-grounding-{index}")],
                confidence: 1.0,
            }
        })
        .collect::<Vec<_>>();

    let node_ids = concepts
        .iter()
        .map(|node| {
            let index = node
                .id
                .rsplit('-')
                .next()
                .unwrap()
                .parse::<usize>()
                .unwrap();
            (index, node.id.clone())
        })
        .collect::<BTreeMap<_, _>>();

    let relations = edges
        .iter()
        .map(|(_, relation, _)| *relation)
        .collect::<BTreeSet<_>>();

    let graph_edges = edges
        .iter()
        .map(|(source, relation, target)| ConceptEdge {
            source: node_ids[source].clone(),
            relation: format!("{prefix}-relation-{relation}"),
            target: node_ids[target].clone(),
            evidence_ids: Vec::new(),
            confidence: 1.0,
        })
        .collect::<Vec<_>>();

    let graph = GroundedConceptGraph {
        nodes: concepts,
        edges: graph_edges,
    };

    let manifest = HdcOntologyManifest {
        schema_version: HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
        scheme_id: SCHEME_ID.into(),
        mapping_provenance_hash: symthaea_communication::content_hash(MAPPING_PROVENANCE.as_bytes()),
        concepts: graph
            .nodes
            .iter()
            .map(|node| {
                let index = node
                    .id
                    .rsplit('-')
                    .next()
                    .unwrap()
                    .parse::<usize>()
                    .unwrap();
                HdcConceptIdentityBinding {
                    node_id: node.id.clone(),
                    concept_id: format!("concept:{index:02}"),
                    kind: node.kind.clone(),
                    grounding_ids: node.grounded_by.clone(),
                }
            })
            .collect(),
        relations: relations
            .into_iter()
            .map(|relation| HdcRelationIdentityBinding {
                local_relation: format!("{prefix}-relation-{relation}"),
                relation_id: format!("relation:{relation}"),
            })
            .collect(),
    };

    (graph, manifest)
}

fn identity_map(
    manifest: &HdcOntologyManifest,
) -> BTreeMap<String, String> {
    manifest
        .concepts
        .iter()
        .map(|binding| (binding.node_id.clone(), binding.concept_id.clone()))
        .collect()
}

fn stable_edge_set(
    edges: &[(usize, usize, usize)],
) -> BTreeSet<(String, String, String)> {
    edges
        .iter()
        .map(|(source, relation, target)| {
            (
                format!("concept:{source:02}"),
                format!("relation:{relation}"),
                format!("concept:{target:02}"),
            )
        })
        .collect()
}

fn execution_revision() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|revision| revision.trim().to_string())
        .filter(|revision| !revision.is_empty())
        .unwrap_or_else(|| "local".into())
}

fn main() -> Result<(), String> {
    let provenance_hash =
        symthaea_communication::content_hash(MAPPING_PROVENANCE.as_bytes());

    // The codebook contains 10 stable identities and 4 predicates. Training
    // graphs expose the atom vocabulary but deliberately omit every held-out
    // edge triple below.
    let training_cases = [
        (
            &[0, 1, 2, 3, 4, 5][..],
            &[
                (0, 0, 1),
                (1, 1, 2),
                (2, 2, 3),
                (3, 3, 4),
                (4, 0, 5),
            ][..],
        ),
        (
            &[1, 2, 3, 5, 6][..],
            &[
                (1, 0, 3),
                (3, 1, 5),
                (5, 2, 6),
                (6, 3, 2),
            ][..],
        ),
        (
            &[0, 4, 5, 6, 8, 9][..],
            &[
                (4, 1, 6),
                (6, 2, 8),
                (8, 3, 0),
                (0, 0, 9),
                (9, 1, 5),
            ][..],
        ),
        (
            &[0, 1, 2, 3, 5, 6, 7, 8, 9][..],
            &[
                (1, 3, 7),
                (7, 0, 5),
                (5, 1, 2),
            ][..],
        ),
    ];

    let training_manifest_graph = make_graph(
        "training",
        &(0..10).collect::<Vec<_>>(),
        &[
            (0, 0, 1),
            (1, 1, 2),
            (2, 2, 3),
            (3, 3, 4),
            (4, 0, 5),
            (1, 0, 3),
            (3, 1, 5),
            (5, 2, 6),
            (6, 3, 2),
            (4, 1, 6),
            (6, 2, 8),
            (8, 3, 0),
            (0, 0, 9),
            (9, 1, 5),
            (1, 3, 7),
            (7, 0, 5),
            (5, 1, 2),
        ],
    );
    let training_manifest = training_manifest_graph.1;

    let training_graphs = training_cases
        .iter()
        .map(|(nodes, edges)| make_graph("training", nodes, edges).0)
        .collect::<Vec<_>>();

    let codebook =
        HdcOntologyCodebook::from_training_graphs(0x6790_2026, &training_graphs, &training_manifest)?;

    let training_edge_set = training_cases
        .iter()
        .flat_map(|(_, edges)| stable_edge_set(edges))
        .collect::<BTreeSet<_>>();

    // Increasing graph load and topology shifts with known identities and unseen stable triples.
    // The first four cases stress load; the next four isolate distinct topology families.
    let held_out_cases = [
        (
            "load-3x2",
            &[0, 2, 5][..],
            &[(0, 3, 2), (2, 0, 5)][..],
        ),
        (
            "load-5x4",
            &[0, 2, 4, 6, 8][..],
            &[
                (0, 2, 4),
                (4, 0, 6),
                (6, 3, 8),
                (8, 1, 2),
            ][..],
        ),
        (
            "load-7x6",
            &[0, 1, 3, 5, 7, 8, 9][..],
            &[
                (0, 1, 3),
                (3, 3, 5),
                (5, 0, 7),
                (7, 2, 8),
                (8, 1, 9),
                (9, 2, 1),
            ][..],
        ),
        (
            "load-9x8",
            &[0, 1, 2, 4, 5, 6, 7, 8, 9][..],
            &[
                (0, 3, 4),
                (4, 2, 1),
                (1, 1, 6),
                (6, 0, 7),
                (7, 3, 8),
                (8, 0, 9),
                (9, 1, 5),
                (5, 2, 2),
            ][..],
        ),
        (
            "topology-out-star",
            &[0, 1, 2, 3, 4, 5, 6][..],
            &[
                (0, 1, 2),
                (0, 2, 3),
                (0, 3, 4),
                (0, 1, 5),
                (0, 2, 6),
                (0, 3, 1),
            ][..],
        ),
        (
            "topology-in-star",
            &[1, 2, 3, 4, 5, 6, 7][..],
            &[
                (2, 0, 1),
                (3, 1, 1),
                (4, 2, 1),
                (5, 3, 1),
                (6, 0, 1),
                (7, 2, 1),
            ][..],
        ),
        (
            "topology-merge-branch",
            &[0, 2, 3, 5, 7, 8, 9][..],
            &[
                (0, 1, 2),
                (3, 2, 7),
                (5, 3, 7),
                (8, 0, 7),
                (9, 2, 7),
                (2, 3, 7),
            ][..],
        ),
        (
            "topology-cycle",
            &[0, 1, 4, 6, 7, 8, 9][..],
            &[
                (0, 1, 4),
                (4, 2, 6),
                (6, 0, 7),
                (7, 1, 8),
                (8, 2, 9),
                (9, 3, 0),
            ][..],
        ),
    ];

    let policy = HdcOntologyDecodePolicy::conservative_default();
    let mut matrix = Vec::with_capacity(held_out_cases.len());

    for (case_id, nodes, edges) in held_out_cases {
        let (expected, source_manifest) = make_graph(case_id, nodes, edges);
        let receiver_prefix = format!("{case_id}-receiver");
        let (receiver_graph, mut receiver_manifest) =
            make_graph(&receiver_prefix, nodes, edges);

        // A receiver has different local node IDs/groundings, but the same
        // stable identity scheme and mapping authority.
        for binding in &mut receiver_manifest.concepts {
            binding.node_id = binding.node_id.replace(
                &receiver_prefix,
                &format!("{case_id}-rx"),
            );
            binding.grounding_ids = vec![format!("receiver-grounding-{}", binding.concept_id)];
        }
        for binding in &mut receiver_manifest.relations {
            binding.local_relation = binding.local_relation.replace(
                &receiver_prefix,
                &format!("{case_id}-rx"),
            );
        }

        // Keep the receiver graph's identity manifests synchronized with its
        // rewritten local IDs/relations.
        let receiver_manifest = HdcOntologyManifest {
            concepts: receiver_manifest.concepts.clone(),
            relations: receiver_manifest.relations.clone(),
            ..receiver_manifest
        };

        let representation = codebook.encode_graph(&expected, &source_manifest)?;
        let decoded = codebook.decode_graph_with_policy(
            &representation,
            &source_manifest,
            &receiver_manifest,
            policy,
        )?;

        let expected_concepts = identity_map(&receiver_manifest);
        let expected_relation_ids = edges
            .iter()
            .map(|(_, relation, _)| format!("relation:{relation}"))
            .collect::<Vec<_>>();

        let metrics = codebook.measure_roundtrip(
            &expected,
            &representation,
            &source_manifest,
            &receiver_manifest,
            &expected_concepts,
            &expected_relation_ids,
            policy,
        )?;

        let held_out_edge_set = stable_edge_set(edges);
        let unseen_edges = held_out_edge_set
            .iter()
            .filter(|edge| !training_edge_set.contains(*edge))
            .count();
        let unseen_edge_fraction = unseen_edges as f64 / edges.len() as f64;
        let distractor_concepts =
            codebook.concept_ids().len().saturating_sub(nodes.len());

        let source_groundings_preserved = {
            let mut expected_groundings = expected
                .nodes
                .iter()
                .map(|node| node.grounded_by.clone())
                .collect::<Vec<_>>();
            let mut observed_groundings = decoded
                .graph
                .nodes
                .iter()
                .map(|node| node.grounded_by.clone())
                .collect::<Vec<_>>();
            expected_groundings.sort();
            observed_groundings.sort();
            expected_groundings == observed_groundings
        };

        if unseen_edge_fraction != 1.0
            || distractor_concepts == 0
            || !metrics.structural_equivalence
            || !metrics.concept_identity_exact
            || !metrics.relation_identity_exact
            || !source_groundings_preserved
            || metrics.node_precision != 1.0
            || metrics.node_recall != 1.0
            || metrics.edge_precision != 1.0
            || metrics.edge_recall != 1.0
            || metrics.node_selection_margin < policy.min_margin
            || metrics.edge_selection_margin < policy.min_margin
        {
            return Err(format!(
                "grounded identity generalization failed for {case_id}: {metrics:#?}"
            ));
        }

        matrix.push(json!({
            "case_id": case_id,
            "nodes": nodes.len(),
            "edges": edges.len(),
            "distractor_concepts": distractor_concepts,
            "unseen_edge_fraction": unseen_edge_fraction,
            "concept_identity_exact": metrics.concept_identity_exact,
            "relation_identity_exact": metrics.relation_identity_exact,
            "source_groundings_preserved": source_groundings_preserved,
            "structural_equivalence": metrics.structural_equivalence,
            "node_precision": metrics.node_precision,
            "node_recall": metrics.node_recall,
            "edge_precision": metrics.edge_precision,
            "edge_recall": metrics.edge_recall,
            "node_min_selected_score": metrics.node_min_selected_score,
            "node_selection_margin": metrics.node_selection_margin,
            "edge_min_selected_score": metrics.edge_min_selected_score,
            "edge_selection_margin": metrics.edge_selection_margin,
            "family": if case_id.starts_with("topology-") { "topology" } else { "load" },
        }));
    }

    let min_node_margin = matrix
        .iter()
        .map(|case| case["node_selection_margin"].as_f64().unwrap())
        .fold(f64::INFINITY, f64::min);
    let min_edge_margin = matrix
        .iter()
        .map(|case| case["edge_selection_margin"].as_f64().unwrap())
        .fold(f64::INFINITY, f64::min);

    let execution_revision =
        execution_revision();

    let output = json!({
        "benchmark": "neurosemantic-hdc-grounded-identity-generalization-n0",
        "benchmark_schema_version": HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
        "execution_revision": execution_revision,
        "adapter_id": HDC_ONTOLOGY_ADAPTER_ID,
        "claim_boundary": "held_out_known_identity_unseen_composition_distractor_load_and_topology_only",
        "codebook": {
            "hash": codebook.codebook_hash(),
            "concept_count": codebook.concept_ids().len(),
            "relation_count": codebook.relation_ids().len(),
        },
        "mapping_provenance_hash": provenance_hash,
        "summary": {
            "cases": matrix.len(),
            "load_cases": 4,
            "topology_cases": 4,
            "all_clean_structurally_equivalent": true,
            "all_concept_identity_exact": true,
            "all_relation_identity_exact": true,
            "all_source_groundings_preserved": true,
            "all_held_out_edges_unseen_in_training": true,
            "max_nodes": 9,
            "max_edges": 8,
            "minimum_node_selection_margin": min_node_margin,
            "minimum_edge_selection_margin": min_edge_margin,
        },
        "cases": matrix,
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&output).map_err(|error| error.to_string())?
    );

    Ok(())
}
