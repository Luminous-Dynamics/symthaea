use std::collections::BTreeMap;

use symthaea_communication::hdc_codec::HdcBinaryFrame;
use symthaea_communication::hdc_ontology::{
    HdcConceptIdentityBinding, HdcOntologyCodebook, HdcOntologyDecodePolicy,
    HdcOntologyManifest, HdcRelationIdentityBinding, HDC_ONTOLOGY_ADAPTER_ID,
    HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
};
use symthaea_communication::{ConceptEdge, ConceptKind, ConceptNode, GroundedConceptGraph};

fn node(id: &str, kind: ConceptKind, label: &str, grounding: &str) -> ConceptNode {
    ConceptNode {
        id: id.into(),
        kind,
        label: Some(label.into()),
        grounded_by: vec![grounding.into()],
        confidence: 1.0,
    }
}

fn edge(source: &str, relation: &str, target: &str) -> ConceptEdge {
    ConceptEdge {
        source: source.into(),
        relation: relation.into(),
        target: target.into(),
        evidence_ids: Vec::new(),
        confidence: 1.0,
    }
}

fn graph(
    agent_id: &str,
    agent_label: &str,
    agent_grounding: &str,
    event_id: &str,
    event_label: &str,
    event_grounding: &str,
    object_id: &str,
    object_label: &str,
    object_grounding: &str,
    relation_a: &str,
    relation_b: &str,
) -> GroundedConceptGraph {
    GroundedConceptGraph {
        nodes: vec![
            node(agent_id, ConceptKind::Agent, agent_label, agent_grounding),
            node(event_id, ConceptKind::Event, event_label, event_grounding),
            node(object_id, ConceptKind::Object, object_label, object_grounding),
        ],
        edges: vec![
            edge(agent_id, relation_a, event_id),
            edge(event_id, relation_b, object_id),
        ],
    }
}

fn manifest(
    graph: &GroundedConceptGraph,
    concept_ids: &[(&str, &str)],
    relation_ids: &[(&str, &str)],
    scheme_id: &str,
    mapping_provenance_hash: &str,
) -> HdcOntologyManifest {
    let concepts = graph
        .nodes
        .iter()
        .map(|node| {
            let concept_id = concept_ids
                .iter()
                .find(|(node_id, _)| *node_id == node.id)
                .map(|(_, concept_id)| *concept_id)
                .unwrap();
            HdcConceptIdentityBinding {
                node_id: node.id.clone(),
                concept_id: concept_id.into(),
                kind: node.kind.clone(),
                grounding_ids: node.grounded_by.clone(),
            }
        })
        .collect();

    HdcOntologyManifest {
        schema_version: HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
        scheme_id: scheme_id.into(),
        mapping_provenance_hash: mapping_provenance_hash.into(),
        concepts,
        relations: relation_ids
            .iter()
            .map(|(local_relation, relation_id)| HdcRelationIdentityBinding {
                local_relation: (*local_relation).into(),
                relation_id: (*relation_id).into(),
            })
            .collect(),
    }
}

fn expected_node_identities() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("receiver-agent".into(), "concept:agent/sender".into()),
        ("receiver-event".into(), "concept:event/approach".into()),
        ("receiver-object".into(), "concept:object/target".into()),
    ])
}

fn expected_edge_identities() -> Vec<(String, String, String)> {
    vec![
        (
            "concept:agent/sender".into(),
            "relation:initiates".into(),
            "concept:event/approach".into(),
        ),
        (
            "concept:event/approach".into(),
            "relation:targets".into(),
            "concept:object/target".into(),
        ),
    ]
}

fn corrupt_frame(frame: &HdcBinaryFrame, flip_probability: f32, seed: u64) -> Result<HdcBinaryFrame, String> {
    let binary = frame.to_binary()?;
    let corrupted = binary.add_noise(flip_probability, seed);
    Ok(HdcBinaryFrame::from_binary(&corrupted))
}

fn verify_result(
    decoded: &symthaea_communication::hdc_ontology::HdcOntologyDecodedGraph,
    expected_nodes: &BTreeMap<String, String>,
    expected_edges: &[(String, String, String)],
) -> bool {
    if &decoded.concept_ids_by_node != expected_nodes {
        return false;
    }

    let observed_edges = decoded
        .graph
        .edges
        .iter()
        .zip(&decoded.relation_ids_by_edge)
        .map(|(edge, relation_id)| {
            let source = decoded
                .concept_ids_by_node
                .get(&edge.source)
                .cloned()
                .unwrap_or_default();
            let target = decoded
                .concept_ids_by_node
                .get(&edge.target)
                .cloned()
                .unwrap_or_default();
            (source, relation_id.clone(), target)
        })
        .collect::<Vec<_>>();

    let mut observed_edges = observed_edges;
    let mut expected_edges = expected_edges.to_vec();
    observed_edges.sort();
    expected_edges.sort();
    observed_edges == expected_edges
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
    let scheme_id = "scheme:adversarial-n0-v1";
    let provenance_hash =
        symthaea_communication::content_hash(b"validated-upstream-identity-map-v1");

    let training = graph(
        "agent-en", "sender", "train-agent",
        "event-en", "approach", "train-event",
        "object-en", "target", "train-object",
        "initiates", "targets",
    );
    let training_manifest = manifest(
        &training,
        &[
            ("agent-en", "concept:agent/sender"),
            ("event-en", "concept:event/approach"),
            ("object-en", "concept:object/target"),
        ],
        &[
            ("initiates", "relation:initiates"),
            ("targets", "relation:targets"),
        ],
        scheme_id,
        &provenance_hash,
    );

    let codebook = HdcOntologyCodebook::from_training_graphs(
        0x6790_A11C_u64,
        &[training],
        &training_manifest,
    )?;

    let source = graph(
        "source-agent", "expéditeur", "source-agent-grounding",
        "source-event", "approche", "source-event-grounding",
        "source-object", "cible", "source-object-grounding",
        "commence", "cible",
    );
    let source_manifest = manifest(
        &source,
        &[
            ("source-agent", "concept:agent/sender"),
            ("source-event", "concept:event/approach"),
            ("source-object", "concept:object/target"),
        ],
        &[
            ("commence", "relation:initiates"),
            ("cible", "relation:targets"),
        ],
        scheme_id,
        &provenance_hash,
    );

    let receiver = graph(
        "receiver-agent", "sender", "receiver-agent-grounding",
        "receiver-event", "approach", "receiver-event-grounding",
        "receiver-object", "target", "receiver-object-grounding",
        "initiates", "targets",
    );
    let receiver_manifest = manifest(
        &receiver,
        &[
            ("receiver-agent", "concept:agent/sender"),
            ("receiver-event", "concept:event/approach"),
            ("receiver-object", "concept:object/target"),
        ],
        &[
            ("initiates", "relation:initiates"),
            ("targets", "relation:targets"),
        ],
        scheme_id,
        &provenance_hash,
    );

    let clean = codebook.encode_graph(&source, &source_manifest)?;
    let policy = HdcOntologyDecodePolicy::conservative_default();
    let expected_nodes = expected_node_identities();
    let expected_edges = expected_edge_identities();

    let corruption_levels = [0.0_f32, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50];

    let mut node_correct_accepts = 0_u32;
    let mut node_abstentions = 0_u32;
    let mut edge_correct_accepts = 0_u32;
    let mut edge_abstentions = 0_u32;

    let mut node_cases = Vec::new();
    for (index, probability) in corruption_levels.iter().copied().enumerate() {
        let mut representation = clean.clone();
        representation.node_frame = corrupt_frame(&clean.node_frame, probability, 0xABCD_0000 + index as u64)?;
        match codebook.decode_graph_with_policy(
            &representation,
            &source_manifest,
            &receiver_manifest,
            policy,
        ) {
            Ok(decoded) => {
                let correct = verify_result(&decoded, &expected_nodes, &expected_edges);
                if !correct {
                    return Err(format!(
                        "node corruption produced accepted but incorrect identity at p={probability}"
                    ));
                }
                node_correct_accepts += 1;
                node_cases.push(serde_json::json!({
                    "flip_probability": probability,
                    "outcome": "accepted_correct",
                }));
            }
            Err(reason) => {
                node_abstentions += 1;
                node_cases.push(serde_json::json!({
                    "flip_probability": probability,
                    "outcome": "abstained",
                    "reason": reason,
                }));
            }
        }
    }

    let mut edge_cases = Vec::new();
    for (index, probability) in corruption_levels.iter().copied().enumerate() {
        let mut representation = clean.clone();
        representation.edge_frame = corrupt_frame(&clean.edge_frame, probability, 0xBCDE_0000 + index as u64)?;
        match codebook.decode_graph_with_policy(
            &representation,
            &source_manifest,
            &receiver_manifest,
            policy,
        ) {
            Ok(decoded) => {
                let correct = verify_result(&decoded, &expected_nodes, &expected_edges);
                if !correct {
                    return Err(format!(
                        "edge corruption produced accepted but incorrect identity at p={probability}"
                    ));
                }
                edge_correct_accepts += 1;
                edge_cases.push(serde_json::json!({
                    "flip_probability": probability,
                    "outcome": "accepted_correct",
                }));
            }
            Err(reason) => {
                edge_abstentions += 1;
                edge_cases.push(serde_json::json!({
                    "flip_probability": probability,
                    "outcome": "abstained",
                    "reason": reason,
                }));
            }
        }
    }

    if node_correct_accepts == 0 || edge_correct_accepts == 0 {
        return Err("clean/noise sweep never produced a correct accepted decode".into());
    }

    let output = serde_json::json!({
        "benchmark": "neurosemantic-hdc-grounded-identity-adversarial-n0",
        "benchmark_schema_version": HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
        "execution_revision": execution_revision(),
        "adapter_id": HDC_ONTOLOGY_ADAPTER_ID,
        "claim_boundary": "identity_confusion_red_team_with_random_bit_corruption_only",
        "policy": {
            "min_score": policy.min_score,
            "min_margin": policy.min_margin,
        },
        "corruption_levels": corruption_levels,
        "summary": {
            "node_correct_accepts": node_correct_accepts,
            "node_abstentions": node_abstentions,
            "edge_correct_accepts": edge_correct_accepts,
            "edge_abstentions": edge_abstentions,
            "confident_wrong_accepts": 0,
            "max_flip_probability_tested": 0.50,
        },
        "node_cases": node_cases,
        "edge_cases": edge_cases,
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&output).map_err(|error| error.to_string())?
    );
    Ok(())
}
