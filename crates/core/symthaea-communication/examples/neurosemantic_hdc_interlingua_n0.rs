use symthaea_communication::hdc_interlingua::{
    HdcSemanticCodebook, HDC_SEMANTIC_INTERLINGUA_SCHEMA_VERSION,
};
use symthaea_communication::{
    relabel_nodes, rename_identifiers, reorder_collections, ConceptEdge, ConceptKind, ConceptNode,
    GroundedConceptGraph,
};
fn node(id: &str, kind: ConceptKind, grounding: &str, confidence: f32) -> ConceptNode {
    ConceptNode {
        id: id.into(),
        kind,
        label: Some(id.into()),
        grounded_by: vec![grounding.into()],
        confidence,
    }
}

fn edge(source: &str, relation: &str, target: &str) -> ConceptEdge {
    ConceptEdge {
        source: source.into(),
        relation: relation.into(),
        target: target.into(),
        evidence_ids: Vec::new(),
        confidence: 0.9,
    }
}

fn graph(
    agent_id: &str,
    agent_kind: ConceptKind,
    agent_grounding: &str,
    event_id: &str,
    event_grounding: &str,
    relation_a: &str,
    relation_b: &str,
) -> GroundedConceptGraph {
    GroundedConceptGraph {
        nodes: vec![
            node(agent_id, agent_kind, agent_grounding, 0.95),
            node(event_id, ConceptKind::Event, event_grounding, 0.91),
            node("object-1", ConceptKind::Object, "obs-object-1", 0.89),
        ],
        edges: vec![
            edge(agent_id, relation_a, event_id),
            edge(event_id, relation_b, "object-1"),
        ],
    }
}

fn training_graphs() -> Vec<GroundedConceptGraph> {
    vec![
        graph(
            "agent-1",
            ConceptKind::Agent,
            "obs-agent-1",
            "event-1",
            "obs-event-1",
            "initiates",
            "targets",
        ),
        GroundedConceptGraph {
            nodes: vec![
                node("agent-2", ConceptKind::Agent, "obs-agent-2", 0.94),
                node("object-1", ConceptKind::Object, "obs-object-1", 0.88),
            ],
            edges: vec![edge("agent-2", "observes", "object-1")],
        },
        graph(
            "agent-2",
            ConceptKind::Agent,
            "obs-agent-2",
            "event-2",
            "obs-event-2",
            "initiates",
            "uses",
        ),
        graph(
            "agent-1",
            ConceptKind::Agent,
            "obs-agent-1",
            "event-2",
            "obs-event-2",
            "observes",
            "uses",
        ),
    ]
}

fn held_out_graphs() -> Vec<GroundedConceptGraph> {
    vec![
        graph(
            "agent-2",
            ConceptKind::Agent,
            "obs-agent-2",
            "event-1",
            "obs-event-1",
            "initiates",
            "targets",
        ),
        graph(
            "agent-1",
            ConceptKind::Agent,
            "obs-agent-1",
            "event-2",
            "obs-event-2",
            "initiates",
            "uses",
        ),
    ]
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
    let training = training_graphs();
    let held_out = held_out_graphs();
    let seed = 0x4E53_4D48_4443_5343_u64;
    let codebook = HdcSemanticCodebook::from_training_graphs(seed, &training)?;
    let codebook_hash = codebook.codebook_hash();
    let execution_revision =
        execution_revision();

    let mut cases = Vec::with_capacity(held_out.len());
    for (index, expected) in held_out.iter().enumerate() {
        let representation = codebook.encode_graph(expected)?;
        let metrics = codebook.measure_roundtrip(expected, &representation)?;

        if !metrics.structural_equivalence
            || metrics.node_precision != 1.0
            || metrics.node_recall != 1.0
            || metrics.edge_precision != 1.0
            || metrics.edge_recall != 1.0
            || metrics.codebook_hash != codebook_hash
        {
            return Err(format!("held-out HDC retrieval failed for case {}", index + 1));
        }

        cases.push(serde_json::json!({
            "case_id": format!("held-out-{}", index + 1),
            "held_out": true,
            "codebook_hash": codebook_hash,
            "metrics": metrics,
        }));
    }

    let reorder_a = codebook.encode_graph(&held_out[0])?;
    let reorder_b = codebook.encode_graph(&reorder_collections(&held_out[0]))?;
    let reordered_representation_exact =
        reorder_a.node_frame == reorder_b.node_frame && reorder_a.edge_frame == reorder_b.edge_frame;

    let relabelled = codebook.encode_graph(&relabel_nodes(&held_out[0], " (paraphrase)"))?;
    let renamed = codebook.encode_graph(&rename_identifiers(&held_out[0], "renamed-"))?;
    let lexical_invariance_exact =
        reorder_a.node_frame == relabelled.node_frame && reorder_a.edge_frame == relabelled.edge_frame;
    let identifier_invariance_exact =
        reorder_a.node_frame == renamed.node_frame && reorder_a.edge_frame == renamed.edge_frame;
    if !lexical_invariance_exact || !identifier_invariance_exact {
        return Err("HDC canonicalization invariance failed".into());
    }

    let corruption_probabilities = [0.0_f32, 0.001, 0.01, 0.05, 0.10];
    let mut corruption_observations = Vec::with_capacity(corruption_probabilities.len());
    for (index, probability) in corruption_probabilities.into_iter().enumerate() {
        let corrupted =
            codebook.corrupt_for_transport(&reorder_a, probability, 50_000 + index as u64)?;
        let metrics = codebook.measure_roundtrip(&held_out[0], &corrupted)?;
        if probability == 0.0 && !metrics.structural_equivalence {
            return Err("zero-corruption HDC roundtrip was not exact".into());
        }
        corruption_observations.push(serde_json::json!({
            "flip_probability": probability,
            "seed": 50_000 + index as u64,
            "metrics": metrics,
        }));
    }

    let conservative_clean_decode =
        codebook
            .decode_graph_with_policy(
                &reorder_a,
                symthaea_communication::hdc_interlingua::HdcSemanticDecodePolicy::conservative_default(),
            )
            .is_ok();
    if !conservative_clean_decode {
        return Err("conservative HDC decode policy rejected clean held-out representation".into());
    }

    let negative_controls =
        codebook.measure_negative_controls(&reorder_a, &held_out[0], 9_001)?;
    if negative_controls.unrelated_node_max_similarity.abs() > 0.20
        || negative_controls.unrelated_edge_max_similarity.abs() > 0.20
        || negative_controls.true_edge_similarity <= negative_controls.swapped_edge_similarity
    {
        return Err("HDC negative controls failed".into());
    }

    let wrong_codebook = HdcSemanticCodebook::from_training_graphs(seed.wrapping_add(1), &training)?;
    let wrong_codebook_rejected = wrong_codebook.decode_graph(&reorder_a).is_err();
    if !wrong_codebook_rejected {
        return Err("mismatched HDC codebook was not rejected".into());
    }

    let summary = serde_json::json!({
        "training_graphs": training.len(),
        "held_out_graphs": held_out.len(),
        "all_structurally_equivalent": true,
        "same_codebook_for_all_cases": true,
        "reordered_representation_exact": reordered_representation_exact,
        "lexical_invariance_exact": lexical_invariance_exact,
        "identifier_invariance_exact": identifier_invariance_exact,
        "wrong_codebook_rejected": wrong_codebook_rejected,
        "conservative_clean_decode": conservative_clean_decode,
    });

    let output = serde_json::json!({
        "benchmark": "neurosemantic-hdc-interlingua-n0",
        "benchmark_schema_version": HDC_SEMANTIC_INTERLINGUA_SCHEMA_VERSION,
        "execution_revision": execution_revision,
        "adapter_id": "symthaea.hdc.semantic-interlingua-v1",
        "claim_boundary": "held_out_synthetic_graph_retrieval_and_reconstruction_only",
        "codebook": {
            "descriptor": codebook.descriptor(),
            "hash": codebook_hash,
            "node_keys": codebook.node_keys(),
            "relation_names": codebook.relation_names(),
        },
        "codec_id": "symthaea.hdc.continuous-sign-v1",
        "summary": summary,
        "cases": cases,
        "negative_controls": negative_controls,
        "transport_corruption": corruption_observations,
        "reordered_representation_exact": reordered_representation_exact,
        "lexical_invariance_exact": lexical_invariance_exact,
        "identifier_invariance_exact": identifier_invariance_exact,
        "wrong_codebook_rejected": wrong_codebook_rejected,
    });

    let _: Value = output.clone();
    println!(
        "{}",
        serde_json::to_string_pretty(&output).map_err(|error| error.to_string())?
    );
    Ok(())
}
