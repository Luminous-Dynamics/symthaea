use std::collections::BTreeMap;

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
            node(
                agent_id,
                ConceptKind::Agent,
                agent_label,
                agent_grounding,
            ),
            node(
                event_id,
                ConceptKind::Event,
                event_label,
                event_grounding,
            ),
            node(
                object_id,
                ConceptKind::Object,
                object_label,
                object_grounding,
            ),
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

    let relations = relation_ids
        .iter()
        .map(|(local_relation, relation_id)| HdcRelationIdentityBinding {
            local_relation: (*local_relation).into(),
            relation_id: (*relation_id).into(),
        })
        .collect();

    HdcOntologyManifest {
        schema_version: HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
        scheme_id: scheme_id.into(),
        mapping_provenance_hash: mapping_provenance_hash.into(),
        concepts,
        relations,
    }
}

fn concept_identity_map(
    graph: &GroundedConceptGraph,
    identities: &[(&str, &str)],
) -> BTreeMap<String, String> {
    graph
        .nodes
        .iter()
        .map(|node| {
            let concept_id = identities
                .iter()
                .find(|(node_id, _)| *node_id == node.id)
                .map(|(_, concept_id)| *concept_id)
                .unwrap();
            (node.id.clone(), concept_id.into())
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
    let scheme_id = "scheme:example-v1";
    let mapping_provenance_hash = symthaea_communication::content_hash(
        b"validated-upstream-identity-map-v1",
    );

    let training = graph(
        "agent-en", "sender", "train-agent",
        "event-en", "approach", "train-event",
        "object-en", "target", "train-object",
        "initiates", "targets",
    );
    let training_concepts = [
        ("agent-en", "concept:agent/sender"),
        ("event-en", "concept:event/approach"),
        ("object-en", "concept:object/target"),
    ];
    let training_relations = [
        ("initiates", "relation:initiates"),
        ("targets", "relation:targets"),
    ];
    let training_manifest = manifest(
        &training,
        &training_concepts,
        &training_relations,
        scheme_id,
        &mapping_provenance_hash,
    );

    let codebook =
        HdcOntologyCodebook::from_training_graphs(0x6790_u64, &[training.clone()], &training_manifest)?;
    let training_representation = codebook.encode_graph(&training, &training_manifest)?;

    let held_out = graph(
        "agent-fr", "expéditeur", "heldout-agent",
        "event-fr", "approche", "heldout-event",
        "object-fr", "cible", "heldout-object",
        "commence", "cible",
    );
    let held_out_concepts = [
        ("agent-fr", "concept:agent/sender"),
        ("event-fr", "concept:event/approach"),
        ("object-fr", "concept:object/target"),
    ];
    let held_out_relations = [
        ("commence", "relation:initiates"),
        ("cible", "relation:targets"),
    ];
    let held_out_manifest = manifest(
        &held_out,
        &held_out_concepts,
        &held_out_relations,
        scheme_id,
        &mapping_provenance_hash,
    );

    let receiver = graph(
        "receiver-agent", "sender", "receiver-agent-grounding",
        "receiver-event", "approach", "receiver-event-grounding",
        "receiver-object", "target", "receiver-object-grounding",
        "initiates", "targets",
    );
    let receiver_concepts = [
        ("receiver-agent", "concept:agent/sender"),
        ("receiver-event", "concept:event/approach"),
        ("receiver-object", "concept:object/target"),
    ];
    let receiver_manifest = manifest(
        &receiver,
        &receiver_concepts,
        &training_relations,
        scheme_id,
        &mapping_provenance_hash,
    );

    let held_out_representation = codebook.encode_graph(&held_out, &held_out_manifest)?;
    let expected_concepts = concept_identity_map(&receiver, &receiver_concepts);
    let expected_relations = vec![
        "relation:initiates".to_string(),
        "relation:targets".to_string(),
    ];

    let policy = HdcOntologyDecodePolicy::conservative_default();

    let mut tampered_source = held_out_manifest.clone();
    tampered_source.concepts[0].grounding_ids = vec!["tampered-source-grounding".into()];
    let source_manifest_hash_rejected = codebook
        .decode_graph_with_policy(
            &held_out_representation,
            &tampered_source,
            &receiver_manifest,
            policy,
        )
        .is_err();

    let decoded = codebook.decode_graph_with_policy(
        &held_out_representation,
        &held_out_manifest,
        &receiver_manifest,
        policy,
    )?;
    let metrics = codebook.measure_roundtrip(
        &held_out,
        &held_out_representation,
        &held_out_manifest,
        &receiver_manifest,
        &expected_concepts,
        &expected_relations,
        policy,
    )?;

    let source_groundings_preserved = {
        let mut expected_groundings = held_out
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
    let receiver_local_ids_used = decoded
        .graph
        .nodes
        .iter()
        .all(|node| node.id.starts_with("receiver-"));

    let same_hdc_frames = training_representation.node_frame == held_out_representation.node_frame
        && training_representation.edge_frame == held_out_representation.edge_frame;
    let new_grounding_allowed_without_recodebook =
        training_manifest.manifest_hash() != held_out_manifest.manifest_hash()
            && training_representation.codebook == held_out_representation.codebook
            && same_hdc_frames;

    let novel = graph(
        "agent-fr", "expéditeur", "heldout-agent",
        "event-fr", "approche", "heldout-event",
        "novel-fr", "nouveau", "heldout-novel",
        "commence", "cible",
    );
    let novel_manifest = manifest(
        &novel,
        &[
            ("agent-fr", "concept:agent/sender"),
            ("event-fr", "concept:event/approach"),
            ("novel-fr", "concept:object/novel"),
        ],
        &held_out_relations,
        scheme_id,
        &mapping_provenance_hash,
    );
    let novel_oov_rejected = codebook.encode_graph(&novel, &novel_manifest).is_err();

    let mut wrong_scheme = held_out_manifest.clone();
    wrong_scheme.scheme_id = "scheme:other-v1".into();
    let wrong_scheme_rejected = codebook
        .encode_graph(&held_out, &wrong_scheme)
        .is_err();

    let mut wrong_mapping_provenance = held_out_manifest.clone();
    wrong_mapping_provenance.mapping_provenance_hash =
        symthaea_communication::content_hash(b"different-authority-revision");
    let wrong_mapping_provenance_rejected = codebook
        .encode_graph(&held_out, &wrong_mapping_provenance)
        .is_err();

    let mut ambiguous_receiver = held_out_manifest.clone();
    ambiguous_receiver.concepts.push(HdcConceptIdentityBinding {
        node_id: "agent-fr-alias".into(),
        concept_id: "concept:agent/sender".into(),
        kind: ConceptKind::Agent,
        grounding_ids: vec!["ambiguous-grounding".into()],
    });
    let ambiguous_receiver_rejected = codebook
        .decode_graph_with_policy(
            &held_out_representation,
            &ambiguous_receiver,
            policy,
        )
        .is_err();

    let mut ambiguous_relation_receiver = held_out_manifest.clone();
    ambiguous_relation_receiver.relations.push(HdcRelationIdentityBinding {
        local_relation: "commencer".into(),
        relation_id: "relation:initiates".into(),
    });
    let ambiguous_receiver_relation_rejected = codebook
        .decode_graph_with_policy(
            &held_out_representation,
            &held_out_manifest,
            &ambiguous_relation_receiver,
            policy,
        )
        .is_err();

    if !metrics.concept_identity_exact
        || !metrics.relation_identity_exact
        || !metrics.structural_equivalence
        || metrics.node_precision != 1.0
        || metrics.node_recall != 1.0
        || metrics.edge_precision != 1.0
        || metrics.edge_recall != 1.0
        || !new_grounding_allowed_without_recodebook
        || !source_manifest_hash_rejected
        || !source_groundings_preserved
        || !receiver_local_ids_used
        || !novel_oov_rejected
        || !wrong_scheme_rejected
        || !wrong_mapping_provenance_rejected
        || !ambiguous_receiver_rejected
        || !ambiguous_receiver_relation_rejected
    {
        return Err("grounded identity N0 acceptance gates failed".into());
    }

    let execution_revision = execution_revision();
    let output = serde_json::json!({
        "benchmark": "neurosemantic-hdc-grounded-identity-n0",
        "benchmark_schema_version": HDC_ONTOLOGY_ADAPTER_SCHEMA_VERSION,
        "execution_revision": execution_revision,
        "adapter_id": HDC_ONTOLOGY_ADAPTER_ID,
        "claim_boundary": "held_out_stable_identity_retrieval_and_grounding_provenance_separation_only",
        "codebook": {
            "descriptor": codebook.descriptor(),
            "hash": codebook.codebook_hash(),
        },
        "training_manifest_hash": training_manifest.manifest_hash(),
        "held_out_manifest_hash": held_out_manifest.manifest_hash(),
        "summary": {
            "same_codebook_for_held_out_groundings": new_grounding_allowed_without_recodebook,
            "same_hdc_frames_for_same_stable_structure": same_hdc_frames,
            "source_manifest_hash_rejected": source_manifest_hash_rejected,
            "source_groundings_preserved": source_groundings_preserved,
            "receiver_local_ids_used": receiver_local_ids_used,
            "source_manifest_differs_from_receiver": held_out_manifest.manifest_hash() != receiver_manifest.manifest_hash(),
            "concept_identity_exact": metrics.concept_identity_exact,
            "relation_identity_exact": metrics.relation_identity_exact,
            "structurally_equivalent": metrics.structural_equivalence,
            "novel_oov_rejected": novel_oov_rejected,
            "wrong_scheme_rejected": wrong_scheme_rejected,
            "wrong_mapping_provenance_rejected": wrong_mapping_provenance_rejected,
            "ambiguous_receiver_rejected": ambiguous_receiver_rejected,
            "ambiguous_receiver_relation_rejected": ambiguous_receiver_relation_rejected,
        },
        "metrics": metrics,
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&output).map_err(|error| error.to_string())?
    );
    Ok(())
}
