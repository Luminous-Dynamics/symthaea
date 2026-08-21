// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_communication::{ConceptEdge, ConceptKind, ConceptNode, GroundedConceptGraph};
use symthaea_interlingua::{
    GraphDelta, GroundedHdcCodec, HdcWireEncoding, HdcWirePacket, ProjectionAttachment,
    SemanticTransferMode, TransferPlanningInput, TransferPolicy, graph_semantic_hash, plan_transfer,
};

fn base_graph() -> GroundedConceptGraph {
    GroundedConceptGraph {
        nodes: vec![
            ConceptNode {
                id: "alice".into(),
                kind: ConceptKind::Agent,
                label: Some("Alice".into()),
                grounded_by: vec!["obs-alice".into()],
                confidence: 0.95,
            },
            ConceptNode {
                id: "reactor".into(),
                kind: ConceptKind::Object,
                label: Some("Reactor A".into()),
                grounded_by: vec!["obs-reactor".into()],
                confidence: 0.98,
            },
        ],
        edges: vec![ConceptEdge {
            source: "alice".into(),
            relation: "observes".into(),
            target: "reactor".into(),
            evidence_ids: vec!["camera-1".into()],
            confidence: 0.85,
        }],
    }
}

fn target_graph() -> GroundedConceptGraph {
    let mut graph = base_graph();
    graph.nodes.push(ConceptNode {
        id: "sensor-s17".into(),
        kind: ConceptKind::Object,
        label: Some("Sensor S17".into()),
        grounded_by: vec!["telemetry-s17".into()],
        confidence: 0.99,
    });
    graph.edges.push(ConceptEdge {
        source: "sensor-s17".into(),
        relation: "measures".into(),
        target: "reactor".into(),
        evidence_ids: vec!["sample-417".into()],
        confidence: 0.93,
    });
    graph
}

#[test]
fn semantic_delta_reconstructs_same_hdc_and_q8_wire_packet() {
    let base = base_graph();
    let target = target_graph();
    let delta = GraphDelta::between(&base, &target).unwrap();
    let reconstructed = delta.apply(&base).unwrap();

    assert_eq!(
        graph_semantic_hash(&reconstructed).unwrap(),
        graph_semantic_hash(&target).unwrap()
    );

    let codec = GroundedHdcCodec::new(1024, "symthaea.scip.interop-test");
    let expected = codec.encode_graph(&target).unwrap();
    let rebuilt = codec.encode_graph(&reconstructed).unwrap();
    assert_eq!(expected.semantic_hash, rebuilt.semantic_hash);
    assert_eq!(expected.values, rebuilt.values);

    let expected_wire =
        HdcWirePacket::encode(&expected, HdcWireEncoding::Q8SymmetricV1).unwrap();
    assert!(expected_wire.reencode_matches(&rebuilt).unwrap());
}

#[test]
fn grounded_transfer_plan_does_not_choose_smaller_text_fallback() {
    let plan = plan_transfer(
        &TransferPlanningInput {
            semantic_reference_bytes: None,
            graph_delta_bytes: Some(420),
            grounded_graph_bytes: 4_000,
            human_text_bytes: Some(90),
            projection_candidates: vec![],
        },
        TransferPolicy::default(),
    )
    .unwrap();

    assert_eq!(plan.semantic, SemanticTransferMode::GraphDelta);
    assert_eq!(plan.projection, ProjectionAttachment::None);
    assert_eq!(plan.total_bytes, 420);
}
