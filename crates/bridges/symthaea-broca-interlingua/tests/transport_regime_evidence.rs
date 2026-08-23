use symthaea_broca_interlingua::{
    BrocaCognitiveContext, BrocaConcept, BrocaDomainContext, BrocaEntity, BrocaEpistemicCube,
    BrocaFidelityExportPolicy, BrocaFidelityInterchangeLimits, BrocaFidelityPlan,
    BrocaRelationMode, BrocaRelationshipStage, BrocaStructuredData, BrocaTranslationPlan,
    HardenedBrocaFidelityPacket, HardenedFidelityBrocaScipAdapter, RendererEpistemicStatus,
    RendererIntent, RendererResponseType,
};
use symthaea_communication::{GroundedConceptGraph, Provenance};
use symthaea_interlingua::{
    GraphDelta, GroundedHdcCodec, HdcWireEncoding, HdcWirePacket, InterchangePayload,
    SemanticReference, SemanticTransferMode, StructuredJsonPayload, TransferPlanningInput,
    TransferPolicy, canonical_graph_bytes, graph_semantic_hash, plan_transfer,
};

const MOOD_TEMPERATURE: f32 = 1.0;

fn provenance(case: &str) -> Provenance {
    Provenance {
        provider: "transport-regime-evidence".into(),
        provider_version: "1".into(),
        model_hash: format!("deterministic-case:{case}"),
        feature_flags: vec![],
        transformations: vec![],
    }
}

fn plan(arousal: f64) -> BrocaFidelityPlan {
    BrocaFidelityPlan {
        base: BrocaTranslationPlan {
            intent: RendererIntent::Answer,
            response_type: RendererResponseType::Report,
            epistemic_status: RendererEpistemicStatus::Probable,
            warmth: 0.72,
            meta_awareness: 0.83,
            coherence: 0.91,
            activated_concepts: (0..8)
                .map(|index| BrocaConcept {
                    name: format!("SYSTEM_CONCEPT_{index}"),
                    activation: 0.92 - index as f32 * 0.05,
                    relevance: 0.90 - index as f32 * 0.04,
                })
                .collect(),
            structured_data: Some(BrocaStructuredData::KeyValue(vec![
                ("reactor".into(), "R-17".into()),
                ("pump".into(), "P-04".into()),
                ("status".into(), "offline".into()),
            ])),
            domain_context: Some(BrocaDomainContext {
                domain: "engineering".into(),
                entities: vec![
                    BrocaEntity {
                        entity_type: "reactor".into(),
                        value: "R-17".into(),
                        confidence: 0.98,
                    },
                    BrocaEntity {
                        entity_type: "pump".into(),
                        value: "P-04".into(),
                        confidence: 0.96,
                    },
                ],
                computed_answer: Some("R-17 and P-04 remain offline pending inspection.".into()),
            }),
            ..Default::default()
        },
        context: BrocaCognitiveContext {
            psi: 0.673,
            valence: 0.10,
            arousal,
            relationship_stage: BrocaRelationshipStage::Attunement,
            relation_mode: BrocaRelationMode::IThou,
            trust: 0.78,
            primitive_tiers: vec!["Strategic".into(), "MetaCognitive".into()],
            domain_epistemic_cube: Some(BrocaEpistemicCube {
                empirical: 3,
                normative: 1,
                materiality: 2,
                harmonic: Some(3),
            }),
            domain_psi: Some(0.594),
        },
    }
}

fn compile(case: &str, arousal: f64) -> HardenedBrocaFidelityPacket {
    HardenedFidelityBrocaScipAdapter::compile_for_text_peer(
        &plan(arousal),
        MOOD_TEMPERATURE,
        provenance(case),
        &BrocaFidelityExportPolicy::default(),
        &BrocaFidelityInterchangeLimits::default(),
    )
    .expect("transport evidence plan must compile without semantic loss")
}

fn grounded_graph(packet: &HardenedBrocaFidelityPacket) -> GroundedConceptGraph {
    match &packet.packet.packet.envelope.payload {
        InterchangePayload::GroundedGraph(graph) => graph.clone(),
        other => panic!("expected grounded graph payload, got {other:?}"),
    }
}

#[test]
fn deterministic_transport_regime_evidence() {
    // One localized cognitive-state update. The graph is intentionally rich so
    // an exact content-addressed delta has a realistic opportunity to beat a
    // full graph transfer without assuming that every semantic change will.
    let base_packet = compile("transport-base", 0.30);
    let target_packet = compile("transport-target", 0.36);
    let base_graph = grounded_graph(&base_packet);
    let target_graph = grounded_graph(&target_packet);

    let base_hash = graph_semantic_hash(&base_graph).unwrap();
    let target_hash = graph_semantic_hash(&target_graph).unwrap();
    assert_ne!(base_hash, target_hash);

    // Full canonical grounded graph: exact, self-contained semantic state.
    let graph_bytes = canonical_graph_bytes(&target_graph).unwrap().len();
    let structured = StructuredJsonPayload::from_graph(&target_graph).unwrap();
    let decoded = structured.decode_graph().unwrap();
    assert_eq!(graph_semantic_hash(&decoded).unwrap(), target_hash);
    assert_eq!(structured.bytes.len(), graph_bytes);

    // Exact graph delta: requires the exact base state and verifies the target
    // semantic hash after reconstruction.
    let delta = GraphDelta::between(&base_graph, &target_graph).unwrap();
    let delta_bytes = delta.canonical_bytes().unwrap().len();
    assert_eq!(delta.estimated_wire_bytes().unwrap(), delta_bytes);
    let reconstructed = delta.apply(&base_graph).unwrap();
    assert_eq!(graph_semantic_hash(&reconstructed).unwrap(), target_hash);
    assert!(
        delta_bytes < graph_bytes,
        "controlled localized update should produce a smaller exact graph delta"
    );

    // Content-address reference: exact only when the receiver already has the
    // target graph. Serialize the actual reference object rather than counting
    // only the raw 64-byte hash.
    let reference = SemanticReference {
        semantic_hash: target_hash.clone(),
    };
    let reference_bytes = serde_json::to_vec(&reference).unwrap().len();

    // Compatibility text remains presentation, not exact machine state sync.
    let text_data_bytes = target_packet.packet.packet.fallback.content.len();
    let text_control_bytes = target_packet.packet.packet.fallback.system_prompt.len();
    let text_total_bytes = text_data_bytes + 1 + text_control_bytes;

    println!("| semantic transfer | bytes | exact semantic recovery | prerequisite |");
    println!("|---|---:|:---:|---|");
    println!(
        "| semantic-reference | {reference_bytes} | true | exact target graph already cached |"
    );
    println!("| graph-delta | {delta_bytes} | true | exact base graph already cached |");
    println!("| grounded-graph | {graph_bytes} | true | none |");
    println!("| text-fallback | {text_total_bytes} | false | human/model presentation path |");

    // HDC is a separate associative projection of the exact grounded graph.
    // F32 is exact projection identity; Q8/Q4 intentionally trade projection
    // precision for size. None of these rows replace the grounded semantic hash.
    let codec = GroundedHdcCodec::standard();
    let hdc = codec.encode_graph(&target_graph).unwrap();
    let f32 = HdcWirePacket::encode(&hdc, HdcWireEncoding::F32LeV1).unwrap();
    let q8 = HdcWirePacket::encode(&hdc, HdcWireEncoding::Q8SymmetricV1).unwrap();
    let q4 = HdcWirePacket::encode(&hdc, HdcWireEncoding::Q4SymmetricV1).unwrap();
    let f32_fidelity = f32.fidelity_against(&hdc).unwrap();
    let q8_fidelity = q8.fidelity_against(&hdc).unwrap();
    let q4_fidelity = q4.fidelity_against(&hdc).unwrap();

    assert!(f32_fidelity.exact);
    assert_eq!(f32_fidelity.cosine_similarity, 1.0);
    assert_eq!(f32_fidelity.max_abs_error, 0.0);
    assert!(q8.reencode_matches(&hdc).unwrap());
    assert!(q4.reencode_matches(&hdc).unwrap());
    assert!(q8.wire_bytes() < f32.wire_bytes());
    assert!(q4.wire_bytes() < q8.wire_bytes());
    assert!(q8_fidelity.cosine_similarity.is_finite());
    assert!(q4_fidelity.cosine_similarity.is_finite());

    println!();
    println!("| HDC projection | body bytes | cosine vs f32 | max abs error | exact projection |");
    println!("|---|---:|---:|---:|:---:|");
    println!(
        "| F32 | {} | {:.6} | {:.6} | {} |",
        f32.wire_bytes(),
        f32_fidelity.cosine_similarity,
        f32_fidelity.max_abs_error,
        f32_fidelity.exact
    );
    println!(
        "| Q8 | {} | {:.6} | {:.6} | {} |",
        q8.wire_bytes(),
        q8_fidelity.cosine_similarity,
        q8_fidelity.max_abs_error,
        q8_fidelity.exact
    );
    println!(
        "| Q4 | {} | {:.6} | {:.6} | {} |",
        q4.wire_bytes(),
        q4_fidelity.cosine_similarity,
        q4_fidelity.max_abs_error,
        q4_fidelity.exact
    );

    // The transfer planner must keep exact semantic synchronization separate
    // from optional projection delivery.
    let mut planning = TransferPlanningInput {
        semantic_reference_bytes: Some(reference_bytes),
        graph_delta_bytes: Some(delta_bytes),
        grounded_graph_bytes: graph_bytes,
        human_text_bytes: Some(text_total_bytes),
        projection_candidates: vec![],
    };
    let plan = plan_transfer(&planning, TransferPolicy::default()).unwrap();
    assert_eq!(plan.semantic, SemanticTransferMode::SemanticReference);

    planning.semantic_reference_bytes = None;
    let plan = plan_transfer(&planning, TransferPolicy::default()).unwrap();
    assert_eq!(plan.semantic, SemanticTransferMode::GraphDelta);

    planning.graph_delta_bytes = None;
    let plan = plan_transfer(&planning, TransferPolicy::default()).unwrap();
    assert_eq!(plan.semantic, SemanticTransferMode::GroundedGraph);
}
