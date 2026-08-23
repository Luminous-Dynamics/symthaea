use symthaea_broca_interlingua::{
    BrocaCognitiveContext, BrocaConcept, BrocaDomainContext, BrocaEntity, BrocaEpistemicCube,
    BrocaFidelityExportPolicy, BrocaFidelityInterchangeLimits, BrocaFidelityPlan,
    BrocaRelationMode, BrocaRelationshipStage, BrocaStructuredData, BrocaTranslationPlan,
    HardenedFidelityBrocaScipAdapter, RendererEpistemicStatus, RendererIntent,
    RendererResponseType,
};
use symthaea_communication::{GroundedConceptGraph, Provenance};
use symthaea_interlingua::{
    GraphDelta, InterchangePayload, SemanticTransferMode, TransferPlanningInput, TransferPolicy,
    canonical_graph_bytes, graph_semantic_hash, plan_transfer,
};

const MOOD_TEMPERATURE: f32 = 1.0;

type Mutator = fn(&mut BrocaFidelityPlan);

struct ChangeCase {
    name: &'static str,
    mutate: Mutator,
}

#[derive(Debug)]
struct ChangeRow {
    name: &'static str,
    graph_bytes: usize,
    delta_bytes: usize,
    ratio: f64,
    remove_nodes: usize,
    upsert_nodes: usize,
    remove_edges: usize,
    add_edges: usize,
    selected: SemanticTransferMode,
}

fn provenance(case: &str) -> Provenance {
    Provenance {
        provider: "semantic-change-curve-evidence".into(),
        provider_version: "1".into(),
        model_hash: format!("deterministic-case:{case}"),
        feature_flags: vec![],
        transformations: vec![],
    }
}

fn base_plan() -> BrocaFidelityPlan {
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
            arousal: 0.30,
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

fn graph(case: &str, plan: &BrocaFidelityPlan) -> GroundedConceptGraph {
    let packet = HardenedFidelityBrocaScipAdapter::compile_for_text_peer(
        plan,
        MOOD_TEMPERATURE,
        provenance(case),
        &BrocaFidelityExportPolicy::default(),
        &BrocaFidelityInterchangeLimits::default(),
    )
    .expect("change-curve plan must compile without semantic loss");

    match packet.packet.packet.envelope.payload {
        InterchangePayload::GroundedGraph(graph) => graph,
        other => panic!("expected grounded graph payload, got {other:?}"),
    }
}

fn scalar_arousal(plan: &mut BrocaFidelityPlan) {
    plan.context.arousal = 0.36;
}

fn relationship_transition(plan: &mut BrocaFidelityPlan) {
    plan.context.relationship_stage = BrocaRelationshipStage::Unity;
    plan.context.trust = 0.91;
}

fn add_concept(plan: &mut BrocaFidelityPlan) {
    plan.base.activated_concepts.push(BrocaConcept {
        name: "SYSTEM_CONCEPT_8".into(),
        activation: 0.74,
        relevance: 0.79,
    });
}

fn replace_one_concept(plan: &mut BrocaFidelityPlan) {
    plan.base.activated_concepts[0].name = "REVISED_CONCEPT_0".into();
}

fn replace_four_concepts(plan: &mut BrocaFidelityPlan) {
    for index in 0..4 {
        plan.base.activated_concepts[index].name = format!("REVISED_CONCEPT_{index}");
    }
}

fn replace_all_concepts(plan: &mut BrocaFidelityPlan) {
    for (index, concept) in plan.base.activated_concepts.iter_mut().enumerate() {
        concept.name = format!("REVISED_CONCEPT_{index}");
    }
}

fn structured_data_update(plan: &mut BrocaFidelityPlan) {
    plan.base.structured_data = Some(BrocaStructuredData::KeyValue(vec![
        ("reactor".into(), "R-17".into()),
        ("pump".into(), "P-04".into()),
        ("status".into(), "inspection-pending".into()),
        ("priority".into(), "high".into()),
    ]));
}

fn domain_result_update(plan: &mut BrocaFidelityPlan) {
    let domain = plan
        .base
        .domain_context
        .as_mut()
        .expect("base plan has domain context");
    domain.computed_answer = Some("R-17 cleared; P-04 remains offline pending inspection.".into());
}

fn domain_entity_addition(plan: &mut BrocaFidelityPlan) {
    let domain = plan
        .base
        .domain_context
        .as_mut()
        .expect("base plan has domain context");
    domain.entities.push(BrocaEntity {
        entity_type: "sensor".into(),
        value: "S-22".into(),
        confidence: 0.94,
    });
}

fn epistemic_transition(plan: &mut BrocaFidelityPlan) {
    plan.base.epistemic_status = RendererEpistemicStatus::Uncertain;
    plan.context.domain_epistemic_cube = Some(BrocaEpistemicCube {
        empirical: 2,
        normative: 2,
        materiality: 2,
        harmonic: Some(2),
    });
    plan.context.domain_psi = Some(0.51);
}

fn broad_transition(plan: &mut BrocaFidelityPlan) {
    plan.base.intent = RendererIntent::Reflect;
    plan.base.response_type = RendererResponseType::Empathic;
    plan.base.epistemic_status = RendererEpistemicStatus::Uncertain;
    plan.base.warmth = 0.90;
    plan.base.meta_awareness = 0.66;
    plan.base.coherence = 0.74;

    for (index, concept) in plan.base.activated_concepts.iter_mut().enumerate() {
        concept.name = format!("BROAD_CONCEPT_{index}");
        concept.activation = 0.58 + index as f32 * 0.025;
        concept.relevance = 0.62 + index as f32 * 0.02;
    }
    plan.base.activated_concepts.push(BrocaConcept {
        name: "BROAD_CONCEPT_8".into(),
        activation: 0.81,
        relevance: 0.84,
    });

    plan.base.structured_data = Some(BrocaStructuredData::KeyValue(vec![
        ("reactor".into(), "R-18".into()),
        ("pump".into(), "P-09".into()),
        ("status".into(), "partial-recovery".into()),
        ("priority".into(), "critical".into()),
    ]));

    plan.base.domain_context = Some(BrocaDomainContext {
        domain: "engineering".into(),
        entities: vec![
            BrocaEntity {
                entity_type: "reactor".into(),
                value: "R-18".into(),
                confidence: 0.91,
            },
            BrocaEntity {
                entity_type: "pump".into(),
                value: "P-09".into(),
                confidence: 0.89,
            },
            BrocaEntity {
                entity_type: "sensor".into(),
                value: "S-22".into(),
                confidence: 0.93,
            },
        ],
        computed_answer: Some(
            "R-18 is stable; P-09 remains degraded while S-22 awaits calibration.".into(),
        ),
    });

    plan.context.psi = 0.421;
    plan.context.valence = -0.35;
    plan.context.arousal = 0.82;
    plan.context.relationship_stage = BrocaRelationshipStage::Unity;
    plan.context.relation_mode = BrocaRelationMode::IThou;
    plan.context.trust = 0.95;
    plan.context.primitive_tiers = vec![
        "Strategic".into(),
        "MetaCognitive".into(),
        "Integrative".into(),
    ];
    plan.context.domain_epistemic_cube = Some(BrocaEpistemicCube {
        empirical: 4,
        normative: 2,
        materiality: 3,
        harmonic: Some(4),
    });
    plan.context.domain_psi = Some(0.711);
}

fn measure_case(base_graph: &GroundedConceptGraph, case: ChangeCase) -> ChangeRow {
    let mut target_plan = base_plan();
    (case.mutate)(&mut target_plan);
    let target_graph = graph(case.name, &target_plan);

    let base_hash = graph_semantic_hash(base_graph).unwrap();
    let target_hash = graph_semantic_hash(&target_graph).unwrap();
    assert_ne!(base_hash, target_hash, "{} must change semantic state", case.name);

    let graph_bytes = canonical_graph_bytes(&target_graph).unwrap().len();
    let delta = GraphDelta::between(base_graph, &target_graph).unwrap();
    let delta_bytes = delta.canonical_bytes().unwrap().len();
    let reconstructed = delta.apply(base_graph).unwrap();
    assert_eq!(graph_semantic_hash(&reconstructed).unwrap(), target_hash);

    let selected = plan_transfer(
        &TransferPlanningInput {
            semantic_reference_bytes: None,
            graph_delta_bytes: Some(delta_bytes),
            grounded_graph_bytes: graph_bytes,
            human_text_bytes: None,
            projection_candidates: vec![],
        },
        TransferPolicy::default(),
    )
    .expect("grounded transfer plan must exist")
    .semantic;

    let expected = if delta_bytes <= graph_bytes {
        SemanticTransferMode::GraphDelta
    } else {
        SemanticTransferMode::GroundedGraph
    };
    assert_eq!(selected, expected);

    ChangeRow {
        name: case.name,
        graph_bytes,
        delta_bytes,
        ratio: delta_bytes as f64 / graph_bytes as f64,
        remove_nodes: delta.remove_nodes.len(),
        upsert_nodes: delta.upsert_nodes.len(),
        remove_edges: delta.remove_edges.len(),
        add_edges: delta.add_edges.len(),
        selected,
    }
}

fn mode_name(mode: SemanticTransferMode) -> &'static str {
    match mode {
        SemanticTransferMode::SemanticReference => "reference",
        SemanticTransferMode::GraphDelta => "delta",
        SemanticTransferMode::GroundedGraph => "full-graph",
        SemanticTransferMode::HumanTextFallback => "text",
    }
}

#[test]
fn deterministic_semantic_change_curve_evidence() {
    let base = base_plan();
    let base_graph = graph("change-curve-base", &base);

    let cases = [
        ChangeCase {
            name: "scalar-arousal",
            mutate: scalar_arousal,
        },
        ChangeCase {
            name: "relationship-transition",
            mutate: relationship_transition,
        },
        ChangeCase {
            name: "concept-addition",
            mutate: add_concept,
        },
        ChangeCase {
            name: "concept-replace-1",
            mutate: replace_one_concept,
        },
        ChangeCase {
            name: "concept-replace-4",
            mutate: replace_four_concepts,
        },
        ChangeCase {
            name: "concept-replace-8",
            mutate: replace_all_concepts,
        },
        ChangeCase {
            name: "structured-data-update",
            mutate: structured_data_update,
        },
        ChangeCase {
            name: "domain-result-update",
            mutate: domain_result_update,
        },
        ChangeCase {
            name: "domain-entity-addition",
            mutate: domain_entity_addition,
        },
        ChangeCase {
            name: "epistemic-transition",
            mutate: epistemic_transition,
        },
        ChangeCase {
            name: "broad-transition",
            mutate: broad_transition,
        },
    ];

    let rows = cases
        .into_iter()
        .map(|case| measure_case(&base_graph, case))
        .collect::<Vec<_>>();

    println!("| semantic change | graph B | delta B | delta/full | -nodes | +nodes | -edges | +edges | planner |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---|");
    for row in &rows {
        println!(
            "| {} | {} | {} | {:.2}% | {} | {} | {} | {} | {} |",
            row.name,
            row.graph_bytes,
            row.delta_bytes,
            row.ratio * 100.0,
            row.remove_nodes,
            row.upsert_nodes,
            row.remove_edges,
            row.add_edges,
            mode_name(row.selected),
        );
    }

    let scalar = rows
        .iter()
        .find(|row| row.name == "scalar-arousal")
        .unwrap();
    let replace_one = rows
        .iter()
        .find(|row| row.name == "concept-replace-1")
        .unwrap();
    let replace_four = rows
        .iter()
        .find(|row| row.name == "concept-replace-4")
        .unwrap();
    let replace_eight = rows
        .iter()
        .find(|row| row.name == "concept-replace-8")
        .unwrap();
    let broad = rows
        .iter()
        .find(|row| row.name == "broad-transition")
        .unwrap();

    // Ratchet only qualitative properties that should remain true for this
    // deterministic evidence corpus. Do not bake in a universal break-even.
    assert!(scalar.ratio < 0.10);
    assert!(replace_one.delta_bytes <= replace_four.delta_bytes);
    assert!(replace_four.delta_bytes <= replace_eight.delta_bytes);
    assert!(broad.delta_bytes > scalar.delta_bytes);
}
