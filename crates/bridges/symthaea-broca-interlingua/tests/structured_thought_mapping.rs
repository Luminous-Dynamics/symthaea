use symthaea::mind::{
    ConstraintType, DomainContext, ETier, EpistemicCube, EpistemicStatus, HTier, MTier, NTier,
    ResponseType, SemanticIntent, StructuredData, StructuredThought,
};
use symthaea_broca_interlingua::{
    BrocaCognitiveContext, BrocaConcept, BrocaConstraint, BrocaConstraintKind, BrocaDomainContext,
    BrocaEntity, BrocaEpistemicCube, BrocaFidelityPlan, BrocaRelationMode, BrocaRelationshipStage,
    BrocaStructuredData, BrocaTranslationPlan, FidelityBrocaScipAdapter, RendererEpistemicStatus,
    RendererIntent, RendererResponseType, StructuredThoughtScipAdapter,
    StructuredThoughtScipPolicy,
};
use symthaea_core::hdc::relational_consciousness::{RelationMode, RelationshipStage};
use symthaea_interlingua::graph_semantic_hash;

/// Reference mapping for the future root-side LLMOrgan integration.
///
/// It intentionally lives in tests today: production dependency direction stays
/// `symthaea -> bridge`-capable instead of `bridge -> symthaea`.
fn plan_from_structured_thought(thought: &StructuredThought) -> BrocaTranslationPlan {
    let structured_data = thought
        .structured_data
        .as_ref()
        .and_then(|data| match data {
            StructuredData::List(items) => Some(BrocaStructuredData::List(items.clone())),
            StructuredData::KeyValue(items) => Some(BrocaStructuredData::KeyValue(items.clone())),
            StructuredData::Numeric { value, unit } => Some(BrocaStructuredData::Numeric {
                value: *value,
                unit: unit.clone(),
            }),
            StructuredData::Code { .. } => Some(BrocaStructuredData::Code),
            StructuredData::None => None,
        });

    BrocaTranslationPlan {
        intent: match thought.semantic_intent {
            SemanticIntent::Acknowledge => RendererIntent::Acknowledge,
            SemanticIntent::Answer => RendererIntent::Answer,
            SemanticIntent::Clarify => RendererIntent::Clarify,
            SemanticIntent::ProposeAction => RendererIntent::ProposeAction,
            SemanticIntent::ExpressUncertainty => RendererIntent::ExpressUncertainty,
            SemanticIntent::Reflect => RendererIntent::Reflect,
            SemanticIntent::Continue => RendererIntent::Continue,
            SemanticIntent::Unknown => RendererIntent::Unknown,
        },
        response_type: match thought.response_type {
            ResponseType::Greeting => RendererResponseType::Greeting,
            ResponseType::Statement => RendererResponseType::Statement,
            ResponseType::Question => RendererResponseType::Question,
            ResponseType::ActionConfirmation => RendererResponseType::ActionConfirmation,
            ResponseType::Report => RendererResponseType::Report,
            ResponseType::Empathic => RendererResponseType::Empathic,
        },
        epistemic_status: match thought.epistemic_status {
            EpistemicStatus::Certain => RendererEpistemicStatus::Certain,
            EpistemicStatus::Probable => RendererEpistemicStatus::Probable,
            EpistemicStatus::Uncertain => RendererEpistemicStatus::Uncertain,
            EpistemicStatus::Unknown => RendererEpistemicStatus::Unknown,
            EpistemicStatus::OutOfDomain => RendererEpistemicStatus::OutOfDomain,
        },
        warmth: thought.emotional_tone.warmth,
        meta_awareness: thought.meta_awareness,
        coherence: thought.coherence,
        activated_concepts: thought
            .activated_concepts
            .iter()
            .map(|concept| BrocaConcept {
                name: concept.name.clone(),
                activation: concept.activation,
                relevance: concept.relevance,
            })
            .collect(),
        structured_data,
        domain_context: thought
            .domain_context
            .as_ref()
            .map(|domain| BrocaDomainContext {
                domain: domain.domain.clone(),
                entities: domain
                    .entities
                    .iter()
                    .map(|(entity_type, value, confidence)| BrocaEntity {
                        entity_type: entity_type.clone(),
                        value: value.clone(),
                        confidence: *confidence,
                    })
                    .collect(),
                computed_answer: domain.computed_answer.clone(),
            }),
        constraints: thought
            .constraints
            .iter()
            .map(|constraint| BrocaConstraint {
                kind: match constraint.constraint_type {
                    ConstraintType::MaxLength => BrocaConstraintKind::MaxLength,
                    ConstraintType::Tone => BrocaConstraintKind::Tone,
                    ConstraintType::MustInclude => BrocaConstraintKind::MustInclude,
                    ConstraintType::MustExclude => BrocaConstraintKind::MustExclude,
                    ConstraintType::Format => BrocaConstraintKind::Format,
                },
                audit_text: constraint.instruction.clone(),
            })
            .collect(),
        original_input: thought.original_input.clone(),
        code_bearing: thought.code_context.is_some()
            || matches!(&thought.structured_data, Some(StructuredData::Code { .. })),
    }
}

fn fidelity_plan_from_structured_thought(thought: &StructuredThought) -> BrocaFidelityPlan {
    BrocaFidelityPlan {
        base: plan_from_structured_thought(thought),
        context: BrocaCognitiveContext {
            psi: thought.psi,
            valence: thought.emotional_tone.valence,
            arousal: thought.emotional_tone.arousal,
            relationship_stage: map_relationship_stage(thought.relationship_stage),
            relation_mode: map_relation_mode(thought.relation_mode),
            trust: thought.trust,
            primitive_tiers: thought.primitive_tiers.clone(),
            domain_epistemic_cube: thought
                .domain_context
                .as_ref()
                .and_then(|domain| domain.cube)
                .map(map_epistemic_cube),
            domain_psi: thought
                .domain_context
                .as_ref()
                .and_then(|domain| domain.psi),
        },
    }
}

fn map_relationship_stage(value: RelationshipStage) -> BrocaRelationshipStage {
    match value {
        RelationshipStage::NoRelation => BrocaRelationshipStage::NoRelation,
        RelationshipStage::Awareness => BrocaRelationshipStage::Awareness,
        RelationshipStage::Contact => BrocaRelationshipStage::Contact,
        RelationshipStage::Attunement => BrocaRelationshipStage::Attunement,
        RelationshipStage::Bonding => BrocaRelationshipStage::Bonding,
        RelationshipStage::Unity => BrocaRelationshipStage::Unity,
    }
}

fn map_relation_mode(value: RelationMode) -> BrocaRelationMode {
    match value {
        RelationMode::IIt => BrocaRelationMode::IIt,
        RelationMode::IThou => BrocaRelationMode::IThou,
    }
}

fn map_epistemic_cube(cube: EpistemicCube) -> BrocaEpistemicCube {
    BrocaEpistemicCube {
        empirical: match cube.e {
            ETier::E0 => 0,
            ETier::E1 => 1,
            ETier::E2 => 2,
            ETier::E3 => 3,
            ETier::E4 => 4,
        },
        normative: match cube.n {
            NTier::N0 => 0,
            NTier::N1 => 1,
            NTier::N2 => 2,
            NTier::N3 => 3,
        },
        materiality: match cube.m {
            MTier::M0 => 0,
            MTier::M1 => 1,
            MTier::M2 => 2,
            MTier::M3 => 3,
        },
        harmonic: cube.h.map(|value| match value {
            HTier::H0 => 0,
            HTier::H1 => 1,
            HTier::H2 => 2,
            HTier::H3 => 3,
            HTier::H4 => 4,
        }),
    }
}

fn thought() -> StructuredThought {
    StructuredThought {
        semantic_intent: SemanticIntent::Answer,
        response_type: ResponseType::Statement,
        epistemic_status: EpistemicStatus::Probable,
        meta_awareness: 0.8,
        coherence: 0.9,
        emotional_tone: symthaea::mind::EmotionalTone {
            warmth: 0.75,
            ..Default::default()
        },
        original_input: Some("private utterance A".into()),
        ..Default::default()
    }
}

#[test]
fn real_structured_thought_maps_to_valid_grounded_plan() {
    let plan = plan_from_structured_thought(&thought());
    let graph = StructuredThoughtScipAdapter::graph(&plan, &StructuredThoughtScipPolicy::default())
        .expect("mapped real StructuredThought should produce a grounded graph");
    assert!(
        graph
            .nodes
            .iter()
            .any(|node| node.label.as_deref() == Some("answer"))
    );
    assert!(graph.nodes[0].grounded_by[0].starts_with("redacted-broca-export:"));
}

#[test]
fn redacted_real_thought_input_does_not_change_semantic_hash() {
    let first = thought();
    let mut second = first.clone();
    second.original_input = Some("private utterance B".into());
    let policy = StructuredThoughtScipPolicy::default();
    let first_graph =
        StructuredThoughtScipAdapter::graph(&plan_from_structured_thought(&first), &policy)
            .unwrap();
    let second_graph =
        StructuredThoughtScipAdapter::graph(&plan_from_structured_thought(&second), &policy)
            .unwrap();
    assert_eq!(
        graph_semantic_hash(&first_graph).unwrap(),
        graph_semantic_hash(&second_graph).unwrap()
    );
}

#[test]
fn real_code_structured_data_maps_to_native_path_marker() {
    let mut thought = thought();
    thought.structured_data = Some(StructuredData::Code {
        language: "rust".into(),
        content: "fn main() {}".into(),
    });
    let plan = plan_from_structured_thought(&thought);
    assert!(plan.code_bearing);
    assert!(
        StructuredThoughtScipAdapter::graph(&plan, &StructuredThoughtScipPolicy::default())
            .is_err()
    );
}

#[test]
fn real_structured_thought_preserves_fidelity_context() {
    let mut thought = thought();
    thought.psi = 0.67;
    thought.emotional_tone.valence = -0.22;
    thought.emotional_tone.arousal = 0.36;
    thought.emotional_tone.warmth = 0.81;
    thought.relationship_stage = RelationshipStage::Attunement;
    thought.relation_mode = RelationMode::IThou;
    thought.trust = 0.79;
    thought.primitive_tiers = vec!["Strategic".into(), "MetaCognitive".into()];
    thought.domain_context = Some(DomainContext {
        domain: "engineering".into(),
        entities: vec![],
        computed_answer: Some("Remain offline.".into()),
        cube: Some(EpistemicCube::with_harmonic(
            ETier::E3,
            NTier::N1,
            MTier::M2,
            HTier::H3,
        )),
        psi: Some(0.59),
    });

    let plan = fidelity_plan_from_structured_thought(&thought);
    let packet = FidelityBrocaScipAdapter::compile_for_text_peer(
        &plan,
        1.0,
        symthaea_communication::Provenance {
            provider: "real-thought-fidelity-test".into(),
            provider_version: "1".into(),
            model_hash: "root-structured-thought".into(),
            feature_flags: vec![],
            transformations: vec![],
        },
        &StructuredThoughtScipPolicy::default(),
    )
    .unwrap();

    let content = &packet.packet.fallback.content;
    assert!(content.contains("attunement"));
    assert!(content.contains("i-thou"));
    assert!(content.contains("Strategic"));
    assert!(content.contains("MetaCognitive"));
    assert!(content.contains("E3/N1/M2/H3"));
    assert!(
        packet
            .packet
            .fallback
            .system_prompt
            .contains("AFFECT CONTROL")
    );
    assert!(
        packet
            .packet
            .fallback
            .system_prompt
            .contains("RELATION CONTROL")
    );
}
