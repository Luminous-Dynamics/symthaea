use symthaea::mind::{
    ConstraintType, EpistemicStatus, ResponseType, SemanticIntent, StructuredData,
    StructuredThought,
};
use symthaea_broca_interlingua::{
    BrocaConcept, BrocaConstraint, BrocaConstraintKind, BrocaDomainContext, BrocaEntity,
    BrocaStructuredData, BrocaTranslationPlan, RendererEpistemicStatus, RendererIntent,
    RendererResponseType, StructuredThoughtScipAdapter, StructuredThoughtScipPolicy,
};
use symthaea_interlingua::graph_semantic_hash;

/// Reference mapping for the future root-side LLMOrgan integration.
///
/// It intentionally lives in tests today: production dependency direction stays
/// `symthaea -> bridge`-capable instead of `bridge -> symthaea`.
fn plan_from_structured_thought(thought: &StructuredThought) -> BrocaTranslationPlan {
    let structured_data = thought.structured_data.as_ref().and_then(|data| match data {
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
        domain_context: thought.domain_context.as_ref().map(|domain| BrocaDomainContext {
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
            || matches!(thought.structured_data, Some(StructuredData::Code { .. })),
    }
}

fn thought() -> StructuredThought {
    let mut thought = StructuredThought::default();
    thought.semantic_intent = SemanticIntent::Answer;
    thought.response_type = ResponseType::Statement;
    thought.epistemic_status = EpistemicStatus::Probable;
    thought.meta_awareness = 0.8;
    thought.coherence = 0.9;
    thought.emotional_tone.warmth = 0.75;
    thought.original_input = Some("private utterance A".into());
    thought
}

#[test]
fn real_structured_thought_maps_to_valid_grounded_plan() {
    let plan = plan_from_structured_thought(&thought());
    let graph = StructuredThoughtScipAdapter::graph(&plan, &StructuredThoughtScipPolicy::default())
        .expect("mapped real StructuredThought should produce a grounded graph");
    assert!(graph.nodes.iter().any(|node| node.label.as_deref() == Some("answer")));
    assert!(graph.nodes[0].grounded_by[0].starts_with("redacted-broca-export:"));
}

#[test]
fn redacted_real_thought_input_does_not_change_semantic_hash() {
    let first = thought();
    let mut second = first.clone();
    second.original_input = Some("private utterance B".into());
    let policy = StructuredThoughtScipPolicy::default();
    let first_graph = StructuredThoughtScipAdapter::graph(
        &plan_from_structured_thought(&first),
        &policy,
    )
    .unwrap();
    let second_graph = StructuredThoughtScipAdapter::graph(
        &plan_from_structured_thought(&second),
        &policy,
    )
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
    assert!(StructuredThoughtScipAdapter::graph(
        &plan,
        &StructuredThoughtScipPolicy::default()
    )
    .is_err());
}
