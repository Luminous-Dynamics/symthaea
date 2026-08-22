use std::fmt::Write as _;

use symthaea_communication::{
    ConceptEdge, ConceptKind, ConceptNode, GroundedConceptGraph, Provenance,
};
use symthaea_interlingua::{
    CognitiveEnvelope, LlmFallbackMode, LlmTextFallback, graph_semantic_hash,
};

use crate::{
    BROCA_SCIP_TRANSFORM_V1, BrocaRendererPolicy, BrocaScipError, BrocaScipPacket,
    BrocaTranslationPlan, StructuredThoughtScipAdapter, StructuredThoughtScipPolicy,
};

pub const BROCA_FIDELITY_CONTEXT_V1: &str = "symthaea.broca-cognitive-context/v1";
pub const BROCA_FIDELITY_TRANSFORM_V1: &str =
    "broca-translation-plan+context->scip-grounded-translation-plan/v1";

/// Additive fidelity profile layered over the stable v1 Broca translation plan.
///
/// Keeping context additive avoids breaking `BrocaTranslationPlan` while still
/// preserving the cognitive state exposed by the legacy StructuredThought text
/// prompt. Executable primitives and code remain outside this profile.
#[derive(Clone, Debug, PartialEq)]
pub struct BrocaFidelityPlan {
    pub base: BrocaTranslationPlan,
    pub context: BrocaCognitiveContext,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BrocaCognitiveContext {
    /// Ψ — Symthaea's composite consciousness estimate (not IIT Phi).
    pub psi: f64,
    pub valence: f64,
    pub arousal: f64,
    pub relationship_stage: BrocaRelationshipStage,
    pub relation_mode: BrocaRelationMode,
    pub trust: f32,
    pub primitive_tiers: Vec<String>,
    pub domain_epistemic_cube: Option<BrocaEpistemicCube>,
    pub domain_psi: Option<f64>,
}

impl Default for BrocaCognitiveContext {
    fn default() -> Self {
        Self {
            psi: 0.0,
            valence: 0.0,
            arousal: 0.0,
            relationship_stage: BrocaRelationshipStage::NoRelation,
            relation_mode: BrocaRelationMode::IIt,
            trust: 0.0,
            primitive_tiers: vec![],
            domain_epistemic_cube: None,
            domain_psi: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BrocaRelationshipStage {
    #[default]
    NoRelation,
    Awareness,
    Contact,
    Attunement,
    Bonding,
    Unity,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BrocaRelationMode {
    #[default]
    IIt,
    IThou,
}

/// Root-independent form of Symthaea's E/N/M/H epistemic classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrocaEpistemicCube {
    /// Empirical tier: 0..=4.
    pub empirical: u8,
    /// Normative tier: 0..=3.
    pub normative: u8,
    /// Materiality tier: 0..=3.
    pub materiality: u8,
    /// Optional harmonic tier: 0..=4.
    pub harmonic: Option<u8>,
}

/// Typed renderer context derived only from bounded numeric/enumerated state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrocaFidelityRendererContext {
    pub valence: f64,
    pub arousal: f64,
    pub warmth: f64,
    pub relationship_stage: BrocaRelationshipStage,
    pub relation_mode: BrocaRelationMode,
    pub trust: f32,
}

impl BrocaFidelityRendererContext {
    pub(crate) fn from_plan(plan: &BrocaFidelityPlan) -> Result<Self, BrocaScipError> {
        validate_fidelity_context(plan)?;
        Ok(Self {
            valence: plan.context.valence,
            arousal: plan.context.arousal,
            warmth: plan.base.warmth,
            relationship_stage: plan.context.relationship_stage,
            relation_mode: plan.context.relation_mode,
            trust: plan.context.trust,
        })
    }

    /// Fixed-format control text. No free-form semantic string can enter here.
    pub fn system_directive(self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "AFFECT CONTROL: valence={:.3}; arousal={:.3}; warmth={:.3}. Affect may shape style only; it must not alter grounded facts.",
            self.valence, self.arousal, self.warmth
        );
        let _ = writeln!(
            out,
            "RELATION CONTROL: stage={}; mode={}; trust={:.3}. Match the relational stance without inventing intimacy, history, permissions, or claims.",
            relationship_stage_name(self.relationship_stage),
            relation_mode_name(self.relation_mode),
            self.trust
        );
        out
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BrocaFidelityPacket {
    pub packet: BrocaScipPacket,
    pub renderer_context: BrocaFidelityRendererContext,
}

pub struct FidelityBrocaScipAdapter;

impl FidelityBrocaScipAdapter {
    pub fn graph(
        plan: &BrocaFidelityPlan,
        policy: &StructuredThoughtScipPolicy,
    ) -> Result<GroundedConceptGraph, BrocaScipError> {
        let mut graph = StructuredThoughtScipAdapter::graph(&plan.base, policy)?;
        enrich_graph_with_fidelity(plan, &mut graph)?;
        Ok(graph)
    }

    pub fn compile_for_text_peer(
        plan: &BrocaFidelityPlan,
        mood_temperature: f32,
        mut provenance: Provenance,
        policy: &StructuredThoughtScipPolicy,
    ) -> Result<BrocaFidelityPacket, BrocaScipError> {
        let renderer = BrocaRendererPolicy::from_plan(&plan.base, mood_temperature)?;
        let renderer_context = BrocaFidelityRendererContext::from_plan(plan)?;
        let graph = Self::graph(plan, policy)?;
        append_fidelity_transforms(&mut provenance);

        let confidence = plan.base.meta_awareness.min(plan.base.coherence) as f32;
        let envelope = CognitiveEnvelope::from_graph(graph, confidence, provenance)?;
        let mut fallback =
            LlmTextFallback::compile(&envelope, None, LlmFallbackMode::FaithfulTranslation)?;
        append_renderer_directives(&mut fallback.system_prompt, renderer, renderer_context);

        Ok(BrocaFidelityPacket {
            packet: BrocaScipPacket {
                envelope,
                fallback,
                renderer,
            },
            renderer_context,
        })
    }
}

/// Enrich a graph that has already been constructed and bounded by the v1
/// adapter. This is `pub(crate)` so the hardened fidelity wrapper can reuse the
/// exact graph rather than allocating/building it twice.
pub(crate) fn enrich_graph_with_fidelity(
    plan: &BrocaFidelityPlan,
    graph: &mut GroundedConceptGraph,
) -> Result<(), BrocaScipError> {
    validate_fidelity_context(plan)?;

    add_numeric_property(graph, "context-psi", plan.context.psi, "has-psi")?;
    add_numeric_property(
        graph,
        "context-meta-awareness",
        plan.base.meta_awareness,
        "has-meta-awareness",
    )?;
    add_numeric_property(
        graph,
        "context-coherence",
        plan.base.coherence,
        "has-coherence",
    )?;
    add_numeric_property(
        graph,
        "context-affect-valence",
        plan.context.valence,
        "has-affect-valence",
    )?;
    add_numeric_property(
        graph,
        "context-affect-arousal",
        plan.context.arousal,
        "has-affect-arousal",
    )?;
    add_numeric_property(
        graph,
        "context-affect-warmth",
        plan.base.warmth,
        "has-affect-warmth",
    )?;
    add_property(
        graph,
        "context-relationship-stage",
        relationship_stage_name(plan.context.relationship_stage),
        "has-relationship-stage",
        1.0,
    );
    add_property(
        graph,
        "context-relation-mode",
        relation_mode_name(plan.context.relation_mode),
        "has-relation-mode",
        1.0,
    );
    add_numeric_property(
        graph,
        "context-trust",
        f64::from(plan.context.trust),
        "has-trust",
    )?;

    for (index, tier) in plan.context.primitive_tiers.iter().enumerate() {
        let id = format!("context-primitive-tier-{index:04}");
        graph.nodes.push(ConceptNode {
            id: id.clone(),
            kind: ConceptKind::Property,
            label: Some(tier.clone()),
            grounded_by: vec![format!("broca-cognitive-context:primitive-tier:{index}")],
            confidence: 1.0,
        });
        graph.edges.push(edge("thought", "has-primitive-tier", &id, 1.0));
    }

    if let Some(cube) = plan.context.domain_epistemic_cube {
        add_property(
            graph,
            "context-domain-epistemic-cube",
            &epistemic_cube_label(cube),
            "has-domain-epistemic-cube",
            1.0,
        );
    }
    if let Some(domain_psi) = plan.context.domain_psi {
        add_numeric_property(
            graph,
            "context-domain-psi",
            domain_psi,
            "has-domain-psi",
        )?;
    }

    refresh_auto_grounding(graph)?;
    Ok(())
}

pub(crate) fn append_fidelity_transforms(provenance: &mut Provenance) {
    for transform in [BROCA_SCIP_TRANSFORM_V1, BROCA_FIDELITY_TRANSFORM_V1] {
        if !provenance
            .transformations
            .iter()
            .any(|item| item == transform)
        {
            provenance.transformations.push(transform.into());
        }
    }
}

pub(crate) fn append_renderer_directives(
    system_prompt: &mut String,
    renderer: BrocaRendererPolicy,
    renderer_context: BrocaFidelityRendererContext,
) {
    system_prompt.push_str("\n\n");
    system_prompt.push_str(&renderer.system_directive());
    system_prompt.push_str(&renderer_context.system_directive());
}

pub(crate) fn validate_fidelity_context(plan: &BrocaFidelityPlan) -> Result<(), BrocaScipError> {
    checked_unit(plan.context.psi, "psi")?;
    checked_range(plan.context.valence, -1.0, 1.0, "affective valence")?;
    checked_unit(plan.context.arousal, "affective arousal")?;
    checked_unit(f64::from(plan.context.trust), "relationship trust")?;
    if let Some(domain_psi) = plan.context.domain_psi {
        checked_unit(domain_psi, "domain psi")?;
    }
    if let Some(cube) = plan.context.domain_epistemic_cube
        && (cube.empirical > 4
            || cube.normative > 3
            || cube.materiality > 3
            || cube.harmonic.is_some_and(|value| value > 4))
    {
        return Err(BrocaScipError::InvalidPlan(
            "epistemic cube tier is outside its defined range".into(),
        ));
    }
    Ok(())
}

fn checked_unit(value: f64, field: &str) -> Result<(), BrocaScipError> {
    checked_range(value, 0.0, 1.0, field)
}

fn checked_range(
    value: f64,
    minimum: f64,
    maximum: f64,
    field: &str,
) -> Result<(), BrocaScipError> {
    if value.is_finite() && (minimum..=maximum).contains(&value) {
        Ok(())
    } else {
        Err(BrocaScipError::InvalidPlan(format!(
            "{field} must be finite and in [{minimum}, {maximum}]"
        )))
    }
}

fn add_numeric_property(
    graph: &mut GroundedConceptGraph,
    id: &str,
    value: f64,
    relation: &str,
) -> Result<(), BrocaScipError> {
    if !value.is_finite() {
        return Err(BrocaScipError::InvalidPlan(format!(
            "{id} must be finite"
        )));
    }
    let label = serde_json::to_string(&value)?;
    add_property(graph, id, &label, relation, 1.0);
    Ok(())
}

fn add_property(
    graph: &mut GroundedConceptGraph,
    id: &str,
    label: &str,
    relation: &str,
    confidence: f32,
) {
    graph.nodes.push(ConceptNode {
        id: id.into(),
        kind: ConceptKind::Property,
        label: Some(label.into()),
        grounded_by: vec![format!("broca-cognitive-context:{id}")],
        confidence,
    });
    graph.edges.push(edge("thought", relation, id, confidence));
}

fn edge(source: &str, relation: &str, target: &str, confidence: f32) -> ConceptEdge {
    ConceptEdge {
        source: source.into(),
        relation: relation.into(),
        target: target.into(),
        evidence_ids: vec![],
        confidence,
    }
}

fn refresh_auto_grounding(graph: &mut GroundedConceptGraph) -> Result<(), BrocaScipError> {
    let root_index = graph
        .nodes
        .iter()
        .position(|node| node.id == "thought")
        .ok_or_else(|| BrocaScipError::InvalidPlan("missing thought root node".into()))?;
    if graph.nodes[root_index].grounded_by.len() == 1
        && graph.nodes[root_index].grounded_by[0].starts_with("redacted-broca-export:")
    {
        graph.nodes[root_index].grounded_by[0] =
            "internal:redacted-broca-fidelity-export/pending".into();
        let semantic_hash = graph_semantic_hash(graph)?;
        graph.nodes[root_index].grounded_by[0] =
            format!("redacted-broca-fidelity-export:{semantic_hash}");
    }
    Ok(())
}

fn relationship_stage_name(value: BrocaRelationshipStage) -> &'static str {
    match value {
        BrocaRelationshipStage::NoRelation => "no-relation",
        BrocaRelationshipStage::Awareness => "awareness",
        BrocaRelationshipStage::Contact => "contact",
        BrocaRelationshipStage::Attunement => "attunement",
        BrocaRelationshipStage::Bonding => "bonding",
        BrocaRelationshipStage::Unity => "unity",
    }
}

fn relation_mode_name(value: BrocaRelationMode) -> &'static str {
    match value {
        BrocaRelationMode::IIt => "i-it",
        BrocaRelationMode::IThou => "i-thou",
    }
}

fn epistemic_cube_label(cube: BrocaEpistemicCube) -> String {
    match cube.harmonic {
        Some(harmonic) => format!(
            "E{}/N{}/M{}/H{harmonic}",
            cube.empirical, cube.normative, cube.materiality
        ),
        None => format!(
            "E{}/N{}/M{}",
            cube.empirical, cube.normative, cube.materiality
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BrocaConcept, BrocaDomainContext, RendererEpistemicStatus, RendererIntent,
        RendererResponseType,
    };

    fn provenance() -> Provenance {
        Provenance {
            provider: "broca-fidelity-test".into(),
            provider_version: "1".into(),
            model_hash: "internal-fidelity-plan".into(),
            feature_flags: vec![],
            transformations: vec![],
        }
    }

    fn plan() -> BrocaFidelityPlan {
        BrocaFidelityPlan {
            base: BrocaTranslationPlan {
                intent: RendererIntent::Answer,
                response_type: RendererResponseType::Statement,
                epistemic_status: RendererEpistemicStatus::Probable,
                warmth: 0.82,
                meta_awareness: 0.74,
                coherence: 0.81,
                activated_concepts: vec![BrocaConcept {
                    name: "reactor".into(),
                    activation: 0.9,
                    relevance: 0.95,
                }],
                domain_context: Some(BrocaDomainContext {
                    domain: "engineering".into(),
                    entities: vec![],
                    computed_answer: Some("Remain offline.".into()),
                }),
                ..Default::default()
            },
            context: BrocaCognitiveContext {
                psi: 0.68,
                valence: -0.2,
                arousal: 0.35,
                relationship_stage: BrocaRelationshipStage::Attunement,
                relation_mode: BrocaRelationMode::IThou,
                trust: 0.77,
                primitive_tiers: vec!["Strategic".into(), "MetaCognitive".into()],
                domain_epistemic_cube: Some(BrocaEpistemicCube {
                    empirical: 3,
                    normative: 1,
                    materiality: 2,
                    harmonic: Some(3),
                }),
                domain_psi: Some(0.61),
            },
        }
    }

    #[test]
    fn fidelity_graph_preserves_cognitive_context() {
        let graph = FidelityBrocaScipAdapter::graph(
            &plan(),
            &StructuredThoughtScipPolicy::default(),
        )
        .unwrap();
        let labels = graph
            .nodes
            .iter()
            .filter_map(|node| node.label.as_deref())
            .collect::<Vec<_>>();
        assert!(labels.contains(&"attunement"));
        assert!(labels.contains(&"i-thou"));
        assert!(labels.contains(&"Strategic"));
        assert!(labels.contains(&"E3/N1/M2/H3"));
        assert!(graph.nodes[0].grounded_by[0].starts_with("redacted-broca-fidelity-export:"));
    }

    #[test]
    fn fidelity_renderer_control_contains_only_typed_context() {
        let packet = FidelityBrocaScipAdapter::compile_for_text_peer(
            &plan(),
            1.0,
            provenance(),
            &StructuredThoughtScipPolicy::default(),
        )
        .unwrap();
        assert!(packet.packet.fallback.system_prompt.contains("AFFECT CONTROL"));
        assert!(packet.packet.fallback.system_prompt.contains("RELATION CONTROL"));
        assert!(packet.packet.fallback.system_prompt.contains("stage=attunement"));
        assert!(packet.packet.fallback.system_prompt.contains("mode=i-thou"));
    }

    #[test]
    fn invalid_epistemic_cube_is_rejected() {
        let mut plan = plan();
        plan.context.domain_epistemic_cube.as_mut().unwrap().empirical = 5;
        assert!(
            FidelityBrocaScipAdapter::graph(&plan, &StructuredThoughtScipPolicy::default())
                .is_err()
        );
    }

    #[test]
    fn invalid_affect_is_rejected_before_export() {
        let mut plan = plan();
        plan.context.valence = f64::NAN;
        assert!(
            FidelityBrocaScipAdapter::graph(&plan, &StructuredThoughtScipPolicy::default())
                .is_err()
        );
    }
}
