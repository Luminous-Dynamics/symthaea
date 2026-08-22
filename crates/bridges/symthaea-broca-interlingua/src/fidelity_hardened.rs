use std::fmt;

use symthaea_communication::Provenance;
use symthaea_interlingua::{
    CognitiveEnvelope, InterchangePayload, LlmFallbackMode, LlmTextFallback,
};

use crate::fidelity::{
    append_fidelity_transforms, append_renderer_directives, enrich_graph_with_fidelity,
    validate_fidelity_context,
};
use crate::{
    BROCA_FIDELITY_CONTEXT_V1, BrocaExportAudit, BrocaFidelityPacket, BrocaFidelityPlan,
    BrocaFidelityRendererContext, BrocaInterchangeLimits, BrocaScipError, BrocaScipPacket,
    HardenedBrocaError, HardenedBrocaScipAdapter, StructuredThoughtScipPolicy,
};

/// Policy for the hardened fidelity path.
///
/// The defaults reject known semantic loss. Experiments may explicitly permit
/// a loss class, but the audit records it and marks the export non-faithful.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrocaFidelityExportPolicy {
    pub base: StructuredThoughtScipPolicy,
    pub allow_legacy_constraint_loss: bool,
    pub allow_concept_truncation: bool,
    pub allow_structured_data_omission: bool,
    pub allow_domain_context_omission: bool,
}

impl Default for BrocaFidelityExportPolicy {
    fn default() -> Self {
        Self {
            base: StructuredThoughtScipPolicy::default(),
            allow_legacy_constraint_loss: false,
            allow_concept_truncation: false,
            allow_structured_data_omission: false,
            allow_domain_context_omission: false,
        }
    }
}

/// Additional pre-allocation ceilings for the additive fidelity context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrocaFidelityInterchangeLimits {
    pub base: BrocaInterchangeLimits,
    pub max_primitive_tiers: usize,
    pub max_primitive_tier_bytes: usize,
    pub max_context_text_bytes: usize,
}

impl Default for BrocaFidelityInterchangeLimits {
    fn default() -> Self {
        Self {
            base: BrocaInterchangeLimits::default(),
            max_primitive_tiers: 64,
            max_primitive_tier_bytes: 8 * 1024,
            max_context_text_bytes: 64 * 1024,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BrocaSemanticLoss {
    LegacyConstraintSemantics { count: usize },
    ActivatedConceptsTruncated { omitted: usize },
    StructuredDataOmitted,
    DomainContextOmitted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrocaFidelityExportAudit {
    pub base: BrocaExportAudit,
    pub context_profile: String,
    pub primitive_tiers_exported: usize,
    pub semantic_losses: Vec<BrocaSemanticLoss>,
    pub faithful_translation: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HardenedBrocaFidelityPacket {
    pub packet: BrocaFidelityPacket,
    pub audit: BrocaFidelityExportAudit,
}

pub struct HardenedFidelityBrocaScipAdapter;

impl HardenedFidelityBrocaScipAdapter {
    pub fn compile_for_text_peer(
        plan: &BrocaFidelityPlan,
        mood_temperature: f32,
        provenance: Provenance,
        policy: &BrocaFidelityExportPolicy,
        limits: &BrocaFidelityInterchangeLimits,
    ) -> Result<HardenedBrocaFidelityPacket, HardenedBrocaFidelityError> {
        validate_fidelity_context(plan)?;
        validate_fidelity_limits(plan, limits)?;

        let losses = detect_semantic_losses(plan, policy);
        reject_unapproved_losses(&losses, policy)?;

        // Build and validate the bounded v1 graph once. We then enrich the
        // owned graph in place rather than constructing the base graph again.
        let base_result = HardenedBrocaScipAdapter::compile_for_text_peer(
            &plan.base,
            mood_temperature,
            provenance,
            &policy.base,
            &limits.base,
        )?;
        let mut base_audit = base_result.audit;
        let base_packet = base_result.packet;
        let renderer = base_packet.renderer;
        let renderer_context = BrocaFidelityRendererContext::from_plan(plan)?;
        let confidence = base_packet.envelope.confidence;
        let mut provenance = base_packet.envelope.provenance;
        let InterchangePayload::GroundedGraph(mut graph) = base_packet.envelope.payload else {
            return Err(HardenedBrocaFidelityError::InvariantViolation(
                "hardened Broca v1 adapter did not return a grounded graph".into(),
            ));
        };

        enrich_graph_with_fidelity(plan, &mut graph)?;
        append_fidelity_transforms(&mut provenance);

        let envelope = CognitiveEnvelope::from_graph(graph, confidence, provenance)?;
        let mut fallback =
            LlmTextFallback::compile(&envelope, None, LlmFallbackMode::FaithfulTranslation)?;
        append_renderer_directives(&mut fallback.system_prompt, renderer, renderer_context);
        append_semantic_loss_directive(&mut fallback.system_prompt, &losses);
        base_audit.semantic_hash = fallback.semantic_hash.clone();

        Ok(HardenedBrocaFidelityPacket {
            packet: BrocaFidelityPacket {
                packet: BrocaScipPacket {
                    envelope,
                    fallback,
                    renderer,
                },
                renderer_context,
            },
            audit: BrocaFidelityExportAudit {
                base: base_audit,
                context_profile: BROCA_FIDELITY_CONTEXT_V1.into(),
                primitive_tiers_exported: plan.context.primitive_tiers.len(),
                faithful_translation: losses.is_empty(),
                semantic_losses: losses,
            },
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HardenedBrocaFidelityError {
    InvalidLimit(String),
    SemanticLossRejected(String),
    InvariantViolation(String),
    Base(String),
    Interchange(String),
}

impl fmt::Display for HardenedBrocaFidelityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimit(value) => write!(f, "invalid Broca fidelity limit: {value}"),
            Self::SemanticLossRejected(value) => {
                write!(f, "Broca fidelity export rejected semantic loss: {value}")
            }
            Self::InvariantViolation(value) => {
                write!(f, "Broca fidelity invariant violated: {value}")
            }
            Self::Base(value) => write!(f, "hardened Broca v1 export failed: {value}"),
            Self::Interchange(value) => write!(f, "Broca fidelity SCIP export failed: {value}"),
        }
    }
}

impl std::error::Error for HardenedBrocaFidelityError {}

impl From<HardenedBrocaError> for HardenedBrocaFidelityError {
    fn from(value: HardenedBrocaError) -> Self {
        Self::Base(value.to_string())
    }
}

impl From<BrocaScipError> for HardenedBrocaFidelityError {
    fn from(value: BrocaScipError) -> Self {
        Self::Interchange(value.to_string())
    }
}

impl From<symthaea_interlingua::InterchangeError> for HardenedBrocaFidelityError {
    fn from(value: symthaea_interlingua::InterchangeError) -> Self {
        Self::Interchange(value.to_string())
    }
}

fn validate_fidelity_limits(
    plan: &BrocaFidelityPlan,
    limits: &BrocaFidelityInterchangeLimits,
) -> Result<(), HardenedBrocaFidelityError> {
    if limits.max_primitive_tiers == 0
        || limits.max_primitive_tier_bytes == 0
        || limits.max_context_text_bytes == 0
    {
        return Err(HardenedBrocaFidelityError::InvalidLimit(
            "all fidelity ceilings must be greater than zero".into(),
        ));
    }
    if limits.max_primitive_tier_bytes > limits.base.max_string_bytes {
        return Err(HardenedBrocaFidelityError::InvalidLimit(
            "max_primitive_tier_bytes cannot exceed the base per-string ceiling".into(),
        ));
    }
    if limits.max_context_text_bytes > limits.base.max_export_text_bytes {
        return Err(HardenedBrocaFidelityError::InvalidLimit(
            "max_context_text_bytes cannot exceed the base aggregate text ceiling".into(),
        ));
    }
    if plan.context.primitive_tiers.len() > limits.max_primitive_tiers {
        return Err(HardenedBrocaFidelityError::InvalidLimit(format!(
            "{} primitive tiers; maximum is {}",
            plan.context.primitive_tiers.len(),
            limits.max_primitive_tiers
        )));
    }

    let mut context_text_bytes = 0usize;
    for tier in &plan.context.primitive_tiers {
        if tier.len() > limits.max_primitive_tier_bytes {
            return Err(HardenedBrocaFidelityError::InvalidLimit(format!(
                "primitive tier is {} bytes; maximum is {}",
                tier.len(),
                limits.max_primitive_tier_bytes
            )));
        }
        context_text_bytes = context_text_bytes.checked_add(tier.len()).ok_or_else(|| {
            HardenedBrocaFidelityError::InvalidLimit(
                "primitive-tier byte accounting overflowed".into(),
            )
        })?;
        if context_text_bytes > limits.max_context_text_bytes {
            return Err(HardenedBrocaFidelityError::InvalidLimit(format!(
                "fidelity context contains {context_text_bytes} text bytes; maximum is {}",
                limits.max_context_text_bytes
            )));
        }
    }
    Ok(())
}

fn detect_semantic_losses(
    plan: &BrocaFidelityPlan,
    policy: &BrocaFidelityExportPolicy,
) -> Vec<BrocaSemanticLoss> {
    let mut losses = Vec::new();
    if !plan.base.constraints.is_empty() {
        losses.push(BrocaSemanticLoss::LegacyConstraintSemantics {
            count: plan.base.constraints.len(),
        });
    }
    if plan.base.activated_concepts.len() > policy.base.max_activated_concepts {
        losses.push(BrocaSemanticLoss::ActivatedConceptsTruncated {
            omitted: plan
                .base
                .activated_concepts
                .len()
                .saturating_sub(policy.base.max_activated_concepts),
        });
    }
    if plan.base.structured_data.is_some() && !policy.base.include_structured_data {
        losses.push(BrocaSemanticLoss::StructuredDataOmitted);
    }
    if plan.base.domain_context.is_some() && !policy.base.include_domain_context {
        losses.push(BrocaSemanticLoss::DomainContextOmitted);
    }
    losses
}

fn reject_unapproved_losses(
    losses: &[BrocaSemanticLoss],
    policy: &BrocaFidelityExportPolicy,
) -> Result<(), HardenedBrocaFidelityError> {
    for loss in losses {
        match loss {
            BrocaSemanticLoss::LegacyConstraintSemantics { count }
                if !policy.allow_legacy_constraint_loss =>
            {
                return Err(HardenedBrocaFidelityError::SemanticLossRejected(format!(
                    "{count} legacy free-form constraint(s) cannot be promoted to trusted renderer control"
                )));
            }
            BrocaSemanticLoss::ActivatedConceptsTruncated { omitted }
                if !policy.allow_concept_truncation =>
            {
                return Err(HardenedBrocaFidelityError::SemanticLossRejected(format!(
                    "{omitted} activated concept(s) would be truncated"
                )));
            }
            BrocaSemanticLoss::StructuredDataOmitted if !policy.allow_structured_data_omission => {
                return Err(HardenedBrocaFidelityError::SemanticLossRejected(
                    "structured data would be omitted by export policy".into(),
                ));
            }
            BrocaSemanticLoss::DomainContextOmitted if !policy.allow_domain_context_omission => {
                return Err(HardenedBrocaFidelityError::SemanticLossRejected(
                    "domain context would be omitted by export policy".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Make explicitly approved loss visible to the receiving text peer as trusted,
/// fixed-vocabulary control metadata. No plan/user string is interpolated here.
fn append_semantic_loss_directive(system_prompt: &mut String, losses: &[BrocaSemanticLoss]) {
    if losses.is_empty() {
        return;
    }

    system_prompt.push_str(
        "\nSEMANTIC LOSS CONTROL: This packet is an intentionally lossy projection. \
         Do not describe it as a complete or fully faithful rendering of the source cognitive state. \
         Omitted semantic classes: ",
    );
    for (index, loss) in losses.iter().enumerate() {
        if index > 0 {
            system_prompt.push_str(", ");
        }
        system_prompt.push_str(match loss {
            BrocaSemanticLoss::LegacyConstraintSemantics { .. } => "legacy-constraint-semantics",
            BrocaSemanticLoss::ActivatedConceptsTruncated { .. } => "activated-concepts",
            BrocaSemanticLoss::StructuredDataOmitted => "structured-data",
            BrocaSemanticLoss::DomainContextOmitted => "domain-context",
        });
    }
    system_prompt.push_str(".\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BrocaCognitiveContext, BrocaConcept, BrocaConstraint, BrocaConstraintKind,
        BrocaDomainContext, BrocaStructuredData, RendererEpistemicStatus, RendererIntent,
        RendererResponseType,
    };

    fn provenance() -> Provenance {
        Provenance {
            provider: "hardened-fidelity-test".into(),
            provider_version: "1".into(),
            model_hash: "fidelity-plan".into(),
            feature_flags: vec![],
            transformations: vec![],
        }
    }

    fn plan() -> BrocaFidelityPlan {
        BrocaFidelityPlan {
            base: crate::BrocaTranslationPlan {
                intent: RendererIntent::Answer,
                response_type: RendererResponseType::Statement,
                epistemic_status: RendererEpistemicStatus::Probable,
                warmth: 0.7,
                meta_awareness: 0.8,
                coherence: 0.9,
                activated_concepts: vec![BrocaConcept {
                    name: "reactor".into(),
                    activation: 0.9,
                    relevance: 0.9,
                }],
                ..Default::default()
            },
            context: BrocaCognitiveContext {
                psi: 0.65,
                valence: 0.1,
                arousal: 0.3,
                primitive_tiers: vec!["Strategic".into()],
                ..Default::default()
            },
        }
    }

    #[test]
    fn loss_free_export_is_marked_faithful() {
        let result = HardenedFidelityBrocaScipAdapter::compile_for_text_peer(
            &plan(),
            1.0,
            provenance(),
            &BrocaFidelityExportPolicy::default(),
            &BrocaFidelityInterchangeLimits::default(),
        )
        .unwrap();
        assert!(result.audit.faithful_translation);
        assert!(result.audit.semantic_losses.is_empty());
        assert_eq!(result.audit.primitive_tiers_exported, 1);
        assert_eq!(
            result.audit.base.semantic_hash,
            result.packet.packet.fallback.semantic_hash
        );
        assert!(
            !result
                .packet
                .packet
                .fallback
                .system_prompt
                .contains("SEMANTIC LOSS CONTROL")
        );
    }

    #[test]
    fn legacy_constraint_semantics_fail_closed_by_default() {
        let mut plan = plan();
        plan.base.constraints.push(BrocaConstraint {
            kind: BrocaConstraintKind::MustInclude,
            audit_text: "mention the shutdown reason".into(),
        });
        assert!(matches!(
            HardenedFidelityBrocaScipAdapter::compile_for_text_peer(
                &plan,
                1.0,
                provenance(),
                &BrocaFidelityExportPolicy::default(),
                &BrocaFidelityInterchangeLimits::default(),
            ),
            Err(HardenedBrocaFidelityError::SemanticLossRejected(_))
        ));
    }

    #[test]
    fn explicitly_allowed_constraint_loss_is_audited_and_peer_visible() {
        let mut plan = plan();
        plan.base.constraints.push(BrocaConstraint {
            kind: BrocaConstraintKind::MustInclude,
            audit_text: "mention the shutdown reason".into(),
        });
        let policy = BrocaFidelityExportPolicy {
            allow_legacy_constraint_loss: true,
            ..Default::default()
        };
        let result = HardenedFidelityBrocaScipAdapter::compile_for_text_peer(
            &plan,
            1.0,
            provenance(),
            &policy,
            &BrocaFidelityInterchangeLimits::default(),
        )
        .unwrap();
        assert!(!result.audit.faithful_translation);
        assert_eq!(
            result.audit.semantic_losses,
            vec![BrocaSemanticLoss::LegacyConstraintSemantics { count: 1 }]
        );
        assert!(
            result
                .packet
                .packet
                .fallback
                .system_prompt
                .contains("SEMANTIC LOSS CONTROL")
        );
        assert!(
            result
                .packet
                .packet
                .fallback
                .system_prompt
                .contains("legacy-constraint-semantics")
        );
    }

    #[test]
    fn concept_truncation_fails_closed_by_default() {
        let mut plan = plan();
        plan.base
            .activated_concepts
            .extend((0..20).map(|index| BrocaConcept {
                name: format!("concept-{index}"),
                activation: 0.5,
                relevance: 0.5,
            }));
        assert!(matches!(
            HardenedFidelityBrocaScipAdapter::compile_for_text_peer(
                &plan,
                1.0,
                provenance(),
                &BrocaFidelityExportPolicy::default(),
                &BrocaFidelityInterchangeLimits::default(),
            ),
            Err(HardenedBrocaFidelityError::SemanticLossRejected(_))
        ));
    }

    #[test]
    fn structured_data_omission_fails_closed_by_default() {
        let mut plan = plan();
        plan.base.structured_data = Some(BrocaStructuredData::List(vec!["shutdown".into()]));
        let policy = BrocaFidelityExportPolicy {
            base: StructuredThoughtScipPolicy {
                include_structured_data: false,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(matches!(
            HardenedFidelityBrocaScipAdapter::compile_for_text_peer(
                &plan,
                1.0,
                provenance(),
                &policy,
                &BrocaFidelityInterchangeLimits::default(),
            ),
            Err(HardenedBrocaFidelityError::SemanticLossRejected(_))
        ));
    }

    #[test]
    fn explicitly_allowed_domain_omission_is_audited_and_peer_visible() {
        let mut plan = plan();
        plan.base.domain_context = Some(BrocaDomainContext {
            domain: "engineering".into(),
            entities: vec![],
            computed_answer: Some("Remain offline.".into()),
        });
        let policy = BrocaFidelityExportPolicy {
            base: StructuredThoughtScipPolicy {
                include_domain_context: false,
                ..Default::default()
            },
            allow_domain_context_omission: true,
            ..Default::default()
        };
        let result = HardenedFidelityBrocaScipAdapter::compile_for_text_peer(
            &plan,
            1.0,
            provenance(),
            &policy,
            &BrocaFidelityInterchangeLimits::default(),
        )
        .unwrap();
        assert!(!result.audit.faithful_translation);
        assert_eq!(
            result.audit.semantic_losses,
            vec![BrocaSemanticLoss::DomainContextOmitted]
        );
        assert!(
            result
                .packet
                .packet
                .fallback
                .system_prompt
                .contains("SEMANTIC LOSS CONTROL")
        );
        assert!(
            result
                .packet
                .packet
                .fallback
                .system_prompt
                .contains("domain-context")
        );
    }

    #[test]
    fn primitive_tier_limits_apply_before_enrichment() {
        let mut plan = plan();
        plan.context.primitive_tiers = vec!["x".repeat(65)];
        let limits = BrocaFidelityInterchangeLimits {
            max_primitive_tier_bytes: 64,
            max_context_text_bytes: 128,
            ..Default::default()
        };
        assert!(matches!(
            HardenedFidelityBrocaScipAdapter::compile_for_text_peer(
                &plan,
                1.0,
                provenance(),
                &BrocaFidelityExportPolicy::default(),
                &limits,
            ),
            Err(HardenedBrocaFidelityError::InvalidLimit(_))
        ));
    }
}
