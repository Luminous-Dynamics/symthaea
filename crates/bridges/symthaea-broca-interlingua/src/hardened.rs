use std::fmt;

use symthaea_communication::Provenance;

use crate::{
    BrocaScipError, BrocaScipPacket, BrocaStructuredData, BrocaTranslationPlan,
    StructuredThoughtScipAdapter, StructuredThoughtScipPolicy,
};

/// Pre-allocation ceilings for exporting a Broca translation plan.
///
/// SCIP validates the finished envelope as well, but these limits are applied
/// before building the intermediate graph so a pathological plan cannot force
/// an arbitrarily large allocation first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrocaInterchangeLimits {
    pub max_activated_concepts: usize,
    pub max_constraints: usize,
    pub max_domain_entities: usize,
    pub max_list_items: usize,
    pub max_key_value_pairs: usize,
    pub max_string_bytes: usize,
    pub max_export_text_bytes: usize,
}

impl Default for BrocaInterchangeLimits {
    fn default() -> Self {
        Self {
            max_activated_concepts: 64,
            max_constraints: 64,
            max_domain_entities: 128,
            max_list_items: 128,
            max_key_value_pairs: 128,
            max_string_bytes: 8 * 1024,
            max_export_text_bytes: 256 * 1024,
        }
    }
}

/// Observable account of what was exported versus deliberately withheld.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrocaExportAudit {
    pub exported_concepts: usize,
    pub omitted_concepts: usize,
    pub exported_constraint_kinds: usize,
    pub exported_constraint_texts: usize,
    pub omitted_constraint_texts: usize,
    pub original_input_exported: bool,
    pub original_input_redacted: bool,
    pub structured_data_exported: bool,
    pub domain_context_exported: bool,
    pub semantic_hash: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HardenedBrocaScipPacket {
    pub packet: BrocaScipPacket,
    pub audit: BrocaExportAudit,
}

/// Preferred boundary for exporting a Broca plan to heterogeneous/text peers.
///
/// This wrapper deliberately validates only data that the selected export
/// policy will reveal. A large private `original_input`, for example, is not
/// inspected or charged against the peer-facing text budget when it is
/// redacted by policy.
pub struct HardenedBrocaScipAdapter;

impl HardenedBrocaScipAdapter {
    pub fn compile_for_text_peer(
        plan: &BrocaTranslationPlan,
        mood_temperature: f32,
        provenance: Provenance,
        policy: &StructuredThoughtScipPolicy,
        limits: &BrocaInterchangeLimits,
    ) -> Result<HardenedBrocaScipPacket, HardenedBrocaError> {
        validate_export(plan, policy, limits)?;
        let packet = StructuredThoughtScipAdapter::compile_for_text_peer(
            plan,
            mood_temperature,
            provenance,
            policy,
        )?;

        let exported_concepts = plan
            .activated_concepts
            .len()
            .min(policy.max_activated_concepts);
        let exported_constraint_texts = if policy.include_constraint_text_for_audit {
            plan.constraints.len()
        } else {
            0
        };
        let original_input_exported = policy.include_original_input && plan.original_input.is_some();

        Ok(HardenedBrocaScipPacket {
            audit: BrocaExportAudit {
                exported_concepts,
                omitted_concepts: plan.activated_concepts.len().saturating_sub(exported_concepts),
                exported_constraint_kinds: plan.constraints.len(),
                exported_constraint_texts,
                omitted_constraint_texts: plan
                    .constraints
                    .len()
                    .saturating_sub(exported_constraint_texts),
                original_input_exported,
                original_input_redacted: !policy.include_original_input
                    && plan.original_input.is_some(),
                structured_data_exported: policy.include_structured_data
                    && plan.structured_data.is_some(),
                domain_context_exported: policy.include_domain_context
                    && plan.domain_context.is_some(),
                semantic_hash: packet.fallback.semantic_hash.clone(),
            },
            packet,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HardenedBrocaError {
    InvalidLimit(String),
    ResourceLimitExceeded(String),
    Interchange(String),
}

impl fmt::Display for HardenedBrocaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimit(value) => write!(f, "invalid Broca interchange limit: {value}"),
            Self::ResourceLimitExceeded(value) => {
                write!(f, "Broca export resource limit exceeded: {value}")
            }
            Self::Interchange(value) => write!(f, "Broca SCIP export failed: {value}"),
        }
    }
}

impl std::error::Error for HardenedBrocaError {}

impl From<BrocaScipError> for HardenedBrocaError {
    fn from(value: BrocaScipError) -> Self {
        Self::Interchange(value.to_string())
    }
}

fn validate_export(
    plan: &BrocaTranslationPlan,
    policy: &StructuredThoughtScipPolicy,
    limits: &BrocaInterchangeLimits,
) -> Result<(), HardenedBrocaError> {
    validate_limits(limits)?;

    if policy.max_activated_concepts > limits.max_activated_concepts {
        return Err(HardenedBrocaError::ResourceLimitExceeded(format!(
            "policy permits {} activated concepts; maximum is {}",
            policy.max_activated_concepts, limits.max_activated_concepts
        )));
    }
    if plan.constraints.len() > limits.max_constraints {
        return Err(HardenedBrocaError::ResourceLimitExceeded(format!(
            "{} constraints; maximum is {}",
            plan.constraints.len(), limits.max_constraints
        )));
    }

    let mut text_bytes = 0usize;
    for concept in plan
        .activated_concepts
        .iter()
        .take(policy.max_activated_concepts)
    {
        charge_string(
            &concept.name,
            "activated concept name",
            limits,
            &mut text_bytes,
        )?;
    }

    if policy.include_constraint_text_for_audit {
        for constraint in &plan.constraints {
            charge_string(
                &constraint.audit_text,
                "constraint audit text",
                limits,
                &mut text_bytes,
            )?;
        }
    }

    if policy.include_original_input
        && let Some(input) = &plan.original_input
    {
        charge_string(input, "original input", limits, &mut text_bytes)?;
    }

    if let Some(grounding_id) = &policy.grounding_id {
        charge_string(grounding_id, "grounding id", limits, &mut text_bytes)?;
    }

    if policy.include_structured_data
        && let Some(data) = &plan.structured_data
    {
        validate_structured_data(data, limits, &mut text_bytes)?;
    }

    if policy.include_domain_context
        && let Some(domain) = &plan.domain_context
    {
        if domain.entities.len() > limits.max_domain_entities {
            return Err(HardenedBrocaError::ResourceLimitExceeded(format!(
                "{} domain entities; maximum is {}",
                domain.entities.len(), limits.max_domain_entities
            )));
        }
        charge_string(&domain.domain, "domain", limits, &mut text_bytes)?;
        for entity in &domain.entities {
            charge_string(
                &entity.entity_type,
                "entity type",
                limits,
                &mut text_bytes,
            )?;
            charge_string(&entity.value, "entity value", limits, &mut text_bytes)?;
        }
        if let Some(answer) = &domain.computed_answer {
            charge_string(answer, "computed answer", limits, &mut text_bytes)?;
        }
    }

    Ok(())
}

fn validate_limits(limits: &BrocaInterchangeLimits) -> Result<(), HardenedBrocaError> {
    if limits.max_activated_concepts == 0
        || limits.max_constraints == 0
        || limits.max_domain_entities == 0
        || limits.max_list_items == 0
        || limits.max_key_value_pairs == 0
        || limits.max_string_bytes == 0
        || limits.max_export_text_bytes == 0
    {
        return Err(HardenedBrocaError::InvalidLimit(
            "all ceilings must be greater than zero".into(),
        ));
    }
    if limits.max_string_bytes > limits.max_export_text_bytes {
        return Err(HardenedBrocaError::InvalidLimit(
            "max_string_bytes cannot exceed max_export_text_bytes".into(),
        ));
    }
    Ok(())
}

fn validate_structured_data(
    data: &BrocaStructuredData,
    limits: &BrocaInterchangeLimits,
    text_bytes: &mut usize,
) -> Result<(), HardenedBrocaError> {
    match data {
        BrocaStructuredData::List(items) => {
            if items.len() > limits.max_list_items {
                return Err(HardenedBrocaError::ResourceLimitExceeded(format!(
                    "{} list items; maximum is {}",
                    items.len(), limits.max_list_items
                )));
            }
            for item in items {
                charge_string(item, "list item", limits, text_bytes)?;
            }
        }
        BrocaStructuredData::KeyValue(pairs) => {
            if pairs.len() > limits.max_key_value_pairs {
                return Err(HardenedBrocaError::ResourceLimitExceeded(format!(
                    "{} key/value pairs; maximum is {}",
                    pairs.len(), limits.max_key_value_pairs
                )));
            }
            for (key, value) in pairs {
                charge_string(key, "structured-data key", limits, text_bytes)?;
                charge_string(value, "structured-data value", limits, text_bytes)?;
            }
        }
        BrocaStructuredData::Numeric { unit, .. } => {
            if let Some(unit) = unit {
                charge_string(unit, "numeric unit", limits, text_bytes)?;
            }
        }
        BrocaStructuredData::Code => {}
    }
    Ok(())
}

fn charge_string(
    value: &str,
    field: &str,
    limits: &BrocaInterchangeLimits,
    total: &mut usize,
) -> Result<(), HardenedBrocaError> {
    if value.len() > limits.max_string_bytes {
        return Err(HardenedBrocaError::ResourceLimitExceeded(format!(
            "{field} is {} bytes; maximum is {}",
            value.len(), limits.max_string_bytes
        )));
    }
    *total = total.checked_add(value.len()).ok_or_else(|| {
        HardenedBrocaError::ResourceLimitExceeded("export text byte count overflowed".into())
    })?;
    if *total > limits.max_export_text_bytes {
        return Err(HardenedBrocaError::ResourceLimitExceeded(format!(
            "export contains {} text bytes; maximum is {}",
            *total, limits.max_export_text_bytes
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BrocaConcept, BrocaConstraint, BrocaConstraintKind, BrocaDomainContext, RendererIntent,
        RendererResponseType, RendererEpistemicStatus,
    };

    fn provenance() -> Provenance {
        Provenance {
            provider: "hardened-broca-test".into(),
            provider_version: "1".into(),
            model_hash: "internal-plan".into(),
            feature_flags: vec![],
            transformations: vec![],
        }
    }

    fn plan() -> BrocaTranslationPlan {
        BrocaTranslationPlan {
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
            structured_data: Some(BrocaStructuredData::List(vec!["alpha".into()])),
            domain_context: Some(BrocaDomainContext {
                domain: "engineering".into(),
                entities: vec![],
                computed_answer: Some("remain offline".into()),
            }),
            constraints: vec![BrocaConstraint {
                kind: BrocaConstraintKind::MustInclude,
                audit_text: "remain offline".into(),
            }],
            original_input: Some("private utterance".into()),
            code_bearing: false,
        }
    }

    #[test]
    fn default_audit_makes_redaction_and_constraint_loss_explicit() {
        let result = HardenedBrocaScipAdapter::compile_for_text_peer(
            &plan(),
            1.0,
            provenance(),
            &StructuredThoughtScipPolicy::default(),
            &BrocaInterchangeLimits::default(),
        )
        .unwrap();
        assert!(result.audit.original_input_redacted);
        assert!(!result.audit.original_input_exported);
        assert_eq!(result.audit.exported_constraint_kinds, 1);
        assert_eq!(result.audit.exported_constraint_texts, 0);
        assert_eq!(result.audit.omitted_constraint_texts, 1);
        assert_eq!(result.audit.semantic_hash, result.packet.fallback.semantic_hash);
    }

    #[test]
    fn huge_redacted_input_does_not_consume_peer_export_budget() {
        let mut plan = plan();
        plan.original_input = Some("x".repeat(100_000));
        let limits = BrocaInterchangeLimits {
            max_string_bytes: 64,
            max_export_text_bytes: 512,
            ..Default::default()
        };
        assert!(HardenedBrocaScipAdapter::compile_for_text_peer(
            &plan,
            1.0,
            provenance(),
            &StructuredThoughtScipPolicy::default(),
            &limits,
        )
        .is_ok());
    }

    #[test]
    fn opted_in_private_input_is_bounded_before_graph_allocation() {
        let mut plan = plan();
        plan.original_input = Some("x".repeat(65));
        let policy = StructuredThoughtScipPolicy {
            include_original_input: true,
            ..Default::default()
        };
        let limits = BrocaInterchangeLimits {
            max_string_bytes: 64,
            max_export_text_bytes: 512,
            ..Default::default()
        };
        assert!(matches!(
            HardenedBrocaScipAdapter::compile_for_text_peer(
                &plan,
                1.0,
                provenance(),
                &policy,
                &limits,
            ),
            Err(HardenedBrocaError::ResourceLimitExceeded(_))
        ));
    }

    #[test]
    fn structured_list_count_is_bounded_before_graph_allocation() {
        let mut plan = plan();
        plan.structured_data = Some(BrocaStructuredData::List(vec!["x".into(); 3]));
        let limits = BrocaInterchangeLimits {
            max_list_items: 2,
            ..Default::default()
        };
        assert!(matches!(
            HardenedBrocaScipAdapter::compile_for_text_peer(
                &plan,
                1.0,
                provenance(),
                &StructuredThoughtScipPolicy::default(),
                &limits,
            ),
            Err(HardenedBrocaError::ResourceLimitExceeded(_))
        ));
    }

    #[test]
    fn policy_cannot_expand_concept_export_past_hard_limit() {
        let policy = StructuredThoughtScipPolicy {
            max_activated_concepts: 4,
            ..Default::default()
        };
        let limits = BrocaInterchangeLimits {
            max_activated_concepts: 3,
            ..Default::default()
        };
        assert!(matches!(
            HardenedBrocaScipAdapter::compile_for_text_peer(
                &plan(),
                1.0,
                provenance(),
                &policy,
                &limits,
            ),
            Err(HardenedBrocaError::ResourceLimitExceeded(_))
        ));
    }

    #[test]
    fn audit_reports_intentional_concept_truncation() {
        let mut plan = plan();
        plan.activated_concepts.extend([
            BrocaConcept {
                name: "pump".into(),
                activation: 0.8,
                relevance: 0.8,
            },
            BrocaConcept {
                name: "valve".into(),
                activation: 0.7,
                relevance: 0.7,
            },
        ]);
        let policy = StructuredThoughtScipPolicy {
            max_activated_concepts: 2,
            ..Default::default()
        };
        let result = HardenedBrocaScipAdapter::compile_for_text_peer(
            &plan,
            1.0,
            provenance(),
            &policy,
            &BrocaInterchangeLimits::default(),
        )
        .unwrap();
        assert_eq!(result.audit.exported_concepts, 2);
        assert_eq!(result.audit.omitted_concepts, 1);
    }
}
