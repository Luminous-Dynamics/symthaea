// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Direct evidence adapter for the cognitive loop's actual `ActivePrimitive` records.
//!
//! Unlike the name-only recovery bridge, this adapter receives the processor's current active
//! records before their activation metadata is lost. It preserves activation strength, typed
//! activation provenance, duration, immutable primitive metadata, and a digest of the exact HDC
//! encoding. None of those facts are silently renamed into the canonical objective axes.

use super::reasoning_context_competition::{ContextCompetitionPolicy, ContextHypothesis};
use super::reasoning_evidence_seeking::{
    plan_with_evidence, EvidenceSeekingPlanReport, EvidenceSeekingPlannerError,
};
use super::reasoning_objective_evidence::{
    CandidateObjectiveEvidence, ObjectiveEvidence, ObjectiveEvidenceError, ObjectiveUnknownReason,
};
use crate::consciousness::{ActivationReason, ActivePrimitive};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use symthaea_core::hdc::primitive_system::{PrimitiveSystem, PrimitiveTier};

pub const ACTIVE_PRIMITIVE_EVIDENCE_ADAPTER_VERSION: &str =
    "rq-006-active-primitive-evidence-adapter-v1";
const OBJECTIVE_SOURCE: &str = "active-primitive-adapter:no-qualified-objective-measurement-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActivePrimitiveActivationEvidence {
    BottomUp { input_similarity: f64 },
    TopDown { goal: String },
    Lateral {
        source_primitive: String,
        affinity: f64,
    },
    Sustained {
        original_reason: Box<ActivePrimitiveActivationEvidence>,
    },
}

impl ActivePrimitiveActivationEvidence {
    fn from_reason(reason: &ActivationReason) -> Self {
        match reason {
            ActivationReason::BottomUp { input_similarity } => Self::BottomUp {
                input_similarity: *input_similarity,
            },
            ActivationReason::TopDown { goal } => Self::TopDown { goal: goal.clone() },
            ActivationReason::Lateral {
                source_primitive,
                affinity,
            } => Self::Lateral {
                source_primitive: source_primitive.clone(),
                affinity: *affinity,
            },
            ActivationReason::Sustained { original_reason } => Self::Sustained {
                original_reason: Box::new(Self::from_reason(original_reason)),
            },
        }
    }

    fn validate(&self) -> Result<(), ActivePrimitiveEvidenceError> {
        match self {
            Self::BottomUp { input_similarity } => {
                validate_unit("activation_reason.input_similarity", *input_similarity)
            }
            Self::TopDown { goal } => require_nonempty("activation_reason.goal", goal),
            Self::Lateral {
                source_primitive,
                affinity,
            } => {
                require_nonempty("activation_reason.source_primitive", source_primitive)?;
                validate_unit("activation_reason.affinity", *affinity)
            }
            Self::Sustained { original_reason } => original_reason.validate(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedPrimitiveMetadata {
    pub name: String,
    pub tier: PrimitiveTier,
    pub domain: String,
    pub definition: String,
    pub is_base: bool,
    pub derivation: Option<String>,
    /// BLAKE3 digest of the exact 16,384-bit primitive encoding.
    pub encoding_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivePrimitiveIdentityAudit {
    pub registry_found: bool,
    pub name_matches_registry: Option<bool>,
    pub tier_matches_registry: Option<bool>,
    pub domain_matches_registry: Option<bool>,
    pub definition_matches_registry: Option<bool>,
    pub encoding_matches_registry: Option<bool>,
    pub base_status_matches_registry: Option<bool>,
    pub derivation_matches_registry: Option<bool>,
    pub immutable_identity_preserved: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivePrimitiveEvidenceProfile {
    pub adapter_version: String,
    pub candidate_id: String,
    pub observed_primitive: ObservedPrimitiveMetadata,
    pub activation: f64,
    pub activation_reason: ActivePrimitiveActivationEvidence,
    pub duration: usize,
    pub identity_audit: ActivePrimitiveIdentityAudit,
    /// Policy objectives remain unknown until a qualified objective-specific adapter measures them.
    pub objective_evidence: CandidateObjectiveEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivePrimitiveEvidenceReport {
    pub adapter_version: String,
    pub profiles: Vec<ActivePrimitiveEvidenceProfile>,
    pub registry_hits: usize,
    pub registry_misses: usize,
    pub immutable_identity_mismatches: usize,
}

impl ActivePrimitiveEvidenceReport {
    pub fn candidate_objective_evidence(&self) -> Vec<CandidateObjectiveEvidence> {
        self.profiles
            .iter()
            .map(|profile| profile.objective_evidence.clone())
            .collect()
    }
}

/// Preserve the exact active records corresponding to the legacy candidate identities.
/// Candidate order is retained so later shadow comparison can bind both paths to the same set.
pub fn adapt_active_primitive_evidence(
    active: &[ActivePrimitive],
    candidate_ids: &[String],
) -> Result<ActivePrimitiveEvidenceReport, ActivePrimitiveEvidenceError> {
    if candidate_ids.is_empty() {
        return Err(ActivePrimitiveEvidenceError::EmptyCandidateSet);
    }

    let mut requested = HashSet::with_capacity(candidate_ids.len());
    for candidate_id in candidate_ids {
        require_nonempty("candidate_id", candidate_id)?;
        if !requested.insert(candidate_id.as_str()) {
            return Err(ActivePrimitiveEvidenceError::DuplicateCandidateId(
                candidate_id.clone(),
            ));
        }
    }

    let mut active_by_name: HashMap<&str, &ActivePrimitive> = HashMap::with_capacity(active.len());
    for primitive in active {
        let name = primitive.primitive.name.as_str();
        if active_by_name.insert(name, primitive).is_some() {
            return Err(ActivePrimitiveEvidenceError::DuplicateActivePrimitive(
                name.to_owned(),
            ));
        }
    }

    let registry = PrimitiveSystem::global();
    let mut profiles = Vec::with_capacity(candidate_ids.len());
    let mut registry_hits = 0usize;
    let mut registry_misses = 0usize;
    let mut immutable_identity_mismatches = 0usize;

    for candidate_id in candidate_ids {
        let active_primitive = active_by_name
            .get(candidate_id.as_str())
            .copied()
            .ok_or_else(|| ActivePrimitiveEvidenceError::MissingActivePrimitive(candidate_id.clone()))?;
        validate_unit("activation", active_primitive.activation)?;
        let activation_reason =
            ActivePrimitiveActivationEvidence::from_reason(&active_primitive.activation_reason);
        activation_reason.validate()?;

        let observed_primitive = metadata_from_primitive(&active_primitive.primitive);
        let identity_audit = if let Some(registry_primitive) = registry.get(candidate_id) {
            registry_hits += 1;
            let registry_metadata = metadata_from_primitive(registry_primitive);
            let name_matches_registry = observed_primitive.name == registry_metadata.name;
            let tier_matches_registry = observed_primitive.tier == registry_metadata.tier;
            let domain_matches_registry = observed_primitive.domain == registry_metadata.domain;
            let definition_matches_registry =
                observed_primitive.definition == registry_metadata.definition;
            let encoding_matches_registry =
                observed_primitive.encoding_digest == registry_metadata.encoding_digest;
            let base_status_matches_registry = observed_primitive.is_base == registry_metadata.is_base;
            let derivation_matches_registry = observed_primitive.derivation == registry_metadata.derivation;
            let immutable_identity_preserved = name_matches_registry
                && tier_matches_registry
                && domain_matches_registry
                && definition_matches_registry
                && encoding_matches_registry
                && base_status_matches_registry
                && derivation_matches_registry;
            if !immutable_identity_preserved {
                immutable_identity_mismatches += 1;
            }
            ActivePrimitiveIdentityAudit {
                registry_found: true,
                name_matches_registry: Some(name_matches_registry),
                tier_matches_registry: Some(tier_matches_registry),
                domain_matches_registry: Some(domain_matches_registry),
                definition_matches_registry: Some(definition_matches_registry),
                encoding_matches_registry: Some(encoding_matches_registry),
                base_status_matches_registry: Some(base_status_matches_registry),
                derivation_matches_registry: Some(derivation_matches_registry),
                immutable_identity_preserved: Some(immutable_identity_preserved),
            }
        } else {
            registry_misses += 1;
            ActivePrimitiveIdentityAudit {
                registry_found: false,
                name_matches_registry: None,
                tier_matches_registry: None,
                domain_matches_registry: None,
                definition_matches_registry: None,
                encoding_matches_registry: None,
                base_status_matches_registry: None,
                derivation_matches_registry: None,
                immutable_identity_preserved: None,
            }
        };

        let objective_evidence = CandidateObjectiveEvidence {
            candidate_id: candidate_id.clone(),
            candidate_label: active_primitive.primitive.name.clone(),
            integration_proxy: unknown_objective()?,
            harmonic_alignment: unknown_objective()?,
            epistemic_grounding: unknown_objective()?,
        };

        profiles.push(ActivePrimitiveEvidenceProfile {
            adapter_version: ACTIVE_PRIMITIVE_EVIDENCE_ADAPTER_VERSION.into(),
            candidate_id: candidate_id.clone(),
            observed_primitive,
            activation: active_primitive.activation,
            activation_reason,
            duration: active_primitive.duration,
            identity_audit,
            objective_evidence,
        });
    }

    Ok(ActivePrimitiveEvidenceReport {
        adapter_version: ACTIVE_PRIMITIVE_EVIDENCE_ADAPTER_VERSION.into(),
        profiles,
        registry_hits,
        registry_misses,
        immutable_identity_mismatches,
    })
}

pub fn plan_active_primitive_evidence(
    hypotheses: &[ContextHypothesis],
    context_policy: ContextCompetitionPolicy,
    active: &[ActivePrimitive],
    candidate_ids: &[String],
) -> Result<(ActivePrimitiveEvidenceReport, EvidenceSeekingPlanReport), ActivePrimitiveEvidenceError> {
    let report = adapt_active_primitive_evidence(active, candidate_ids)?;
    let objective_evidence = report.candidate_objective_evidence();
    let plan = plan_with_evidence(hypotheses, context_policy, &objective_evidence)?;
    Ok((report, plan))
}

fn metadata_from_primitive(
    primitive: &symthaea_core::hdc::primitive_system::Primitive,
) -> ObservedPrimitiveMetadata {
    ObservedPrimitiveMetadata {
        name: primitive.name.clone(),
        tier: primitive.tier,
        domain: primitive.domain.clone(),
        definition: primitive.definition.clone(),
        is_base: primitive.is_base,
        derivation: primitive.derivation.clone(),
        encoding_digest: blake3::hash(&primitive.encoding.0).to_hex().to_string(),
    }
}

fn unknown_objective() -> Result<ObjectiveEvidence, ObjectiveEvidenceError> {
    ObjectiveEvidence::unknown(
        OBJECTIVE_SOURCE,
        Vec::new(),
        ObjectiveUnknownReason::NotMeasured,
    )
}

fn require_nonempty(
    field: &'static str,
    value: &str,
) -> Result<(), ActivePrimitiveEvidenceError> {
    if value.trim().is_empty() {
        Err(ActivePrimitiveEvidenceError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_unit(field: &'static str, value: f64) -> Result<(), ActivePrimitiveEvidenceError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ActivePrimitiveEvidenceError::InvalidUnitValue { field, value })
    }
}

#[derive(Debug)]
pub enum ActivePrimitiveEvidenceError {
    EmptyField(&'static str),
    EmptyCandidateSet,
    DuplicateCandidateId(String),
    DuplicateActivePrimitive(String),
    MissingActivePrimitive(String),
    InvalidUnitValue { field: &'static str, value: f64 },
    Objective(ObjectiveEvidenceError),
    Planner(EvidenceSeekingPlannerError),
}

impl fmt::Display for ActivePrimitiveEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "required field `{field}` is empty"),
            Self::EmptyCandidateSet => write!(f, "active primitive adapter requires candidates"),
            Self::DuplicateCandidateId(id) => write!(f, "candidate id `{id}` is duplicated"),
            Self::DuplicateActivePrimitive(id) => {
                write!(f, "active primitive `{id}` appears more than once")
            }
            Self::MissingActivePrimitive(id) => {
                write!(f, "candidate `{id}` has no corresponding current ActivePrimitive")
            }
            Self::InvalidUnitValue { field, value } => {
                write!(f, "`{field}` must be finite and within [0, 1], got {value}")
            }
            Self::Objective(err) => write!(f, "objective evidence error: {err}"),
            Self::Planner(err) => write!(f, "active primitive V3 planning failed: {err}"),
        }
    }
}

impl std::error::Error for ActivePrimitiveEvidenceError {}

impl From<ObjectiveEvidenceError> for ActivePrimitiveEvidenceError {
    fn from(value: ObjectiveEvidenceError) -> Self {
        Self::Objective(value)
    }
}

impl From<EvidenceSeekingPlannerError> for ActivePrimitiveEvidenceError {
    fn from(value: EvidenceSeekingPlannerError) -> Self {
        Self::Planner(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::reasoning_evidence_seeking::{EvidenceRequestKind, EvidenceSeekingOutcome};
    use crate::consciousness::context_aware_evolution::ReasoningContext;

    fn active(name: &str, activation: f64) -> ActivePrimitive {
        let primitive = PrimitiveSystem::global()
            .get(name)
            .unwrap_or_else(|| panic!("fixture primitive `{name}` must exist"))
            .clone();
        ActivePrimitive {
            primitive,
            activation,
            activation_reason: ActivationReason::BottomUp {
                input_similarity: activation,
            },
            duration: 3,
        }
    }

    fn hypothesis() -> ContextHypothesis {
        ContextHypothesis {
            context: ReasoningContext::ScientificReasoning,
            support: 0.9,
            source: "fixture-context".into(),
            evidence_refs: vec!["query".into()],
        }
    }

    #[test]
    fn actual_activation_and_encoding_identity_are_preserved() {
        let actives = [active("NSM_KNOW", 0.73)];
        let report = adapt_active_primitive_evidence(&actives, &["NSM_KNOW".into()]).unwrap();
        let profile = &report.profiles[0];
        assert_eq!(profile.activation, 0.73);
        assert_eq!(profile.duration, 3);
        assert!(matches!(
            profile.activation_reason,
            ActivePrimitiveActivationEvidence::BottomUp { input_similarity } if input_similarity == 0.73
        ));
        assert!(!profile.observed_primitive.encoding_digest.is_empty());
        assert_eq!(profile.identity_audit.immutable_identity_preserved, Some(true));
    }

    #[test]
    fn active_metadata_still_does_not_become_policy_objective_evidence() {
        let actives = [active("NSM_KNOW", 0.73)];
        let report = adapt_active_primitive_evidence(&actives, &["NSM_KNOW".into()]).unwrap();
        assert_eq!(report.profiles[0].objective_evidence.observed_axes(), 0);
    }

    #[test]
    fn registry_mismatch_is_explicit() {
        let mut altered = active("NSM_KNOW", 0.73);
        altered.primitive.definition.push_str(" altered");
        let report = adapt_active_primitive_evidence(&[altered], &["NSM_KNOW".into()]).unwrap();
        assert_eq!(report.immutable_identity_mismatches, 1);
        assert_eq!(
            report.profiles[0].identity_audit.immutable_identity_preserved,
            Some(false)
        );
        assert_eq!(
            report.profiles[0].identity_audit.definition_matches_registry,
            Some(false)
        );
    }

    #[test]
    fn exact_candidate_set_rejects_missing_active_identity() {
        let actives = [active("NSM_KNOW", 0.73)];
        assert!(matches!(
            adapt_active_primitive_evidence(&actives, &["NSM_DO".into()]),
            Err(ActivePrimitiveEvidenceError::MissingActivePrimitive(_))
        ));
    }

    #[test]
    fn v3_requests_objective_measurements_from_actual_active_records() {
        let actives = [active("NSM_KNOW", 0.73), active("NSM_DO", 0.61)];
        let candidate_ids = vec!["NSM_KNOW".into(), "NSM_DO".into()];
        let (report, plan) = plan_active_primitive_evidence(
            &[hypothesis()],
            ContextCompetitionPolicy::development_v1(),
            &actives,
            &candidate_ids,
        )
        .unwrap();
        assert_eq!(report.profiles.len(), 2);
        let EvidenceSeekingOutcome::NeedEvidence { requests, .. } = plan.outcome else {
            panic!("expected NeedEvidence");
        };
        assert_eq!(requests.len(), 6);
        assert!(requests.iter().all(|request| matches!(
            &request.kind,
            EvidenceRequestKind::ObjectiveMeasurement { .. }
        )));
    }
}
