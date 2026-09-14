// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence bridge from the live reconstructed meta-reasoning candidates to V3.
//!
//! The current cognitive-loop boundary carries only active primitive names into the later
//! meta-reasoning phase. The legacy builder then reconstructs `CandidatePrimitive` values with
//! neutral/default objective fields. This module recovers only what can actually be recovered from
//! the canonical primitive registry and marks everything else unknown.
//!
//! Registry metadata is identity evidence, not objective evidence. In particular, active-set
//! membership does not establish integration fitness, harmonic alignment, or epistemic grounding.

use super::reasoning_context_competition::{ContextCompetitionPolicy, ContextHypothesis};
use super::reasoning_evidence_seeking::{
    plan_with_evidence, EvidenceSeekingPlanReport, EvidenceSeekingPlannerError,
};
use super::reasoning_objective_evidence::{
    CandidateObjectiveEvidence, ObjectiveEvidence, ObjectiveEvidenceError, ObjectiveUnknownReason,
};
use crate::consciousness::primitive_evolution::CandidatePrimitive;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use symthaea_core::hdc::primitive_system::{PrimitiveSystem, PrimitiveTier};

pub const LIVE_PRIMITIVE_EVIDENCE_BRIDGE_VERSION: &str =
    "rq-006-live-primitive-evidence-bridge-v1";
const LIVE_CANDIDATE_SOURCE: &str = "cognitive-loop-live-candidate-v1";
const REGISTRY_SOURCE: &str = "canonical-primitive-registry-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryPrimitiveMetadata {
    pub name: String,
    pub tier: PrimitiveTier,
    pub domain: String,
    pub definition: String,
    pub is_base: bool,
    pub derivation: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveCandidateIdentityAudit {
    pub registry_found: bool,
    pub tier_matches_registry: Option<bool>,
    pub definition_matches_registry: Option<bool>,
    pub immutable_identity_preserved: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LivePrimitiveEvidenceProfile {
    pub bridge_version: String,
    pub candidate_id: String,
    pub candidate_label: String,
    /// Candidate inclusion proves only that this name reached the live meta-reasoning candidate set.
    pub active_membership_observed: bool,
    /// The earlier `ActivePrimitive.activation` value was discarded by the phase boundary.
    pub activation_strength: Option<f64>,
    /// The earlier `ActivePrimitive.activation_reason` was discarded by the phase boundary.
    pub activation_reason_available: bool,
    /// The earlier `ActivePrimitive.duration` was discarded by the phase boundary.
    pub duration: Option<usize>,
    pub registry_metadata: Option<RegistryPrimitiveMetadata>,
    pub identity_audit: LiveCandidateIdentityAudit,
    pub objective_evidence: CandidateObjectiveEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LivePrimitiveEvidenceReport {
    pub bridge_version: String,
    pub profiles: Vec<LivePrimitiveEvidenceProfile>,
    pub registry_hits: usize,
    pub registry_misses: usize,
    pub immutable_identity_mismatches: usize,
}

impl LivePrimitiveEvidenceReport {
    pub fn candidate_objective_evidence(&self) -> Vec<CandidateObjectiveEvidence> {
        self.profiles
            .iter()
            .map(|profile| profile.objective_evidence.clone())
            .collect()
    }
}

/// Recover immutable registry identity without pretending discarded live measurements still exist.
pub fn recover_live_primitive_evidence(
    candidates: &[CandidatePrimitive],
) -> Result<LivePrimitiveEvidenceReport, LivePrimitiveEvidenceError> {
    if candidates.is_empty() {
        return Err(LivePrimitiveEvidenceError::EmptyCandidateSet);
    }
    let registry = PrimitiveSystem::global();
    let mut seen = HashSet::with_capacity(candidates.len());
    let mut profiles = Vec::with_capacity(candidates.len());
    let mut registry_hits = 0usize;
    let mut registry_misses = 0usize;
    let mut immutable_identity_mismatches = 0usize;

    for candidate in candidates {
        if candidate.name.trim().is_empty() {
            return Err(LivePrimitiveEvidenceError::EmptyCandidateId);
        }
        if !seen.insert(candidate.name.as_str()) {
            return Err(LivePrimitiveEvidenceError::DuplicateCandidateId(
                candidate.name.clone(),
            ));
        }

        let (registry_metadata, identity_audit) = if let Some(primitive) = registry.get(&candidate.name)
        {
            registry_hits += 1;
            let tier_matches_registry = candidate.tier == primitive.tier;
            let definition_matches_registry = candidate.definition == primitive.definition;
            let immutable_identity_preserved = tier_matches_registry && definition_matches_registry;
            if !immutable_identity_preserved {
                immutable_identity_mismatches += 1;
            }
            (
                Some(RegistryPrimitiveMetadata {
                    name: primitive.name.clone(),
                    tier: primitive.tier,
                    domain: primitive.domain.clone(),
                    definition: primitive.definition.clone(),
                    is_base: primitive.is_base,
                    derivation: primitive.derivation.clone(),
                }),
                LiveCandidateIdentityAudit {
                    registry_found: true,
                    tier_matches_registry: Some(tier_matches_registry),
                    definition_matches_registry: Some(definition_matches_registry),
                    immutable_identity_preserved: Some(immutable_identity_preserved),
                },
            )
        } else {
            registry_misses += 1;
            (
                None,
                LiveCandidateIdentityAudit {
                    registry_found: false,
                    tier_matches_registry: None,
                    definition_matches_registry: None,
                    immutable_identity_preserved: None,
                },
            )
        };

        // The live candidate object's numeric fields are not accepted as measurements here:
        // current production construction assigns one global unified_psi to every candidate,
        // neutral 0.5 harmonic alignment, and a default epistemic coordinate. Those are exactly the
        // placeholders this bridge is meant to stop laundering into evidence.
        let objective_evidence = CandidateObjectiveEvidence {
            candidate_id: candidate.name.clone(),
            candidate_label: candidate.name.clone(),
            integration_proxy: unknown_objective("integration proxy")?,
            harmonic_alignment: unknown_objective("harmonic alignment")?,
            epistemic_grounding: unknown_objective("epistemic grounding")?,
        };

        profiles.push(LivePrimitiveEvidenceProfile {
            bridge_version: LIVE_PRIMITIVE_EVIDENCE_BRIDGE_VERSION.into(),
            candidate_id: candidate.name.clone(),
            candidate_label: candidate.name.clone(),
            active_membership_observed: true,
            activation_strength: None,
            activation_reason_available: false,
            duration: None,
            registry_metadata,
            identity_audit,
            objective_evidence,
        });
    }

    Ok(LivePrimitiveEvidenceReport {
        bridge_version: LIVE_PRIMITIVE_EVIDENCE_BRIDGE_VERSION.into(),
        profiles,
        registry_hits,
        registry_misses,
        immutable_identity_mismatches,
    })
}

/// Run the V3 planner over the truthful evidence surface recoverable from today's live candidate
/// boundary. This is a measurement/probe API and has no behavior authority.
pub fn plan_live_primitive_evidence(
    hypotheses: &[ContextHypothesis],
    context_policy: ContextCompetitionPolicy,
    candidates: &[CandidatePrimitive],
) -> Result<(LivePrimitiveEvidenceReport, EvidenceSeekingPlanReport), LivePrimitiveEvidenceError> {
    let evidence_report = recover_live_primitive_evidence(candidates)?;
    let objective_evidence = evidence_report.candidate_objective_evidence();
    let plan = plan_with_evidence(hypotheses, context_policy, &objective_evidence)?;
    Ok((evidence_report, plan))
}

fn unknown_objective(label: &'static str) -> Result<ObjectiveEvidence, ObjectiveEvidenceError> {
    ObjectiveEvidence::unknown(
        format!("{LIVE_CANDIDATE_SOURCE}:{label}:{REGISTRY_SOURCE}"),
        Vec::new(),
        ObjectiveUnknownReason::NotMeasured,
    )
}

#[derive(Debug)]
pub enum LivePrimitiveEvidenceError {
    EmptyCandidateSet,
    EmptyCandidateId,
    DuplicateCandidateId(String),
    Objective(ObjectiveEvidenceError),
    Planner(EvidenceSeekingPlannerError),
}

impl fmt::Display for LivePrimitiveEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCandidateSet => write!(f, "live primitive evidence requires candidates"),
            Self::EmptyCandidateId => write!(f, "live primitive candidate id cannot be empty"),
            Self::DuplicateCandidateId(id) => write!(f, "live primitive candidate id `{id}` is duplicated"),
            Self::Objective(err) => write!(f, "live primitive objective evidence is malformed: {err}"),
            Self::Planner(err) => write!(f, "live primitive V3 planning failed: {err}"),
        }
    }
}

impl std::error::Error for LivePrimitiveEvidenceError {}

impl From<ObjectiveEvidenceError> for LivePrimitiveEvidenceError {
    fn from(value: ObjectiveEvidenceError) -> Self {
        Self::Objective(value)
    }
}

impl From<EvidenceSeekingPlannerError> for LivePrimitiveEvidenceError {
    fn from(value: EvidenceSeekingPlannerError) -> Self {
        Self::Planner(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::context_aware_evolution::ReasoningContext;
    use crate::consciousness::epistemic_tiers::EpistemicCoordinate;
    use crate::hdc::BinaryHV;
    use super::super::reasoning_context_competition::ContextHypothesis;
    use super::super::reasoning_evidence_seeking::{EvidenceRequestKind, EvidenceSeekingOutcome};

    fn live_candidate(name: &str) -> CandidatePrimitive {
        CandidatePrimitive {
            name: name.into(),
            tier: PrimitiveTier::NSM,
            definition: name.into(),
            fitness: 0.7,
            encoding: BinaryHV::random(42),
            epistemic_coordinate: EpistemicCoordinate::default(),
            harmonic_alignment: 0.5,
        }
    }

    fn scientific_hypothesis() -> ContextHypothesis {
        ContextHypothesis {
            context: ReasoningContext::ScientificReasoning,
            support: 0.9,
            source: "fixture-context".into(),
            evidence_refs: vec!["query".into()],
        }
    }

    #[test]
    fn registry_metadata_does_not_become_objective_evidence() {
        let report = recover_live_primitive_evidence(&[live_candidate("NSM_KNOW")]).unwrap();
        assert_eq!(report.profiles.len(), 1);
        let profile = &report.profiles[0];
        assert!(profile.active_membership_observed);
        assert!(profile.activation_strength.is_none());
        assert_eq!(profile.objective_evidence.observed_axes(), 0);
    }

    #[test]
    fn legacy_reconstruction_is_audited_against_registry_identity() {
        let report = recover_live_primitive_evidence(&[live_candidate("NSM_KNOW")]).unwrap();
        let profile = &report.profiles[0];
        assert!(profile.identity_audit.registry_found);
        // Whether the current placeholder happens to match a specific primitive is not assumed;
        // the important theorem is that the comparison is explicit and counted.
        assert_eq!(
            report.immutable_identity_mismatches,
            usize::from(profile.identity_audit.immutable_identity_preserved == Some(false))
        );
    }

    #[test]
    fn multiple_live_candidates_request_measurements_instead_of_using_placeholders() {
        let candidates = [live_candidate("NSM_KNOW"), live_candidate("NSM_DO")];
        let (_evidence, plan) = plan_live_primitive_evidence(
            &[scientific_hypothesis()],
            ContextCompetitionPolicy::development_v1(),
            &candidates,
        )
        .unwrap();
        let EvidenceSeekingOutcome::NeedEvidence { requests, .. } = plan.outcome else {
            panic!("expected NeedEvidence");
        };
        assert_eq!(requests.len(), 6);
        assert!(requests.iter().all(|request| matches!(
            request.kind,
            EvidenceRequestKind::ObjectiveMeasurement { .. }
        )));
    }

    #[test]
    fn duplicate_live_candidate_identity_fails_closed() {
        let candidates = [live_candidate("NSM_KNOW"), live_candidate("NSM_KNOW")];
        assert!(matches!(
            recover_live_primitive_evidence(&candidates),
            Err(LivePrimitiveEvidenceError::DuplicateCandidateId(_))
        ));
    }
}
