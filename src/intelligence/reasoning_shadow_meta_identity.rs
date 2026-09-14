// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Identity-preserving compatibility layer for the live matched meta-reasoning shadow.
//!
//! The cognitive loop historically constructs lightweight `CandidatePrimitive` values from active
//! primitive names, but those values may carry placeholder tier/definition/HDC encodings. The
//! matched-evidence shadow measures the processor's exact `ActivePrimitive` records. Same-name
//! objects must not therefore be assumed to share immutable identity.
//!
//! This layer resolves that boundary without changing the legacy scoring inputs. Immediately before
//! the active-evidence path is delegated, each uniquely matched candidate receives the live
//! primitive's tier, definition, and exact HDC encoding. Fitness, epistemic coordinate, harmonic
//! alignment, ordering, and candidate name are preserved exactly.
//!
//! Missing or duplicate live identities are never guessed. Such candidates remain unchanged so the
//! inner identity/evidence audits continue to fail closed while historical reasoning remains
//! available.

use super::reasoning_shadow_meta::{
    ShadowMetaObservation, ShadowMetaStats, V3ShadowObservation,
};
use super::reasoning_shadow_meta_matched::{
    MatchedShadowQualifiedMetaReasoner, MatchedShadowStats, MatchedV3ShadowObservation,
};
use crate::consciousness::meta_reasoning::{
    MetaCognitiveReasoner, MetaReasoningConfig, MetaReasoningResult,
};
use crate::consciousness::primitive_evolution::{CandidatePrimitive, EvolutionConfig};
use crate::consciousness::primitive_reasoning::ReasoningChain;
use crate::consciousness::ActivePrimitive;
use anyhow::Result;
use std::collections::HashMap;

pub const LIVE_META_IDENTITY_BINDING_VERSION: &str = "rq-006y-live-meta-identity-binding-v1";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CandidateIdentityBindingReport {
    pub binding_version: String,
    pub attempted: usize,
    pub bound: usize,
    pub missing: usize,
    pub ambiguous: usize,
}

impl CandidateIdentityBindingReport {
    pub fn fully_bound(&self) -> bool {
        self.attempted > 0
            && self.bound == self.attempted
            && self.missing == 0
            && self.ambiguous == 0
    }
}

/// Public compatibility wrapper used by the cognitive loop.
///
/// All reasoning authority remains in `inner`; this layer only repairs the identity representation
/// of candidates immediately before the active-evidence call.
pub struct IdentityBoundShadowQualifiedMetaReasoner {
    inner: MatchedShadowQualifiedMetaReasoner,
    last_identity_binding_report: Option<CandidateIdentityBindingReport>,
}

impl IdentityBoundShadowQualifiedMetaReasoner {
    pub fn new(
        evolution_config: EvolutionConfig,
        meta_config: MetaReasoningConfig,
    ) -> Result<Self> {
        Ok(Self {
            inner: MatchedShadowQualifiedMetaReasoner::new(evolution_config, meta_config)?,
            last_identity_binding_report: None,
        })
    }

    /// Compatibility path without an `ActivePrimitive` snapshot. No identity repair is attempted.
    pub fn meta_reason(
        &mut self,
        query: &str,
        primitives: Vec<CandidatePrimitive>,
        chain: &mut ReasoningChain,
    ) -> Result<MetaReasoningResult> {
        self.last_identity_binding_report = None;
        self.inner.meta_reason(query, primitives, chain)
    }

    /// Bind every uniquely resolvable candidate to the exact live primitive identity before
    /// delegating. Scoring inputs are deliberately preserved byte-for-byte/value-for-value.
    pub fn meta_reason_with_active_evidence(
        &mut self,
        query: &str,
        primitives: Vec<CandidatePrimitive>,
        active_primitives: &[ActivePrimitive],
        chain: &mut ReasoningChain,
    ) -> Result<MetaReasoningResult> {
        let (bound_primitives, report) =
            bind_candidate_identities(primitives, active_primitives);
        self.last_identity_binding_report = Some(report);
        self.inner.meta_reason_with_active_evidence(
            query,
            bound_primitives,
            active_primitives,
            chain,
        )
    }

    pub fn legacy(&self) -> &MetaCognitiveReasoner {
        self.inner.legacy()
    }

    pub fn shadow_stats(&self) -> &ShadowMetaStats {
        self.inner.shadow_stats()
    }

    pub fn last_shadow_observation(&self) -> Option<&ShadowMetaObservation> {
        self.inner.last_shadow_observation()
    }

    pub fn last_v3_shadow_observation(&self) -> Option<&V3ShadowObservation> {
        self.inner.last_v3_shadow_observation()
    }

    pub fn canonical_next_sequence(&self) -> u64 {
        self.inner.canonical_next_sequence()
    }

    pub fn matched_shadow_stats(&self) -> &MatchedShadowStats {
        self.inner.matched_shadow_stats()
    }

    pub fn last_matched_v3_shadow_observation(&self) -> Option<&MatchedV3ShadowObservation> {
        self.inner.last_matched_v3_shadow_observation()
    }

    pub fn last_identity_binding_report(&self) -> Option<&CandidateIdentityBindingReport> {
        self.last_identity_binding_report.as_ref()
    }
}

fn bind_candidate_identities(
    mut candidates: Vec<CandidatePrimitive>,
    active_primitives: &[ActivePrimitive],
) -> (Vec<CandidatePrimitive>, CandidateIdentityBindingReport) {
    let mut active_by_name: HashMap<&str, Option<&ActivePrimitive>> = HashMap::new();
    for active in active_primitives {
        let name = active.primitive.name.as_str();
        active_by_name
            .entry(name)
            .and_modify(|slot| *slot = None)
            .or_insert(Some(active));
    }

    let mut report = CandidateIdentityBindingReport {
        binding_version: LIVE_META_IDENTITY_BINDING_VERSION.into(),
        attempted: candidates.len(),
        ..CandidateIdentityBindingReport::default()
    };

    for candidate in &mut candidates {
        match active_by_name.get(candidate.name.as_str()) {
            Some(Some(active)) => {
                // These are identity-bearing fields only. The legacy optimizer's candidate scoring
                // inputs are intentionally left unchanged.
                candidate.tier = active.primitive.tier;
                candidate.definition = active.primitive.definition.clone();
                candidate.encoding = active.primitive.encoding;
                report.bound += 1;
            }
            Some(None) => report.ambiguous += 1,
            None => report.missing += 1,
        }
    }

    (candidates, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::epistemic_tiers::EpistemicCoordinate;
    use crate::consciousness::ActivationReason;
    use crate::hdc::BinaryHV;
    use symthaea_core::hdc::primitive_system::{PrimitiveSystem, PrimitiveTier};

    fn active(name: &str, activation: f64) -> ActivePrimitive {
        ActivePrimitive {
            primitive: PrimitiveSystem::global()
                .get(name)
                .unwrap_or_else(|| panic!("fixture primitive `{name}` must exist"))
                .clone(),
            activation,
            activation_reason: ActivationReason::BottomUp {
                input_similarity: activation,
            },
            duration: 2,
        }
    }

    fn placeholder_candidate(name: &str) -> CandidatePrimitive {
        CandidatePrimitive {
            name: name.into(),
            tier: PrimitiveTier::NSM,
            definition: name.into(),
            fitness: 0.731,
            encoding: BinaryHV::random(42),
            epistemic_coordinate: EpistemicCoordinate::axiom(),
            harmonic_alignment: 0.617,
        }
    }

    #[test]
    fn exact_live_identity_replaces_only_identity_fields() {
        let original = placeholder_candidate("NSM_KNOW");
        let original_fitness = original.fitness;
        let original_epistemic = original.epistemic_coordinate;
        let original_harmonic = original.harmonic_alignment;
        let actives = [active("NSM_KNOW", 0.8)];

        let (bound, report) = bind_candidate_identities(vec![original], &actives);
        let candidate = &bound[0];
        let live = &actives[0].primitive;

        assert_eq!(report.attempted, 1);
        assert_eq!(report.bound, 1);
        assert!(report.fully_bound());
        assert_eq!(candidate.name, live.name);
        assert_eq!(candidate.tier, live.tier);
        assert_eq!(candidate.definition, live.definition);
        assert_eq!(candidate.encoding, live.encoding);
        assert_eq!(candidate.fitness, original_fitness);
        assert_eq!(candidate.epistemic_coordinate, original_epistemic);
        assert_eq!(candidate.harmonic_alignment, original_harmonic);
    }

    #[test]
    fn missing_live_identity_is_not_invented() {
        let original = placeholder_candidate("NSM_DO");
        let original_encoding = original.encoding;
        let actives = [active("NSM_KNOW", 0.8)];

        let (bound, report) = bind_candidate_identities(vec![original], &actives);
        assert_eq!(report.bound, 0);
        assert_eq!(report.missing, 1);
        assert!(!report.fully_bound());
        assert_eq!(bound[0].encoding, original_encoding);
    }

    #[test]
    fn duplicate_live_identity_is_ambiguous_not_arbitrarily_selected() {
        let original = placeholder_candidate("NSM_KNOW");
        let original_encoding = original.encoding;
        let mut second = active("NSM_KNOW", 0.7);
        second.primitive.encoding = BinaryHV::random(99_991);
        let actives = [active("NSM_KNOW", 0.8), second];

        let (bound, report) = bind_candidate_identities(vec![original], &actives);
        assert_eq!(report.bound, 0);
        assert_eq!(report.ambiguous, 1);
        assert!(!report.fully_bound());
        assert_eq!(bound[0].encoding, original_encoding);
    }

    #[test]
    fn live_path_repairs_placeholder_identity_before_matched_shadow() {
        let mut reasoner = IdentityBoundShadowQualifiedMetaReasoner::new(
            EvolutionConfig::default(),
            MetaReasoningConfig::default(),
        )
        .unwrap();
        let actives = [active("NSM_KNOW", 0.8), active("NSM_DO", 0.6)];
        let mut chain = ReasoningChain::new(BinaryHV::random(22_500));

        let result = reasoner.meta_reason_with_active_evidence(
            "evidence experiment research theory scientific",
            vec![
                placeholder_candidate("NSM_KNOW"),
                placeholder_candidate("NSM_DO"),
            ],
            &actives,
            &mut chain,
        );

        assert!(result.is_ok());
        let binding = reasoner.last_identity_binding_report().unwrap();
        assert!(binding.fully_bound());
        assert_eq!(binding.bound, 2);
        let matched = reasoner.last_matched_v3_shadow_observation().unwrap();
        assert_eq!(matched.cross_plane_identity_mismatches, 0);
        assert_ne!(matched.matched_outcome, super::super::reasoning_shadow_meta::V3ShadowOutcomeKind::Error);
    }
}
